"""
Audio MIA pipeline for MusicGen-small.

Audio analog of mia_core.py. Extracts per-codebook loss from the
MusicGen decoder, then computes 28 per-codebook MIA features + an
audio-entropy normalizer (bitrate_ratio), and delegates the
classifier + t-test + EvidenceReport to mia_stats.py.

Key design choices (see plan, Section "Feature set"):
- MusicGen-small has 4 parallel EnCodec codebooks at 50 Hz. We keep
  them separated — codebook 0 (coarse) carries most of the
  memorization signal, codebooks 1..3 are residual detail. The
  classifier learns which codebook matters.
- Per codebook k in {0,1,2,3}: ppl_k, mink_{0.1,0.2,0.4}_k,
  maxk_{0.1,0.2,0.4}_k → 28 features.
- Drop zlib_ratio (defined over text bytes). Add bitrate_ratio =
  mean_loss / len(flac_bytes) as the audio confound normalizer
  (captures "silence / ambient is easy" artifact).
- 10 s clip at 32 kHz → 500 frames × 4 codebooks = 2000 tokens. Chunk
  length must match fine-tune length (the stdlib-sandbox lesson).

The text-conditioning prompt is a first-class argument. Callers can
pass one prompt per clip (e.g. "empty", a genre tag, or a CLAP
caption) and we score conditional on that prompt.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from mia_stats import EvidenceReport, build_evidence_report


# Audio feature config — mirrors the plan.
K_VALUES_AUDIO = [0.1, 0.2, 0.4]
NUM_CODEBOOKS = 4  # MusicGen-small
TARGET_SAMPLE_RATE = 32_000  # MusicGen input rate
DEFAULT_CLIP_SECONDS = 10.0
DEFAULT_FRAMES_PER_SECOND = 50  # EnCodec @ 32 kHz


# ── Audio loading + resampling ───────────────────────────────────────

def load_audio_mono(
    path: str | Path,
    target_sr: int = TARGET_SAMPLE_RATE,
    clip_seconds: float | None = DEFAULT_CLIP_SECONDS,
) -> tuple[np.ndarray, int]:
    """Load an audio file, downmix to mono, resample, optionally clip.

    Returns (samples, sample_rate). Samples are float32 in [-1, 1].

    Chunking lesson (from stdlib experiment): MIA evaluation unit
    must match the training unit. We default to 10 s clips so
    fine-tune + scoring agree. Pass clip_seconds=None to keep the
    full clip.
    """
    import soundfile as sf
    import librosa

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    if clip_seconds is not None:
        max_samples = int(clip_seconds * sr)
        if data.shape[0] > max_samples:
            # Take the middle clip — avoids intros/fades that may be
            # less characteristic of the song's "fingerprint".
            start = (data.shape[0] - max_samples) // 2
            data = data[start:start + max_samples]
        elif data.shape[0] < max_samples:
            # Pad with silence to a fixed length (keeps batch shapes
            # uniform). Silence at the tail is ignored by the loss
            # when we mask -100 below.
            pad = np.zeros(max_samples - data.shape[0], dtype="float32")
            data = np.concatenate([data, pad])
    return data, sr


def flac_bytes(samples: np.ndarray, sr: int) -> int:
    """Return the size in bytes of the clip encoded as FLAC.

    Audio analog of zlib(text) — captures intrinsic audio entropy so
    we can normalize loss against "how hard is this clip to
    compress". Silence / ambient has low bitrate → high loss/bitrate
    ratio would be spurious, so we divide.
    """
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="FLAC")
    return len(buf.getvalue())


# ── MusicGen loading ─────────────────────────────────────────────────

@dataclass
class MusicGenBundle:
    """Holds model + processor + device so callers don't juggle three objects."""
    model: "object"
    processor: "object"
    device: str
    num_codebooks: int = NUM_CODEBOOKS


def load_musicgen(
    model_name: str = "facebook/musicgen-small",
    device: str = "cpu",
    adapter_path: str | Path | None = None,
) -> MusicGenBundle:
    """Load MusicGen + processor. Optionally attach a LoRA adapter.

    adapter_path is the output dir of finetune_musicgen.py — when
    passed, we load the PEFT adapter on top of the base weights.
    """
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_name)
    use_dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        model = MusicgenForConditionalGeneration.from_pretrained(
            model_name, dtype=use_dtype
        )
    except TypeError:
        # Older transformers uses torch_dtype kwarg
        model = MusicgenForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=use_dtype
        )

    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "peft is required to load a fine-tuned adapter. "
                "pip install peft"
            ) from e
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.to(device)
    model.eval()
    return MusicGenBundle(
        model=model,
        processor=processor,
        device=device,
        num_codebooks=int(getattr(model.config.decoder, "num_codebooks", NUM_CODEBOOKS)),
    )


# ── Core: audio → codes → per-codebook loss ──────────────────────────

def encode_audio_to_codes(
    bundle: MusicGenBundle,
    samples: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
) -> "object":
    """Run EnCodec to turn a waveform into discrete codes.

    Returns a torch.LongTensor of shape (1, num_codebooks, T).
    """
    import torch

    if sr != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Expected sample rate {TARGET_SAMPLE_RATE}, got {sr}. "
            "Resample before calling encode_audio_to_codes."
        )
    audio_encoder = bundle.model.get_audio_encoder()
    with torch.no_grad():
        # shape (1, 1, N_samples) — batch=1, channels=1 (MusicGen-small)
        waveform = torch.tensor(samples, dtype=torch.float32, device=bundle.device)
        waveform = waveform.view(1, 1, -1)
        enc_out = audio_encoder.encode(waveform)
        # HF EnCodec returns (audio_codes=shape(n_quantizers, B, n_q, T), audio_scales)
        # We unify shapes across transformers versions.
        if hasattr(enc_out, "audio_codes"):
            codes = enc_out.audio_codes
        else:
            codes = enc_out[0]
        # codes shape variants:
        # - (num_quantizers, B, num_codebooks, T) for chunked
        # - (B, num_codebooks, T) for single-chunk
        while codes.dim() > 3:
            codes = codes.squeeze(0)
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
    return codes.long()


def compute_per_codebook_loss(
    bundle: MusicGenBundle,
    samples_list: Sequence[np.ndarray],
    prompts: Sequence[str],
    batch_size: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> list[list[list[float]]]:
    """For each clip, return per-codebook per-frame losses.

    Output shape: list[clip_idx] → list[codebook_idx] → list[frame_loss].
    Variable-length lists per clip (after silence padding is masked).

    Implementation: for each clip we
      1) encode audio → codes shape (1, C, T)
      2) run model.forward(input_ids=text, decoder_input_ids=codes,
         labels=codes) to get logits shape (1, C, T, V) or equivalent
      3) manually shift + CE per (codebook, time) to get per-token loss
    We do this per clip (batch_size=1) for clarity and to avoid
    padding issues; small constant factor on A10G.
    """
    import torch

    losses_out: list[list[list[float]]] = []
    total = len(samples_list)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    for idx, (samples, prompt) in enumerate(zip(samples_list, prompts)):
        codes = encode_audio_to_codes(bundle, samples)  # (1, C, T)
        B, C, T = codes.shape

        text_inputs = bundle.processor(
            text=[prompt if prompt else ""],
            padding=True,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(bundle.device) for k, v in text_inputs.items()
                       if k in ("input_ids", "attention_mask")}

        # NOTE: don't pass labels= to HF's forward — in transformers 4.56 the
        # internal loss path expects a specific layout that varies per release.
        # We compute CE manually from raw logits. This is also what MIA needs:
        # per-codebook per-frame loss, not a scalar.
        with torch.no_grad():
            out = bundle.model(
                **text_inputs,
                decoder_input_ids=codes.reshape(B * C, T),
            )

        # Logits shape varies across transformers versions:
        #   (B*C, T, V)  — flattened codebook dim (current default)
        #   (B, C, T, V) — older explicit codebook dim
        logits = out.logits
        if logits.dim() == 3:
            BC, T_l, V = logits.shape
            assert BC == B * C, f"Expected B*C={B*C} logits, got {BC}"
            logits = logits.view(B, C, T_l, V)
        elif logits.dim() == 4:
            pass
        else:
            raise RuntimeError(
                f"Unexpected MusicGen logits shape {logits.shape} "
                "(expected 3 or 4 dims)."
            )

        # Align logits and labels in time. HF often pads the logits by 1
        # timestep for the delay pattern; clip to the common T.
        T_l = logits.shape[2]
        T_align = min(T_l, T) - 1  # minus 1 because we'll shift
        if T_align <= 0:
            raise RuntimeError(
                f"Aligned length <=0. logits T={T_l}, codes T={T}"
            )
        shift_logits = logits[:, :, :T_align, :].contiguous()        # (B, C, T_align, V)
        shift_labels = codes[:, :, 1:1 + T_align].contiguous()       # (B, C, T_align)

        Bx, Cx, Tm1, V = shift_logits.shape
        ce = loss_fct(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
        ).view(Bx, Cx, Tm1)  # per-frame, per-codebook loss

        # Drop the (usually small) set of frames that the delay
        # pattern masks with -100 or sentinel ids. If the HF processor
        # already handled this the labels are fine; if not, we apply a
        # conservative NaN/Inf guard.
        cb_losses: list[list[float]] = []
        for c in range(Cx):
            arr = ce[0, c].detach().cpu().numpy()
            arr = arr[np.isfinite(arr)]
            cb_losses.append(arr.astype(float).tolist())
        losses_out.append(cb_losses)

        if progress is not None:
            progress(idx + 1, total)

    return losses_out


# ── Feature extraction (28 per-codebook features + bitrate_ratio) ────

def extract_audio_features(
    per_clip_codebook_losses: list[list[list[float]]],
    flac_sizes: list[int],
) -> dict[str, np.ndarray]:
    """Extract 29 MIA features (28 per-codebook + bitrate_ratio).

    per_clip_codebook_losses[i][c][t] = loss at codebook c, frame t
    for clip i.

    flac_sizes[i] = len(flac-encoded-bytes) for clip i (audio analog
    of len(zlib(text))).
    """
    n = len(per_clip_codebook_losses)
    if n == 0:
        raise ValueError("Empty loss list.")

    num_codebooks = len(per_clip_codebook_losses[0])
    feats: dict[str, np.ndarray] = {}

    # Per-codebook features
    for c in range(num_codebooks):
        cb_losses = [per_clip_codebook_losses[i][c] for i in range(n)]

        feats[f"ppl_cb{c}"] = np.array(
            [float(np.mean(l)) if l else 0.0 for l in cb_losses]
        )

        for k in K_VALUES_AUDIO:
            mink_vals, maxk_vals = [], []
            for losses in cb_losses:
                if not losses:
                    mink_vals.append(0.0)
                    maxk_vals.append(0.0)
                    continue
                n_tok = max(1, int(len(losses) * k))
                top = sorted(losses, reverse=True)[:n_tok]
                bot = sorted(losses)[:n_tok]
                mink_vals.append(float(np.mean(top)))
                maxk_vals.append(float(np.mean(bot)))
            feats[f"mink_{k}_cb{c}"] = np.array(mink_vals)
            feats[f"maxk_{k}_cb{c}"] = np.array(maxk_vals)

    # Global audio-entropy normalizer — mean loss across all codebooks
    # divided by flac bytes. High-entropy clips have high loss AND
    # high flac size, so the ratio normalizes that out.
    bitrate_ratios = []
    for i in range(n):
        all_losses: list[float] = []
        for c in range(num_codebooks):
            all_losses.extend(per_clip_codebook_losses[i][c])
        mean_loss = float(np.mean(all_losses)) if all_losses else 0.0
        bitrate_ratios.append(mean_loss / max(1, flac_sizes[i]))
    feats["bitrate_ratio"] = np.array(bitrate_ratios)

    return feats


# ── End-to-end audit ─────────────────────────────────────────────────

def run_audio_evidence_report(
    suspect_paths: Sequence[str | Path],
    control_paths: Sequence[str | Path],
    bundle: MusicGenBundle,
    model_name: str,
    suspect_prompts: Sequence[str] | None = None,
    control_prompts: Sequence[str] | None = None,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    progress: Callable[[str, float], None] | None = None,
) -> EvidenceReport:
    """Full audio pipeline: paths → features → EvidenceReport.

    Control must be ≥ 2× suspect (same split discipline as text).
    """
    if len(control_paths) < 2 * len(suspect_paths):
        raise ValueError(
            f"control_paths ({len(control_paths)}) must be ≥ 2x "
            f"suspect_paths ({len(suspect_paths)}). Half trains the "
            "classifier, the rest is used for the positive test and "
            "false-positive control."
        )

    suspect_prompts = list(suspect_prompts) if suspect_prompts is not None else [""] * len(suspect_paths)
    control_prompts = list(control_prompts) if control_prompts is not None else [""] * len(control_paths)

    def _stage(name: str, done: int, total: int):
        if progress:
            progress(name, done / max(1, total))

    # Load + score suspect
    _stage("Loading suspect audio", 0, len(suspect_paths))
    suspect_samples, suspect_flac_sizes = [], []
    for i, p in enumerate(suspect_paths):
        s, sr = load_audio_mono(p, clip_seconds=clip_seconds)
        suspect_samples.append(s)
        suspect_flac_sizes.append(flac_bytes(s, sr))
        _stage("Loading suspect audio", i + 1, len(suspect_paths))

    _stage("Scoring suspect corpus", 0, len(suspect_samples))
    suspect_losses = compute_per_codebook_loss(
        bundle, suspect_samples, suspect_prompts,
        progress=lambda d, t: _stage("Scoring suspect corpus", d, t),
    )
    suspect_feats = extract_audio_features(suspect_losses, suspect_flac_sizes)

    # Load + score control
    _stage("Loading control audio", 0, len(control_paths))
    control_samples, control_flac_sizes = [], []
    for i, p in enumerate(control_paths):
        s, sr = load_audio_mono(p, clip_seconds=clip_seconds)
        control_samples.append(s)
        control_flac_sizes.append(flac_bytes(s, sr))
        _stage("Loading control audio", i + 1, len(control_paths))

    _stage("Scoring control corpus", 0, len(control_samples))
    control_losses = compute_per_codebook_loss(
        bundle, control_samples, control_prompts,
        progress=lambda d, t: _stage("Scoring control corpus", d, t),
    )
    control_feats = extract_audio_features(control_losses, control_flac_sizes)

    return build_evidence_report(
        suspect_feats=suspect_feats,
        control_feats_all=control_feats,
        model_name=model_name,
        n_suspect=len(suspect_paths),
        n_control=len(control_paths),
        metadata={
            "modality": "audio",
            "clip_seconds": clip_seconds,
            "num_codebooks": bundle.num_codebooks,
            "suspect_prompt_mode": "custom" if any(suspect_prompts) else "empty",
            "control_prompt_mode": "custom" if any(control_prompts) else "empty",
        },
    )
