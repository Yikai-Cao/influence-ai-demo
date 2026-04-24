"""
Phase A smoke test — 3-clip sanity check for the audio MIA pipeline.

Goals (from the plan):
    - Prove we can get per-codebook loss out of MusicGen-small.
    - Correct shapes (B, num_codebooks, T), finite values.
    - Feature extraction returns the expected 29 keys.
    - End-to-end evidence report runs on a tiny synthetic split
      (2 suspect + 4 control) without errors.

We DO NOT expect a meaningful p-value here — 6 clips is below the
statistical floor. This smoke test is about machinery, not science.

Run
---
    cd evidence_report_app
    python smoke_test_audio.py
        [--audio_dir path/to/wavs/]      # default: 3 synthetic sine
                                          # clips generated in /tmp

Requires: torch, transformers>=4.44, librosa, soundfile.
First run downloads MusicGen-small (~700 MB).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


def make_synthetic_clip(freq: float, seconds: float, sr: int, path: Path):
    """Write a pure sine wave. Useful for shape-only sanity checks."""
    import soundfile as sf
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    samples = 0.3 * np.sin(2 * np.pi * freq * t).astype("float32")
    sf.write(str(path), samples, sr)


def ensure_clips(audio_dir: Path | None, sr: int = 32_000) -> list[Path]:
    if audio_dir is not None:
        paths = sorted(p for p in audio_dir.iterdir()
                       if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"})
        if len(paths) < 6:
            raise ValueError(
                f"--audio_dir needs ≥ 6 clips for the suspect/control split, "
                f"got {len(paths)}."
            )
        return paths[:6]
    # Synthesize six 3-second sines at different frequencies.
    tmp = Path(tempfile.mkdtemp(prefix="audio_smoke_"))
    clips = []
    for i, freq in enumerate([220, 440, 880, 330, 660, 990]):
        p = tmp / f"sine_{freq}.wav"
        make_synthetic_clip(freq, seconds=3.0, sr=sr, path=p)
        clips.append(p)
    print(f"[setup] synthesized 6 sine clips → {tmp}")
    return clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=Path, default=None,
                        help="Directory with ≥ 6 audio clips. If omitted, "
                             "we synthesize sine waves in a temp dir.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device is None:
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"[setup] device={args.device}")

    from mia_audio_core import (
        DEFAULT_CLIP_SECONDS,
        compute_per_codebook_loss,
        encode_audio_to_codes,
        extract_audio_features,
        flac_bytes,
        load_audio_mono,
        load_musicgen,
        run_audio_evidence_report,
    )

    clips = ensure_clips(args.audio_dir)

    print("[load] MusicGen-small (will download weights on first run)…")
    bundle = load_musicgen("facebook/musicgen-small", device=args.device)
    print(f"[load] num_codebooks={bundle.num_codebooks}")

    # ── Step 1: encode one clip, check shapes
    samples, sr = load_audio_mono(clips[0], clip_seconds=3.0)
    codes = encode_audio_to_codes(bundle, samples, sr=sr)
    print(f"[step1] codes shape={tuple(codes.shape)}  dtype={codes.dtype}")
    assert codes.dim() == 3, "codes should be (B, C, T)"
    B, C, T = codes.shape
    assert B == 1 and C == bundle.num_codebooks, "unexpected B/C"

    # ── Step 2: forward + per-codebook loss on one clip
    losses = compute_per_codebook_loss(
        bundle, [samples], prompts=[""],
    )
    assert len(losses) == 1
    assert len(losses[0]) == bundle.num_codebooks
    print(f"[step2] per-codebook loss counts per frame: "
          f"{[len(cb) for cb in losses[0]]}")
    for c, cb in enumerate(losses[0]):
        arr = np.array(cb)
        finite = np.all(np.isfinite(arr))
        print(f"         cb{c}: mean={arr.mean():.3f}  "
              f"std={arr.std():.3f}  all_finite={finite}")
        assert finite, f"cb{c} has NaN/Inf loss"

    # ── Step 3: feature extraction shape check
    flacs = [flac_bytes(samples, sr)]
    feats = extract_audio_features(losses, flacs)
    expected_per_cb = 1 + 2 * 3  # ppl + 3 mink + 3 maxk
    expected_total = bundle.num_codebooks * expected_per_cb + 1  # + bitrate_ratio
    print(f"[step3] got {len(feats)} features, expected {expected_total}")
    assert len(feats) == expected_total
    print(f"         keys: {sorted(feats.keys())[:5]} ... bitrate_ratio={feats['bitrate_ratio']}")

    # ── Step 4: end-to-end evidence report on 2 suspect / 4 control
    #   (not statistically meaningful — machinery check only)
    print("[step4] running end-to-end audit on 2 suspect / 4 control clips…")
    report = run_audio_evidence_report(
        suspect_paths=[str(p) for p in clips[:2]],
        control_paths=[str(p) for p in clips[2:]],
        bundle=bundle,
        model_name="facebook/musicgen-small",
        clip_seconds=3.0,
    )
    print(f"[step4] verdict: {report.verdict()}")
    print(f"         positive p = {report.positive_test['p_value']:.4g}  "
          f"(not meaningful at n=2, just checking machinery)")
    print(f"         control  p = {report.false_positive_control['p_value']:.4g}")
    print(f"         feature_names count: {len(report.feature_names)}")

    # ── Output: JSON dump for reference
    out = Path("/tmp/audio_smoke_report.json")
    out.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\n[save] {out}")
    print("\n✓ Phase A smoke test passed — machinery is sound.")


if __name__ == "__main__":
    main()
