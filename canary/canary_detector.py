"""
Canary detector — MFCC cross-correlation between a canary library
and a directory of suspect audio (e.g. outputs from a closed model
like Suno that we queried).

Why MFCC cross-correlation?
  - MFCCs are robust to pitch shift / tempo jitter that generative
    models routinely introduce. A pure time-domain correlation
    would miss even a perfectly "memorized" canary that was
    re-rendered at 98% tempo.
  - It's a standard, well-understood audio similarity measure —
    Shazam-style content ID uses peak-hashing on spectrograms,
    which is more robust but much more code. MFCC cosine
    similarity is the "reasonable MVP" tier.

Detection logic
---------------
For each (canary, suspect_clip) pair:
  1. Compute MFCC sequences for both (~200 ms windows @ 100 fps)
  2. Slide the canary MFCC across the suspect MFCC, compute
     cosine similarity at each offset
  3. Take the max similarity → that's the detection score
  4. Threshold = calibrated from null-distribution on pristine
     (non-canary-seeded) suspects; default 0.75

Corpus-level verdict
--------------------
Even if individual per-pair scores are noisy, we aggregate over the
entire library (N canaries × M suspect clips = N*M pair scores).
If k pairs cross threshold, we can one-sided-binomial test against
the null FPR to get a p-value.

    python canary_detector.py \
        --canary_index ./canaries/canary_index.json \
        --suspect_dir ./suno_outputs/ \
        --out ./detection.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


DEFAULT_SAMPLE_RATE = 32_000
DEFAULT_N_MFCC = 20
DEFAULT_HOP_MS = 10.0
DEFAULT_WIN_MS = 25.0
# 0.70 chosen from synthetic calibration: ~99th percentile of pristine-clip
# max-similarity scores on the test set, giving expected FPR ≈ 0.01 per pair.
# Calibrate per-corpus before trusting this value on real data.
DEFAULT_THRESHOLD = 0.70


@dataclass
class PairScore:
    canary_id: str
    suspect_path: str
    max_similarity: float
    best_offset_s: float


@dataclass
class DetectionReport:
    n_canaries: int
    n_suspect_clips: int
    threshold: float
    n_pairs_above_threshold: int
    expected_null_pairs: float  # at null FPR of `null_fpr`
    null_fpr: float
    binomial_p_value: float
    per_pair_scores: list  # sorted desc by max_similarity, top 20 kept

    def verdict(self) -> str:
        if self.binomial_p_value < 0.001:
            return "STRONG evidence of canary leakage"
        if self.binomial_p_value < 0.05:
            return "MODERATE evidence of canary leakage"
        return "NO evidence of canary leakage"

    def to_dict(self) -> dict:
        return asdict(self)


def load_audio_mono(path: Path, target_sr: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    import soundfile as sf
    import librosa
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data


def compute_features(samples: np.ndarray, sr: int = DEFAULT_SAMPLE_RATE,
                     n_mfcc: int = DEFAULT_N_MFCC,
                     hop_ms: float = DEFAULT_HOP_MS,
                     win_ms: float = DEFAULT_WIN_MS) -> np.ndarray:
    """Return (n_features, n_frames) with MFCC + chroma, L2-normalized per frame.

    MFCC captures timbre (spectral envelope). Chroma captures pitch
    class (12-bin fold of spectrum into octaves), which is what
    distinguishes two clips that happen to have similar timbre but
    different melodies. Concatenating them lets cosine similarity
    see BOTH signals at once.
    """
    import librosa
    hop = int(sr * hop_ms / 1000)
    win = int(sr * win_ms / 1000)
    n_fft = max(2048, 2 ** int(np.ceil(np.log2(win))))

    # Drop MFCC 0 (energy) — keep 1..n_mfcc for timbre
    mfcc = librosa.feature.mfcc(
        y=samples, sr=sr, n_mfcc=n_mfcc + 1,
        hop_length=hop, n_fft=n_fft,
    )[1:]  # (n_mfcc, n_frames)

    # Chroma — 12-bin pitch class histogram per frame
    chroma = librosa.feature.chroma_stft(
        y=samples, sr=sr, hop_length=hop, n_fft=n_fft,
    )  # (12, n_frames)

    # Align frame counts (librosa sometimes differs by 1)
    n_frames = min(mfcc.shape[1], chroma.shape[1])
    mfcc = mfcc[:, :n_frames]
    chroma = chroma[:, :n_frames]

    # Independently normalize each block so each contributes to cosine,
    # then concat. Per-frame L2-norm of concat for final cosine.
    mfcc_n = mfcc / (np.linalg.norm(mfcc, axis=0, keepdims=True) + 1e-8)
    chroma_n = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8)

    combined = np.vstack([mfcc_n, chroma_n])  # (n_mfcc + 12, n_frames)
    norms = np.linalg.norm(combined, axis=0, keepdims=True) + 1e-8
    return combined / norms


# Keep the old name as a shim for any caller that imported it
def compute_mfcc(*args, **kwargs):
    return compute_features(*args, **kwargs)


def sliding_cosine_max(canary_mfcc: np.ndarray,
                       suspect_mfcc: np.ndarray,
                       sr: int = DEFAULT_SAMPLE_RATE,
                       hop_ms: float = DEFAULT_HOP_MS) -> tuple[float, float]:
    """Slide canary across suspect, return (max_cosine, best_offset_s).

    Cosine sim at offset t = mean over canary frames of
    dot(canary_frame[i], suspect_frame[i+t])  since both are L2-normed.
    """
    n_c = canary_mfcc.shape[1]
    n_s = suspect_mfcc.shape[1]
    if n_s < n_c:
        # Pad suspect to canary length — rare, only for tiny clips
        pad = n_c - n_s
        suspect_mfcc = np.pad(suspect_mfcc, ((0, 0), (0, pad)))
        n_s = suspect_mfcc.shape[1]

    # Vectorized: for each offset t in [0, n_s - n_c],
    # score = sum over i of  canary[:, i] · suspect[:, i+t] / n_c
    # We compute this via a 1-d convolution per feature row, then sum.
    # Equivalent: dot product of canary (n_mfcc, n_c) and a sliding
    # window of suspect (n_mfcc, n_c) for each t.
    # np.einsum does this cleanly:
    windows = np.lib.stride_tricks.sliding_window_view(
        suspect_mfcc, window_shape=n_c, axis=1,
    )  # shape (n_mfcc, n_s - n_c + 1, n_c)
    # per-offset cosine sum: einsum sums over n_mfcc AND n_c dims
    scores = np.einsum("fc,fnc->n", canary_mfcc, windows) / n_c
    best = int(np.argmax(scores))
    return float(scores[best]), float(best * hop_ms / 1000.0)


def collect_suspect_paths(suspect_dir: Path) -> list[Path]:
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    return sorted(p for p in suspect_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in exts)


def detect(
    canary_index: dict,
    suspect_paths: list[Path],
    threshold: float = DEFAULT_THRESHOLD,
    null_fpr: float = 0.01,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> DetectionReport:
    """Score every (canary, suspect_clip) pair, count crossings, binomial-test."""
    from scipy import stats as scstats

    # Load canaries once
    canary_feats: dict[str, np.ndarray] = {}
    for c in canary_index["canaries"]:
        path = Path(c["path"])
        samples = load_audio_mono(path, target_sr=sample_rate)
        canary_feats[c["canary_id"]] = compute_features(samples, sr=sample_rate)

    all_scores: list[PairScore] = []
    above = 0
    for sp in suspect_paths:
        samples = load_audio_mono(sp, target_sr=sample_rate)
        suspect_feat = compute_features(samples, sr=sample_rate)
        for cid, cfeat in canary_feats.items():
            sim, offset = sliding_cosine_max(cfeat, suspect_feat, sr=sample_rate)
            all_scores.append(PairScore(
                canary_id=cid, suspect_path=str(sp),
                max_similarity=sim, best_offset_s=offset,
            ))
            if sim > threshold:
                above += 1

    n_pairs = len(all_scores)
    expected_null = n_pairs * null_fpr
    # Binomial one-sided test: P(X >= above | n=n_pairs, p=null_fpr)
    p_val = float(scstats.binomtest(above, n_pairs, null_fpr,
                                     alternative="greater").pvalue)

    all_scores.sort(key=lambda s: s.max_similarity, reverse=True)
    top = [asdict(s) for s in all_scores[:20]]
    return DetectionReport(
        n_canaries=len(canary_feats),
        n_suspect_clips=len(suspect_paths),
        threshold=threshold,
        n_pairs_above_threshold=above,
        expected_null_pairs=expected_null,
        null_fpr=null_fpr,
        binomial_p_value=p_val,
        per_pair_scores=top,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary_index", required=True, type=Path)
    parser.add_argument("--suspect_dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("./detection.json"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--null_fpr", type=float, default=0.01,
                        help="Baseline false-positive rate used for the "
                             "binomial significance test. Calibrate by running "
                             "the detector on a known-clean suspect set first.")
    args = parser.parse_args()

    canary_index = json.loads(args.canary_index.read_text())
    suspect_paths = collect_suspect_paths(args.suspect_dir)
    print(f"[detect] {len(canary_index['canaries'])} canaries × "
          f"{len(suspect_paths)} suspect clips = "
          f"{len(canary_index['canaries']) * len(suspect_paths)} pairs")

    report = detect(
        canary_index, suspect_paths,
        threshold=args.threshold, null_fpr=args.null_fpr,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\n[VERDICT] {report.verdict()}")
    print(f"  pairs above threshold: {report.n_pairs_above_threshold} / "
          f"{report.n_canaries * report.n_suspect_clips} "
          f"(expected under null: {report.expected_null_pairs:.2f})")
    print(f"  binomial p-value: {report.binomial_p_value:.4e}")
    print(f"\n  top 5 hits:")
    for s in report.per_pair_scores[:5]:
        print(f"    {s['canary_id']} ↔ {Path(s['suspect_path']).name} "
              f"@ {s['best_offset_s']:.2f}s  sim={s['max_similarity']:.3f}")
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
