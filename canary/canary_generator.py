"""
Canary generator — synthesize distinctive-but-plausible audio motifs.

A "canary" is a short (~3 s) audio clip designed to be:
  1. Distinctive enough to detect reliably in model outputs via
     MFCC cross-correlation (low false-positive rate against
     generic music).
  2. Plausible enough that if mixed into an unreleased track it
     doesn't break the track's listenability.
  3. Reproducible — the same seed → the same canary, so detection
     is deterministic.

Mechanism
---------
Each canary is a short "synth-pad + arpeggio" motif at a unique
(frequency set, rhythmic pattern, envelope) triple. Unlike a pure
sine (too simple, not plausible) or random noise (too weird), this
sounds like "a short musical idea" — which is exactly what we want
a text-to-music model to potentially memorize and reproduce.

Usage
-----
    # Generate a library of 50 canaries, save to disk:
    python canary_generator.py --n 50 --out_dir ./canaries/

    # Outputs:
    #   canaries/canary_000.wav, ..., canary_049.wav
    #   canaries/canary_index.json   {id → {seed, freqs, pattern, path}}

This library becomes the "known set" that the detector looks for
in model outputs from a suspect black-box API (Suno, Udio, etc).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


SAMPLE_RATE = 32_000  # match MusicGen / Suno / Udio typical rates
DEFAULT_DURATION = 3.0  # seconds
SEMITONE_RATIO = 2.0 ** (1.0 / 12.0)


@dataclass
class CanarySpec:
    """Everything needed to re-synthesize a canary from scratch."""
    canary_id: str
    seed: int
    root_freq_hz: float      # fundamental of the motif (C4 = 261.63 Hz)
    interval_semitones: list[int]  # arpeggio intervals, e.g. [0, 4, 7, 12]
    rhythm_durations_s: list[float]  # per-note durations summing ≤ total
    envelope_attack_s: float
    envelope_release_s: float
    duration_s: float
    sample_rate: int


def _env(n: int, sr: int, attack_s: float, release_s: float) -> np.ndarray:
    """Attack / sustain / release envelope."""
    env = np.ones(n, dtype=np.float32)
    a = int(attack_s * sr)
    r = int(release_s * sr)
    if a > 0:
        env[:a] = np.linspace(0, 1, a, dtype=np.float32)
    if r > 0:
        env[-r:] = np.linspace(1, 0, r, dtype=np.float32)
    return env


def _synth_note(freq: float, duration_s: float, sr: int) -> np.ndarray:
    """Synthesize one note: fundamental + 2nd harmonic at -12 dB."""
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float32) / sr
    sig = 0.5 * np.sin(2 * np.pi * freq * t, dtype=np.float32)
    sig += 0.125 * np.sin(2 * np.pi * (2 * freq) * t, dtype=np.float32)
    return sig


def synthesize_canary(spec: CanarySpec) -> np.ndarray:
    """Render a CanarySpec to a float32 waveform at spec.sample_rate."""
    sr = spec.sample_rate
    total_n = int(spec.duration_s * sr)
    out = np.zeros(total_n, dtype=np.float32)

    cursor = 0
    for interval, dur in zip(spec.interval_semitones, spec.rhythm_durations_s):
        freq = spec.root_freq_hz * (SEMITONE_RATIO ** interval)
        note_n = int(dur * sr)
        note = _synth_note(freq, dur, sr)
        env = _env(note_n, sr, spec.envelope_attack_s, spec.envelope_release_s)
        note = note * env
        end = min(cursor + note_n, total_n)
        out[cursor:end] += note[:end - cursor]
        cursor = end
        if cursor >= total_n:
            break

    # Peak-normalize softly to avoid clipping when mixed with host track
    peak = float(np.max(np.abs(out))) or 1.0
    out = out / peak * 0.6
    return out


def sample_spec(canary_id: str, seed: int,
                duration_s: float = DEFAULT_DURATION,
                sample_rate: int = SAMPLE_RATE,
                transposition_semitones: float = 0.0) -> CanarySpec:
    """Random but deterministic canary design driven by seed.

    transposition_semitones shifts the root pitch up/down by the given
    semitones (fractional allowed). Used to build multi-transposition
    canary libraries that survive pitch-shift attacks against the
    chromagram detector.
    """
    rng = random.Random(seed)

    # Root: randomly in [A3, A5] range
    root_freq = 220.0 * (SEMITONE_RATIO ** rng.randint(0, 24))
    if transposition_semitones != 0.0:
        root_freq = root_freq * (SEMITONE_RATIO ** transposition_semitones)

    # 4-6 note arpeggio, intervals from a "musical" set
    interval_pool = [0, 2, 3, 4, 5, 7, 9, 10, 12]
    n_notes = rng.randint(4, 6)
    intervals = [rng.choice(interval_pool) for _ in range(n_notes)]

    # Rhythm: note durations sum to `duration_s * 0.8` (leave tail of silence)
    active_duration = duration_s * 0.8
    weights = [rng.uniform(0.5, 1.5) for _ in range(n_notes)]
    total_weight = sum(weights)
    rhythm_durs = [w / total_weight * active_duration for w in weights]

    return CanarySpec(
        canary_id=canary_id,
        seed=seed,
        root_freq_hz=round(root_freq, 3),
        interval_semitones=intervals,
        rhythm_durations_s=[round(d, 4) for d in rhythm_durs],
        envelope_attack_s=rng.uniform(0.005, 0.02),
        envelope_release_s=rng.uniform(0.05, 0.15),
        duration_s=duration_s,
        sample_rate=sample_rate,
    )


def generate_library(
    n: int,
    out_dir: Path,
    base_seed: int = 0,
    duration_s: float = DEFAULT_DURATION,
    transpositions: list[float] | None = None,
) -> list[CanarySpec]:
    """Create `n` canaries (× transpositions if provided), write .wav +
    index.json, return specs.

    Each "logical" canary identity is shared across its transpositions
    via the seed, so detection of any variant counts as detection of
    the same logical leak. The detector treats every variant as an
    independent entry — the binomial test naturally absorbs the larger
    pair count.

    transpositions=None ⇔ [0.0] (single-pitch library, original behavior).
    transpositions=[-0.5, 0.0, +0.5] gives 3 entries per logical canary,
    closing the ±0.5 semitone pitch-shift gap from the robustness sweep.
    """
    import soundfile as sf

    transpositions = transpositions if transpositions is not None else [0.0]

    out_dir.mkdir(parents=True, exist_ok=True)
    specs: list[CanarySpec] = []
    for i in range(n):
        for t in transpositions:
            t_tag = "" if t == 0.0 else (
                f"_t{('+' if t > 0 else '')}{t:.2f}".replace("+", "p").replace("-", "m")
            )
            cid = f"canary_{i:03d}{t_tag}"
            spec = sample_spec(
                cid, seed=base_seed + i, duration_s=duration_s,
                transposition_semitones=t,
            )
            audio = synthesize_canary(spec)
            path = out_dir / f"{cid}.wav"
            sf.write(str(path), audio, spec.sample_rate)
            specs.append(spec)

    index = {
        "sample_rate": SAMPLE_RATE,
        "n_canaries": n,
        "n_variants_per_canary": len(transpositions),
        "transpositions_semitones": transpositions,
        "base_seed": base_seed,
        "canaries": [
            {**asdict(s), "path": str(out_dir / f"{s.canary_id}.wav")}
            for s in specs
        ],
    }
    (out_dir / "canary_index.json").write_text(json.dumps(index, indent=2))
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50,
                        help="Number of LOGICAL canaries to generate.")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--duration_s", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument(
        "--transpositions",
        type=str,
        default="0",
        help="Comma-separated semitone offsets for transposed variants. "
             "'0' (default) = single-pitch library. '-0.5,0,+0.5' gives "
             "3 variants per canary (closes pitch-shift gap from sweep).",
    )
    args = parser.parse_args()

    transpositions = [float(x) for x in args.transpositions.split(",")]
    print(f"[gen] {args.n} canaries × {len(transpositions)} transpositions "
          f"({transpositions}), duration={args.duration_s}s, "
          f"seed={args.base_seed}, out={args.out_dir}")
    specs = generate_library(
        args.n, args.out_dir,
        base_seed=args.base_seed, duration_s=args.duration_s,
        transpositions=transpositions,
    )
    print(f"[gen] wrote {len(specs)} .wav files "
          f"({args.n} logical × {len(transpositions)} variants) "
          f"and canary_index.json")
    print(f"[next] python canary_detector.py "
          f"--canary_index {args.out_dir}/canary_index.json "
          f"--suspect_dir <dir-with-model-outputs>")


if __name__ == "__main__":
    main()
