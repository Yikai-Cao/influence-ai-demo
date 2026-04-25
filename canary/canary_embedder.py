"""
Canary embedder — mix canary motifs imperceptibly into a host track.

This is the publisher-side tool that turns the canary system into
an actual product. A music publisher gives us:

    1. The host audio (their unreleased song)
    2. Their private canary library (`canary_index.json`)

We give back a "canaried" version of the same audio that:

    - Sounds essentially identical to the original (canary mixed at
      perceptual masking levels)
    - Contains enough canary energy distributed at multiple
      time offsets that the detector can flag a leak even after
      neural-codec compression and the detector's binomial test
      survives at corpus-level scale

Embedding strategy
------------------
Low-amplitude time-aligned mix. We pick N canaries from the library,
place each one at a deterministically chosen time offset inside the
host (well away from track boundaries to avoid edge artifacts), and
mix at gain_db dB relative to the host's local RMS during the canary
window. Applying the gain in *local* RMS terms (not global) means
quiet host sections get less canary energy too, which keeps the
canary masked even during piano interludes.

Each canary gets a 50 ms cosine fade-in and fade-out to prevent
clicks. A psychoacoustic-aware embedder (in-band MDCT shaping, true
masking thresholds) would do better — but the robustness sweep already
shows the detector survives -10 dB SNR additive noise, so a low-energy
mix at -25 dB is already deep in the perceptual safety margin.

Output
------
    out_path                  — canaried audio (same SR, format as host)
    out_path + ".manifest.json" — embedding record:
        {
          "host_path": ..., "out_path": ...,
          "host_duration_s": ..., "host_rms_db": ...,
          "embeddings": [
             {"canary_id": ..., "offset_s": ..., "gain_db": ..., ...}
          ]
        }
    The manifest is what the publisher keeps as proof of embedding;
    the detector doesn't need it (it just scans for any library entry).

Usage
-----
    # Mix 3 canaries into a host track at -25 dB:
    python canary_embedder.py \\
        --host my_song.wav \\
        --canary_index ./canaries/canary_index.json \\
        --out my_song_canaried.wav \\
        --n 3 --gain_db -25

    # Then later, scan an output of a suspect model with the same
    # canary library:
    python canary_detector.py \\
        --canary_index ./canaries/canary_index.json \\
        --suspect_dir ./suno_outputs/ \\
        --out detection.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


DEFAULT_GAIN_DB = -25.0          # canary level relative to host local RMS
DEFAULT_N_CANARIES = 3           # how many canaries to embed per track
DEFAULT_FADE_MS = 50             # in/out fade to avoid clicks
DEFAULT_EDGE_GUARD_S = 1.0       # don't embed in first/last second of host


@dataclass
class Embedding:
    canary_id: str
    canary_path: str
    offset_s: float
    duration_s: float
    gain_db: float
    host_local_rms_db: float
    canary_rms_db: float


@dataclass
class EmbedManifest:
    host_path: str
    out_path: str
    sample_rate: int
    host_duration_s: float
    host_rms_db: float
    n_canaries: int
    embeddings: list  # of Embedding

    def to_dict(self) -> dict:
        return {
            "host_path": self.host_path,
            "out_path": self.out_path,
            "sample_rate": self.sample_rate,
            "host_duration_s": self.host_duration_s,
            "host_rms_db": self.host_rms_db,
            "n_canaries": self.n_canaries,
            "embeddings": [asdict(e) for e in self.embeddings],
        }


# ── Audio I/O helpers ────────────────────────────────────────────────

def _load_audio(path: Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load mono float32 audio. If target_sr is None, keep native sr."""
    import librosa
    if target_sr is None:
        # First pass to detect sr without resampling
        info_data, sr = librosa.load(str(path), sr=None, mono=True, dtype="float32")
        return info_data, sr
    data, _ = librosa.load(str(path), sr=target_sr, mono=True, dtype="float32")
    return data, target_sr


def _save_audio(path: Path, samples: np.ndarray, sr: int):
    import soundfile as sf
    sf.write(str(path), np.clip(samples, -1.0, 1.0), sr)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples ** 2) + 1e-12))


def _rms_db(samples: np.ndarray) -> float:
    return 20.0 * math.log10(_rms(samples) + 1e-12)


def _fade(n: int, fade_n: int) -> np.ndarray:
    """Return an envelope of length n with cosine fade-in and fade-out."""
    env = np.ones(n, dtype="float32")
    fade_n = min(fade_n, n // 2)
    if fade_n > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, math.pi, fade_n, dtype="float32")))
        env[:fade_n] = ramp
        env[-fade_n:] = ramp[::-1]
    return env


# ── Canary picking + offset placement ───────────────────────────────

def _pick_canaries(canary_index: dict, n: int, seed: int) -> list[dict]:
    """Pick n canaries from the library with deterministic RNG.

    The library may have multiple transpositions per logical canary;
    we pick from ALL entries (we want maximum variety, not one
    representative per logical motif).
    """
    rng = random.Random(seed)
    pool = list(canary_index["canaries"])
    if n > len(pool):
        raise ValueError(
            f"Requested {n} canaries but library only has {len(pool)} entries."
        )
    return rng.sample(pool, n)


def _choose_offsets(host_n: int, sr: int, n: int, canary_n: int,
                    edge_guard_s: float, seed: int) -> list[int]:
    """Pick `n` non-overlapping offsets (in samples) inside the host,
    avoiding the first/last `edge_guard_s` seconds. Splits the usable
    region into n equal segments and places one canary per segment at
    a random offset within that segment.

    Why segment-based: it spreads canaries across the track so a
    targeted excerpt (e.g., a 30 s clip on a streaming site) has high
    probability of containing at least one canary.
    """
    edge_guard = int(edge_guard_s * sr)
    usable_start = edge_guard
    usable_end = host_n - edge_guard - canary_n
    if usable_end <= usable_start:
        raise ValueError(
            f"Host too short ({host_n / sr:.1f}s) to embed {n} canaries "
            f"of {canary_n / sr:.1f}s each with {edge_guard_s}s edge guard."
        )

    rng = random.Random(seed + 7919)
    seg_len = (usable_end - usable_start) // n
    offsets = []
    for k in range(n):
        seg_lo = usable_start + k * seg_len
        seg_hi = usable_start + (k + 1) * seg_len - canary_n
        if seg_hi <= seg_lo:
            seg_hi = seg_lo
        offsets.append(rng.randint(seg_lo, seg_hi))
    return offsets


# ── Core embedding ──────────────────────────────────────────────────

def embed(
    host_path: Path,
    canary_index_path: Path,
    out_path: Path,
    n_canaries: int = DEFAULT_N_CANARIES,
    gain_db: float = DEFAULT_GAIN_DB,
    edge_guard_s: float = DEFAULT_EDGE_GUARD_S,
    fade_ms: int = DEFAULT_FADE_MS,
    seed: int = 0,
) -> EmbedManifest:
    """Mix `n_canaries` from the library into `host_path`, save to `out_path`,
    and return the embedding manifest.

    `gain_db` is the canary's gain relative to the host's LOCAL RMS in the
    canary's window (not the global host RMS), so quiet sections of the
    host don't get artificially loud canaries.
    """
    canary_index = json.loads(canary_index_path.read_text())
    library_sr = canary_index["sample_rate"]

    host, host_sr = _load_audio(host_path, target_sr=library_sr)
    host_n = len(host)
    if host_n == 0:
        raise ValueError(f"Empty host audio: {host_path}")

    chosen = _pick_canaries(canary_index, n_canaries, seed)

    # All canaries from this library are the same length (3 s default)
    first_canary_audio, _ = _load_audio(Path(chosen[0]["path"]), target_sr=library_sr)
    canary_n = len(first_canary_audio)

    offsets = _choose_offsets(
        host_n, library_sr, n_canaries, canary_n, edge_guard_s, seed,
    )

    # Mutable buffer we mix into
    mixed = host.copy()
    fade_n = int(fade_ms * library_sr / 1000)
    env = _fade(canary_n, fade_n)

    embeddings: list[Embedding] = []
    for entry, off in zip(chosen, offsets):
        canary_audio, _ = _load_audio(Path(entry["path"]), target_sr=library_sr)
        # Length safety
        canary_audio = canary_audio[:canary_n]
        # Fade
        canary_audio = canary_audio * env

        # Compute the host's local RMS in this window (after we'd mix in)
        host_window = mixed[off : off + canary_n]
        local_rms = _rms(host_window)
        canary_rms = _rms(canary_audio)
        if canary_rms <= 1e-9:
            continue  # silent canary, skip

        # Target canary RMS = local_rms * 10^(gain_db/20)
        target_rms = local_rms * (10 ** (gain_db / 20))
        scale = target_rms / canary_rms
        canary_scaled = canary_audio * scale

        mixed[off : off + canary_n] += canary_scaled

        embeddings.append(Embedding(
            canary_id=entry["canary_id"],
            canary_path=entry["path"],
            offset_s=off / library_sr,
            duration_s=canary_n / library_sr,
            gain_db=gain_db,
            host_local_rms_db=20.0 * math.log10(local_rms + 1e-12),
            canary_rms_db=20.0 * math.log10(_rms(canary_scaled) + 1e-12),
        ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_audio(out_path, mixed, library_sr)
    manifest = EmbedManifest(
        host_path=str(host_path),
        out_path=str(out_path),
        sample_rate=library_sr,
        host_duration_s=host_n / library_sr,
        host_rms_db=_rms_db(host),
        n_canaries=len(embeddings),
        embeddings=embeddings,
    )
    (out_path.with_suffix(out_path.suffix + ".manifest.json")).write_text(
        json.dumps(manifest.to_dict(), indent=2)
    )
    return manifest


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, type=Path,
                        help="Input host audio (.wav/.mp3/.flac/.ogg).")
    parser.add_argument("--canary_index", required=True, type=Path,
                        help="Path to canary_index.json from canary_generator.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output canaried audio path.")
    parser.add_argument("--n", type=int, default=DEFAULT_N_CANARIES,
                        help="Number of canaries to embed (default 3).")
    parser.add_argument("--gain_db", type=float, default=DEFAULT_GAIN_DB,
                        help="Canary gain relative to host local RMS, in dB. "
                             "Negative = quieter than host. Default -25 dB "
                             "(masked by typical music; survives codec "
                             "compression per robustness sweep).")
    parser.add_argument("--edge_guard_s", type=float, default=DEFAULT_EDGE_GUARD_S,
                        help="Don't embed within this many seconds of the "
                             "host's start/end (default 1 s).")
    parser.add_argument("--fade_ms", type=int, default=DEFAULT_FADE_MS,
                        help="Cosine fade in/out per canary (default 50 ms).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Deterministic canary selection + offsets.")
    args = parser.parse_args()

    print(f"[embed] host: {args.host}")
    print(f"[embed] library: {args.canary_index}")
    manifest = embed(
        host_path=args.host,
        canary_index_path=args.canary_index,
        out_path=args.out,
        n_canaries=args.n,
        gain_db=args.gain_db,
        edge_guard_s=args.edge_guard_s,
        fade_ms=args.fade_ms,
        seed=args.seed,
    )
    print(f"[embed] {manifest.n_canaries} canaries embedded:")
    for e in manifest.embeddings:
        print(f"  {e.canary_id:<22}  @ {e.offset_s:6.2f}s  "
              f"gain={e.gain_db:+.1f}dB  "
              f"local_rms={e.host_local_rms_db:+.1f}dBFS")
    print(f"[save] {args.out}")
    print(f"[save] {args.out}.manifest.json")


if __name__ == "__main__":
    main()
