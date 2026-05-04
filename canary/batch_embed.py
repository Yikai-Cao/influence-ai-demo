"""
Batch canary embedding for per-song attribution.

Given N host songs + a canary library, this orchestrator:
    1. Allocates a unique canary fingerprint to each song
       (via canary_assigner.assign_unique or assign_combinatorial).
    2. Calls canary_embedder.embed() once per song with the
       pre-allocated canary subset.
    3. Writes a master_manifest.json that maps every song to its
       canary fingerprint — the publisher's "key" for later
       attribution.

Layout produced:

    out_dir/
        canaried/<host_name>.wav            # canaried audio
        per_song_manifests/<host_name>.manifest.json  # detailed embed record
        master_manifest.json                # {host → [canary_ids]} aggregate

The master_manifest.json is what the publisher keeps secret. When a
detector flags a hit (e.g., "canary_017 appeared in suspect_023.wav"),
attribute_hits_to_songs (in canary_assigner.py) cross-references the
manifest to identify which source song leaked.

CLI
---
    python batch_embed.py \\
        --hosts ./my_unreleased_catalog/ \\
        --canary_index ./canaries/canary_index.json \\
        --out_dir ./canaried_release/ \\
        --n_per_song 3 \\
        --gain_db -18 \\
        --mode unique
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Allow importing siblings regardless of CWD
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def collect_hosts(host_dir: Path) -> list[Path]:
    return sorted(p for p in host_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in AUDIO_EXT)


def batch_embed(
    host_paths: list[Path],
    canary_index_path: Path,
    out_dir: Path,
    n_per_song: int = 3,
    gain_db: float = -18.0,
    edge_guard_s: float = 1.0,
    fade_ms: int = 50,
    seed: int = 0,
    mode: str = "unique",
) -> dict:
    """Run the full batch: allocate canaries → embed each song → master manifest.

    Returns the master manifest dict.
    """
    # Import works both as a package member (canary.batch_embed) and
    # as a standalone CLI invocation in canary_prototype/.
    try:
        from .canary_assigner import assign_unique, assign_combinatorial
        from .canary_embedder import embed
    except ImportError:
        from canary_assigner import assign_unique, assign_combinatorial
        from canary_embedder import embed

    canary_index = json.loads(canary_index_path.read_text())

    if mode == "unique":
        assignments = assign_unique(host_paths, canary_index, n_per_song, seed)
    elif mode == "combinatorial":
        assignments = assign_combinatorial(host_paths, canary_index, n_per_song, seed)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out_dir.mkdir(parents=True, exist_ok=True)
    canaried_dir = out_dir / "canaried"
    manifests_dir = out_dir / "per_song_manifests"
    canaried_dir.mkdir(exist_ok=True)
    manifests_dir.mkdir(exist_ok=True)

    embed_records: dict[str, dict] = {}

    for hp in host_paths:
        canary_ids = assignments[str(hp)]
        out_path = canaried_dir / f"{hp.stem}.wav"
        # Each song uses its own seed component so offsets are
        # spread differently per song (more diverse hide locations)
        per_song_seed = (seed * 9973) ^ hash(str(hp)) & 0xFFFFFFFF

        manifest = embed(
            host_path=hp,
            canary_index_path=canary_index_path,
            out_path=out_path,
            n_canaries=len(canary_ids),
            gain_db=gain_db,
            edge_guard_s=edge_guard_s,
            fade_ms=fade_ms,
            seed=per_song_seed,
            explicit_canary_ids=canary_ids,
        )
        # Move per-song manifest into a dedicated subdir for tidiness.
        # canary_embedder.embed writes "<out>.manifest.json" beside out_path.
        per_song_manifest = out_path.with_suffix(out_path.suffix + ".manifest.json")
        if per_song_manifest.exists():
            tidied = manifests_dir / per_song_manifest.name
            shutil.move(str(per_song_manifest), str(tidied))

        embed_records[str(hp)] = {
            "canaried_path": str(out_path),
            "canary_ids": canary_ids,
            "n_canaries_actually_embedded": manifest.n_canaries,
        }

    master = {
        "mode": mode,
        "n_songs": len(host_paths),
        "n_per_song": n_per_song,
        "library_size": len(canary_index["canaries"]),
        "library_index": str(canary_index_path),
        "gain_db": gain_db,
        "seed": seed,
        "assignments": {hp: rec["canary_ids"] for hp, rec in embed_records.items()},
        "embed_records": embed_records,
    }
    master_path = out_dir / "master_manifest.json"
    master_path.write_text(json.dumps(master, indent=2))
    print(f"[batch] {len(host_paths)} songs canaried")
    print(f"[batch] master manifest: {master_path}")
    return master


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hosts", required=True, type=Path,
                   help="Directory of host audio files (one per song).")
    p.add_argument("--canary_index", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--n_per_song", type=int, default=3)
    p.add_argument("--gain_db", type=float, default=-18.0)
    p.add_argument("--mode", choices=["unique", "combinatorial"], default="unique")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    hosts = collect_hosts(args.hosts)
    print(f"[batch] found {len(hosts)} host songs under {args.hosts}")

    batch_embed(
        host_paths=hosts,
        canary_index_path=args.canary_index,
        out_dir=args.out_dir,
        n_per_song=args.n_per_song,
        gain_db=args.gain_db,
        seed=args.seed,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
