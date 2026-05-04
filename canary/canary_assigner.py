"""
Per-song canary assignment for attribution-grade canarying.

The base canary system gives publishers corpus-level detection
("some song from your catalog leaked"). To get per-song attribution
("song 142 specifically leaked"), each song must carry a unique
canary fingerprint.

Two designs are supported here:

    Mode "unique"
        Each song gets a disjoint set of canaries (no overlap with any
        other song). Library size must be ≥ N_songs × n_per_song.
        Detection of a single canary uniquely identifies the source
        song. Strongest attribution; biggest library cost.

    Mode "combinatorial"
        Each song gets an n_per_song subset, drawn so that no two
        songs share ALL their canaries. Library can be much smaller
        (C(K, n_per_song) ≥ N_songs). A single hit narrows to a
        candidate set; multiple hits in one suspect output usually
        identify a unique source song. Cheaper library, slightly
        more complex attribution.

Mode "unique" is the default — simpler logic, more obvious to
explain to a publisher, the tradeoff is just "buy a bigger library".

Returns a master manifest:

    {
        "mode": "unique",
        "n_per_song": 3,
        "n_songs": 1000,
        "library_size": 3000,
        "seed": 0,
        "assignments": {
            "<host_path_1>": ["canary_017", "canary_142", "canary_889"],
            "<host_path_2>": ["canary_003", "canary_045", "canary_201"],
            ...
        }
    }

Used by batch_embed.py to drive the per-song embedding loop and by
canary_detector lookup helpers to turn a hit into a source-song claim.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path


# Pitch-transposition suffix regex: '_tm0.50' / '_tp1.00' / etc.
# Used to collapse transposition variants down to their LOGICAL canary
# so a single logical canary is owned by exactly one song.
_TRANSP_SUFFIX = re.compile(r"_t[mp]\d+\.\d+$")


def logical_canary_id(canary_id: str) -> str:
    """Strip a transposition suffix if present.

    ``canary_017_tm0.50`` → ``canary_017``
    ``canary_017_tp1.00`` → ``canary_017``
    ``canary_017``        → ``canary_017``
    """
    return _TRANSP_SUFFIX.sub("", canary_id)


def group_by_logical(canary_index: dict) -> dict[str, list[str]]:
    """{logical_id → [variant canary_ids]}, preserving variant order."""
    groups: dict[str, list[str]] = defaultdict(list)
    for entry in canary_index["canaries"]:
        groups[logical_canary_id(entry["canary_id"])].append(entry["canary_id"])
    return dict(groups)


def assign_unique(
    host_paths: list[Path],
    canary_index: dict,
    n_per_song: int = 3,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Disjoint canary assignment — each LOGICAL canary used by at most one song.

    n_per_song means "logical canaries per song". The returned IDs are
    the *canary IDs to embed* (one variant per logical canary, picked
    deterministically — preferring the 0-pitch variant when it exists).

    Attribution works via `logical_canary_id()`: a hit on ANY pitch
    variant of an assigned logical canary maps back to the owning song,
    even though only one variant was physically embedded. This keeps
    pitch-shift robustness intact while staying within host duration
    budgets (no need to fit all 3 transposition variants per song).

    Returns {str(host_path) → [canary_id_to_embed, ...]} of length
    n_per_song. Use `logical_canary_id()` on returned IDs to recover
    the logical canary an entry represents.
    """
    n_songs = len(host_paths)
    logical_groups = group_by_logical(canary_index)
    n_logical = len(logical_groups)
    needed = n_songs * n_per_song
    if n_logical < needed:
        raise ValueError(
            f"Library has {n_logical} LOGICAL canaries but unique mode needs "
            f"{n_songs} songs × {n_per_song} per song = {needed}. "
            "Generate a larger library (more logical canaries — counted by "
            "unique seed, not by transposition variant) or switch to "
            "combinatorial mode."
        )

    rng = random.Random(seed)
    logicals = sorted(logical_groups.keys())
    rng.shuffle(logicals)

    def _pick_variant_to_embed(variants: list[str], logical_id: str) -> str:
        """Prefer the 0-pitch (suffix-less) variant; else the first."""
        for v in variants:
            if v == logical_id:
                return v
        return variants[0]

    assignments: dict[str, list[str]] = {}
    cursor = 0
    for hp in host_paths:
        chosen_logicals = logicals[cursor:cursor + n_per_song]
        embed_ids = [
            _pick_variant_to_embed(logical_groups[lc], lc)
            for lc in chosen_logicals
        ]
        assignments[str(hp)] = embed_ids
        cursor += n_per_song
    return assignments


def assign_combinatorial(
    host_paths: list[Path],
    canary_index: dict,
    n_per_song: int = 3,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Each song gets a unique n_per_song subset (overlapping subsets allowed).

    Library can be smaller — only needs C(K, n_per_song) ≥ N_songs.
    Detection of a single canary narrows to a candidate set;
    detection of all n_per_song canaries in one suspect uniquely
    identifies the source song.
    """
    from math import comb
    n_songs = len(host_paths)
    n_total = len(canary_index["canaries"])
    capacity = comb(n_total, n_per_song)
    if capacity < n_songs:
        raise ValueError(
            f"Library has only C({n_total}, {n_per_song}) = {capacity} "
            f"unique subsets but {n_songs} songs requested. Generate a "
            "larger library or increase n_per_song."
        )

    rng = random.Random(seed)
    used: set[tuple[int, ...]] = set()
    assignments: dict[str, list[str]] = {}
    canary_ids = [c["canary_id"] for c in canary_index["canaries"]]
    indices = list(range(n_total))

    for hp in host_paths:
        # Sample subsets without replacement until we hit a fresh one.
        # For typical library sizes this completes in O(1) attempts;
        # cap retries to fail fast on degenerate cases.
        for _ in range(100):
            picked = tuple(sorted(rng.sample(indices, n_per_song)))
            if picked not in used:
                used.add(picked)
                assignments[str(hp)] = [canary_ids[i] for i in picked]
                break
        else:
            raise RuntimeError(
                f"Could not find a unique subset for {hp} after 100 tries — "
                "library is near capacity, try a larger one."
            )
    return assignments


# ── Detection-side: hit → song attribution ──────────────────────────

def build_canary_to_songs_index(
    master_manifest: dict,
) -> dict[str, list[str]]:
    """Reverse the assignment map: {canary_id → [host_path, ...]}.

    Keyed by raw canary_id (variant-level). Use
    build_logical_canary_to_songs for hit-attribution — pitch-shift
    transpositions of an embedded canary still carry the same logical
    ownership.
    """
    rev: dict[str, list[str]] = defaultdict(list)
    for host_path, canaries in master_manifest["assignments"].items():
        for cid in canaries:
            rev[cid].append(host_path)
    return dict(rev)


def build_logical_canary_to_songs(
    master_manifest: dict,
) -> dict[str, list[str]]:
    """{logical_canary_id → [host_path, ...]} — what attribution should use.

    A hit on `canary_017_tp0.50` belongs to whoever owns logical canary
    `canary_017` regardless of which variant they actually embedded.
    """
    rev: dict[str, list[str]] = defaultdict(list)
    for host_path, canaries in master_manifest["assignments"].items():
        for cid in canaries:
            rev[logical_canary_id(cid)].append(host_path)
    return dict(rev)


def attribute_hits_to_songs(
    detection_report: dict,
    master_manifest: dict,
    threshold: float | None = None,
) -> dict:
    """Turn detector hits into source-song attribution.

    For each (canary, suspect_clip) pair above threshold, look up which
    songs had that canary embedded. With unique-mode assignment, exactly
    one song. With combinatorial mode, possibly many — we then look at
    the FULL set of canaries that hit in this suspect clip and find the
    one source song whose subset matches best.

    Returns:
        {
            "by_suspect": {
                "<suspect_clip>": {
                    "source_songs": ["song_142"],   # ranked candidates
                    "matched_canaries": [{"canary_id": ..., "sim": ..., "offset_s": ...}],
                }
            },
            "summary": {
                "n_suspects_attributed": 7,
                "n_unique_source_songs": 5,
                "uniquely_identified_songs": [...],
            },
        }
    """
    if threshold is None:
        threshold = detection_report.get("threshold", 0.55)
    logical_to_songs = build_logical_canary_to_songs(master_manifest)

    # Group hits by suspect clip
    by_suspect: dict[str, list[dict]] = defaultdict(list)
    for pair in detection_report["per_pair_scores"]:
        if pair["max_similarity"] >= threshold:
            by_suspect[pair["suspect_path"]].append(pair)

    out: dict = {"by_suspect": {}}
    all_attributed = set()
    n_per_song = master_manifest.get("n_per_song", 3)

    for suspect, hits in by_suspect.items():
        # Collapse pitch-transposition variants of the same logical
        # canary into a single hit before counting — a +0.5 / 0 / -0.5
        # set of three matches is one logical canary, not three.
        hit_logicals = {logical_canary_id(h["canary_id"]) for h in hits}

        # Score each candidate song by how many of its assigned LOGICAL
        # canaries surfaced as hits.
        candidate_scores: dict[str, int] = {}
        for cid in hit_logicals:
            for s in logical_to_songs.get(cid, []):
                candidate_scores[s] = candidate_scores.get(s, 0) + 1

        # Rank candidates by hit count, then by fraction-of-subset
        ranked = sorted(
            candidate_scores.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        if not ranked:
            continue

        # A song is "uniquely identified" only if it has at least 2
        # logical-canary hits AND strictly dominates the runner-up.
        # The 2-hit floor screens out single-canary cross-firings (one
        # spurious match against an unassigned canary by spectral
        # coincidence), which are the dominant noise mode at typical
        # detection thresholds.
        min_hits_for_unique = max(2, n_per_song - 1)
        top_score = ranked[0][1]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0
        unique = (top_score >= min_hits_for_unique
                  and top_score > runner_up)

        out["by_suspect"][suspect] = {
            "source_songs_ranked": [
                {"host_path": s, "canaries_hit": c, "n_per_song": n_per_song}
                for s, c in ranked
            ],
            "uniquely_identified": unique,
            "matched_canaries": [
                {
                    "canary_id": h["canary_id"],
                    "max_similarity": h["max_similarity"],
                    "best_offset_s": h["best_offset_s"],
                }
                for h in hits
            ],
        }
        if unique:
            all_attributed.add(ranked[0][0])

    out["summary"] = {
        "n_suspects_with_hits": len(out["by_suspect"]),
        "n_uniquely_identified_source_songs": len(all_attributed),
        "uniquely_identified_songs": sorted(all_attributed),
    }
    return out
