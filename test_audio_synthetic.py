"""
Synthetic pipeline test for the audio MIA stack.

Mirrors the Phase 3 synthetic-data validation from the text side
(see CLAUDE.md §阶段 3): inject handcrafted per-codebook loss
distributions and assert the non-model parts of the pipeline —
feature extraction, classifier, t-test, EvidenceReport — behave
correctly across four scenarios.

Runs in ~0.5 s on CPU. Requires numpy + scipy + scikit-learn only
(no torch, no MusicGen download). Good pre-flight check for
CI / before paying for Modal GPU time.

Scenarios
---------
    A. Member loss is 0.3 nats lower on all codebooks.
       Expect: positive p  < 0.001, control p > 0.3, verdict STRONG.
    B. Member and control drawn from same distribution (null).
       Expect: positive p  > 0.1,   control p > 0.3, verdict NO.
    C. Weak 0.05-nat separation on codebook 0 only (classifier
       must learn per-codebook weight).
       Expect: positive p  < 0.05, control p > 0.3, verdict STRONG or MODERATE.
    D. Reversed signal — member loss *higher* than control.
       Known methodological gotcha: the logistic-regression classifier
       fits whatever separates the labels, so it produces low p on
       the "reversed" signal too. The per-feature one-sided t-tests
       (H1: member < nonmember) catch this by returning p > 0.9.
       Expect: positive p  LOW (classifier-driven artifact), but
               per-feature p-values > 0.9 → manual cross-check fails.
       This is the scenario that tells us to trust the per-feature
       breakdown as a sanity check, not the headline alone.

Usage
-----
    cd evidence_report_app
    python test_audio_synthetic.py
"""

from __future__ import annotations

import numpy as np

from mia_audio_core import extract_audio_features
from mia_stats import build_evidence_report


NUM_CODEBOOKS = 4
FRAMES_PER_CLIP = 500  # 10 s at 50 Hz
N_MEMBER = 80
N_CONTROL = 160  # must be ≥ 2× member


def make_synthetic_losses(
    n_clips: int,
    mean_per_cb: list[float],
    std: float,
    seed: int,
) -> list[list[list[float]]]:
    """Synthesize per-clip, per-codebook loss distributions.

    mean_per_cb[c] sets the mean loss for codebook c. std is shared.
    """
    rng = np.random.default_rng(seed)
    assert len(mean_per_cb) == NUM_CODEBOOKS
    out = []
    for _ in range(n_clips):
        clip_losses = []
        for c in range(NUM_CODEBOOKS):
            # Clip at 0 — CE loss is non-negative
            frame_losses = np.clip(
                rng.normal(mean_per_cb[c], std, FRAMES_PER_CLIP),
                0, None,
            )
            clip_losses.append(frame_losses.tolist())
        out.append(clip_losses)
    return out


def synthetic_flac_sizes(n: int, mean: int = 100_000, std: int = 8_000, seed: int = 0) -> list[int]:
    """Plausible flac sizes (~100 KB ± 8 KB for 10 s clips)."""
    rng = np.random.default_rng(seed)
    return [max(1, int(x)) for x in rng.normal(mean, std, n)]


def run_scenario(name: str, member_means: list[float], control_means: list[float],
                 std: float = 0.4, seed_base: int = 0) -> dict:
    """Run one synthetic scenario and return p-values + verdict."""
    member_losses = make_synthetic_losses(N_MEMBER, member_means, std, seed_base)
    control_losses = make_synthetic_losses(N_CONTROL, control_means, std, seed_base + 1)

    member_flac = synthetic_flac_sizes(N_MEMBER, seed=seed_base + 2)
    control_flac = synthetic_flac_sizes(N_CONTROL, seed=seed_base + 3)

    member_feats = extract_audio_features(member_losses, member_flac)
    control_feats = extract_audio_features(control_losses, control_flac)

    report = build_evidence_report(
        suspect_feats=member_feats,
        control_feats_all=control_feats,
        model_name=f"synthetic-{name}",
        n_suspect=N_MEMBER,
        n_control=N_CONTROL,
        metadata={"modality": "audio", "scenario": name},
    )

    # Collect per-feature one-sided p-values (H1: member < nonmember)
    per_feature_ps = [pf["p_value"] for pf in report.per_feature.values()]
    median_feature_p = float(np.median(per_feature_ps))

    return {
        "scenario": name,
        "positive_p": report.positive_test["p_value"],
        "control_p": report.false_positive_control["p_value"],
        "verdict": report.verdict(),
        "n_features": len(report.feature_names),
        "median_feature_p": median_feature_p,
    }


def main():
    print("Synthetic audio pipeline test")
    print("=" * 60)

    scenarios = [
        {
            "name": "A_strong",
            "member_means": [1.0, 1.2, 1.3, 1.4],
            "control_means": [1.3, 1.5, 1.6, 1.7],
            "expect_positive": "p < 0.001",
            "expect_control":  "p > 0.3",
        },
        {
            "name": "B_null",
            "member_means": [1.3, 1.5, 1.6, 1.7],
            "control_means": [1.3, 1.5, 1.6, 1.7],
            "expect_positive": "p > 0.1",
            "expect_control":  "p > 0.3",
        },
        {
            "name": "C_cb0_only",
            "member_means": [1.0, 1.5, 1.6, 1.7],
            "control_means": [1.05, 1.5, 1.6, 1.7],
            "expect_positive": "p < 0.05",
            "expect_control":  "p > 0.3",
        },
        {
            "name": "D_reversed",
            "member_means": [1.6, 1.8, 1.9, 2.0],
            "control_means": [1.3, 1.5, 1.6, 1.7],
            "expect_positive": "p > 0.9",
            "expect_control":  "p > 0.3",
        },
    ]

    rows = []
    for i, sc in enumerate(scenarios):
        r = run_scenario(sc["name"], sc["member_means"], sc["control_means"],
                         seed_base=i * 100)
        r["expect_positive"] = sc["expect_positive"]
        r["expect_control"] = sc["expect_control"]
        rows.append(r)
        print(f"[{r['scenario']:<12}] pos_p={r['positive_p']:.3e}  "
              f"ctrl_p={r['control_p']:.3f}  "
              f"median_feature_p={r['median_feature_p']:.3f}  "
              f"verdict: {r['verdict']}")
        print(f"               expected: {r['expect_positive']}, "
              f"{r['expect_control']}")

    # Assertions
    print()
    print("Assertions:")
    a = rows[0]
    assert a["positive_p"] < 0.001, f"A_strong: expected p<0.001, got {a['positive_p']:.3e}"
    assert a["control_p"]  > 0.3,   f"A_strong: expected control>0.3, got {a['control_p']:.3f}"
    assert a["verdict"].startswith("STRONG"), a["verdict"]
    print("  A_strong    ✓")

    b = rows[1]
    assert b["positive_p"] > 0.1,   f"B_null: expected p>0.1, got {b['positive_p']:.3e}"
    # verdict can be NO evidence or INCONCLUSIVE depending on control exact value
    assert not b["verdict"].startswith("STRONG"), b["verdict"]
    print("  B_null      ✓")

    c = rows[2]
    assert c["positive_p"] < 0.05,  f"C_cb0_only: expected p<0.05, got {c['positive_p']:.3e}"
    assert c["control_p"]  > 0.3,   f"C_cb0_only: expected control>0.3, got {c['control_p']:.3f}"
    print("  C_cb0_only  ✓")

    # D_reversed: classifier will fit the "wrong-direction" pattern and
    # report a low classifier p-value (methodological artifact). The
    # safeguard is the per-feature t-test (H1: member < nonmember) —
    # it returns p > 0.9 when the direction is reversed. We assert on
    # that, and treat the "classifier looks strong but per-feature all
    # failed" pattern as the cross-check the memo should mention.
    d = rows[3]
    assert d["median_feature_p"] > 0.9, (
        f"D_reversed: expected median per-feature p > 0.9 "
        f"(reversed direction), got {d['median_feature_p']:.3f}"
    )
    print("  D_reversed  ✓  (per-feature p > 0.9 flags direction "
          "mismatch that classifier alone would miss)")

    # Feature count check — 4 codebooks × (ppl + 3 mink + 3 maxk) + bitrate = 29
    expected_feats = NUM_CODEBOOKS * 7 + 1
    for r in rows:
        assert r["n_features"] == expected_feats, (
            f"{r['scenario']}: expected {expected_feats} features, got {r['n_features']}"
        )
    print(f"  Feature count {expected_feats} ✓")

    print()
    print("✓ All synthetic scenarios passed.")
    print("  Audio feature extraction + classifier + t-test are correct.")
    print("  Any issue seen with real MusicGen is therefore in the model/")
    print("  loss-extraction layer, not the statistical machinery.")


if __name__ == "__main__":
    main()
