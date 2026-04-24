"""
Shared statistics layer for MIA evidence reports.

Domain-agnostic pieces used by both text (mia_core.py) and audio
(mia_audio_core.py): logistic-regression classifier, Welch's t-test,
EvidenceReport dataclass, and the split/train/test/report orchestration.

Keep this file text-free and audio-free — it operates on dicts of
numpy feature vectors and knows nothing about tokens, codecs, or captions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import stats


# ── Classifier + stats ────────────────────────────────────────────────

def train_classifier(
    member_feats: dict[str, np.ndarray],
    nonmember_feats: dict[str, np.ndarray],
    feature_names: list[str],
):
    """Logistic regression on (member=1, non-member=0). Returns (clf, mu, sd)."""
    X_m = np.column_stack([member_feats[f] for f in feature_names])
    X_n = np.column_stack([nonmember_feats[f] for f in feature_names])
    X = np.vstack([X_m, X_n])
    y = np.concatenate([np.ones(len(X_m)), np.zeros(len(X_n))])

    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    Xn = np.clip((X - mu) / sd, -3, 3)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(Xn, y)
    return clf, mu, sd


def classifier_scores(clf, feats, feature_names, mu, sd) -> np.ndarray:
    X = np.column_stack([feats[f] for f in feature_names])
    Xn = np.clip((X - mu) / sd, -3, 3)
    return clf.predict_proba(Xn)[:, 1]


def one_sided_t_test_higher(a: np.ndarray, b: np.ndarray) -> dict:
    """H1: mean(a) > mean(b)."""
    t_stat, p_val = stats.ttest_ind(a, b, alternative="greater", equal_var=False)
    return {
        "a_mean": float(np.mean(a)),
        "b_mean": float(np.mean(b)),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }


# ── End-to-end report ─────────────────────────────────────────────────

@dataclass
class EvidenceReport:
    """Corpus-level MIA evidence report. Domain-agnostic."""
    model_name: str
    n_suspect: int
    n_control: int
    positive_test: dict = field(default_factory=dict)
    false_positive_control: dict = field(default_factory=dict)
    per_feature: dict = field(default_factory=dict)
    feature_names: list = field(default_factory=list)
    # Optional domain-specific metadata (e.g., "modality": "audio",
    # "suspect_caption_mode": "empty"). Keep free-form so UI/memo layer
    # can render without coupling back.
    metadata: dict = field(default_factory=dict)

    def verdict(self) -> str:
        p_pos = self.positive_test.get("p_value", 1.0)
        p_ctrl = self.false_positive_control.get("p_value", 0.0)
        if p_pos < 0.01 and p_ctrl > 0.3:
            return "STRONG evidence of training-set membership"
        if p_pos < 0.1 and p_ctrl > 0.3:
            return "MODERATE evidence of training-set membership"
        if p_ctrl <= 0.3:
            return "INCONCLUSIVE — control test failed (possible distribution mismatch)"
        return "NO evidence of training-set membership"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceReport":
        return cls(**d)


def build_evidence_report(
    suspect_feats: dict[str, np.ndarray],
    control_feats_all: dict[str, np.ndarray],
    model_name: str,
    n_suspect: int,
    n_control: int,
    metadata: dict | None = None,
) -> EvidenceReport:
    """Run split → train classifier → positive test + FPR control → per-feature t-tests.

    Expects TWO feature dicts of equal key set, shape (n_suspect,) and
    (n_control,). Splits control in half for classifier training + testing,
    then splits the remainder in half again for the FPR control.

    Callers should have already run per-passage feature extraction. The
    split logic + statistical tests are the same as the original
    mia_core.run_evidence_report.
    """
    feature_names = sorted(suspect_feats.keys())

    half_s = len(suspect_feats[feature_names[0]]) // 2
    half_c = len(control_feats_all[feature_names[0]]) // 2

    clf_train_member = {f: v[:half_s] for f, v in suspect_feats.items()}
    clf_train_nonmember = {f: v[:half_c] for f, v in control_feats_all.items()}
    test_member = {f: v[half_s:] for f, v in suspect_feats.items()}
    ctrl_remainder = {f: v[half_c:] for f, v in control_feats_all.items()}
    m = len(ctrl_remainder[feature_names[0]]) // 2
    test_nonmember_A = {f: v[:m] for f, v in ctrl_remainder.items()}
    test_nonmember_B = {f: v[m:] for f, v in ctrl_remainder.items()}

    clf, mu, sd = train_classifier(clf_train_member, clf_train_nonmember, feature_names)

    s_member = classifier_scores(clf, test_member, feature_names, mu, sd)
    s_A = classifier_scores(clf, test_nonmember_A, feature_names, mu, sd)
    s_B = classifier_scores(clf, test_nonmember_B, feature_names, mu, sd)

    positive = one_sided_t_test_higher(s_member, s_A)
    control = one_sided_t_test_higher(s_A, s_B)

    per_feature = {}
    for f in feature_names:
        t_stat, p_val = stats.ttest_ind(
            test_member[f], test_nonmember_A[f], alternative="less", equal_var=False
        )
        per_feature[f] = {
            "member_mean": float(np.mean(test_member[f])),
            "nonmember_mean": float(np.mean(test_nonmember_A[f])),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
        }

    return EvidenceReport(
        model_name=model_name,
        n_suspect=n_suspect,
        n_control=n_control,
        positive_test=positive,
        false_positive_control=control,
        per_feature=per_feature,
        feature_names=feature_names,
        metadata=metadata or {},
    )
