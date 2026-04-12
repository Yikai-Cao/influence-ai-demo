"""
Core MIA pipeline for the evidence-report demo.

Refactored from phase1_reproduction/run_full_features.py into a reusable library:
- extract 16 features from a single forward pass
- train logistic-regression classifier on suspect + control
- run corpus-level one-sided t-test -> p-value

Inputs are plain lists of strings. The caller handles file parsing / UI.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np
from scipy import stats

K_VALUES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6]


# ── Feature extraction ────────────────────────────────────────────────

def compute_per_token_loss(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 512,
    batch_size: int = 4,
    device: str = "cpu",
    progress: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """Return per-token cross-entropy loss for each input text."""
    import torch

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    out_losses: list[list[float]] = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        labels = enc["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        shift_logits = out.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size(0), shift_labels.size(1))
        for row_losses, row_labels in zip(losses.tolist(), shift_labels.tolist()):
            kept = [l for l, lab in zip(row_losses, row_labels) if lab != -100]
            if kept:
                out_losses.append(kept)
        if progress is not None:
            progress(min(i + batch_size, total), total)
    return out_losses


def extract_16_features(per_token_losses: list[list[float]], texts: list[str]) -> dict[str, np.ndarray]:
    """Extract 16 MIA features (ppl, zlib_ratio, 7 Min-K%, 7 Max-K%)."""
    n = len(per_token_losses)
    feats: dict[str, np.ndarray] = {}
    feats["ppl"] = np.array([float(np.mean(l)) for l in per_token_losses])

    zlib_ratios = []
    for losses, text in zip(per_token_losses, texts[:n]):
        z = len(zlib.compress(text.encode("utf-8")))
        zlib_ratios.append(float(np.mean(losses)) / max(1, z))
    feats["zlib_ratio"] = np.array(zlib_ratios)

    for k in K_VALUES:
        vals = []
        for losses in per_token_losses:
            n_tok = max(1, int(len(losses) * k))
            top = sorted(losses, reverse=True)[:n_tok]
            vals.append(float(np.mean(top)))
        feats[f"k_min_probs_{k}"] = np.array(vals)

    for k in K_VALUES:
        vals = []
        for losses in per_token_losses:
            n_tok = max(1, int(len(losses) * k))
            bot = sorted(losses)[:n_tok]
            vals.append(float(np.mean(bot)))
        feats[f"k_max_probs_{k}"] = np.array(vals)

    return feats


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
    model_name: str
    n_suspect: int
    n_control: int
    positive_test: dict = field(default_factory=dict)
    false_positive_control: dict = field(default_factory=dict)
    per_feature: dict = field(default_factory=dict)
    feature_names: list = field(default_factory=list)

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


def run_evidence_report(
    suspect_texts: list[str],
    control_texts: list[str],
    model,
    tokenizer,
    model_name: str,
    max_length: int = 512,
    batch_size: int = 4,
    device: str = "cpu",
    progress: Callable[[str, float], None] | None = None,
) -> EvidenceReport:
    """Run the full pipeline and return an EvidenceReport.

    Requires 2x control texts relative to suspect — half train the classifier,
    the other half serves as test (non-member A) and false-positive control (B).
    """
    if len(control_texts) < 2 * len(suspect_texts):
        raise ValueError(
            f"control_texts ({len(control_texts)}) must be ≥ 2x suspect_texts "
            f"({len(suspect_texts)}). Half trains the classifier, the rest is used "
            f"for the positive test and false-positive control."
        )

    def _cb(done, total, stage):
        if progress:
            progress(stage, done / max(1, total))

    if progress:
        progress("Scoring suspect corpus", 0.0)
    suspect_losses = compute_per_token_loss(
        model, tokenizer, suspect_texts, max_length, batch_size, device,
        progress=lambda d, t: _cb(d, t, "Scoring suspect corpus"),
    )
    if progress:
        progress("Scoring control corpus", 0.0)
    control_losses = compute_per_token_loss(
        model, tokenizer, control_texts, max_length, batch_size, device,
        progress=lambda d, t: _cb(d, t, "Scoring control corpus"),
    )

    suspect_feats = extract_16_features(suspect_losses, suspect_texts)
    control_feats_all = extract_16_features(control_losses, control_texts)

    feature_names = sorted(suspect_feats.keys())

    half_s = len(suspect_feats["ppl"]) // 2
    half_c = len(control_feats_all["ppl"]) // 2

    clf_train_member = {f: v[:half_s] for f, v in suspect_feats.items()}
    clf_train_nonmember = {f: v[:half_c] for f, v in control_feats_all.items()}
    test_member = {f: v[half_s:] for f, v in suspect_feats.items()}
    ctrl_remainder = {f: v[half_c:] for f, v in control_feats_all.items()}
    m = len(ctrl_remainder["ppl"]) // 2
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
        n_suspect=len(suspect_texts),
        n_control=len(control_texts),
        positive_test=positive,
        false_positive_control=control,
        per_feature=per_feature,
        feature_names=feature_names,
    )


def load_model(model_name: str, device: str = "cpu"):
    """Load HF causal LM + tokenizer. Returns (model, tokenizer, device)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs = {"torch_dtype": use_dtype}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=use_dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer
