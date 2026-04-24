"""
Text MIA pipeline for the evidence-report demo.

- extract 16 features from a single forward pass (perplexity, zlib_ratio,
  7 Min-K%, 7 Max-K%)
- delegate classifier + t-test + EvidenceReport to mia_stats.py
- run corpus-level one-sided t-test → p-value

Inputs are plain lists of strings. The caller handles file parsing / UI.
"""

from __future__ import annotations

import zlib
from typing import Callable

import numpy as np

from mia_stats import (
    EvidenceReport,
    build_evidence_report,
    classifier_scores,
    one_sided_t_test_higher,
    train_classifier,
)

# Re-export for backward compatibility with existing callers (app.py, report.py).
__all__ = [
    "K_VALUES",
    "EvidenceReport",
    "classifier_scores",
    "compute_per_token_loss",
    "extract_16_features",
    "load_model",
    "one_sided_t_test_higher",
    "run_evidence_report",
    "train_classifier",
]

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


# ── End-to-end report ─────────────────────────────────────────────────

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
    """Run the full text pipeline and return an EvidenceReport.

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

    return build_evidence_report(
        suspect_feats=suspect_feats,
        control_feats_all=control_feats_all,
        model_name=model_name,
        n_suspect=len(suspect_texts),
        n_control=len(control_texts),
        metadata={"modality": "text"},
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
