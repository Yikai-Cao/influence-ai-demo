"""Render an EvidenceReport as Markdown for download."""

from __future__ import annotations

from datetime import datetime

from mia_core import EvidenceReport


def format_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def render_markdown(r: EvidenceReport) -> str:
    pos = r.positive_test
    ctrl = r.false_positive_control
    verdict = r.verdict()

    lines = []
    lines.append(f"# Data Attribution Evidence Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.utcnow().isoformat()}Z  ")
    lines.append(f"**Model audited:** `{r.model_name}`  ")
    lines.append(f"**Suspect corpus size:** {r.n_suspect} passages  ")
    lines.append(f"**Control corpus size:** {r.n_control} passages  ")
    lines.append("")
    lines.append(f"## Verdict")
    lines.append("")
    lines.append(f"**{verdict}**")
    lines.append("")
    lines.append("## Headline statistics")
    lines.append("")
    lines.append("| Test | p-value | Interpretation |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Positive (suspect vs control) | **{format_p(pos['p_value'])}** | "
        f"p < 0.1 indicates suspect corpus looks like training data |"
    )
    lines.append(
        f"| False-positive control (control-A vs control-B) | **{format_p(ctrl['p_value'])}** | "
        f"p > 0.3 indicates pipeline is not producing spurious signal |"
    )
    lines.append("")
    lines.append(
        f"Classifier score on suspect: {pos['a_mean']:.4f}  "
        f"Classifier score on control: {pos['b_mean']:.4f}  "
        f"t = {pos['t_stat']:+.3f}"
    )
    lines.append("")
    lines.append("## Per-feature breakdown")
    lines.append("")
    lines.append("| Feature | Suspect mean loss | Control mean loss | t | p |")
    lines.append("|---|---:|---:|---:|---:|")
    for f in r.feature_names:
        pf = r.per_feature[f]
        lines.append(
            f"| `{f}` | {pf['member_mean']:.4f} | {pf['nonmember_mean']:.4f} | "
            f"{pf['t_stat']:+.3f} | {format_p(pf['p_value'])} |"
        )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "16 MIA features (perplexity, zlib ratio, 7 Min-K% variants, 7 Max-K% variants) "
        "are extracted from a single forward pass of the audited model. A logistic "
        "regression classifier is trained on half the corpus to separate suspect from "
        "control; scores on the held-out half are compared with a one-sided Welch's "
        "t-test. Methodology follows Maini et al., *LLM Dataset Inference* (NeurIPS 2024)."
    )
    lines.append("")
    lines.append("## Assumptions and limitations")
    lines.append("")
    lines.append(
        "- **Control corpus quality is critical.** The control set must be drawn from "
        "the same distribution as the suspect corpus but known not to be in the "
        "training set. Distribution mismatch (topic, length, style) can produce "
        "spurious positives or negatives.\n"
        "- **Gray-box access.** The method requires per-token log-probabilities from "
        "the model. It does not work on pure black-box APIs.\n"
        "- **Statistical claim only.** A significant p-value indicates the suspect "
        "corpus is statistically distinguishable from the control under this model — "
        "strong evidence, but not proof, of training-set inclusion.\n"
        "- **Effective sample size.** Passages from the same document may be "
        "correlated; the reported p-value assumes independence across passages."
    )
    lines.append("")
    return "\n".join(lines)
