"""Render an EvidenceReport as Markdown for download.

Handles both text and audio reports by branching on
`report.metadata.get('modality')`. Text reports render exactly as
before (no regression).
"""

from __future__ import annotations

from datetime import datetime

from mia_stats import EvidenceReport  # shared across text + audio


def format_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _unit(modality: str) -> str:
    """Passages (text) vs clips (audio)."""
    return "clips" if modality == "audio" else "passages"


def _method_paragraph(modality: str, r: EvidenceReport) -> str:
    if modality == "audio":
        n_cb = r.metadata.get("num_codebooks", 4)
        clip_s = r.metadata.get("clip_seconds")
        clip_note = f" Each clip is scored over its first {clip_s:.0f} s." if clip_s else ""
        return (
            f"Per-codebook MIA features (perplexity + Min-K% + Max-K% at "
            f"k ∈ {{0.1, 0.2, 0.4}}) are extracted for each of the audio "
            f"model's {n_cb} EnCodec codebooks from a single forward pass, "
            f"plus a global bitrate-normalized loss (`bitrate_ratio`)."
            f"{clip_note} A logistic regression classifier is trained on "
            f"half the corpus to separate suspect from control; scores on "
            f"the held-out half are compared with a one-sided Welch's "
            f"t-test. Methodology adapted from Maini et al., *LLM Dataset "
            f"Inference* (NeurIPS 2024) to audio."
        )
    return (
        "16 MIA features (perplexity, zlib ratio, 7 Min-K% variants, "
        "7 Max-K% variants) are extracted from a single forward pass of "
        "the audited model. A logistic regression classifier is trained "
        "on half the corpus to separate suspect from control; scores on "
        "the held-out half are compared with a one-sided Welch's t-test. "
        "Methodology follows Maini et al., *LLM Dataset Inference* "
        "(NeurIPS 2024)."
    )


def _limitations(modality: str) -> str:
    common = (
        "- **Control corpus quality is critical.** The control set must be "
        "drawn from the same distribution as the suspect corpus but known "
        "not to be in the training set. Distribution mismatch (topic, "
        "length, style) can produce spurious positives or negatives.\n"
        "- **Gray-box access.** The method requires per-token (or "
        "per-codec-token) log-probabilities from the model. It does not "
        "work on pure black-box APIs.\n"
        "- **Statistical claim only.** A significant p-value indicates "
        "the suspect corpus is statistically distinguishable from the "
        "control under this model — strong evidence, but not proof, of "
        "training-set inclusion.\n"
    )
    if modality == "audio":
        return common + (
            "- **Prompt conditioning matters.** MusicGen is text-conditional. "
            "The same audio scored with an empty prompt vs. its real caption "
            "can yield very different losses; check `metadata.prompt_mode` "
            "before comparing runs.\n"
            "- **Chunk alignment.** Scoring clips of a different length than "
            "what the model saw during training can mask the memorization "
            "signal. Fine-tune length should equal scoring length.\n"
            "- **Classifier direction check.** The logistic classifier fits "
            "whatever separates the two groups. If the per-feature one-sided "
            "t-tests (H1: member < nonmember) mostly return p > 0.5 while "
            "the headline p is low, the headline may be fitting a reversed "
            "distribution, not genuine membership signal — flag as "
            "inconclusive.\n"
        )
    return common + (
        "- **Effective sample size.** Passages from the same document may "
        "be correlated; the reported p-value assumes independence across "
        "passages.\n"
    )


def render_markdown(r: EvidenceReport) -> str:
    pos = r.positive_test
    ctrl = r.false_positive_control
    verdict = r.verdict()
    modality = r.metadata.get("modality", "text")
    unit = _unit(modality)

    lines = []
    lines.append(f"# Data Attribution Evidence Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.utcnow().isoformat()}Z  ")
    lines.append(f"**Modality:** {modality}  ")
    lines.append(f"**Model audited:** `{r.model_name}`  ")
    lines.append(f"**Suspect corpus size:** {r.n_suspect} {unit}  ")
    lines.append(f"**Control corpus size:** {r.n_control} {unit}  ")
    if modality == "audio":
        if r.metadata.get("clip_seconds"):
            lines.append(f"**Clip length:** {r.metadata['clip_seconds']:.0f} s  ")
        if r.metadata.get("suspect_prompt_mode"):
            lines.append(f"**Suspect prompt mode:** "
                         f"`{r.metadata['suspect_prompt_mode']}`  ")
        if r.metadata.get("num_codebooks"):
            lines.append(f"**EnCodec codebooks:** "
                         f"{r.metadata['num_codebooks']}  ")
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
        f"| False-positive control (control-A vs control-B) | "
        f"**{format_p(ctrl['p_value'])}** | "
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
    loss_col_label = ("Suspect mean (loss / ratio)" if modality == "audio"
                      else "Suspect mean loss")
    ctrl_col_label = ("Control mean (loss / ratio)" if modality == "audio"
                      else "Control mean loss")
    lines.append(f"| Feature | {loss_col_label} | {ctrl_col_label} | t | p |")
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
    lines.append(_method_paragraph(modality, r))
    lines.append("")
    lines.append("## Assumptions and limitations")
    lines.append("")
    lines.append(_limitations(modality))
    lines.append("")
    return "\n".join(lines)
