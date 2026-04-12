"""
Streamlit evidence-report demo for Influence AI.

Upload a suspect corpus + control corpus, pick an audited model, run MIA,
and download a p-value-backed evidence report.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import streamlit as st

from mia_core import load_model, run_evidence_report, EvidenceReport
from report import render_markdown, format_p

APP_DIR = Path(__file__).parent
EXAMPLES_DIR = APP_DIR / "examples"


@st.cache_resource(show_spinner="Loading model weights (cached across users)…")
def cached_load_model(model_name: str, device: str):
    """Cache model + tokenizer across sessions so concurrent users share one copy."""
    return load_model(model_name, device=device)


def load_example_jsonl(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        t = obj.get("text") or obj.get("content") or ""
        if t:
            out.append(t)
    return out


st.set_page_config(page_title="Influence AI — Evidence Report", layout="wide")

st.title("Influence AI — Data Attribution Evidence Report")
st.caption(
    "Membership inference on a suspect corpus against an audited language model. "
    "Method: Maini et al., *LLM Dataset Inference* (NeurIPS 2024), 16-feature subset "
    "with learned logistic regression."
)

st.warning(
    "**This hosted demo uses Pythia-160m on CPU** for a fast, shareable UI walkthrough. "
    "Pythia-160m does not memorize The Pile heavily enough to produce a decisive "
    "p-value — expect an **INCONCLUSIVE or NO evidence** verdict on the example corpus. "
    "The headline result from our experiments (**p ≈ 4.46 × 10⁻⁵ on Pile Wikipedia**) "
    "requires Pythia-6.9B on an A10G GPU via the Modal backend — see the "
    "[README](https://github.com/Yikai-Cao/influence-ai-demo) to run that locally."
)

# ── Sidebar: config ───────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")
    model_name = st.selectbox(
        "Audited model (HuggingFace)",
        [
            "EleutherAI/pythia-160m-deduped",
            "EleutherAI/pythia-1.4b-deduped",
            "EleutherAI/pythia-2.8b-deduped",
            "EleutherAI/pythia-6.9b-deduped",
            "custom…",
        ],
        index=0,
        help="For a shareable laptop demo, pick 160m. Larger models need a GPU backend.",
    )
    if model_name == "custom…":
        model_name = st.text_input("Custom model name", value="EleutherAI/pythia-160m-deduped")

    backend = st.radio(
        "Backend",
        ["local (CPU/GPU)", "Modal A10G (cloud)"],
        index=0,
        help="Local is fine for pythia-160m smoke tests. Use Modal A10G for 1.4B+ "
             "models — A10G has 24 GB VRAM, ~$1.10/hr, ~10-15 min per audit.",
    )
    is_modal = backend.startswith("Modal")
    device = "cuda" if is_modal else st.radio("Local device", ["cpu", "cuda"], index=0, horizontal=True)
    max_length = st.slider("Max tokens per passage", 128, 1024, 512, step=128)
    batch_size = st.slider("Batch size", 1, 16, 4 if is_modal else 4)

    st.divider()
    st.markdown(
        "**Corpus requirements**  \n"
        "• Suspect: ≥ 50 passages recommended  \n"
        "• Control: must be ≥ 2× suspect size  \n"
        "• File formats: `.txt` (one passage per line) or `.jsonl` with a `text` field"
    )


# ── File upload ───────────────────────────────────────────────────────

def parse_corpus(uploaded) -> list[str]:
    if uploaded is None:
        return []
    raw = uploaded.read().decode("utf-8", errors="replace")
    if uploaded.name.lower().endswith(".jsonl"):
        out = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text") or obj.get("content") or ""
            if text:
                out.append(text)
        return out
    return [l for l in raw.splitlines() if l.strip()]


example_suspect = EXAMPLES_DIR / "pile_wikipedia_suspect.jsonl"
example_control = EXAMPLES_DIR / "pile_wikipedia_control.jsonl"
example_available = example_suspect.exists() and example_control.exists()

if example_available:
    st.info(
        "**Example corpus available.** 500 suspect + 1000 control Wikipedia passages "
        "from the Pile (`pratyushmaini/llm_dataset_inference`). Pick a demo size "
        "that fits your patience on CPU."
    )
    demo_size = st.radio(
        "Demo size",
        ["Quick (50 / 100, ~2 min)", "Medium (150 / 300, ~6 min)", "Full (500 / 1000, ~20 min)"],
        index=0,
        horizontal=True,
        help="Subsamples the bundled corpus. Quick is enough to see the pipeline; "
             "Full matches Phase 1b exactly.",
    )
    _size_to_n = {"Quick": 50, "Medium": 150, "Full": 500}
    n_suspect = next(v for k, v in _size_to_n.items() if demo_size.startswith(k))

    if st.button("Load Pile Wikipedia example"):
        suspect_all = load_example_jsonl(example_suspect)
        control_all = load_example_jsonl(example_control)
        st.session_state["example_suspect"] = suspect_all[:n_suspect]
        st.session_state["example_control"] = control_all[:2 * n_suspect]
        st.session_state.pop("report", None)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Suspect corpus")
    st.caption("Content you believe may have been used to train the model.")
    suspect_file = st.file_uploader("Upload .txt or .jsonl", type=["txt", "jsonl"], key="suspect")

with col_b:
    st.subheader("Control corpus")
    st.caption("Similar-distribution content known *not* to be in training (e.g. post-cutoff).")
    control_file = st.file_uploader("Upload .txt or .jsonl", type=["txt", "jsonl"], key="control")

suspect_texts = parse_corpus(suspect_file) or st.session_state.get("example_suspect", [])
control_texts = parse_corpus(control_file) or st.session_state.get("example_control", [])

if suspect_texts and "example_suspect" in st.session_state and not suspect_file:
    st.caption("Using bundled Pile Wikipedia example corpus.")

if suspect_texts or control_texts:
    c1, c2 = st.columns(2)
    c1.metric("Suspect passages", len(suspect_texts))
    c2.metric("Control passages", len(control_texts))
    if control_texts and len(control_texts) < 2 * len(suspect_texts):
        st.warning(
            f"Control ({len(control_texts)}) must be at least 2× suspect "
            f"({len(suspect_texts)}) — need {2 * len(suspect_texts)}."
        )


# ── Run ───────────────────────────────────────────────────────────────

run_btn = st.button(
    "Run audit",
    type="primary",
    disabled=not (suspect_texts and control_texts and len(control_texts) >= 2 * len(suspect_texts)),
)

if run_btn:
    if is_modal:
        status = st.status(f"Running on Modal A10G ({model_name}) …", expanded=True)
        with status:
            st.write("Connecting to Modal app `influence-ai-evidence-report` …")
            try:
                from modal_backend import app as modal_app, run_audit_remote
                st.write(
                    f"Dispatching {len(suspect_texts)} suspect + {len(control_texts)} "
                    "control passages. Expect ~10–15 min including cold start."
                )
                with modal_app.run():
                    result_dict = run_audit_remote.remote(
                        suspect_texts,
                        control_texts,
                        model_name=model_name,
                        max_length=max_length,
                        batch_size=batch_size,
                    )
                report = EvidenceReport.from_dict(result_dict)
                status.update(label="Modal run complete", state="complete")
                st.session_state["report"] = report
            except Exception as e:
                status.update(label=f"Modal run failed: {type(e).__name__}", state="error")
                st.exception(e)
                st.info(
                    "Modal setup required: `pip install modal` then `modal setup`. "
                    "Make sure you're authenticated to the `ykcao` workspace."
                )
    else:
        status = st.status(f"Loading {model_name} …", expanded=True)
        progress_bar = st.progress(0.0)

        def progress_cb(stage: str, frac: float):
            status.update(label=f"{stage} ({frac * 100:.0f}%)")
            progress_bar.progress(min(1.0, frac))

        with status:
            st.write("Loading model weights (cached after first run)…")
            model, tokenizer = cached_load_model(model_name, device)
            st.write("Running MIA pipeline…")
            report = run_evidence_report(
                suspect_texts,
                control_texts,
                model,
                tokenizer,
                model_name=model_name,
                max_length=max_length,
                batch_size=batch_size,
                device=device,
                progress=progress_cb,
            )
        status.update(label="Done", state="complete")
        progress_bar.progress(1.0)
        st.session_state["report"] = report


# ── Render results ────────────────────────────────────────────────────

if "report" in st.session_state:
    r = st.session_state["report"]
    st.header("Results")

    verdict = r.verdict()
    color = {
        "STRONG": "red",
        "MODERATE": "orange",
        "INCONCLUSIVE": "gray",
        "NO": "green",
    }
    badge_color = next((v for k, v in color.items() if verdict.startswith(k)), "gray")
    st.markdown(f":{badge_color}[**{verdict}**]")

    c1, c2, c3 = st.columns(3)
    c1.metric("Positive p-value", format_p(r.positive_test["p_value"]),
              help="p < 0.1 → suspect corpus looks like training data")
    c2.metric("Control p-value", format_p(r.false_positive_control["p_value"]),
              help="p > 0.3 → pipeline is not producing spurious signal")
    c3.metric("Classifier t-stat", f"{r.positive_test['t_stat']:+.3f}")

    with st.expander("Per-feature breakdown", expanded=False):
        import pandas as pd
        rows = []
        for f in r.feature_names:
            pf = r.per_feature[f]
            rows.append({
                "feature": f,
                "suspect_mean": pf["member_mean"],
                "control_mean": pf["nonmember_mean"],
                "t": pf["t_stat"],
                "p": pf["p_value"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

    md = render_markdown(r)
    st.download_button(
        "Download evidence report (Markdown)",
        data=md.encode("utf-8"),
        file_name="evidence_report.md",
        mime="text/markdown",
    )

    with st.expander("Raw JSON"):
        st.json(r.to_dict())

    st.markdown("---")
    st.markdown(
        "### Reading this report\n"
        "A **positive p-value < 0.1** means the suspect corpus has a statistically "
        "distinguishable profile from the control under this model — strong evidence "
        "of training-set inclusion. The **control test** (p > 0.3) ensures this is "
        "not a pipeline artifact. If the control test fails, the suspect/control "
        "corpora likely have a distribution mismatch — the result is inconclusive."
    )
