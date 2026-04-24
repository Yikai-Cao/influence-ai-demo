"""
Streamlit evidence-report demo for Influence AI.

Two tabs:
- Text — upload suspect/control corpus, pick HF LM, run MIA (Phase 1b).
- Audio — upload suspect/control audio clips, score against MusicGen
  (Phase E; requires local librosa/soundfile/peft install or Modal A10G,
  so not available on the hosted CPU-only demo).

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from mia_core import load_model, run_evidence_report, EvidenceReport
from report import render_markdown, format_p

APP_DIR = Path(__file__).parent
EXAMPLES_DIR = APP_DIR / "examples"


# ── Shared: cached model loaders ──────────────────────────────────────

@st.cache_resource(show_spinner="Loading text model weights (cached across users)…")
def cached_load_text_model(model_name: str, device: str):
    return load_model(model_name, device=device)


@st.cache_resource(show_spinner="Loading MusicGen weights (~700 MB, first load is slow)…")
def cached_load_musicgen(model_name: str, device: str, adapter_path: str | None):
    """Audio model load. Lazy-imports audio deps so the Text tab keeps
    working when librosa/soundfile/peft are not installed."""
    from mia_audio_core import load_musicgen
    return load_musicgen(model_name, device=device, adapter_path=adapter_path)


# ── Helpers ───────────────────────────────────────────────────────────

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


def parse_text_corpus(uploaded) -> list[str]:
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


def save_uploaded_audio(uploaded_files, subdir: str) -> list[Path]:
    """Persist Streamlit UploadedFile list to a tempdir; return paths."""
    if not uploaded_files:
        return []
    tmp = Path(tempfile.mkdtemp(prefix=f"audio_{subdir}_"))
    paths = []
    for f in uploaded_files:
        p = tmp / f.name
        p.write_bytes(f.getbuffer())
        paths.append(p)
    return paths


def render_report_panel(r: EvidenceReport):
    """Shared result panel for text + audio."""
    verdict = r.verdict()
    color = {"STRONG": "red", "MODERATE": "orange",
             "INCONCLUSIVE": "gray", "NO": "green"}
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


# ── Page chrome ───────────────────────────────────────────────────────

st.set_page_config(page_title="Influence AI — Evidence Report", layout="wide")
st.title("Influence AI — Data Attribution Evidence Report")
st.caption(
    "Membership inference on a suspect corpus against an audited model. "
    "Method: Maini et al., *LLM Dataset Inference* (NeurIPS 2024), 16-feature subset "
    "with learned logistic regression. Audio extension uses MusicGen's 4 EnCodec codebooks."
)

tab_text, tab_audio = st.tabs(["📄 Text (Pythia)", "🎵 Audio (MusicGen, beta)"])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — TEXT (Phase 1b, unchanged)
# ══════════════════════════════════════════════════════════════════════

with tab_text:
    st.warning(
        "**This hosted demo uses Pythia-160m on CPU** for a fast, shareable UI walkthrough. "
        "Pythia-160m does not memorize The Pile heavily enough to produce a decisive "
        "p-value — expect an **INCONCLUSIVE or NO evidence** verdict on the example corpus. "
        "The headline result from our experiments (**p ≈ 4.46 × 10⁻⁵ on Pile Wikipedia**) "
        "requires Pythia-6.9B on an A10G GPU via the Modal backend — see the "
        "[README](https://github.com/Yikai-Cao/influence-ai-demo) to run that locally."
    )

    with st.sidebar:
        st.header("Text tab — Configuration")
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
            key="text_model_name",
        )
        if model_name == "custom…":
            model_name = st.text_input(
                "Custom model name",
                value="EleutherAI/pythia-160m-deduped",
                key="text_custom_model",
            )

        backend = st.radio(
            "Backend",
            ["local (CPU/GPU)", "Modal A10G (cloud)"],
            index=0,
            help="Local is fine for pythia-160m smoke tests. Use Modal A10G for 1.4B+ "
                 "models — A10G has 24 GB VRAM, ~$1.10/hr, ~10-15 min per audit.",
            key="text_backend",
        )
        is_modal = backend.startswith("Modal")
        device = ("cuda" if is_modal
                  else st.radio("Local device", ["cpu", "cuda"],
                                index=0, horizontal=True, key="text_device"))
        max_length = st.slider("Max tokens per passage", 128, 1024, 512, step=128,
                               key="text_max_length")
        batch_size = st.slider("Batch size", 1, 16, 4, key="text_batch_size")

        st.divider()
        st.markdown(
            "**Corpus requirements**  \n"
            "• Suspect: ≥ 50 passages recommended  \n"
            "• Control: must be ≥ 2× suspect size  \n"
            "• File formats: `.txt` (one passage per line) or `.jsonl` with a `text` field"
        )

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
            index=0, horizontal=True,
            help="Subsamples the bundled corpus. Quick is enough to see the pipeline; "
                 "Full matches Phase 1b exactly.",
            key="text_demo_size",
        )
        _size_to_n = {"Quick": 50, "Medium": 150, "Full": 500}
        n_suspect = next(v for k, v in _size_to_n.items() if demo_size.startswith(k))

        if st.button("Load Pile Wikipedia example", key="text_load_example"):
            suspect_all = load_example_jsonl(example_suspect)
            control_all = load_example_jsonl(example_control)
            st.session_state["example_suspect"] = suspect_all[:n_suspect]
            st.session_state["example_control"] = control_all[:2 * n_suspect]
            st.session_state.pop("report", None)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Suspect corpus")
        st.caption("Content you believe may have been used to train the model.")
        suspect_file = st.file_uploader("Upload .txt or .jsonl",
                                        type=["txt", "jsonl"],
                                        key="text_suspect_upload")
    with col_b:
        st.subheader("Control corpus")
        st.caption("Similar-distribution content known *not* to be in training.")
        control_file = st.file_uploader("Upload .txt or .jsonl",
                                        type=["txt", "jsonl"],
                                        key="text_control_upload")

    suspect_texts = parse_text_corpus(suspect_file) or st.session_state.get("example_suspect", [])
    control_texts = parse_text_corpus(control_file) or st.session_state.get("example_control", [])

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

    run_btn = st.button(
        "Run audit",
        type="primary",
        disabled=not (suspect_texts and control_texts
                      and len(control_texts) >= 2 * len(suspect_texts)),
        key="text_run_btn",
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
                            suspect_texts, control_texts,
                            model_name=model_name,
                            max_length=max_length, batch_size=batch_size,
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
                model, tokenizer = cached_load_text_model(model_name, device)
                st.write("Running MIA pipeline…")
                report = run_evidence_report(
                    suspect_texts, control_texts,
                    model, tokenizer,
                    model_name=model_name,
                    max_length=max_length, batch_size=batch_size,
                    device=device, progress=progress_cb,
                )
            status.update(label="Done", state="complete")
            progress_bar.progress(1.0)
            st.session_state["report"] = report

    if "report" in st.session_state:
        r = st.session_state["report"]
        st.header("Results")
        render_report_panel(r)
        st.markdown("---")
        st.markdown(
            "### Reading this report\n"
            "A **positive p-value < 0.1** means the suspect corpus has a statistically "
            "distinguishable profile from the control under this model — strong evidence "
            "of training-set inclusion. The **control test** (p > 0.3) ensures this is "
            "not a pipeline artifact. If the control test fails, the suspect/control "
            "corpora likely have a distribution mismatch — the result is inconclusive."
        )


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — AUDIO (Phase E, MusicGen-small)
# ══════════════════════════════════════════════════════════════════════

with tab_audio:
    st.info(
        "**Audio tab — Track A (gray-box MIA on MusicGen).** Click the "
        "**'Load audio example'** button below to see a pre-computed "
        "report instantly. Live MusicGen inference on the hosted CPU "
        "takes 5-10 min — recommended only if you want to try your own "
        "clips. For the headline Phase B result (p<0.01 after fine-tune), "
        "see the GitHub README — that runs on Modal A10G (~$30). "
        "We also have **Track B (canary/data-poisoning)** for closed "
        "models like Suno/Udio — see `canary_prototype/` in the repo."
    )

    audio_example_report_path = APP_DIR / "examples" / "audio_demo_report.json"
    audio_example_suspect_dir = APP_DIR / "examples" / "audio_demo_suspect"
    audio_example_control_dir = APP_DIR / "examples" / "audio_demo_control"
    audio_example_available = audio_example_report_path.exists()

    if audio_example_available:
        col_ex1, col_ex2 = st.columns([1, 3])
        with col_ex1:
            if st.button("🎵 Load audio example",
                          help="Loads a pre-computed MIA report on 5 suspect + "
                               "10 control synthetic clips through base "
                               "MusicGen-small. Instant — no live inference.",
                          key="audio_load_example"):
                report_dict = json.loads(audio_example_report_path.read_text())
                st.session_state["audio_report"] = EvidenceReport.from_dict(report_dict)
        with col_ex2:
            st.caption(
                "Demo uses **base MusicGen-small + 5 suspect + 10 control clips** "
                "on synthetic audio. Expected verdict is **NO evidence** — "
                "base model + no memorization signal = honest null result. "
                "The real Phase B result (p ≈ 10⁻⁵ after fine-tune) needs Modal GPU."
            )

    audio_deps_ok = True
    try:
        import librosa  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError:
        audio_deps_ok = False
        st.warning(
            "**Live audio inference unavailable** on this host — missing "
            "`librosa` / `soundfile`. Viewing pre-computed example above still "
            "works. To run live locally: `pip install librosa soundfile peft` "
            "and restart Streamlit."
        )

    with st.expander("Audio configuration", expanded=True):
        audio_model_name = st.text_input(
            "Base model (HuggingFace)",
            value="facebook/musicgen-small",
            key="audio_model_name",
        )
        audio_adapter_path = st.text_input(
            "LoRA adapter path (optional — output of finetune_musicgen.py)",
            value="",
            help="Leave empty to score against the base model. Required "
                 "for Phase B / reproducing strong signals.",
            key="audio_adapter_path",
        )
        audio_backend = st.radio(
            "Backend",
            ["local (CPU/GPU)", "Modal A10G (cloud)"],
            index=0,
            help="Local only works after installing the audio deps. Modal is "
                 "recommended for n ≥ 50 clips.",
            key="audio_backend",
        )
        audio_is_modal = audio_backend.startswith("Modal")
        audio_device = (
            "cuda" if audio_is_modal
            else st.radio("Local device", ["cpu", "mps", "cuda"],
                          index=0, horizontal=True, key="audio_device")
        )
        prompt_mode = st.radio(
            "Prompt mode",
            ["empty", "neutral (\"music\")", "custom per clip"],
            index=0,
            help="MusicGen is text-conditional. 'empty' is the safest baseline. "
                 "Custom caption matching the clip's genre/mood may amplify "
                 "signal but also adds variance.",
            key="audio_prompt_mode",
        )
        clip_seconds = st.slider(
            "Clip length (s)", 3, 30, 10, step=1,
            help="Must match the clip length used during fine-tune. Default "
                 "10 s matches the Phase B plan.",
            key="audio_clip_seconds",
        )

    st.markdown("#### Upload clips")
    col_as, col_ac = st.columns(2)
    with col_as:
        st.subheader("Suspect audio")
        st.caption("Clips you believe may have been in the model's training set.")
        audio_suspect_files = st.file_uploader(
            "Upload audio files (.wav/.mp3/.flac/.ogg/.m4a)",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            accept_multiple_files=True,
            key="audio_suspect_upload",
        )
    with col_ac:
        st.subheader("Control audio")
        st.caption("Same-distribution clips known *not* to be in training.")
        audio_control_files = st.file_uploader(
            "Upload audio files",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            accept_multiple_files=True,
            key="audio_control_upload",
        )

    n_suspect_audio = len(audio_suspect_files or [])
    n_control_audio = len(audio_control_files or [])

    if n_suspect_audio or n_control_audio:
        c1, c2 = st.columns(2)
        c1.metric("Suspect clips", n_suspect_audio)
        c2.metric("Control clips", n_control_audio)
        if n_control_audio and n_control_audio < 2 * n_suspect_audio:
            st.warning(
                f"Control ({n_control_audio}) must be at least 2× suspect "
                f"({n_suspect_audio}) — need {2 * n_suspect_audio}."
            )

    audio_run_btn = st.button(
        "Run audio audit",
        type="primary",
        disabled=not (audio_deps_ok and n_suspect_audio
                      and n_control_audio >= 2 * n_suspect_audio),
        key="audio_run_btn",
    )

    if audio_run_btn:
        with st.status("Preparing audio …", expanded=True) as status:
            st.write("Saving uploaded clips to a tempdir …")
            suspect_paths = save_uploaded_audio(audio_suspect_files, "suspect")
            control_paths = save_uploaded_audio(audio_control_files, "control")

            if prompt_mode.startswith("empty"):
                s_prompts = [""] * len(suspect_paths)
                c_prompts = [""] * len(control_paths)
            elif prompt_mode.startswith("neutral"):
                s_prompts = ["music"] * len(suspect_paths)
                c_prompts = ["music"] * len(control_paths)
            else:
                # Custom per-clip — use filename stem as caption
                s_prompts = [p.stem.replace("_", " ") for p in suspect_paths]
                c_prompts = [p.stem.replace("_", " ") for p in control_paths]

            if audio_is_modal:
                st.write("Dispatching to Modal … (not yet wired for uploaded "
                         "audio — use `modal volume put` to stage clips and "
                         "run modal_audio_backend.py directly for now)")
                st.info(
                    "The Modal integration for the audio tab expects clips "
                    "pre-loaded on a Modal Volume. See `AUDIO_README.md` "
                    "for the recommended workflow: "
                    "`modal volume put fma-small …` then "
                    "`modal run modal_audio_backend.py::finetune_and_audit`."
                )
                status.update(label="Modal dispatch incomplete — see info above",
                              state="error")
            else:
                try:
                    from mia_audio_core import run_audio_evidence_report
                    st.write(f"Loading MusicGen ({audio_model_name}) …")
                    adapter = audio_adapter_path.strip() or None
                    bundle = cached_load_musicgen(audio_model_name, audio_device, adapter)
                    st.write(
                        f"Scoring {len(suspect_paths)} suspect + "
                        f"{len(control_paths)} control clips — "
                        "this can take several minutes on CPU/MPS."
                    )

                    progress_bar = st.progress(0.0)
                    def audio_progress(stage: str, frac: float):
                        status.update(label=f"{stage} ({frac * 100:.0f}%)")
                        progress_bar.progress(min(1.0, frac))

                    report = run_audio_evidence_report(
                        suspect_paths=[str(p) for p in suspect_paths],
                        control_paths=[str(p) for p in control_paths],
                        bundle=bundle,
                        model_name=audio_model_name + ("+lora" if adapter else ""),
                        suspect_prompts=s_prompts,
                        control_prompts=c_prompts,
                        clip_seconds=float(clip_seconds),
                        progress=audio_progress,
                    )
                    status.update(label="Audio audit complete", state="complete")
                    progress_bar.progress(1.0)
                    st.session_state["audio_report"] = report
                except Exception as e:
                    status.update(label=f"Audio run failed: {type(e).__name__}",
                                  state="error")
                    st.exception(e)

    if "audio_report" in st.session_state:
        r = st.session_state["audio_report"]
        st.header("Results")
        render_report_panel(r)
        st.markdown("---")
        st.markdown(
            "### Reading this audio report\n"
            "- The classifier **p-value** is the headline, but in audio a low classifier "
            "p-value with mostly per-feature p > 0.5 means the classifier may be fitting "
            "a reversed distribution (suspect has *higher* loss). Always check the "
            "per-feature breakdown for direction.\n"
            "- Prompt mode matters. Empty-prompt is the safest baseline; per-clip captions "
            "amplify the signal when MusicGen was fine-tuned with those captions.\n"
            "- This maps to **Level 3** in Sureel's 5-level attribution taxonomy — "
            "corpus-level yes/no, not per-track attribution (that's Level 4, future work).\n"
        )
