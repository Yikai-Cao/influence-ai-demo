"""
Streamlit evidence-report demo for Influence AI.

Three tabs:
- Text  — upload suspect/control corpus, pick HF LM, run MIA (Phase 1b).
- Audio — upload suspect/control audio clips, score against MusicGen
  (Phase E; requires local librosa/soundfile/peft install or Modal A10G,
  so not available on the hosted CPU-only demo).
- Canary — generate a private canary library, embed canaries into a
  host track, scan suspect model outputs for leaks. Pure-CPU; works
  on the hosted demo. Plan B wedge for closed black-box models like
  Suno/Udio.

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


PISTACHIO_ATTRIBUTION = (
    "Example track: *Pistachio Ice Cream Ragtime* by Lena Orsa "
    "([Freesound 442789](https://freesound.org/s/442789/)), shipped via "
    "`librosa.example('pistachio')` — CC-licensed."
)
SYNTHETIC_FALLBACK_ATTRIBUTION = (
    "Example track: synthesized arpeggio (CC0). The CC-BY/PD librosa "
    "example wasn't reachable in this environment, so this is the fallback."
)


def _load_pistachio_track(target_sr: int = 32_000):
    """Lazy-cached load of librosa's `pistachio` PD music example.

    Returns (samples float32, sr). Raises if librosa can't fetch it
    (e.g. no internet on first call). Caller should try/except and
    fall back to a synthetic clip.
    """
    if not hasattr(_load_pistachio_track, "_cache"):
        import librosa
        path = librosa.example("pistachio")
        samples, sr = librosa.load(path, sr=target_sr, mono=True, dtype="float32")
        _load_pistachio_track._cache = (samples, sr)
    return _load_pistachio_track._cache


def _make_synthetic_host(seed: int, duration_s: float, sr: int):
    """Fallback synth host. Used when librosa example unavailable."""
    import numpy as np
    from canary import canary_generator as cg

    target_n = int(duration_s * sr)
    rng = np.random.default_rng(seed)
    chunks = []
    chunk_idx = 0
    while sum(len(c) for c in chunks) < target_n:
        spec = cg.sample_spec(
            f"example_host_chunk_{chunk_idx}",
            seed=10_000 + seed * 100 + chunk_idx,
            duration_s=4.0,
        )
        chunk = cg.synthesize_canary(spec)
        chunk = chunk + rng.normal(0, 0.015, len(chunk)).astype("float32")
        chunks.append(chunk)
        chunk_idx += 1
    samples = np.concatenate(chunks)[:target_n].astype("float32")
    return samples


def _make_example_host_bytes(
    seed: int = 42, duration_s: float = 30.0,
) -> tuple[str, bytes, str]:
    """Return (filename, wav_bytes, attribution_md).

    Tries real CC-licensed music first (`librosa.example('pistachio')`);
    falls back to the synthetic arpeggio if librosa can't fetch it
    (offline environments, blocked egress, etc).
    """
    import io
    import numpy as np
    import soundfile as sf

    from canary import canary_generator as cg

    target_sr = cg.SAMPLE_RATE  # 32 kHz to match canary library

    # Try real music first
    try:
        track, sr = _load_pistachio_track(target_sr=target_sr)
        target_n = int(duration_s * sr)
        if len(track) >= target_n:
            samples = track[:target_n].copy()
        else:
            # Pistachio is ~30 s. If shorter than requested, loop with seam.
            n_repeats = (target_n // len(track)) + 1
            samples = np.tile(track, n_repeats)[:target_n].copy()
        fade_n = int(0.1 * sr)
        samples[:fade_n] *= np.linspace(0, 1, fade_n, dtype="float32")
        samples[-fade_n:] *= np.linspace(1, 0, fade_n, dtype="float32")
        peak = float(np.max(np.abs(samples))) or 1.0
        samples = (samples / peak * 0.7).astype("float32")
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV")
        return (
            "example_pistachio_30s.wav",
            buf.getvalue(),
            PISTACHIO_ATTRIBUTION,
        )
    except Exception as _real_err:  # noqa: F841
        pass

    # Synthetic fallback
    sr = target_sr
    samples = _make_synthetic_host(seed, duration_s, sr)
    fade_n = int(0.1 * sr)
    samples[:fade_n] *= np.linspace(0, 1, fade_n, dtype="float32")
    samples[-fade_n:] *= np.linspace(1, 0, fade_n, dtype="float32")
    peak = float(np.max(np.abs(samples))) or 1.0
    samples = (samples / peak * 0.6).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return (
        "example_synth_30s.wav",
        buf.getvalue(),
        SYNTHETIC_FALLBACK_ATTRIBUTION,
    )


def _make_example_suspect_set(
    library_zip_bytes: bytes,
    n_leaked: int = 3,
    n_clean: int = 3,
    seed: int = 7,
    apply_codec: bool = True,
) -> tuple[list[tuple[str, bytes]], str]:
    """Generate a small example suspect-output set so visitors can run
    Mode 3 (Scan) end-to-end without bringing their own files.

    Returns (blobs, attribution_md) where blobs is a list of
    (filename, wav_bytes) tuples mixing:

      n_leaked clips: synthetic host with a known canary mixed in at
        high gain (~0.7), then run through the `neural_codec_aggressive`
        pipeline. Simulates a model that *memorized* a canary and
        regurgitated it during generation.

      n_clean clips: synthetic pristine host, codec-compressed but
        with no canary planted. Simulates non-leaked outputs.

    The detector run on this set should fire STRONG on the leaked clips
    and stay clean on the pristine ones.
    """
    import io
    import json
    import random
    import zipfile
    from pathlib import Path
    import numpy as np
    import soundfile as sf

    from canary import canary_generator as cg

    # Extract the library so we know which canaries to plant
    tmp_lib = Path(tempfile.mkdtemp(prefix="canary_example_lib_"))
    with zipfile.ZipFile(io.BytesIO(library_zip_bytes)) as zf:
        zf.extractall(tmp_lib)
    idx_path = next(tmp_lib.rglob("canary_index.json"))
    index = json.loads(idx_path.read_text())
    library_canaries = index["canaries"]

    sr = cg.SAMPLE_RATE
    clip_seconds = 6.0  # 6s clips so 6 of them fit comfortably in 30s pistachio
    clip_n = int(clip_seconds * sr)

    rng_picker = random.Random(seed)

    # Try to use the real CC-licensed track as the host source.
    # Each suspect clip is a different segment of the same track, so
    # the leaked + clean clips share musical character (which is the
    # realistic case for "many model outputs from the same training").
    using_real_host = False
    real_track_samples = None
    try:
        real_track_samples, real_track_sr = _load_pistachio_track(target_sr=sr)
        using_real_host = True
    except Exception:
        pass

    def _slice_real_host(idx: int, total_clips: int) -> np.ndarray:
        """Get the idx-th 6-second segment of pistachio, evenly spread."""
        track_len = len(real_track_samples)
        max_start = max(0, track_len - clip_n)
        if total_clips <= 1:
            start = 0
        else:
            start = int(idx * max_start / (total_clips - 1))
        end = start + clip_n
        if end > track_len:
            # Loop with seam if we run off the end (shouldn't happen
            # for pistachio @ 30 s, 6 clips × 6 s, but defensive)
            tail = end - track_len
            seg = np.concatenate([real_track_samples[start:],
                                   real_track_samples[:tail]])
        else:
            seg = real_track_samples[start:end].copy()
        return seg.astype("float32")

    def _synth_host_fallback(host_seed: int) -> np.ndarray:
        spec = cg.sample_spec(
            f"suspect_host_{host_seed}", seed=20_000 + host_seed,
            duration_s=clip_seconds,
        )
        clip = cg.synthesize_canary(spec)
        rng = np.random.default_rng(host_seed)
        clip = clip + rng.normal(0, 0.02, len(clip)).astype("float32")
        return clip[:clip_n].astype("float32")

    total_clips = n_leaked + n_clean

    def _get_host(idx: int, fallback_seed: int) -> np.ndarray:
        if using_real_host:
            return _slice_real_host(idx, total_clips)
        return _synth_host_fallback(fallback_seed)

    def _codec(clip: np.ndarray, c_seed: int) -> np.ndarray:
        if not apply_codec:
            return clip
        # Light codec is realistic for a high-bitrate model output —
        # lowpass at 12 kHz, downsample to 24 kHz, 35 dB-SNR noise.
        # The aggressive variant (8 kHz lowpass + 16 kHz + μ-law) is
        # the worst-case stress test in our robustness sweep, but
        # it's overkill for a demo — leaves only 4/144 pair scores
        # above threshold on a 3-leaked-3-clean set, which dilutes
        # the binomial. Light codec keeps the demo verdict crisp
        # while still validating codec robustness.
        from canary.transforms import neural_codec_light
        return neural_codec_light(clip, sr, seed=c_seed)

    out: list[tuple[str, bytes]] = []

    # Leaked clips: pick a random canary, mix at high gain into a host,
    # then codec-degrade. This is the synthetic memorization-leak case.
    for i in range(n_leaked):
        chosen = rng_picker.choice(library_canaries)
        canary_audio, c_sr = sf.read(chosen["path"], dtype="float32",
                                      always_2d=False)
        if canary_audio.ndim == 2:
            canary_audio = canary_audio.mean(axis=1)
        host = _get_host(idx=i, fallback_seed=200 + i)
        # Mix the canary into a random offset, replacing the host there
        leak_offset = rng_picker.uniform(0.5, clip_seconds - 4.0)
        leak_gain = rng_picker.uniform(0.5, 0.85)
        start = int(leak_offset * sr)
        end = min(start + len(canary_audio), clip_n)
        host[start:end] = (
            leak_gain * canary_audio[:end - start]
            + 0.15 * host[start:end]
        )
        host = _codec(host, c_seed=300 + i)
        buf = io.BytesIO()
        sf.write(buf, host, sr, format="WAV")
        out.append((f"leaked_{i:02d}.wav", buf.getvalue()))

    # Clean clips: just codec-compressed pristine hosts, no canary
    for i in range(n_clean):
        host = _get_host(idx=n_leaked + i, fallback_seed=500 + i)
        host = _codec(host, c_seed=600 + i)
        buf = io.BytesIO()
        sf.write(buf, host, sr, format="WAV")
        out.append((f"clean_{i:02d}.wav", buf.getvalue()))

    import shutil
    shutil.rmtree(tmp_lib, ignore_errors=True)

    if using_real_host:
        attribution = (
            PISTACHIO_ATTRIBUTION + " "
            "Each suspect clip is a different 6-second segment of the same "
            "track — leaked clips have a canary planted, clean clips don't. "
            f"All clips run through `neural_codec_light` to simulate codec output."
        )
    else:
        attribution = (
            SYNTHETIC_FALLBACK_ATTRIBUTION + " "
            "Suspect clips are synthetic; leaked have planted canaries, "
            "clean don't, all codec-compressed."
        )
    return out, attribution


def _decode_audio_bytes(b: bytes) -> tuple:
    """Decode .wav bytes → (mono float32 array, sr). Used by the canary
    tab's spectrogram visualization."""
    import io
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data.astype("float32"), sr


def _make_spec_panel(samples, sr: int, title: str, canary_windows=None,
                     time_range=None):
    """Render one mel-spectrogram subplot. Returns matplotlib Figure.

    canary_windows: list of (start_s, end_s) for cyan overlay rectangles.
    time_range: optional (start_s, end_s) to zoom into.
    """
    import numpy as np
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt

    if time_range is not None:
        start_s, end_s = time_range
        i0 = max(0, int(start_s * sr))
        i1 = min(len(samples), int(end_s * sr))
        samples = samples[i0:i1]
        time_offset = start_s
    else:
        time_offset = 0.0

    fig, ax = plt.subplots(figsize=(7, 3.2))
    S = librosa.feature.melspectrogram(y=samples, sr=sr, n_mels=128,
                                        fmax=min(sr // 2, 16000))
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(
        S_db, sr=sr, x_axis="time", y_axis="mel",
        cmap="magma", ax=ax, vmin=-60, vmax=0,
    )
    # Re-label x-axis with absolute timestamps when we're zoomed in
    if time_offset > 0:
        from matplotlib.ticker import FixedLocator
        ticks = list(ax.get_xticks())
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels([f"{t + time_offset:.1f}" for t in ticks])
    ax.set_title(title, fontsize=11)
    if canary_windows:
        for (start, end) in canary_windows:
            rel_start = start - time_offset
            rel_end = end - time_offset
            # Skip windows outside the visible range
            if time_range is not None:
                if rel_end < 0 or rel_start > (time_range[1] - time_range[0]):
                    continue
            # facecolor="cyan" + alpha for translucent fill,
            # edgecolor (no conflicting `color` kwarg) for the box outline
            ax.axvspan(
                max(0, rel_start), rel_end,
                facecolor="cyan", alpha=0.25,
                edgecolor="cyan", linewidth=1.5,
            )
    fig.tight_layout()
    return fig


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

tab_text, tab_audio, tab_canary = st.tabs([
    "📄 Text (Pythia)",
    "🎵 Audio (MusicGen, beta)",
    "🕵️ Canary (Plan B)",
])


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


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — CANARY (Plan B for closed black-box models)
# ══════════════════════════════════════════════════════════════════════

with tab_canary:
    st.info(
        "**Plan B — canary detection for closed black-box models** "
        "(Suno, Udio, ElevenLabs). The audit MIA on the other two tabs "
        "needs gray-box log-probabilities, which closed audio models "
        "don't expose. Canarying is the workaround: register a private "
        "library of distinctive motifs, mix them imperceptibly into "
        "your unreleased catalog, then later scan suspect model outputs "
        "for any of the planted canaries. Pure-CPU; runs on this demo."
    )

    canary_deps_ok = True
    try:
        import librosa  # noqa: F401
        import soundfile as sf  # noqa: F401
    except ImportError:
        canary_deps_ok = False
        st.error("Audio deps missing on this host — `pip install librosa soundfile`.")

    canary_mode = st.radio(
        "Step:",
        [
            "1️⃣ Generate library",
            "2️⃣ Embed in track",
            "3️⃣ Scan suspect outputs",
        ],
        horizontal=True,
        key="canary_mode",
    )

    # ── Helpers (lazy imports so this tab loads fast) ────────────────────

    def _zip_dir(dir_path: Path) -> bytes:
        import io, zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in dir_path.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(dir_path))
        return buf.getvalue()

    def _unzip_to_tmp(zip_bytes: bytes) -> Path:
        import io, zipfile
        tmp = Path(tempfile.mkdtemp(prefix="canary_lib_"))
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(tmp)
        return tmp

    # ── Mode 1 — Generate library ────────────────────────────────────────

    if canary_mode.startswith("1️⃣"):
        st.markdown("### Generate a private canary library")
        st.caption(
            "Each canary is a 3-second synth motif. The library is "
            "deterministic — the same `(N, transpositions, seed)` always "
            "produces the same canaries. **Keep the library private** — "
            "it's the key that proves embedding + enables detection."
        )
        # Direct mapping from display label → list, so we don't parse
        # display strings (avoids the `+0.5` not-valid-JSON gotcha).
        TRANSP_OPTIONS = {
            "single pitch (no mitigation)": [0.0],
            "±0.5 semitones (recommended)":  [-0.5, 0.0, 0.5],
            "±1 semitone (covers cover/remix)": [-1.0, -0.5, 0.0, 0.5, 1.0],
        }
        c1, c2, c3 = st.columns(3)
        n_logical = c1.number_input("N logical canaries", 4, 50, 16,
                                    key="canary_n_logical")
        seed = c2.number_input("Seed", 0, 99999, 0, key="canary_seed")
        transp_choice = c3.radio(
            "Transpositions",
            list(TRANSP_OPTIONS.keys()),
            index=1,
            help="Multiple transpositions close the pitch-shift attack "
                 "found in our robustness sweep. ±0.5 semitones survives "
                 "every codec / pitch transform tested. ±1 adds margin "
                 "against intentional cover/remix transposition.",
            key="canary_transp",
        )

        if st.button("Generate library", type="primary",
                      disabled=not canary_deps_ok, key="canary_gen_btn"):
            with st.status("Synthesizing canaries…", expanded=True) as status:
                from canary import canary_generator as cg
                transpositions = TRANSP_OPTIONS[transp_choice]
                tmp = Path(tempfile.mkdtemp(prefix="canary_gen_"))
                out_dir = tmp / "canaries"
                cg.generate_library(
                    int(n_logical), out_dir,
                    base_seed=int(seed),
                    transpositions=transpositions,
                )
                n_total = int(n_logical) * len(transpositions)
                st.write(f"  generated {n_total} canary entries in {out_dir}")
                zip_bytes = _zip_dir(out_dir)
                st.session_state["canary_library_zip"] = zip_bytes
                st.session_state["canary_library_meta"] = {
                    "n_logical": int(n_logical),
                    "n_total": n_total,
                    "transpositions": transpositions,
                    "seed": int(seed),
                }
                status.update(label="Library ready", state="complete")

        if "canary_library_zip" in st.session_state:
            meta = st.session_state["canary_library_meta"]
            st.success(
                f"Library: {meta['n_logical']} logical × "
                f"{len(meta['transpositions'])} transpositions = "
                f"{meta['n_total']} entries  |  seed={meta['seed']}"
            )
            st.download_button(
                "⬇ Download canary library (zip)",
                data=st.session_state["canary_library_zip"],
                file_name=f"canary_library_n{meta['n_logical']}_seed{meta['seed']}.zip",
                mime="application/zip",
                key="canary_lib_download",
            )
            with st.expander("Preview a few canaries", expanded=False):
                tmp = _unzip_to_tmp(st.session_state["canary_library_zip"])
                idx = json.loads((tmp / "canary_index.json").read_text())
                for c in idx["canaries"][:3]:
                    st.caption(f"`{c['canary_id']}` — root_freq={c['root_freq_hz']:.1f} Hz")
                    st.audio(c["path"])

    # ── Mode 2 — Embed canaries ──────────────────────────────────────────

    elif canary_mode.startswith("2️⃣"):
        st.markdown("### Embed canaries into a host track")
        st.caption(
            "Mix N canaries from your library into an unreleased track "
            "at low gain (default −25 dB → ~28 dB perceptual SNR, "
            "essentially imperceptible). The output is a `.wav` plus a "
            "`.manifest.json` recording exactly which canaries went where."
        )

        lib_source = st.radio(
            "Library source",
            ["Use library generated this session", "Upload library zip"],
            key="canary_embed_lib_source",
            disabled="canary_library_zip" not in st.session_state,
        )
        lib_zip_bytes = None
        if lib_source.startswith("Use") and "canary_library_zip" in st.session_state:
            lib_zip_bytes = st.session_state["canary_library_zip"]
            st.caption("Using the library generated above (Step 1️⃣).")
        else:
            uploaded_lib = st.file_uploader(
                "Library zip (from Step 1️⃣)", type=["zip"],
                key="canary_embed_lib_upload",
            )
            if uploaded_lib:
                lib_zip_bytes = uploaded_lib.getvalue()

        host_file = st.file_uploader(
            "Host audio (your unreleased track) — .wav / .mp3 / .flac",
            type=["wav", "mp3", "flac", "ogg"],
            key="canary_embed_host",
        )

        # Provide an example host so demo visitors can complete the
        # full flow without bringing their own audio
        ex_col_a, ex_col_b = st.columns([1, 3])
        with ex_col_a:
            if st.button("🎼 Use example host",
                          help="Load a 30s real instrumental music clip "
                               "(CC-licensed via librosa) as the host. "
                               "Falls back to a synthetic clip if the "
                               "real track can't be fetched. Useful if "
                               "you don't have an audio file handy.",
                          disabled=not canary_deps_ok,
                          key="canary_embed_example_host_btn"):
                with st.spinner("Loading example host…"):
                    name, example_bytes, attribution = _make_example_host_bytes(
                        seed=42, duration_s=30.0,
                    )
                st.session_state["canary_example_host_blob"] = (
                    name, example_bytes,
                )
                st.session_state["canary_example_host_attribution"] = attribution
        with ex_col_b:
            if "canary_example_host_blob" in st.session_state:
                attr = st.session_state.get(
                    "canary_example_host_attribution",
                    "Example host loaded.",
                )
                st.caption(
                    f"📎 `{st.session_state['canary_example_host_blob'][0]}` "
                    f"loaded. {attr} Upload your own file to override."
                )

        # Resolve the effective host: upload takes precedence over example
        effective_host = None
        if host_file is not None:
            effective_host = (host_file.name, bytes(host_file.getbuffer()))
        elif "canary_example_host_blob" in st.session_state:
            effective_host = st.session_state["canary_example_host_blob"]

        c1, c2, c3 = st.columns(3)
        n_to_embed = c1.number_input("N canaries to embed", 1, 20, 3,
                                     key="canary_embed_n")
        gain_db = c2.slider("Gain (dB rel. host RMS)", -30, 0, -25,
                            help="Lower = more imperceptible. -25 dB "
                                 "recommended for release-quality.",
                            key="canary_embed_gain")
        embed_seed = c3.number_input("Embed seed", 0, 99999, 0,
                                     key="canary_embed_seed")

        if st.button("Embed canaries", type="primary",
                      disabled=not (canary_deps_ok and lib_zip_bytes
                                     and effective_host is not None),
                      key="canary_embed_btn"):
            with st.status("Embedding…", expanded=True) as status:
                from canary import canary_embedder as ce

                lib_tmp = _unzip_to_tmp(lib_zip_bytes)
                lib_index_path = next(lib_tmp.rglob("canary_index.json"))

                host_tmp = Path(tempfile.mkdtemp(prefix="canary_host_"))
                host_name, host_bytes_in = effective_host
                host_path = host_tmp / host_name
                host_path.write_bytes(host_bytes_in)
                out_path = host_tmp / f"canaried_{host_path.stem}.wav"

                manifest = ce.embed(
                    host_path=host_path,
                    canary_index_path=lib_index_path,
                    out_path=out_path,
                    n_canaries=int(n_to_embed),
                    gain_db=float(gain_db),
                    seed=int(embed_seed),
                )

                canaried_bytes = out_path.read_bytes()
                manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
                manifest_bytes = manifest_path.read_bytes()
                st.session_state["canary_canaried_bytes"] = canaried_bytes
                st.session_state["canary_canaried_name"] = out_path.name
                st.session_state["canary_manifest_bytes"] = manifest_bytes
                st.session_state["canary_manifest"] = manifest.to_dict()
                st.session_state["canary_host_bytes"] = host_path.read_bytes()
                st.session_state["canary_host_name"] = host_path.name
                status.update(label=f"{manifest.n_canaries} canaries embedded",
                              state="complete")

        if "canary_canaried_bytes" in st.session_state:
            m = st.session_state["canary_manifest"]
            st.success(
                f"{m['n_canaries']} canaries placed in "
                f"{m['host_duration_s']:.1f}s host. "
                f"Host RMS: {m['host_rms_db']:.1f} dBFS."
            )

            st.markdown("**A/B preview** — listen for the canaries:")
            c_a, c_b = st.columns(2)
            with c_a:
                st.caption("Original host")
                st.audio(st.session_state["canary_host_bytes"])
            with c_b:
                st.caption("Canaried")
                st.audio(st.session_state["canary_canaried_bytes"])

            # ── Spectrogram comparison ───────────────────────────────────
            st.markdown(
                "**Spectrogram comparison** — cyan rectangles mark the "
                "canary windows. Even when masked under the host so your "
                "ears can't pick them out, the canary's spectral signature "
                "is still present and matchable by the detector."
            )
            try:
                import matplotlib.pyplot as plt
                host_arr, host_sr = _decode_audio_bytes(
                    st.session_state["canary_host_bytes"])
                canaried_arr, canaried_sr = _decode_audio_bytes(
                    st.session_state["canary_canaried_bytes"])
                canary_windows = [
                    (e["offset_s"], e["offset_s"] + e["duration_s"])
                    for e in m.get("embeddings", [])
                ]
                spec_a, spec_b = st.columns(2)
                with spec_a:
                    fig_host = _make_spec_panel(
                        host_arr, host_sr,
                        title="Original host (no canary)",
                        canary_windows=None,
                    )
                    st.pyplot(fig_host, use_container_width=True)
                    plt.close(fig_host)
                with spec_b:
                    fig_canaried = _make_spec_panel(
                        canaried_arr, canaried_sr,
                        title="Canaried (cyan = canary windows)",
                        canary_windows=canary_windows,
                    )
                    st.pyplot(fig_canaried, use_container_width=True)
                    plt.close(fig_canaried)

                # Zoom into the first canary window
                if canary_windows:
                    first = canary_windows[0]
                    pad = 0.5  # half-second context on each side
                    zoom_start = max(0, first[0] - pad)
                    zoom_end = min(len(canaried_arr) / canaried_sr,
                                    first[1] + pad)
                    with st.expander(
                        f"🔍 Zoom into first canary window "
                        f"({first[0]:.2f}s – {first[1]:.2f}s)",
                        expanded=False,
                    ):
                        zoom_a, zoom_b = st.columns(2)
                        with zoom_a:
                            fig_zh = _make_spec_panel(
                                host_arr, host_sr,
                                title="Original host (zoom)",
                                canary_windows=None,
                                time_range=(zoom_start, zoom_end),
                            )
                            st.pyplot(fig_zh, use_container_width=True)
                            plt.close(fig_zh)
                        with zoom_b:
                            fig_zc = _make_spec_panel(
                                canaried_arr, canaried_sr,
                                title="Canaried (zoom)",
                                canary_windows=canary_windows,
                                time_range=(zoom_start, zoom_end),
                            )
                            st.pyplot(fig_zc, use_container_width=True)
                            plt.close(fig_zc)
                        st.caption(
                            "Comparing the same time window in the original "
                            "vs canaried spectrograms reveals the canary's "
                            "spectral fingerprint inside the cyan box. "
                            "If you can't see it visually at very low gains, "
                            "the detector still finds it via MFCC + chroma "
                            "cosine similarity."
                        )
            except Exception as _spec_err:
                st.caption(
                    f"(Spectrogram visualization unavailable: "
                    f"{type(_spec_err).__name__}: {_spec_err}. "
                    f"Audio + manifest still downloadable below.)"
                )

            with st.expander("Embedding manifest", expanded=False):
                st.json(m)

            cd1, cd2 = st.columns(2)
            cd1.download_button(
                "⬇ Download canaried .wav",
                data=st.session_state["canary_canaried_bytes"],
                file_name=st.session_state["canary_canaried_name"],
                mime="audio/wav",
                key="canary_canaried_download",
            )
            cd2.download_button(
                "⬇ Download manifest JSON",
                data=st.session_state["canary_manifest_bytes"],
                file_name=st.session_state["canary_canaried_name"] + ".manifest.json",
                mime="application/json",
                key="canary_manifest_download",
            )

    # ── Mode 3 — Scan suspect outputs ────────────────────────────────────

    else:
        st.markdown("### Scan suspect model outputs for leaks")
        st.caption(
            "Upload audio you suspect was generated by a model trained on "
            "your canaried catalog. We compute MFCC + chromagram sliding "
            "cosine similarity at every (canary × suspect) pair, then "
            "binomial-test the count of pairs above threshold against the "
            "calibrated null FPR. Output is a corpus-level p-value."
        )

        lib_source = st.radio(
            "Library source",
            ["Use library generated this session", "Upload library zip"],
            key="canary_scan_lib_source",
            disabled="canary_library_zip" not in st.session_state,
        )
        lib_zip_bytes = None
        if lib_source.startswith("Use") and "canary_library_zip" in st.session_state:
            lib_zip_bytes = st.session_state["canary_library_zip"]
        else:
            uploaded_lib = st.file_uploader(
                "Library zip", type=["zip"],
                key="canary_scan_lib_upload",
            )
            if uploaded_lib:
                lib_zip_bytes = uploaded_lib.getvalue()

        suspect_files = st.file_uploader(
            "Suspect audio outputs (multiple)",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            accept_multiple_files=True,
            key="canary_scan_suspects",
        )

        # "Use example suspect outputs" — only meaningful if a library
        # is in scope, since the leaked clips need to plant entries
        # from THAT library to be detectable.
        ex_col_a, ex_col_b = st.columns([1, 3])
        with ex_col_a:
            if st.button("🎯 Use example outputs",
                          help="Generate 3 leaked + 3 clean suspect clips "
                               "from the loaded library, with codec "
                               "compression applied. Lets you see what "
                               "the detector says on a known ground-truth "
                               "set. Uses real CC-licensed music as host "
                               "where available; falls back to synthetic.",
                          disabled=not (canary_deps_ok and lib_zip_bytes),
                          key="canary_scan_example_btn"):
                with st.spinner("Synthesizing example suspect outputs…"):
                    blobs, attribution = _make_example_suspect_set(
                        lib_zip_bytes, n_leaked=3, n_clean=3, seed=7,
                    )
                    st.session_state["canary_example_suspect_blobs"] = blobs
                    st.session_state["canary_example_suspect_attribution"] = attribution
        with ex_col_b:
            if "canary_example_suspect_blobs" in st.session_state:
                names = [name for (name, _) in
                         st.session_state["canary_example_suspect_blobs"]]
                attr = st.session_state.get(
                    "canary_example_suspect_attribution",
                    "Example suspect set loaded.",
                )
                st.caption(
                    f"📎 {len(names)} clips loaded "
                    f"(3 leaked + 3 clean). {attr} "
                    "Upload your own files to override."
                )

        # Resolve effective suspect set: upload takes precedence
        effective_suspects: list[tuple[str, bytes]] = []
        if suspect_files:
            for f in suspect_files:
                effective_suspects.append((f.name, bytes(f.getbuffer())))
        elif "canary_example_suspect_blobs" in st.session_state:
            effective_suspects = list(
                st.session_state["canary_example_suspect_blobs"]
            )

        c1, c2 = st.columns(2)
        threshold = c1.slider("Detection threshold (cosine)", 0.50, 0.95, 0.70,
                              step=0.01, key="canary_scan_threshold")
        null_fpr = c2.slider("Null FPR (for binomial test)",
                              0.001, 0.10, 0.01, step=0.001, format="%.3f",
                              key="canary_scan_null_fpr")

        if st.button("Run scan", type="primary",
                      disabled=not (canary_deps_ok and lib_zip_bytes
                                     and effective_suspects),
                      key="canary_scan_btn"):
            with st.status(f"Scanning {len(effective_suspects)} clips…",
                            expanded=True) as status:
                from canary import canary_detector as cd

                lib_tmp = _unzip_to_tmp(lib_zip_bytes)
                lib_index_path = next(lib_tmp.rglob("canary_index.json"))
                canary_index = json.loads(lib_index_path.read_text())

                suspect_dir = Path(tempfile.mkdtemp(prefix="canary_suspect_"))
                suspect_paths = []
                for (name, blob) in effective_suspects:
                    p = suspect_dir / name
                    p.write_bytes(blob)
                    suspect_paths.append(p)

                report = cd.detect(
                    canary_index, suspect_paths,
                    threshold=float(threshold),
                    null_fpr=float(null_fpr),
                )
                st.session_state["canary_detection_report"] = report.to_dict()
                status.update(label="Scan complete", state="complete")

        if "canary_detection_report" in st.session_state:
            r = st.session_state["canary_detection_report"]
            verdict = (
                "STRONG evidence of canary leakage" if r["binomial_p_value"] < 0.001
                else "MODERATE evidence of canary leakage" if r["binomial_p_value"] < 0.05
                else "NO evidence of canary leakage"
            )
            color = "red" if "STRONG" in verdict else (
                "orange" if "MODERATE" in verdict else "green")
            st.markdown(f":{color}[**{verdict}**]")

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Binomial p-value", format_p(r["binomial_p_value"]))
            mc2.metric("Pairs above threshold",
                        f"{r['n_pairs_above_threshold']} / "
                        f"{r['n_canaries'] * r['n_suspect_clips']}")
            mc3.metric("Expected null pairs",
                        f"{r['expected_null_pairs']:.2f}")

            with st.expander("Top 20 (canary, suspect) similarity hits"):
                import pandas as pd
                df = pd.DataFrame(r["per_pair_scores"])
                df["suspect_path"] = df["suspect_path"].apply(
                    lambda p: Path(p).name)
                st.dataframe(df, use_container_width=True)

            st.download_button(
                "⬇ Download detection report (JSON)",
                data=json.dumps(r, indent=2).encode("utf-8"),
                file_name="canary_detection_report.json",
                mime="application/json",
                key="canary_detection_download",
            )

    # ── Tab footer caveats ──────────────────────────────────────────────

    st.markdown("---")
    st.markdown(
        "### Important caveats\n"
        "- **Only protects forward-looking content.** Catalogs already "
        "published can't be retroactively canaried.\n"
        "- **Detection works for memorization-style leaks** (model trained "
        "on canaried audio reproduces canary motifs at high gain in its "
        "outputs). Validated end-to-end at p ≤ 10⁻¹⁰ across 18 codec / "
        "compression / pitch transforms — see `test_canary_robustness.py`.\n"
        "- **Verbatim file copy at low embed gain is NOT detected** by "
        "this content-similarity detector. That requires a matched-filter "
        "detector — different attack class, separate roadmap item.\n"
        "- **Arms race risk.** Model trainers who suspect canaries can "
        "filter via deduplication / anomaly detection on their training "
        "data. Watermarking literature is full of defeated schemes.\n"
        "- Library is private. Don't share `canary_index.json` publicly."
    )
