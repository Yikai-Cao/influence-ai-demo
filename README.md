---
title: Influence AI Demo
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# Influence AI — Data Attribution Evidence Report (Demo)

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/ykcao/influence-ai-demo)

**[Try the live demo →](https://huggingface.co/spaces/ykcao/influence-ai-demo)**
(hosted on HuggingFace Spaces, free CPU tier — three tabs, all interactive)

A Streamlit app that helps copyright holders (especially music
publishers) investigate whether their content was used to train AI
models. Three tabs covering the realistic access-level matrix:

| Tab | What it does | Best for |
|---|---|---|
| **📄 Text (Pythia)** | Corpus-level MIA on HuggingFace text language models | Open LMs (Pythia, OPT, GPT-NeoX) |
| **🎵 Audio (MusicGen, beta)** | Audio analog targeting `facebook/musicgen-small` | Open music models (MusicGen, Stable Audio Open) |
| **🕵️ Canary (Plan B)** | Generate / embed / scan canaries — black-box leak detection | **Closed models (Suno, Udio, ElevenLabs)** |

> *Did this model train on my catalog?* — same question, three
> different access levels, three different tools.

**Headline experimental result** (Pythia-6.9B vs Pile-Wikipedia, 500
suspect + 1000 control): **p ≈ 4.46 × 10⁻⁵** (positive) /
**p ≈ 0.80** (false-positive control). Statistically decisive
evidence of training-set inclusion under the audited model.

Prototype built for Stanford Lean Launchpad. Methodology follows
[Maini et al., *LLM Dataset Inference: Did you train on my dataset?*
(NeurIPS 2024)](https://arxiv.org/abs/2406.06443) — 16-feature subset
(text) or 29-feature per-codebook subset (audio) from a single forward
pass, combined with a learned logistic regression classifier and a
one-sided Welch's t-test.

---

## The three tabs

### 📄 Text (Pythia) — Plan A for text LMs

Upload a suspect corpus + control corpus → score against any HF causal
LM → get a p-value verdict.

- Click **"Load Pile Wikipedia example"** to load 500 suspect + 1000
  control passages and see the full pipeline run.
- Default audited model is `pythia-160m-deduped` on CPU (UI smoke test
  — does not produce meaningful signal). For the headline 4.46 × 10⁻⁵
  result, switch the backend to **Modal A10G** and pick
  `pythia-6.9b-deduped`.

Input format: `.txt` (one passage per line) or `.jsonl` (one JSON
object per line with a `text` or `content` field). Control corpus
must be at least 2× the suspect size.

### 🎵 Audio (MusicGen, beta) — Plan A for music

Audio analog targeting `facebook/musicgen-small`. The hosted demo
includes a **pre-computed example report** (click "Load audio
example") so visitors can see the report layout instantly without
hosted MusicGen inference.

For live audio runs against your own clips, install the audio deps
locally (`pip install librosa soundfile peft`) and run
`streamlit run app.py`. For the strong-signal Phase B fine-tuned
result, see `phase1_reproduction/AUDIO_README.md` in the parent
project repository.

### 🕵️ Canary (Plan B) — for closed black-box models

Three sub-modes that work entirely on the free hosted CPU tier:

1. **Generate library** — pick N + transpositions + seed, click
   Generate, download a private `.zip` containing the synthesized
   canary motifs and `canary_index.json`.
2. **Embed in track** — upload your unreleased song + the library zip,
   pick gain (default −25 dB), embed N canaries, listen to the A/B
   preview, download the canaried `.wav` plus the embedding
   `.manifest.json`.
3. **Scan suspect outputs** — upload the library + multiple suspect
   audio files (e.g. outputs you got from a closed model API), tune
   threshold + null FPR sliders, click Run scan → corpus-level
   binomial p-value, top-20 (canary, suspect) similarity hits, and a
   downloadable JSON report.

Robustness sweep (locally, see `canary/` subpackage): the detector
survives **18/18 realistic codec / compression / pitch transforms**
when paired with a multi-transposition library, including an
"aggressive neural codec" pipeline (lowpass to 8 kHz + downsample to
16 kHz + 25 dB-SNR noise + μ-law 8-bit).

---

## Quickstart (run locally)

```bash
git clone https://github.com/Yikai-Cao/influence-ai-demo.git
cd influence-ai-demo
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. All three tabs work immediately on a
fresh install.

---

## How the MIA tabs work (Text + Audio)

1. **Score** every passage / clip through the audited model,
   extracting per-token cross-entropy losses.
2. **Extract features** — 16 per-passage (text: ppl, zlib ratio,
   7 Min-K%, 7 Max-K%) or 29 per-clip (audio: ppl + Min-K% + Max-K%
   for each of MusicGen's 4 EnCodec codebooks + 1 bitrate ratio).
3. **Train a logistic regression classifier** on one half of the
   suspect + control to separate member from non-member.
4. **Apply the classifier** to the held-out half, run a one-sided
   Welch's t-test → **positive p-value**.
5. **False-positive control** — compare control-half-A vs
   control-half-B → **control p-value** (should be > 0.3).

The verdict badge (STRONG / MODERATE / INCONCLUSIVE / NO) combines
both p-values.

## How the Canary tab works

Canaries are short distinctive musical motifs you plant in your
catalog *before* publication. If a generative model later trains on
your canaried tracks and reproduces canary-like content in its
outputs, we detect that statistically.

1. **Generate** a private library of canary motifs (deterministic —
   same `(N, transpositions, seed)` → same library, every time).
2. **Embed** canaries into your unreleased audio at low gain (−25 dB
   default → ~28 dB perceptual SNR, essentially imperceptible). The
   manifest records exactly which canaries went where.
3. **Detect** by computing MFCC + chromagram sliding cosine similarity
   between every canary and every suspect clip; aggregate with a
   one-sided binomial test against a calibrated null FPR.

Multi-transposition libraries (each canary registered at ±0.5
semitones) close the pitch-shift attack identified by the robustness
sweep — see the local `canary/` subpackage and parent project's
`canary_prototype/README.md` for the full design rationale and A/B
results (Mode B survives all 18 transforms tested).

---

## Running large models on Modal (Plan A only)

Pythia-160m on CPU is a UI smoke test — does not produce meaningful
signal. For the headline 4.46 × 10⁻⁵ result, you need Pythia-6.9B on
GPU.

```bash
pip install modal
modal setup      # one-time auth
```

Then in the Text tab sidebar, pick **Modal A10G (cloud)** and model
`EleutherAI/pythia-6.9b-deduped`. Each audit ≈ 10–15 minutes including
cold start, ~$0.20 at A10G spot pricing.

The Audio tab's Modal path is the CLI in
`phase1_reproduction/modal_audio_backend.py` (Phase B fine-tune +
audit, ~$3, ~2 h) — the in-app Modal button is intentionally limited
because uploading audio to Modal Volumes via a web UI isn't a clean
pattern.

---

## Layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — three tabs (Text, Audio, Canary) |
| `mia_stats.py` | Shared stats layer — classifier, t-test, `EvidenceReport` |
| `mia_core.py` | **Text** MIA pipeline (16 features) |
| `mia_audio_core.py` | **Audio** MIA pipeline (29 per-codebook features) |
| `report.py` | Markdown report renderer (text + audio aware) |
| `modal_backend.py` | Optional Modal A10G backend for large text models |
| `canary/` | **Canary subpackage** (generator, embedder, detector, transforms) |
| `prepare_example_corpora.py` | One-shot downloader for the Pile Wikipedia example |
| `examples/pile_wikipedia_{suspect,control}.jsonl` | Bundled text example |
| `examples/audio_demo_report.json` | Bundled pre-computed audio report |
| `smoke_test.py` / `smoke_test_audio.py` | Headless end-to-end pipeline tests |
| `test_audio_synthetic.py` | Synthetic 4-scenario stats test (no MusicGen download) |
| `requirements.txt` | All deps (Streamlit + torch + transformers + librosa + soundfile + peft + scipy + sklearn) |

---

## Limitations — please read before making claims

The methods have real technical constraints. We document them
explicitly so customers and reviewers know what's claimed and what
isn't.

### MIA tabs (Text + Audio)

- **Control corpus quality is decisive.** Control must be drawn from
  the same distribution as suspect but known *not* to be in training.
  Distribution mismatch can flip the signal. If the control p-value
  fails (< 0.3), the headline is not trustworthy.
- **Gray-box access required.** Per-token log-probabilities from the
  audited model. Does **not** work on pure black-box APIs (Suno,
  Udio) — that's what the Canary tab is for.
- **Statistical claim, not proof.** A significant p-value is strong
  evidence, not court-ready proof, of training-set inclusion.
- **Small models produce no signal.** Pythia-160m on Pile, or base
  MusicGen-small on synthetic audio, both correctly produce *NO
  evidence* verdicts in the hosted demo. Real headline numbers
  require Pythia-1.4B+ (or fine-tuned MusicGen) on GPU.

### Canary tab (Plan B)

- **Forward-looking only.** Catalogs already published cannot be
  retroactively canaried.
- **Detects memorization-style leaks.** A model trained on your
  canaried audio that *regurgitates* the canary motif in its output:
  ✅ detected, validated end-to-end at p ≤ 10⁻¹⁰ across 18 codec /
  compression / pitch transforms.
- **Does not detect verbatim file copies at low embed gain.** A
  matched-filter detector would handle this — separate code path,
  flagged as roadmap.
- **Arms-race risk.** Model trainers who notice canaries can filter
  them via deduplication / anomaly detection.
- **Library is private.** Don't share `canary_index.json` publicly.

---

## Citation

If you build on this, please cite the underlying method:

```bibtex
@inproceedings{maini2024llm,
  title={LLM Dataset Inference: Did you train on my dataset?},
  author={Maini, Pratyush and Jia, Hengrui and Papernot, Nicolas and Dziedzic, Adam},
  booktitle={NeurIPS},
  year={2024}
}
```

For the canary methodology, the relevant precedent is Sablayrolles
et al., *Radioactive Data: Tracing Through Training* (ICML 2020).

## License

MIT. See `LICENSE`.
