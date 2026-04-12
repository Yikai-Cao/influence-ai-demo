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
(hosted on HuggingFace Spaces, free CPU tier — runs Pythia-160m for a UI walkthrough)

A Streamlit app that takes a **suspect corpus** and a **control corpus**, runs
membership inference against an audited language model, and produces a
p-value-backed evidence report answering:

> *Was this content used to train the model?*

Prototype built for Stanford Lean Launchpad. Methodology follows
[Maini et al., *LLM Dataset Inference: Did you train on my dataset?*
(NeurIPS 2024)](https://arxiv.org/abs/2406.06443) — specifically a
16-feature subset extracted from a single forward pass, combined with a
learned logistic regression classifier and a one-sided Welch's t-test.

On Pythia-6.9B against 500 Pile-Wikipedia training passages vs 1000 val
passages, this pipeline produces **p ≈ 4.46 × 10⁻⁵** (positive) with
**p ≈ 0.80** (false-positive control) — statistically decisive evidence of
training-set membership, with the control confirming the pipeline is clean.

## Quickstart

```bash
git clone https://github.com/Yikai-Cao/influence-ai-demo.git
cd influence-ai-demo
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Default audited model is `pythia-160m-deduped`
on CPU — enough to smoke-test the UI end-to-end with the bundled example.

Click **"Load Pile Wikipedia example"** to load 500 suspect + 1000 control
Wikipedia passages and see the full pipeline run.

## How it works

1. **Score** every passage through the audited model, extracting per-token
   cross-entropy losses.
2. **Extract 16 features** per passage from those losses: perplexity, zlib
   ratio, and 14 Min-K% / Max-K% variants.
3. **Train a logistic regression classifier** on one half of the suspect +
   control corpora to separate member from non-member.
4. **Apply the classifier** to the held-out half; run a one-sided
   Welch's t-test comparing suspect vs control scores → **positive p-value**.
5. **Run a false-positive control test** comparing control-half-A vs
   control-half-B → **control p-value** (should be > 0.3 if the pipeline is
   clean).

The verdict badge (STRONG / MODERATE / INCONCLUSIVE / NO) combines both
p-values.

## Running large models on Modal

Pythia-160m is a UI smoke test — it does not produce meaningful signal.
The result in the intro (p ≈ 4.46e-05) requires Pythia-6.9B, which needs
an A10G GPU (24 GB VRAM).

Wire in the [Modal](https://modal.com) backend:

```bash
pip install modal
modal setup      # one-time auth
```

Then in the sidebar, pick **Modal A10G (cloud)** and model
`EleutherAI/pythia-6.9b-deduped`. Each audit ≈ 10–15 minutes including
cold start, ~$0.20 at A10G spot pricing.

## Input format

- `.txt` — one passage per line
- `.jsonl` — one JSON object per line with a `text` (or `content`) field

**Requirement:** the control corpus must be **at least 2× the suspect
corpus size**. Half trains the classifier; the remainder is split into
the positive test set and a false-positive control group.

## Layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `mia_core.py` | Feature extraction, classifier, t-test, `EvidenceReport` |
| `report.py` | Markdown report renderer |
| `modal_backend.py` | Optional Modal A10G backend for large models |
| `prepare_example_corpora.py` | One-shot downloader for bundled example |
| `examples/` | 500-suspect + 1000-control Pile Wikipedia JSONL |
| `smoke_test.py` | Headless end-to-end pipeline test |

## Limitations — please read before making claims

This is a research prototype. The method has real technical constraints:

- **Control corpus quality is decisive.** The control set must be drawn
  from the same distribution as the suspect corpus but known *not* to be
  in the training set. Distribution mismatch (topic, length, style) can
  produce spurious positives or reverse the signal. If the control
  p-value fails (< 0.3), the headline p-value is not trustworthy.
- **Gray-box access required.** The method needs per-token
  log-probabilities from the audited model. It does **not** work on pure
  black-box APIs, and most commercial audio models (Suno, MusicGen) do
  not expose this interface.
- **Passage independence is assumed.** Chunks from the same document are
  correlated; the reported p-value assumes independence across passages.
  For strict use, aggregate to document level first.
- **Statistical claim, not proof.** A significant p-value indicates the
  suspect corpus is statistically distinguishable from the control under
  this model — strong evidence, but not court-ready proof, of
  training-set inclusion.
- **Small models produce no signal.** The method relies on the model
  memorizing its training data. Pythia-160m on The Pile does not
  memorize enough for a 16-feature lightweight MIA to detect. You need
  Pythia-1.4B+ (preferably 6.9B+) to see meaningful results.

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

## License

MIT. See `LICENSE`.
