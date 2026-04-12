"""Headless smoke test: exercise the full mia_core pipeline end-to-end.

Uses Pythia-160m + a small slice of the Pile wikipedia subset. Does NOT
assert a significant p-value — per CLAUDE.md, 160m + 16 features is too
weak. Purpose: confirm the pipeline runs without errors and emits a
well-formed EvidenceReport.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mia_core import load_model, run_evidence_report
from report import render_markdown


def main():
    n_suspect = 20
    n_control = 40  # must be >= 2 * n_suspect
    model_name = "EleutherAI/pythia-160m-deduped"

    print(f"[smoke] loading Pile wikipedia slice (n_suspect={n_suspect}, n_control={n_control})")
    from datasets import load_dataset
    ds_train = load_dataset("pratyushmaini/llm_dataset_inference", "wikipedia", split="train")
    ds_val = load_dataset("pratyushmaini/llm_dataset_inference", "wikipedia", split="val")
    suspect = [r["text"] for r in ds_train.select(range(n_suspect))]
    control = [r["text"] for r in ds_val.select(range(n_control))]

    print(f"[smoke] loading {model_name}")
    t0 = time.time()
    model, tokenizer = load_model(model_name, device="cpu")
    print(f"[smoke] model loaded in {time.time() - t0:.1f}s")

    def cb(stage, frac):
        print(f"  [{stage}] {frac * 100:.0f}%")

    t0 = time.time()
    report = run_evidence_report(
        suspect, control, model, tokenizer,
        model_name=model_name, max_length=256, batch_size=4, device="cpu",
        progress=cb,
    )
    print(f"[smoke] pipeline done in {time.time() - t0:.1f}s")

    print("\n--- VERDICT ---")
    print(report.verdict())
    print(f"positive p = {report.positive_test['p_value']:.4g}")
    print(f"control  p = {report.false_positive_control['p_value']:.4g}")
    print(f"features   = {len(report.feature_names)}")

    md = render_markdown(report)
    out = Path(__file__).parent / "smoke_report.md"
    out.write_text(md)
    print(f"\n[smoke] wrote {out} ({len(md)} chars)")
    print("[smoke] PASS — pipeline executed end-to-end")


if __name__ == "__main__":
    main()
