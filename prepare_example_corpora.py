"""Download + save example suspect/control corpora for the demo.

Pulls 500 train passages (suspect) and 1000 val passages (control) from
`pratyushmaini/llm_dataset_inference` wikipedia subset. These are the exact
slices Phase 1b used to achieve p=4.46e-05 on Pythia-6.9B.

Run once to generate `examples/pile_wikipedia_{suspect,control}.jsonl`.
"""

from __future__ import annotations

import json
from pathlib import Path


def main():
    from datasets import load_dataset

    out_dir = Path(__file__).parent / "examples"
    out_dir.mkdir(exist_ok=True)

    print("[prepare] loading pratyushmaini/llm_dataset_inference wikipedia ...")
    ds_train = load_dataset("pratyushmaini/llm_dataset_inference", "wikipedia", split="train")
    ds_val = load_dataset("pratyushmaini/llm_dataset_inference", "wikipedia", split="val")

    n_suspect, n_control = 500, 1000
    print(f"[prepare] writing {n_suspect} suspect + {n_control} control passages")

    suspect_path = out_dir / "pile_wikipedia_suspect.jsonl"
    control_path = out_dir / "pile_wikipedia_control.jsonl"

    with suspect_path.open("w") as f:
        for row in ds_train.select(range(n_suspect)):
            f.write(json.dumps({"text": row["text"]}) + "\n")

    with control_path.open("w") as f:
        for row in ds_val.select(range(n_control)):
            f.write(json.dumps({"text": row["text"]}) + "\n")

    print(f"[prepare] wrote {suspect_path} ({suspect_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[prepare] wrote {control_path} ({control_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
