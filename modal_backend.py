"""
Modal backend for the evidence-report demo.

Lets the Streamlit app offload heavy inference (Pythia-6.9B, etc.) to a cloud GPU
instead of the local machine. Call `run_audit_remote` from app.py when the user
picks a model that's too large for CPU.

Usage (standalone test):
    modal run modal_backend.py::run_audit \\
        --suspect-path suspect.txt --control-path control.txt \\
        --model-name EleutherAI/pythia-6.9b-deduped
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("influence-ai-evidence-report")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "scipy",
        "numpy",
        "accelerate",
        "scikit-learn",
    )
    .add_local_dir(str(Path(__file__).parent), remote_path="/root/app")
)

volume = modal.Volume.from_name("influence-ai-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={"/results": volume},
)
def run_audit_remote(
    suspect_texts: list[str],
    control_texts: list[str],
    model_name: str = "EleutherAI/pythia-6.9b-deduped",
    max_length: int = 512,
    batch_size: int = 4,
) -> dict:
    import sys
    sys.path.insert(0, "/root/app")

    from mia_core import load_model, run_evidence_report

    model, tokenizer = load_model(model_name, device="cuda")
    report = run_evidence_report(
        suspect_texts,
        control_texts,
        model,
        tokenizer,
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
        device="cuda",
    )
    return report.to_dict()


@app.local_entrypoint()
def run_audit(
    suspect_path: str,
    control_path: str,
    model_name: str = "EleutherAI/pythia-6.9b-deduped",
):
    suspect = [l for l in Path(suspect_path).read_text().splitlines() if l.strip()]
    control = [l for l in Path(control_path).read_text().splitlines() if l.strip()]
    result = run_audit_remote.remote(suspect, control, model_name=model_name)
    print(json.dumps(result, indent=2))
