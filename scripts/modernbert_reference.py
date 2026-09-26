# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.48", "safetensors", "numpy"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Dump HF ModernBERT per-layer hidden states as a parity fixture for the Burn port.

    uv run scripts/modernbert_reference.py [--model models/modernbert-base] [--out tests/fixtures/modernbert_ref.safetensors]

Cases:
  short  - one short sequence (all layers see everything; local == global)
  long   - one sequence longer than the 128-token local window
  padded - a right-padded batch of two, exercising the key padding mask

For each case the fixture holds `<case>.input_ids`, `<case>.attention_mask` and
`<case>.hidden_<i>` for HF's `hidden_states` tuple: index 0 is the embedding output,
1..n-1 are layer outputs, and index n is the final-normed last layer.
"""

import argparse
import json
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer

SHORT = "Unusual outbound SSH connection from build agent ci-runner-07 to 203.0.113.44 at 03:12 UTC."
MEDIUM = (
    "Deployment pipeline for payments-api promoted build 4812 to production after the canary "
    "stage reported a 0.2% error rate, within the 1% budget."
)
LONG = " ".join(
    [
        "Telemetry: the identity provider logged 37 failed MFA challenges for user svc-backup",
        "across three regions within ten minutes, followed by a successful login from a new ASN.",
        "The session then enumerated IAM roles, attached AdministratorAccess to a dormant role,",
        "and created two access keys. CloudTrail shows the keys were used from a residential proxy",
        "to list S3 buckets tagged finance and to disable object versioning on one of them.",
        "No change ticket references these actions, and the on-call engineer did not approve them.",
    ]
    * 3
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/modernbert-base")
    ap.add_argument("--out", default="tests/fixtures/modernbert_ref.safetensors")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, attn_implementation="eager", torch_dtype=torch.float32)
    model.eval()

    cases = {"short": [SHORT], "long": [LONG], "padded": [SHORT, MEDIUM]}
    tensors = {}
    summary = {}
    for name, texts in cases.items():
        enc = tok(texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        tensors[f"{name}.input_ids"] = enc["input_ids"].to(torch.int64).contiguous()
        tensors[f"{name}.attention_mask"] = enc["attention_mask"].to(torch.int64).contiguous()
        for i, h in enumerate(out.hidden_states):
            tensors[f"{name}.hidden_{i}"] = h.to(torch.float32).contiguous()
        summary[name] = {"shape": list(enc["input_ids"].shape), "hidden_states": len(out.hidden_states)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"transformers": transformers.__version__, "torch": torch.__version__, "cases": json.dumps(summary)}
    save_file(tensors, str(out_path), metadata=meta)
    print(f"wrote {out_path}: {summary} (transformers {transformers.__version__})")


if __name__ == "__main__":
    main()
