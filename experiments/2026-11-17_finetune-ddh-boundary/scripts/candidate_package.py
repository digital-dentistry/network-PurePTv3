"""A candidate model package for evaluation only: the fine-tuned instance checkpoint + the base package's semantic
model and model.json.  Runs in ddh-inference-server:dev (uses /app/package_toothsegnet.slim, the same slimming as a
real package).  Writes outside ./models; a real package is made with package_toothsegnet.py once a run is accepted.

usage: python candidate_package.py <train checkpoint> <base package dir> <out dir> <label>
"""
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/app")
from package_toothsegnet import slim  # noqa: E402


def main():
    ck_path, base, out, label = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    out.mkdir(parents=True, exist_ok=True)
    inst, note = slim(ck_path, "instance")
    torch.save(inst, out / "instance.pt")
    shutil.copy(base / "semantic.pt", out / "semantic.pt")
    spec = json.loads((base / "model.json").read_text())
    spec["version"] = label
    spec["description"] = f"candidate {label}: {spec.get('description', '')}"
    spec.pop("sha256", None)
    spec["candidate"] = {"instance": note | {"val_at_save": None}, "base": str(base)}
    (out / "model.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1, default=str))
    print(f"{out}: epoch {note['epoch']}")


if __name__ == "__main__":
    main()
