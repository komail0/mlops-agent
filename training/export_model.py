"""Train on one dataset version and export the model for the inference image.

    python -m training.export_model --version v1 --out models

Writes models/model.joblib and models/model_info.json (metrics and dataset provenance).
The models/ folder is not tracked by Git; it is rebuilt from the DVC-versioned data.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import joblib

from training.make_data import build
from training.seed_mlflow import tags_for_version
from training.train import train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    parser.add_argument("--out", default="models")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "data.csv"
        build(args.version).to_csv(csv, index=False)
        model, metrics = train(csv, args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    info = {"metrics": metrics, "dataset": tags_for_version(args.version), "model": args.model}
    (out / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"exported {args.model} trained on {args.version}: f1={metrics['f1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
