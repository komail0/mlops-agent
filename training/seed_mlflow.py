"""Seed MLflow with real training runs on both dataset versions.

Each run trains on the exact bytes of one DVC dataset version and records provenance read
from Git history (`git show data-vN:...`), so the recorded tags are true for that data.

    python -m training.seed_mlflow            # uses MLFLOW_TRACKING_URI
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from training.make_data import build
from training.train import log_to_mlflow, train

EXPERIMENT = "mlops-agent-demo"
PLAN = [
    ("v1", "logreg", 42),
    ("v1", "rf", 42),
    ("v2", "logreg", 42),
    ("v2", "rf", 42),
    ("v1", "logreg", 7),
]


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, timeout=30
    ).stdout.strip()


def tags_for_version(version: str, dvc_path: str = "data/dataset.csv.dvc") -> dict[str, str]:
    ref = f"data-{version}"
    text = _git(["show", f"{ref}:{dvc_path}"])
    md5 = re.search(r"md5:\s*([0-9a-f]{32})", text).group(1)
    return {
        "dataset.version": version,
        "dvc.md5": md5,
        "dvc.dataset_path": dvc_path.removesuffix(".dvc"),
        "dvc.dataset_commit": _git(["rev-parse", ref + "^{commit}"]),
        "git.commit": _git(["rev-parse", "HEAD"]),
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        for version, model_type, seed in PLAN:
            csv = Path(tmp) / f"{version}.csv"
            build(version).to_csv(csv, index=False)
            model, metrics = train(csv, model_type, seed)
            run_id = log_to_mlflow(
                metrics,
                {"model_type": model_type, "seed": seed},
                tags_for_version(version),
                EXPERIMENT,
                model=model,
                run_name=f"{model_type}-{version}-seed{seed}",
            )
            print(f"{run_id} {version} {model_type} f1={metrics['f1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
