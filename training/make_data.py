"""Generate the demo dataset versions (deterministic).

v1: the scikit-learn breast-cancer dataset as a clean CSV.
v2: the same rows with a share of labels flipped, simulating degraded data.

The course material names no dataset, so a small public tabular one is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

LABEL = "target"


def build(version: str, flip_fraction: float = 0.3, seed: int = 7) -> pd.DataFrame:
    frame = load_breast_cancer(as_frame=True).frame
    if version == "v1":
        return frame
    if version == "v2":
        rng = np.random.default_rng(seed)
        out = frame.copy()
        idx = rng.choice(len(out), size=int(len(out) * flip_fraction), replace=False)
        out.loc[out.index[idx], LABEL] = 1 - out.loc[out.index[idx], LABEL]
        return out
    raise ValueError(f"unknown version '{version}' (use v1 or v2)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["v1", "v2"], required=True)
    parser.add_argument("--out", default="data/dataset.csv")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build(args.version).to_csv(out, index=False)
    print(f"wrote {out} ({args.version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
