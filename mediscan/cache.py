"""Disk cache for the full analysis (≈2 minutes to compute from scratch).

The cache key is a fingerprint of the CSV's bytes, the library versions and the analysis
version, so any change to the data or the environment triggers a clean recomputation.
Deleting the .cache folder is always safe.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ANALYSIS_VERSION = "mediscan-2026-10-22-v4"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def fingerprint(csv_path: str | Path) -> str:
    h = hashlib.sha256(Path(csv_path).read_bytes())
    h.update(f"{ANALYSIS_VERSION}|{sklearn.__version__}|{np.__version__}|{pd.__version__}".encode())
    return h.hexdigest()[:20]


def load_or_run(df: pd.DataFrame, csv_path: str | Path):
    """Return (Analysis, source) where source is 'cache' or 'computed'."""
    from .pipeline import run_all

    path = CACHE_DIR / f"analysis_{fingerprint(csv_path)}.pkl"
    if path.exists():
        try:
            with path.open("rb") as fh:
                return pickle.load(fh), "cache"
        except Exception:
            path.unlink(missing_ok=True)  # stale or incompatible cache -> recompute
    A = run_all(df)
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        for old in CACHE_DIR.glob("analysis_*.pkl"):
            old.unlink(missing_ok=True)
        with path.open("wb") as fh:
            pickle.dump(A, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass  # read-only disk etc. — the app still works, it just recomputes next time
    return A, "computed"


if __name__ == "__main__":  # python -m mediscan.cache  -> build the cache before a demo
    import time

    from .data import load_and_validate, locate_csv

    t0 = time.time()
    path = locate_csv()
    df, _ = load_and_validate(path)
    A, src = load_or_run(df, path)
    print(f"Analysis ready ({src}) in {time.time() - t0:.0f}s -> {CACHE_DIR}")
