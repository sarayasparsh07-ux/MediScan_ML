"""Loading and validation of the MediScan dataset.

The CSV is already cleaned (see 00_EDA_Report.md). This module therefore
*verifies* rather than re-cleans: every check is reported, nothing is
silently altered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    EXPECTED_CLASS_COUNTS,
    EXPECTED_COLUMNS,
    EXPECTED_N_ROWS,
    FEATURES,
    ID_COL,
    TARGET,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_PATHS = [
    PROJECT_ROOT / "data" / "mediscan.csv",
    PROJECT_ROOT / "mediscan.csv",
    Path.cwd() / "mediscan.csv",
    Path.cwd() / "data" / "mediscan.csv",
]


class DataValidationError(RuntimeError):
    """Raised when the dataset cannot be used safely for modelling."""


@dataclass
class Check:
    name: str
    status: str  # "pass" | "warn" | "fail"
    detail: str


@dataclass
class ValidationReport:
    path: str
    n_rows: int
    n_cols: int
    checks: list[Check] = field(default_factory=list)

    @property
    def n_fail(self) -> int:
        return sum(c.status == "fail" for c in self.checks)

    @property
    def n_warn(self) -> int:
        return sum(c.status == "warn" for c in self.checks)

    def add(self, name: str, ok: bool, detail: str, severity: str = "fail") -> None:
        self.checks.append(Check(name, "pass" if ok else severity, detail))

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.__dict__ for c in self.checks]).rename(
            columns={"name": "Check", "status": "Status", "detail": "Finding"}
        )


def locate_csv(explicit: str | Path | None = None) -> Path:
    """Find mediscan.csv robustly (explicit path, data/ folder, project root, cwd)."""
    paths = [Path(explicit)] if explicit else []
    paths += CANDIDATE_PATHS
    for p in paths:
        if p.is_file():
            return p
    tried = "\n".join(f"  • {p}" for p in paths)
    raise FileNotFoundError(
        "Could not find mediscan.csv. Place it in the project's data/ folder.\nTried:\n" + tried
    )


def load_and_validate(path: str | Path | None = None) -> tuple[pd.DataFrame, ValidationReport]:
    """Load the CSV and run every start-up check. Never modifies observations."""
    csv_path = locate_csv(path)
    df = pd.read_csv(csv_path)
    rep = ValidationReport(path=str(csv_path), n_rows=len(df), n_cols=df.shape[1])

    # 1. Schema
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    rep.add(
        "Expected columns present",
        not missing_cols,
        "All 10 expected columns found."
        if not missing_cols
        else f"Missing columns: {missing_cols}",
    )
    rep.add(
        "No unexpected columns",
        not extra_cols,
        "No extra columns." if not extra_cols else f"Unexpected columns (ignored for modelling): {extra_cols}",
        severity="warn",
    )
    if missing_cols:
        raise DataValidationError(f"Schema mismatch — missing columns {missing_cols}.")

    # 2. Size vs EDA report
    rep.add(
        "Row count matches EDA report",
        len(df) == EXPECTED_N_ROWS,
        f"{len(df)} rows (EDA report: {EXPECTED_N_ROWS}).",
        severity="warn",
    )

    # 3. Types
    non_numeric = [c for c in FEATURES + [TARGET] if not pd.api.types.is_numeric_dtype(df[c])]
    rep.add(
        "Predictors and target are numeric",
        not non_numeric,
        "All 8 predictors and the target are numeric."
        if not non_numeric
        else f"Non-numeric columns: {non_numeric}",
    )
    if non_numeric:
        raise DataValidationError(f"Non-numeric modelling columns: {non_numeric}")

    # 4. Missing values
    n_missing = int(df[EXPECTED_COLUMNS].isna().sum().sum())
    by_col = df[EXPECTED_COLUMNS].isna().sum()
    rep.add(
        "No missing values",
        n_missing == 0,
        "0 missing values (EDA report: medians already imputed)."
        if n_missing == 0
        else f"{n_missing} missing values: {by_col[by_col > 0].to_dict()}",
    )

    # 5. Duplicates
    n_dup_rows = int(df.duplicated().sum())
    n_dup_ids = int(df[ID_COL].duplicated().sum())
    rep.add("No duplicate rows", n_dup_rows == 0, f"{n_dup_rows} fully duplicated rows.", severity="warn")
    rep.add("Unique patient IDs", n_dup_ids == 0, f"{n_dup_ids} repeated patient IDs.", severity="warn")

    # 6. Target labels
    labels = sorted(df[TARGET].dropna().unique().tolist())
    rep.add(
        "Target labels are {0, 1}",
        set(labels) <= {0, 1} and len(labels) == 2,
        f"Observed labels: {labels} (0 = Benign, 1 = Malignant).",
    )
    counts = df[TARGET].value_counts().to_dict()
    rep.add(
        "Class counts match EDA report",
        {int(k): int(v) for k, v in counts.items()} == EXPECTED_CLASS_COUNTS,
        f"Benign {counts.get(0, 0)}, Malignant {counts.get(1, 0)} "
        f"(EDA report: {EXPECTED_CLASS_COUNTS[0]} / {EXPECTED_CLASS_COUNTS[1]}).",
        severity="warn",
    )

    # 7. Impossible values (EDA report: negatives were clipped)
    neg = {c: int((df[c] < 0).sum()) for c in FEATURES if (df[c] < 0).any()}
    rep.add(
        "No negative measurements",
        not neg,
        "All measurements ≥ 0 (EDA report: negative noise already clipped)."
        if not neg
        else f"Negative values found: {neg}",
        severity="warn",
    )
    nonfinite = int((~np.isfinite(df[FEATURES].to_numpy(dtype=float))).sum())
    rep.add("All measurements finite", nonfinite == 0, f"{nonfinite} non-finite values.")

    # 8. Leakage guard: identifier must not be numeric-encodable as a feature
    rep.add(
        "Identifier excluded from predictors",
        ID_COL not in FEATURES and TARGET not in FEATURES,
        f"'{ID_COL}' and '{TARGET}' are excluded from the 8-feature predictor matrix.",
    )

    if rep.n_fail:
        failed = [c.name for c in rep.checks if c.status == "fail"]
        raise DataValidationError(f"Dataset failed validation: {failed}")
    return df, rep


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X = the 8 predictors only; y = diagnosis. patient_id never enters X."""
    X = df[FEATURES].astype(float).copy()
    y = df[TARGET].astype(int).copy()
    assert ID_COL not in X.columns and TARGET not in X.columns
    return X, y
