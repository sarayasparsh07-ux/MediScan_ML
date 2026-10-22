"""Statistical significance tests and decision-threshold tuning.

Significance
------------
* McNemar's exact test — paired comparison of two classifiers on the SAME held-out
  patients. Only discordant patients matter: b = A right & B wrong, c = A wrong & B right.
  Under H0 (equal error rates) b ~ Binomial(b + c, 0.5). Run on all patients (accuracy),
  malignant only (recall / missed cancers) or benign only (specificity / false alarms).
* Corrected resampled t-test (Nadeau & Bengio, 2003) — paired comparison across the
  repeated CV folds. All models were scored on identical folds, so per-fold differences
  are paired; the variance is inflated by (1/J + n_test/n_train) because CV folds overlap.
* Holm–Bonferroni adjustment when many pairs are tested at once.

Thresholds
----------
Out-of-fold (OOF) scores on the TRAINING set only are used to choose a threshold; the
held-out test set is used once, afterwards, to report what that choice does. The decision
rule everywhere is "malignant if score > threshold", which reproduces scikit-learn's own
predict() at the default threshold (0.5 for probabilities, 0 for SVM decision values).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .config import MODEL_ORDER
from .pipeline import CV_SPLITS, compute_metrics

SUBSETS = {"all": "All test patients (accuracy)", "malignant": "Malignant only (missed cancers)",
           "benign": "Benign only (false alarms)"}


# ======================================================================== McNemar
def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact (binomial) McNemar test. Returns discordant counts, p-value and the smallest p achievable."""
    a, b_ = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b = int((a & ~b_).sum())   # A right, B wrong
    c = int((~a & b_).sum())   # A wrong, B right
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sps.binom.cdf(min(b, c), n, 0.5))
    p_min = 1.0 if n == 0 else min(1.0, 2 * 0.5 ** n)  # most extreme split possible with n discordant
    both_right = int((a & b_).sum())
    both_wrong = int((~a & ~b_).sum())
    return dict(b=b, c=c, n_discordant=n, p=float(p), p_min=float(p_min),
                both_right=both_right, both_wrong=both_wrong)


def _subset_mask(y_test: np.ndarray, subset: str) -> np.ndarray:
    y = np.asarray(y_test)
    return {"all": np.ones_like(y, bool), "malignant": y == 1, "benign": y == 0}[subset]


def holm(pvals: pd.Series) -> pd.Series:
    """Holm–Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    p = pvals.dropna().sort_values()
    m = len(p)
    adj, running = {}, 0.0
    for i, (k, v) in enumerate(p.items()):
        running = max(running, min(1.0, (m - i) * v))
        adj[k] = running
    return pd.Series(adj).reindex(pvals.index)


def mcnemar_table(results: dict, y_test, subset: str = "all", models: list[str] | None = None) -> pd.DataFrame:
    """All pairwise McNemar tests among `models` on one patient subset, with Holm adjustment."""
    models = [m for m in (models or MODEL_ORDER) if results[m].ok]
    mask = _subset_mask(y_test, subset)
    y = np.asarray(y_test)[mask]
    rows = []
    for m1, m2 in combinations(models, 2):
        c1 = results[m1].y_pred[mask] == y
        c2 = results[m2].y_pred[mask] == y
        r = mcnemar_exact(c1, c2)
        rows.append({"Model A": m1, "Model B": m2, **r, "n": int(mask.sum()),
                     "A correct": int(c1.sum()), "B correct": int(c2.sum())})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(df["p"]).values
    return df


def pvalue_matrix(tbl: pd.DataFrame, models: list[str], col: str = "p_holm") -> pd.DataFrame:
    M = pd.DataFrame(np.nan, index=models, columns=models)
    for _, r in tbl.iterrows():
        M.loc[r["Model A"], r["Model B"]] = M.loc[r["Model B"], r["Model A"]] = r[col]
    return M


# ======================================================================== corrected resampled t-test
def corrected_resampled_ttest(diffs: np.ndarray, n_train: int, n_test: int) -> dict:
    """Nadeau & Bengio (2003) corrected repeated k-fold t-test on paired per-fold differences."""
    d = np.asarray(diffs, float)
    J = len(d)
    mean = float(d.mean())
    var = float(d.var(ddof=1)) if J > 1 else 0.0
    se = np.sqrt((1.0 / J + n_test / n_train) * var)
    if se == 0:
        t, p = (0.0, 1.0) if mean == 0 else (np.inf * np.sign(mean), 0.0)
    else:
        t = mean / se
        p = float(2 * sps.t.sf(abs(t), df=J - 1))
    tcrit = sps.t.ppf(0.975, df=J - 1)
    return dict(mean_diff=mean, se=float(se), t=float(t), p=float(p), df=J - 1,
                ci_low=mean - tcrit * se, ci_high=mean + tcrit * se)


def cv_ttest_table(cv_long: pd.DataFrame, metric: str, n_train_total: int,
                   models: list[str] | None = None) -> pd.DataFrame:
    """Pairwise corrected t-tests on CV fold scores (folds are identical across models)."""
    models = models or MODEL_ORDER
    d = cv_long[cv_long["metric"] == metric].pivot(index="fold", columns="Model", values="value")
    n_test = n_train_total / CV_SPLITS
    n_train = n_train_total - n_test
    rows = []
    for m1, m2 in combinations([m for m in models if m in d.columns], 2):
        r = corrected_resampled_ttest(d[m1] - d[m2], n_train, n_test)
        rows.append({"Model A": m1, "Model B": m2, "mean A": d[m1].mean(), "mean B": d[m2].mean(), **r})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(df["p"]).values
    return df


# ======================================================================== threshold tuning
# One protocol shared with mediscan/ensembles.py and the notebook:
#   * scores are malignant PROBABILITIES for every model (SVMs via Platt calibration),
#   * thresholds come from the grid 0.05, 0.10, ..., 0.95, rule "malignant if P > t",
#   * the threshold is chosen on 5-fold out-of-fold predictions for the 600 TRAINING patients,
#   * the 200 test patients are used once, afterwards, to report the effect.
from .ensembles import THRESHOLDS  # noqa: E402


def default_threshold(score_source: str = "probability") -> float:
    return 0.5


def predict_at(scores: np.ndarray, t: float) -> np.ndarray:
    return (np.asarray(scores) > t).astype(int)


def threshold_curve(y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray = THRESHOLDS) -> pd.DataFrame:
    """Vectorised metrics for every threshold (rule: malignant if score > threshold)."""
    y = np.asarray(y).astype(bool)
    t = np.asarray(thresholds, float)
    P = np.asarray(scores, float)[None, :] > t[:, None]          # thresholds × patients
    tp = (P & y).sum(1); fp = (P & ~y).sum(1)
    fn = y.sum() - tp; tn = (~y).sum() - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = tp / (tp + fn)
        spec = tn / (tn + fp)
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan)
    return pd.DataFrame({"threshold": t, "recall": recall, "specificity": spec, "precision": prec,
                         "FN": fn, "FP": fp, "accuracy": (tp + tn) / len(y)})


def choose_threshold(curve: pd.DataFrame, rule: str, target_recall: float = 0.99,
                     fn_fp_ratio: float = 10.0) -> tuple[float, str]:
    """Pick a threshold from the OOF (training) curve only.

    rule="target_recall": the HIGHEST threshold whose OOF recall ≥ target (fewest false alarms
                          among thresholds that meet the sensitivity target).
    rule="min_cost":      the threshold minimising FN·ratio + FP on OOF predictions
                          (ties → higher threshold, i.e. fewer false alarms).
    """
    c = curve.copy()
    if rule == "target_recall":
        ok = c[c["recall"] >= target_recall - 1e-12]
        if ok.empty:
            return float(c["threshold"].min()), f"No grid threshold reaches {target_recall:.0%} OOF recall; using the lowest."
        return float(ok["threshold"].max()), f"Highest threshold with OOF recall ≥ {target_recall:.0%}."
    c["cost"] = c["FN"] * fn_fp_ratio + c["FP"]
    best = c[np.isclose(c["cost"], c["cost"].min())]
    return float(best["threshold"].max()), f"Minimises OOF penalty FN×{fn_fp_ratio:g} + FP."


def score_source(name: str) -> str:
    return "calibrated probability (Platt)" if "SVM" in name else "probability"


def tune_model(name: str, A, rule: str, target_recall: float = 0.99, fn_fp_ratio: float = 10.0) -> dict:
    """Choose the threshold on training OOF probabilities, then evaluate default vs tuned on the test set."""
    oof, test = A.ens["oof"][name], A.ens["test"][name]
    ytr, yte = A.split.y_train.to_numpy(), A.split.y_test.to_numpy()
    curve = threshold_curve(ytr, oof)
    t, why = choose_threshold(curve, rule, target_recall, fn_fp_ratio)
    return dict(model=name, source=score_source(name), default=0.5, threshold=t, why=why, curve=curve,
                oof_default=compute_metrics(ytr, predict_at(oof, 0.5)),
                oof_tuned=compute_metrics(ytr, predict_at(oof, t)),
                test_default=compute_metrics(yte, predict_at(test, 0.5)),
                test_tuned=compute_metrics(yte, predict_at(test, t)))


def tune_all(A, rule: str, target_recall: float = 0.99, fn_fp_ratio: float = 10.0) -> pd.DataFrame:
    rows = []
    for name in MODEL_ORDER:
        if not A.results[name].ok:
            continue
        t = tune_model(name, A, rule, target_recall, fn_fp_ratio)
        rows.append({
            "Model": name, "Score": t["source"], "Default t": t["default"], "Tuned t": t["threshold"],
            "OOF recall (tuned)": t["oof_tuned"]["recall"], "OOF specificity (tuned)": t["oof_tuned"]["specificity"],
            "Test FN default": t["test_default"]["FN"], "Test FN tuned": t["test_tuned"]["FN"],
            "Test FP default": t["test_default"]["FP"], "Test FP tuned": t["test_tuned"]["FP"],
            "Test recall tuned": t["test_tuned"]["recall"], "Test precision tuned": t["test_tuned"]["precision"],
            "Test accuracy tuned": t["test_tuned"]["accuracy"],
        })
    return pd.DataFrame(rows)
