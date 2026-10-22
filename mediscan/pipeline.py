"""One consistent, leakage-safe ML pipeline for every MediScan experiment.

Protocol (from 02_Student_Guide.md):
  * X = 8 predictors, y = diagnosis (1 = malignant = positive class)
  * train_test_split(test_size=0.25, random_state=1, stratify=y)
  * StandardScaler fitted on the TRAINING rows only
  * all six baseline models trained on scaled train, evaluated on the SAME scaled test rows

Everything else in this file (unscaled RBF, 2-feature demo, PCA, CV, feature
engineering, MLP) is an *additional experiment* and is returned separately so
it can never leak into the baseline leaderboard.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    make_scorer,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import FEATURES, MODEL_ORDER, POS_LABEL, SPLIT_RANDOM_STATE, TEST_SIZE

CV_SPLITS, CV_REPEATS, CV_SEED = 5, 3, 42


# ------------------------------------------------------------------ split
@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    X_train_s: np.ndarray
    X_test_s: np.ndarray


def make_split(X: pd.DataFrame, y: pd.Series) -> Split:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)  # TRAIN ONLY — no test leakage
    return Split(X_tr, X_te, y_tr, y_te, scaler, scaler.transform(X_tr), scaler.transform(X_te))


# ------------------------------------------------------------------ models
def build_models() -> dict[str, object]:
    """Fresh, unfitted instances of the six required baseline configurations (exact spec)."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Naive Bayes": GaussianNB(),
        "SVM (Linear)": SVC(kernel="linear"),
        "SVM (RBF)": SVC(kernel="rbf", C=3),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=0),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=0),
    }


def continuous_score(model, X) -> tuple[np.ndarray | None, str]:
    """Continuous malignancy score for ROC/PR analysis (never hard labels)."""
    if hasattr(model, "predict_proba"):
        try:
            classes = list(model.classes_)
            return model.predict_proba(X)[:, classes.index(POS_LABEL)], "predict_proba"
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        return model.decision_function(X), "decision_function"
    return None, "none"


# ------------------------------------------------------------------ metrics
def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def compute_metrics(y_true, y_pred, score=None) -> dict:
    """All class-specific metrics treat MALIGNANT (1) as the positive class."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    n = tn + fp + fn + tp
    precision = _safe_div(tp, tp + fp)  # NaN if model never predicts malignant
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else float("nan")
    )
    m = dict(
        TN=tn, FP=fp, FN=fn, TP=tp, n=n,
        accuracy=_safe_div(tp + tn, n),
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        balanced_accuracy=(recall + specificity) / 2,
        errors=fp + fn,
        cm=cm,
        roc_auc=float("nan"),
        pr_auc=float("nan"),
    )
    if score is not None and len(np.unique(y_true)) == 2:
        m["roc_auc"] = float(roc_auc_score(y_true, score))
        m["pr_auc"] = float(average_precision_score(y_true, score, pos_label=POS_LABEL))
    return m


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n (used to show test-set uncertainty)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class ModelResult:
    name: str
    ok: bool
    model: object | None = None
    y_pred: np.ndarray | None = None
    score: np.ndarray | None = None
    score_source: str = "none"
    metrics: dict = field(default_factory=dict)
    error: str | None = None


def fit_and_evaluate(name, model, X_tr, y_tr, X_te, y_te) -> ModelResult:
    """Fit one model and evaluate it. Failures are captured and reported, never zero-filled."""
    try:
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        score, src = continuous_score(model, X_te)
        return ModelResult(name, True, model, pred, score, src, compute_metrics(y_te, pred, score))
    except Exception as exc:  # surfaced in the UI, never hidden
        return ModelResult(name, False, error=f"{type(exc).__name__}: {exc}")


def evaluate_baseline(split: Split) -> dict[str, ModelResult]:
    out = {}
    for name, model in build_models().items():
        out[name] = fit_and_evaluate(name, model, split.X_train_s, split.y_train, split.X_test_s, split.y_test)
    return out


# ------------------------------------------------------------------ leaderboard & rankings
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy", "roc_auc", "pr_auc"]


def leaderboard(results: dict[str, ModelResult]) -> pd.DataFrame:
    rows = []
    for name in MODEL_ORDER:
        r = results[name]
        if not r.ok:
            rows.append({"Model": name, "Status": "FAILED", "Error": r.error})
            continue
        m = r.metrics
        rlo, rhi = wilson_ci(m["TP"], m["TP"] + m["FN"])
        rows.append(
            {
                "Model": name, "Status": "OK",
                **{k: m[k] for k in METRIC_COLS},
                "FN": m["FN"], "FP": m["FP"], "TP": m["TP"], "TN": m["TN"],
                "errors": m["errors"], "n": m["n"],
                "recall_ci_low": rlo, "recall_ci_high": rhi,
                "score_source": r.score_source,
            }
        )
    return pd.DataFrame(rows)


def tie_aware_rank(values: pd.Series, decimals: int = 3, ascending: bool = False) -> pd.Series:
    """Competition ranking (1, 1, 3) on values rounded to the DISPLAYED precision.

    decimals=3 on a proportion == one decimal place in percent, which is what the
    dashboard displays, so two models that look identical share a rank.
    """
    return values.round(decimals).rank(method="min", ascending=ascending).astype(int)


def ranking(lb: pd.DataFrame, metric: str, count_col: str, count_label: str) -> pd.DataFrame:
    ok = lb[lb["Status"] == "OK"].copy()
    ok["Rank"] = tie_aware_rank(ok[metric])
    ok = ok.sort_values(["Rank", count_col, "Model"])
    ok["Tied"] = ok["Rank"].duplicated(keep=False)
    return ok[["Rank", "Model", metric, count_col, "Tied"]].rename(columns={count_col: count_label})


def leaders(lb: pd.DataFrame, metric: str, higher_is_better: bool = True, decimals: int = 3) -> list[str]:
    """All models tied for best on a metric (at displayed precision)."""
    ok = lb[lb["Status"] == "OK"]
    vals = ok[metric].round(decimals)
    best = vals.max() if higher_is_better else vals.min()
    return ok.loc[vals == best, "Model"].tolist()


# ------------------------------------------------------------------ curves
def curves(results: dict[str, ModelResult], y_test) -> dict:
    out = {}
    for name, r in results.items():
        if r.ok and r.score is not None:
            fpr, tpr, _ = roc_curve(y_test, r.score, pos_label=POS_LABEL)
            prec, rec, _ = precision_recall_curve(y_test, r.score, pos_label=POS_LABEL)
            out[name] = dict(fpr=fpr, tpr=tpr, precision=prec, recall=rec)
    return out


# ------------------------------------------------------------------ scaling experiment
def scaling_experiment(split: Split, results: dict) -> dict:
    """RBF-SVM (C=3) on RAW features vs the baseline scaled RBF-SVM, plus the same test for all six models."""
    scaled_result = results["SVM (RBF)"]
    raw = fit_and_evaluate(
        "SVM (RBF) — unscaled", SVC(kernel="rbf", C=3),
        split.X_train, split.y_train, split.X_test, split.y_test,
    )
    # Diagnose *why*: share of total raw variance carried by each feature.
    var = split.X_train.var()
    share = (var / var.sum()).sort_values(ascending=False)
    # Average squared-distance contribution between random pairs of training patients
    rng = np.random.default_rng(0)
    A = split.X_train.to_numpy()
    i, j = rng.integers(0, len(A), 4000), rng.integers(0, len(A), 4000)
    d2 = (A[i] - A[j]) ** 2
    dist_share = pd.Series(d2.mean(0) / d2.mean(0).sum(), index=FEATURES).sort_values(ascending=False)
    As = split.X_train_s
    d2s = (As[i] - As[j]) ** 2
    dist_share_scaled = pd.Series(d2s.mean(0) / d2s.mean(0).sum(), index=FEATURES)
    # Every required model, raw vs scaled (shows WHICH models are scale-sensitive).
    all_rows = []
    for name, mdl in build_models().items():
        r_raw = fit_and_evaluate(name, mdl, split.X_train, split.y_train, split.X_test, split.y_test)
        r_sc = results[name]
        if r_raw.ok and r_sc.ok:
            all_rows.append({"Model": name, "Unscaled accuracy": r_raw.metrics["accuracy"],
                             "Scaled accuracy": r_sc.metrics["accuracy"],
                             "Unscaled recall": r_raw.metrics["recall"], "Scaled recall": r_sc.metrics["recall"],
                             "Unscaled FN": r_raw.metrics["FN"], "Scaled FN": r_sc.metrics["FN"]})
    all_models = pd.DataFrame(all_rows)
    all_models["Accuracy change"] = all_models["Scaled accuracy"] - all_models["Unscaled accuracy"]
    return dict(unscaled=raw, scaled=scaled_result, variance_share=share, all_models=all_models,
                distance_share_raw=dist_share, distance_share_scaled=dist_share_scaled)


# ------------------------------------------------------------------ PCA (visualisation only)
def pca_projection(split: Split) -> dict:
    pca = PCA(n_components=2).fit(split.X_train_s)  # fitted on scaled TRAIN only
    full = PCA().fit(split.X_train_s)
    tr = pca.transform(split.X_train_s)
    te = pca.transform(split.X_test_s)
    loadings = pd.DataFrame(pca.components_.T, index=FEATURES, columns=["PC1", "PC2"])
    # How separable is the 2-D projection? (diagnostic only — NOT the leaderboard)
    sep = {}
    for nm, mdl in {"SVM (Linear)": SVC(kernel="linear"), "SVM (RBF)": SVC(kernel="rbf", C=3)}.items():
        mdl.fit(tr, split.y_train)
        sep[nm] = float((mdl.predict(te) == split.y_test.to_numpy()).mean())
    return dict(
        train=tr, test=te, evr=pca.explained_variance_ratio_, evr_all=full.explained_variance_ratio_,
        loadings=loadings, separability=sep,
    )


# ------------------------------------------------------------------ 2-feature boundary demo
TWO_FEATURES = ["cell_size", "cell_shape"]


def two_feature_demo(split: Split, grid_n: int = 220) -> dict:
    """Diagnostic boundary demo on (cell_size, cell_shape) ONLY — separate from the leaderboard.

    All six model types are refit on just these two (training-scaled) features so the
    *shape* of each boundary can be seen directly.
    """
    sc2 = StandardScaler().fit(split.X_train[TWO_FEATURES])
    a = sc2.transform(split.X_train[TWO_FEATURES])
    b = sc2.transform(split.X_test[TWO_FEATURES])
    lo = np.minimum(split.X_train[TWO_FEATURES].min(), split.X_test[TWO_FEATURES].min()) - 0.6
    hi = np.maximum(split.X_train[TWO_FEATURES].max(), split.X_test[TWO_FEATURES].max()) + 0.6
    gx = np.linspace(lo.iloc[0], hi.iloc[0], grid_n)
    gy = np.linspace(lo.iloc[1], hi.iloc[1], grid_n)
    XX, YY = np.meshgrid(gx, gy)
    grid = sc2.transform(pd.DataFrame(np.c_[XX.ravel(), YY.ravel()], columns=TWO_FEATURES))
    models = {}
    for name, mdl in build_models().items():
        mdl.fit(a, split.y_train)
        pred_te = mdl.predict(b)
        Z = mdl.predict(grid).reshape(XX.shape)
        models[name] = dict(
            Z=Z,
            train_acc=float((mdl.predict(a) == split.y_train.to_numpy()).mean()),
            test_acc=float((pred_te == split.y_test.to_numpy()).mean()),
            test_metrics=compute_metrics(split.y_test, pred_te),
        )
    centre = split.X_train.loc[split.y_train == 0, TWO_FEATURES].mean()
    radius = lambda d: np.hypot(d[TWO_FEATURES[0]] - centre.iloc[0], d[TWO_FEATURES[1]] - centre.iloc[1])
    return dict(gx=gx, gy=gy, models=models, centre=centre,
                radius_train=radius(split.X_train), radius_test=radius(split.X_test))


# ------------------------------------------------------------------ cross-validation robustness
def _specificity(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else np.nan


CV_SCORING = {
    "accuracy": "accuracy",
    "recall": make_scorer(lambda y, p: _safe_div(((y == 1) & (p == 1)).sum(), (y == 1).sum())),
    "precision": make_scorer(lambda y, p: _safe_div(((y == 1) & (p == 1)).sum(), (p == 1).sum())),
    "specificity": make_scorer(_specificity),
}


def cross_validation(split: Split) -> pd.DataFrame:
    """Repeated stratified 5-fold CV on the TRAINING set only; scaler refit inside every fold."""
    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=CV_SEED)
    rows = []
    for name, mdl in build_models().items():
        pipe = Pipeline([("scale", StandardScaler()), ("model", mdl)])
        res = cross_validate(pipe, split.X_train, split.y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1)
        for k in CV_SCORING:
            for fold, v in enumerate(res[f"test_{k}"]):
                rows.append({"Model": name, "metric": k, "fold": fold, "value": float(v)})
    return pd.DataFrame(rows)


def cv_summary(cv_long: pd.DataFrame) -> pd.DataFrame:
    s = cv_long.groupby(["Model", "metric"])["value"].agg(["mean", "std"]).unstack("metric")
    s.columns = [f"{m}_{stat}" for stat, m in s.columns]
    return s.reindex(MODEL_ORDER)


# ------------------------------------------------------------------ feature importance
def feature_importance(split: Split, results: dict[str, ModelResult], scoring: str = "recall") -> pd.DataFrame:
    """Permutation importance on the held-out test set (interpretation only, not model selection)."""
    rows = []
    for name in MODEL_ORDER:
        r = results[name]
        if not r.ok:
            continue
        pi = permutation_importance(
            r.model, split.X_test_s, split.y_test, n_repeats=20, random_state=0, scoring=scoring
        )
        for f, mu, sd in zip(FEATURES, pi.importances_mean, pi.importances_std):
            rows.append({"Model": name, "feature": f, "mean": float(mu), "std": float(sd)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ optional: feature engineering
def _append_cell_squares(Z):
    """Append squared *training-centred, training-scaled* cell_size and cell_shape.

    Because StandardScaler (fitted on training folds only) has already centred these
    columns, z_size² + z_shape² is a leakage-free proxy for 'distance from the centre
    of the cell cloud' — exactly the ring structure a straight line cannot see.
    """
    Z = np.asarray(Z)
    return np.c_[Z, Z[:, 0] ** 2, Z[:, 1] ** 2]


ENGINEERED = {
    "LR + squared cell terms": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("sq", FunctionTransformer(_append_cell_squares)),
        ("model", LogisticRegression(max_iter=2000)),
    ]),
    "Linear SVM + squared cell terms": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("sq", FunctionTransformer(_append_cell_squares)),
        ("model", SVC(kernel="linear")),
    ]),
    "LR + full degree-2 polynomial": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("model", LogisticRegression(max_iter=5000)),
    ]),
}
ENGINEERED_NFEATURES = {
    "LR + squared cell terms": 10,
    "Linear SVM + squared cell terms": 10,
    "LR + full degree-2 polynomial": 44,
}


def feature_engineering_experiment(split: Split) -> pd.DataFrame:
    """CV on training data + one held-out evaluation. Reported separately from the baseline."""
    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=CV_SEED)
    rows = []
    for name, factory in ENGINEERED.items():
        res = cross_validate(factory(), split.X_train, split.y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1)
        held = fit_and_evaluate(name, factory(), split.X_train, split.y_train, split.X_test, split.y_test)
        m = held.metrics if held.ok else {}
        rows.append({
            "Variant": name,
            "Input features": ENGINEERED_NFEATURES[name],
            "CV accuracy": res["test_accuracy"].mean(), "CV accuracy sd": res["test_accuracy"].std(),
            "CV recall": res["test_recall"].mean(), "CV recall sd": res["test_recall"].std(),
            "Test accuracy": m.get("accuracy", np.nan), "Test recall": m.get("recall", np.nan),
            "Test precision": m.get("precision", np.nan),
            "FN": m.get("FN", np.nan), "FP": m.get("FP", np.nan),
            "ok": held.ok, "error": held.error,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ optional: ANN extension
MLP_ACTIVATIONS = ["relu", "tanh", "logistic"]
MLP_HIDDEN = (16, 8)


def mlp_experiment(split: Split) -> dict:
    """Compare hidden-layer activations by CV on TRAINING data; evaluate only the CV-selected one on test."""
    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=CV_SEED)
    rows, conv_warn = [], {}
    for act in MLP_ACTIVATIONS:
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, activation=act, solver="adam",
                                  max_iter=3000, random_state=0)),
        ])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            res = cross_validate(pipe, split.X_train, split.y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1)
        conv_warn[act] = sum(issubclass(x.category, ConvergenceWarning) for x in w)
        rows.append({
            "Hidden activation": act,
            "CV accuracy": res["test_accuracy"].mean(), "CV accuracy sd": res["test_accuracy"].std(),
            "CV recall": res["test_recall"].mean(), "CV recall sd": res["test_recall"].std(),
            "CV precision": res["test_precision"].mean(),
            "Convergence warnings": conv_warn[act],
        })
    cvdf = pd.DataFrame(rows)
    # Select by CV recall (clinical priority), tie-break on CV accuracy — never by test score.
    best = cvdf.sort_values(["CV recall", "CV accuracy"], ascending=False).iloc[0]["Hidden activation"]
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, activation=best, solver="adam",
                              max_iter=3000, random_state=0)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        held = fit_and_evaluate(f"MLP ({best})", pipe, split.X_train, split.y_train, split.X_test, split.y_test)
    return dict(cv=cvdf, selected=best, held_out=held)


# ------------------------------------------------------------------ clinical cost analysis
def cost_table(lb: pd.DataFrame, fn_cost: float, fp_cost: float) -> pd.DataFrame:
    ok = lb[lb["Status"] == "OK"].copy()
    ok["Illustrative cost"] = ok["FN"] * fn_cost + ok["FP"] * fp_cost
    ok["Cost rank"] = ok["Illustrative cost"].rank(method="min").astype(int)
    return ok.sort_values(["Illustrative cost", "FN"])[["Cost rank", "Model", "FN", "FP", "Illustrative cost"]]


def cost_scenarios(lb: pd.DataFrame, ratios=None) -> pd.DataFrame:
    """For each FN:FP penalty ratio, which model(s) minimise FN·ratio + FP·1?"""
    ratios = ratios if ratios is not None else np.round(np.arange(1, 30.01, 0.5), 2)
    ok = lb[lb["Status"] == "OK"]
    rows = []
    for r in ratios:
        cost = ok["FN"] * r + ok["FP"]
        best = ok.loc[cost == cost.min(), "Model"].tolist()
        for _, row in ok.iterrows():
            rows.append({"ratio": float(r), "Model": row["Model"], "cost": float(row["FN"] * r + row["FP"]),
                         "is_best": row["Model"] in best, "best": " & ".join(best)})
    return pd.DataFrame(rows)


def cv_expected_errors(cv: pd.DataFrame, prevalence: float, n: int = 1000) -> pd.DataFrame:
    """Expected FN and FP per `n` patients from CROSS-VALIDATED recall and specificity.

    Uses 600 training patients × 15 folds instead of one 200-patient split, so a
    single-patient difference on the test set cannot decide the result on its own.
    """
    out = pd.DataFrame(index=cv.index)
    out["FN"] = n * prevalence * (1 - cv["recall_mean"])
    out["FP"] = n * (1 - prevalence) * (1 - cv["specificity_mean"])
    return out.rename_axis("Model").reset_index()


def cost_scenarios_from_counts(counts: pd.DataFrame, ratios=None) -> pd.DataFrame:
    """Same as cost_scenarios but from any table with Model/FN/FP columns (e.g. CV expected errors)."""
    ratios = ratios if ratios is not None else np.round(np.arange(1, 30.01, 0.5), 2)
    rows = []
    for r in ratios:
        cost = counts["FN"] * r + counts["FP"]
        best = counts.loc[np.isclose(cost, cost.min()), "Model"].tolist()
        for _, row in counts.iterrows():
            rows.append({"ratio": float(r), "Model": row["Model"], "cost": float(row["FN"] * r + row["FP"]),
                         "is_best": row["Model"] in best, "best": " & ".join(best)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ everything
@dataclass
class Analysis:
    df: pd.DataFrame
    split: Split
    results: dict
    lb: pd.DataFrame
    curves: dict
    scaling: dict
    pca: dict
    two_feat: dict
    cv_long: pd.DataFrame
    cv: pd.DataFrame
    importance_recall: pd.DataFrame
    importance_accuracy: pd.DataFrame
    engineered: pd.DataFrame
    mlp: dict
    ens: dict  # combinations, stacking, cost-tuned thresholds, fresh-fold check (see ensembles.py)


def run_all(df: pd.DataFrame) -> Analysis:
    from .data import split_features_target
    from . import ensembles

    X, y = split_features_target(df)
    split = make_split(X, y)
    results = evaluate_baseline(split)
    lb = leaderboard(results)
    cv_long = cross_validation(split)
    return Analysis(
        df=df, split=split, results=results, lb=lb,
        curves=curves(results, split.y_test),
        scaling=scaling_experiment(split, results),
        pca=pca_projection(split),
        two_feat=two_feature_demo(split),
        cv_long=cv_long, cv=cv_summary(cv_long),
        importance_recall=feature_importance(split, results, "recall"),
        importance_accuracy=feature_importance(split, results, "accuracy"),
        engineered=feature_engineering_experiment(split),
        mlp=mlp_experiment(split),
        ens=ensembles.run(split),
    )
