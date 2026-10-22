"""Live quality-assurance checks (Section 18 of the brief).

Run inside the dashboard (Methodology page) AND by tests/test_pipeline.py, so the
same assertions that the professor sees are the ones the test suite enforces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

from .config import FEATURES, ID_COL, MODEL_ORDER, SPLIT_RANDOM_STATE, TARGET, TEST_SIZE
from .pipeline import Analysis, ranking


def run_checks(A: Analysis) -> pd.DataFrame:
    s = A.split
    rows = []

    def add(group, name, ok, detail=""):
        rows.append({"Group": group, "Check": name, "Result": "PASS" if bool(ok) else "FAIL", "Detail": detail})

    # ---------------- data & modelling
    add("Data & modelling", "CSV loaded", len(A.df) > 0, f"{len(A.df)} rows")
    add("Data & modelling", "All 8 predictors in baseline", list(s.X_train.columns) == FEATURES, ", ".join(s.X_train.columns))
    add("Data & modelling", "patient_id excluded", ID_COL not in s.X_train.columns)
    add("Data & modelling", "diagnosis is sole target", TARGET not in s.X_train.columns and s.y_train.name == TARGET)
    add("Data & modelling", "Target mapping 0/1", set(np.unique(A.df[TARGET])) == {0, 1}, "0 = Benign, 1 = Malignant")
    n = len(A.df)
    exp_test = int(np.ceil(n * TEST_SIZE))
    add("Data & modelling", "25% stratified test split", len(s.X_test) == exp_test and
        abs(s.y_test.mean() - A.df[TARGET].mean()) < 0.01,
        f"test n = {len(s.X_test)}, malignant share train {s.y_train.mean():.3f} / test {s.y_test.mean():.3f}, random_state={SPLIT_RANDOM_STATE}")
    add("Data & modelling", "Train/test disjoint", len(set(s.X_train.index) & set(s.X_test.index)) == 0)
    add("Data & modelling", "Scaler fitted on train only",
        np.allclose(s.scaler.mean_, s.X_train.mean().to_numpy()) and not np.allclose(s.scaler.mean_, A.df[FEATURES].mean().to_numpy()),
        "scaler mean = training mean ≠ full-data mean")
    add("Data & modelling", "Scaled train has mean≈0, sd≈1",
        np.allclose(s.X_train_s.mean(0), 0, atol=1e-9) and np.allclose(s.X_train_s.std(0), 1, atol=1e-9))
    ok_models = [m for m in MODEL_ORDER if A.results[m].ok]
    add("Data & modelling", "All six required models trained", len(ok_models) == 6,
        f"{len(ok_models)}/6 succeeded" + ("" if len(ok_models) == 6 else
                                            f" — failed: {[m for m in MODEL_ORDER if not A.results[m].ok]}"))
    add("Data & modelling", "Every model scored on same test rows",
        all(len(A.results[m].y_pred) == len(s.y_test) for m in ok_models))
    add("Data & modelling", "Unscaled RBF experiment separate",
        "SVM (RBF) — unscaled" not in set(A.lb["Model"]) and A.scaling["unscaled"].ok)
    add("Data & modelling", "PCA fitted on scaled train",
        A.pca["train"].shape == (len(s.X_train), 2) and A.pca["test"].shape == (len(s.X_test), 2))
    add("Data & modelling", "2-feature demo kept out of leaderboard", len(A.lb) == 6)

    # ---------------- evaluation
    for m in ok_models:
        r = A.results[m]
        y, p = s.y_test, r.y_pred
        me = r.metrics
        cm = confusion_matrix(y, p, labels=[0, 1])
        agree = (
            np.isclose(me["accuracy"], accuracy_score(y, p))
            and np.isclose(me["recall"], recall_score(y, p, pos_label=1))
            and np.isclose(me["precision"], precision_score(y, p, pos_label=1, zero_division=np.nan))
            and np.isclose(me["specificity"], recall_score(y, p, pos_label=0))
            and cm.sum() == len(y)
            and (cm.ravel() == [me["TN"], me["FP"], me["FN"], me["TP"]]).all()
            and np.isclose(me["recall"], me["TP"] / (me["TP"] + me["FN"]))
        )
        add("Evaluation", f"{m}: metrics match sklearn & matrix", agree,
            f"TN {me['TN']} · FP {me['FP']} · FN {me['FN']} · TP {me['TP']} (sum {cm.sum()})")
    for metric, col in [("recall", "FN"), ("precision", "FP"), ("accuracy", "errors")]:
        rk = ranking(A.lb, metric, col, col)
        mono = (rk[metric].round(3).diff().dropna() <= 1e-12).all()
        ties_ok = (rk.groupby(rk[metric].round(3))["Rank"].nunique() == 1).all()
        add("Evaluation", f"{metric.title()} ranking ordered & tie-aware", mono and ties_ok,
            " > ".join(f"{r.Model} ({r.Rank})" for r in rk.itertuples()))

    # ---------------- reproducibility
    from .pipeline import build_models, fit_and_evaluate
    rerun = fit_and_evaluate("RF", build_models()["Random Forest"], s.X_train_s, s.y_train, s.X_test_s, s.y_test)
    add("Application", "Deterministic rerun (Random Forest)",
        rerun.ok and (rerun.y_pred == A.results["Random Forest"].y_pred).all(), "identical predictions on refit")

    # ---------------- significance tests & thresholds
    from sklearn.model_selection import RepeatedStratifiedKFold

    from .pipeline import CV_REPEATS, CV_SEED, CV_SPLITS
    from .stats import mcnemar_exact

    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=CV_SEED)
    f1 = [te.tolist() for _, te in cv.split(s.X_train, s.y_train)]
    f2 = [te.tolist() for _, te in cv.split(s.X_train, s.y_train)]
    add("Significance & thresholds", "CV folds identical across models (paired t-test valid)", f1 == f2,
        f"{len(f1)} folds regenerated identically from random_state={CV_SEED}")
    a = A.results["Random Forest"].y_pred == s.y_test.to_numpy()
    b = A.results["Naive Bayes"].y_pred == s.y_test.to_numpy()
    m1, m2 = mcnemar_exact(a, b), mcnemar_exact(b, a)
    add("Significance & thresholds", "McNemar symmetric (A,B) = (B,A)",
        m1["p"] == m2["p"] and (m1["b"], m1["c"]) == (m2["c"], m2["b"]),
        f"RF vs NB: b={m1['b']}, c={m1['c']}, p={m1['p']:.3f}")
    non_svm = [m for m in ok_models if "SVM" not in m]
    repro = all(((A.ens["test"][m] > 0.5).astype(int) == A.results[m].y_pred).all() for m in non_svm)
    add("Significance & thresholds", "Probability > 0.5 reproduces predict()", repro,
        "LR, NB, DT, RF; SVMs use Platt-calibrated probabilities for thresholds and voting")
    add("Significance & thresholds", "Thresholds chosen on training OOF only",
        all(len(A.ens["oof"][m]) == len(s.y_train) for m in ok_models),
        f"{len(s.y_train)} out-of-fold training probabilities per model; test set used only to report")

    # ---------------- model combinations
    E = A.ens
    vc = E["vote_check"]
    add("Model combinations", "Soft vote = averaged member probabilities", vc["max_abs_diff"] < 1e-9,
        f"real VotingClassifier vs average for {' + '.join(vc['members'])}: max |diff| = {vc['max_abs_diff']:.1e}")
    add("Model combinations", "All 50 combinations + 6 singles + stacking scored",
        len(E["candidates"]) == 57, f"{len(E['candidates'])} candidates in stage 1")
    w = E["winner"]
    add("Model combinations", "Winner chosen on fresh folds, not stage-1 folds",
        w["Candidate"] == E["stage2"].sort_values(["Fresh cost (mean)", "Size"]).iloc[0]["Candidate"],
        f"{w['Candidate']} @ {w['Threshold']:.2f}: fresh-fold cost {w['Fresh cost (mean)']:.1f}")
    add("Model combinations", "Random Forest kept as reference in stage 2",
        "Random Forest" in set(E["stage2"]["Candidate"]))

    # ---------------- calibration (independent recomputation of the stored Brier score)
    cal = E.get("calibration", {})
    if "Random Forest" in cal:
        yv = np.asarray(E["y_train"], float)
        p_rf = np.asarray(E["oof"]["Random Forest"], float)
        brier = float(np.mean((p_rf - yv) ** 2))
        rel = cal["Random Forest"]["oof"]["reliability"]
        add("Calibration", "Brier & reliability bins recomputed independently",
            np.isclose(brier, cal["Random Forest"]["oof"]["brier"]) and int(rel["n"].sum()) == len(yv),
            f"Random Forest out-of-fold Brier {brier:.4f}; bins cover {int(rel['n'].sum())} of {len(yv)} patients; "
            f"{len(cal)} configurations checked")
    else:
        add("Calibration", "Brier & reliability bins recomputed independently", False, "calibration missing")
    return pd.DataFrame(rows)
