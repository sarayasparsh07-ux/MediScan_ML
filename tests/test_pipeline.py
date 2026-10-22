"""Pipeline correctness tests — Section 18 checklist plus independent recomputation.

Run:  python -m pytest tests/test_pipeline.py -q
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mediscan.config import MODEL_ORDER
from mediscan.data import load_and_validate
from mediscan.pipeline import build_models, ranking, run_all, tie_aware_rank, compute_metrics
from mediscan.qa import run_checks
from mediscan.recommendation import recommend, professor_paragraph


@pytest.fixture(scope="module")
def A():
    df, rep = load_and_validate()
    assert rep.n_fail == 0
    return run_all(df)


def test_live_qa_checks_all_pass(A):
    qa = run_checks(A)
    failed = qa[qa["Result"] != "PASS"]
    assert failed.empty, failed.to_string()


def test_independent_recompute_matches_student_guide(A):
    """Re-implement Student Guide Part A–C from scratch; every number must match the app."""
    df = A.df
    X = df.drop(columns=["patient_id", "diagnosis"]); y = df["diagnosis"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
    sc = StandardScaler().fit(Xtr)
    for name, model in build_models().items():
        model.fit(sc.transform(Xtr), ytr)
        p = model.predict(sc.transform(Xte))
        m = A.results[name].metrics
        assert np.isclose(m["accuracy"], accuracy_score(yte, p)), name
        assert np.isclose(m["recall"], recall_score(yte, p)), name
        assert (confusion_matrix(yte, p).ravel() == [m["TN"], m["FP"], m["FN"], m["TP"]]).all(), name


@pytest.mark.parametrize("name,acc,rec", [
    # INSTRUCTOR_NOTES.md reference run ("~" values, so tolerance ±0.02)
    ("Logistic Regression", 0.73, 0.69), ("SVM (Linear)", 0.73, 0.70), ("Naive Bayes", 0.96, 0.99),
    ("SVM (RBF)", 0.97, 0.97), ("Decision Tree", 0.97, 0.97), ("Random Forest", 0.97, 0.98),
])
def test_matches_instructor_reference(A, name, acc, rec):
    m = A.results[name].metrics
    assert abs(m["accuracy"] - acc) <= 0.02 and abs(m["recall"] - rec) <= 0.02, (name, m["accuracy"], m["recall"])


def test_scaling_demo_matches_reference(A):
    un = A.scaling["unscaled"].metrics
    assert abs(un["accuracy"] - 0.61) <= 0.02 and abs(un["recall"] - 0.40) <= 0.02


def test_two_feature_demo_reference(A):
    tf = A.two_feat["models"]
    assert abs(tf["SVM (Linear)"]["test_acc"] - 0.54) <= 0.02
    assert tf["SVM (RBF)"]["test_acc"] >= 0.97


def test_tie_aware_ranking():
    s = pd.Series([0.975, 0.975, 0.97, 0.9601, 0.96])
    assert tie_aware_rank(s).tolist() == [1, 1, 3, 4, 4]


def test_metrics_safe_when_no_positive_predictions():
    m = compute_metrics(np.array([0, 1, 1, 0]), np.array([0, 0, 0, 0]))
    assert np.isnan(m["precision"]) and m["recall"] == 0 and np.isnan(m["f1"]) and m["FN"] == 2


def test_rankings_sorted(A):
    for metric, col in [("recall", "FN"), ("precision", "FP"), ("accuracy", "errors")]:
        r = ranking(A.lb, metric, col, col)
        assert r["Rank"].is_monotonic_increasing
        assert (r[metric].round(3).diff().dropna() <= 0).all()


def test_recommendation_is_data_driven(A):
    rec0 = recommend(A.lb, A.cv, fn_tolerance=0)
    min_fn = A.lb["FN"].min()
    assert int(A.lb.set_index("Model").loc[rec0.recommended, "FN"]) == min_fn
    rec1 = recommend(A.lb, A.cv, fn_tolerance=1)
    assert rec1.recommended in rec1.gate_survivors
    surv = A.lb.set_index("Model").loc[rec1.gate_survivors]
    assert surv.loc[rec1.recommended, "FP"] == surv["FP"].min()
    para = professor_paragraph(rec1, A.lb, A.cv, A.scaling)
    assert rec1.recommended in para and "not establish clinical validity" in para


def test_experiments_never_enter_leaderboard(A):
    assert list(A.lb["Model"]) == MODEL_ORDER
    assert len(A.engineered) == 3 and A.mlp["selected"] in {"relu", "tanh", "logistic"}


# ---------------------------------------------------------------- significance & thresholds
from mediscan import stats as S


def test_mcnemar_exact_known_values():
    # b=1, c=4 -> two-sided exact p = 2*P(X<=1 | n=5, 0.5) = 2*(6/32) = 0.375
    a = np.array([1, 0, 0, 0, 0, 1, 1], bool)
    b = np.array([0, 1, 1, 1, 1, 1, 1], bool)
    r = S.mcnemar_exact(a, b)
    assert (r["b"], r["c"]) == (1, 4) and np.isclose(r["p"], 0.375)
    assert S.mcnemar_exact(a, a)["p"] == 1.0  # identical predictions -> no evidence


def test_holm_adjustment():
    p = pd.Series([0.01, 0.04, 0.03], index=list("abc"))
    adj = S.holm(p)
    assert np.allclose(adj[["a", "c", "b"]].values, [0.03, 0.06, 0.06])


def test_corrected_ttest_is_more_conservative_than_naive():
    rng = np.random.default_rng(0)
    d = rng.normal(0.01, 0.02, 15)
    corr = S.corrected_resampled_ttest(d, 480, 120)
    from scipy import stats as sps
    naive_p = sps.ttest_1samp(d, 0).pvalue
    assert corr["p"] >= naive_p and corr["df"] == 14


def test_threshold_rules_respect_oof_target(A):
    for target in (0.97, 0.99, 1.0):
        t = S.tune_model("Random Forest", A, "target_recall", target_recall=target)
        assert t["oof_tuned"]["recall"] >= target - 1e-12
    t = S.tune_model("Random Forest", A, "min_cost", fn_fp_ratio=10)
    c = t["curve"].assign(cost=lambda d: d["FN"] * 10 + d["FP"])
    chosen = c.loc[np.isclose(c["threshold"], t["threshold"]), "cost"].iloc[0]
    assert np.isclose(chosen, c["cost"].min())


def test_top_models_not_significantly_different_on_test(A):
    tb = S.mcnemar_table(A.results, A.split.y_test.to_numpy(), "all",
                         ["Naive Bayes", "SVM (RBF)", "Decision Tree", "Random Forest"])
    assert (tb["p"] > 0.05).all()
    lin = S.mcnemar_table(A.results, A.split.y_test.to_numpy(), "all", ["Logistic Regression", "Random Forest"])
    assert lin["p"].iloc[0] < 0.001  # linear vs curved is highly significant


# ---------------------------------------------------------------- model combinations (ensembles.py)
from mediscan import ensembles as ENS
from mediscan.recommendation import cautious_choice, combination_sentences


def test_combination_search_is_complete_and_exact(A):
    E = A.ens
    assert len(E["candidates"]) == 57                                   # 6 singles + 50 votes + 1 stack
    assert (E["candidates"]["Type"] == "Soft vote").sum() == 50
    assert E["vote_check"]["max_abs_diff"] < 1e-9                       # averaging == VotingClassifier


def test_two_stage_selection_rules(A):
    E = A.ens
    s2 = E["stage2"]
    assert "Random Forest" in set(s2["Candidate"])                      # reference always re-checked
    assert E["winner"]["Candidate"] == s2.sort_values(["Fresh cost (mean)", "Size"]).iloc[0]["Candidate"]
    assert all(t in ENS.THRESHOLDS for t in s2["Threshold"])
    # stage-1 thresholds really are the lowest-cost grid point on the OOF predictions
    for _, r in E["candidates"].iterrows():
        p = E["oof"][r["Candidate"]] if r["Type"] == "Stacking" else np.mean([E["oof"][m] for m in r["members"]], 0)
        t, fn, fp = ENS.best_threshold(E["y_train"], p)
        assert np.isclose(t, r["Tuned threshold"]) and ENS.cost(fn, fp) == r["CV cost tuned"]


def test_thresholds_consistent_between_pages(A):
    rf_combo = A.ens["candidates"].set_index("Candidate").loc["Random Forest", "Tuned threshold"]
    assert np.isclose(S.tune_model("Random Forest", A, "min_cost", fn_fp_ratio=10)["threshold"], rf_combo)


def test_recommendation_text_matches_numbers(A):
    txt = combination_sentences(A.ens)
    w = A.ens["winner"]
    assert w["Candidate"] in txt and f"{w['Threshold']:.2f}" in txt
    c = cautious_choice(A.ens)
    assert c is None or c["Fresh missed (mean)"] <= w["Fresh missed (mean)"]


def test_scaling_all_models_only_rbf_changes(A):
    am = A.scaling["all_models"].set_index("Model")
    changed = am.index[am["Accuracy change"].abs() > 1e-9].tolist()
    assert changed == ["SVM (RBF)"]
