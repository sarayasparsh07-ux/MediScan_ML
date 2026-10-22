"""Model combinations (soft voting, stacking) and cost-tuned decision thresholds.

Shared by the dashboard AND MediScan_Full_Project.ipynb so both report identical numbers.

Protocol (stated before any result is read)
-------------------------------------------
* Every model is a Pipeline(StandardScaler -> model): scaling is refit inside every CV fold.
* SVMs have no native probability, so they are wrapped in CalibratedClassifierCV (Platt
  scaling, internal 5-fold CV) wherever a probability is needed (voting, thresholds).
* Soft voting = the plain average of the members' malignant probabilities. Because every
  member is deterministic, the out-of-fold (OOF) probability of a vote equals the average
  of the members' OOF probabilities on the same folds — verified exactly in the notebook.
  That lets all 50 combinations of 2–4 models be scored without refitting.
* Decision rule: malignant if P(malignant) > t. At t = 0.5 this reproduces predict().
* Threshold grid: 0.05, 0.10, …, 0.95. Clinical cost = 10 × missed cancers + 1 × false alarm
  (an illustrative assumption). Ties go to the higher threshold (fewer false alarms).

Selection (two stages, so the winner is not graded on the data that chose it)
  Stage 1  5-fold OOF on the 600 training patients (seed 42): tune each of the 56
           candidates (6 single models + 50 combinations); shortlist the 8 lowest costs,
           plus Random Forest (best single model) as the reference.
  Stage 2  Fresh repeated CV (10 repeats × 5 folds, seed 2026) with the stage-1 thresholds
           held fixed. Recommend the lowest mean fresh-fold cost.
  Test     The 200 held-out patients are used once, to report — never to choose.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as sps
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import LINEAR_MODELS, MODEL_ORDER
from .pipeline import build_models

COST_FN, COST_FP = 10, 1
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
OOF_SEED, FRESH_SEED, FRESH_REPEATS, N_SPLITS = 42, 2026, 10, 5
SHORTLIST = 8
SHORT = {"Logistic Regression": "LR", "Naive Bayes": "NB", "SVM (Linear)": "SVM-lin",
         "SVM (RBF)": "SVM-RBF", "Decision Tree": "DT", "Random Forest": "RF"}
STACK_MEMBERS = ("Naive Bayes", "SVM (RBF)", "Random Forest")
STACK_NAME = "Stack: NB + SVM-RBF + RF → LR"


# ------------------------------------------------------------------ building blocks
def prob_model(name: str):
    """A probability-producing version of a required model (SVMs get Platt calibration)."""
    m = build_models()[name]
    return CalibratedClassifierCV(m, cv=5, ensemble=False) if isinstance(m, SVC) else m


def member_pipeline(name: str) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", prob_model(name))])


def stacking_pipeline() -> Pipeline:
    stack = StackingClassifier([(SHORT[n], prob_model(n)) for n in STACK_MEMBERS],
                               final_estimator=LogisticRegression(max_iter=2000), cv=5,
                               stack_method="predict_proba")
    return Pipeline([("scale", StandardScaler()), ("model", stack)])


def label(combo: tuple[str, ...]) -> str:
    return combo[0] if len(combo) == 1 else " + ".join(SHORT[m] for m in combo)


def counts(y: np.ndarray, p: np.ndarray, t: float) -> tuple[int, int, int, int]:
    pred = np.asarray(p) > t
    y = np.asarray(y).astype(bool)
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
    return tp, fp, int(y.sum()) - tp, int((~y).sum()) - fp   # TP, FP, FN, TN


def cost(fn, fp) -> float:
    return COST_FN * fn + COST_FP * fp


def best_threshold(y, p) -> tuple[float, int, int]:
    """Lowest-cost threshold on the grid; ties -> higher threshold."""
    best = None
    for t in THRESHOLDS:
        _, fp, fn, _ = counts(y, p, t)
        key = (cost(fn, fp), -t)
        if best is None or key < best[0]:
            best = (key, float(t), fn, fp)
    return best[1], best[2], best[3]


def threshold_curve(y, p) -> pd.DataFrame:
    rows = []
    for t in THRESHOLDS:
        tp, fp, fn, tn = counts(y, p, t)
        rows.append({"threshold": float(t), "FN": fn, "FP": fp, "cost": cost(fn, fp),
                     "recall": tp / (tp + fn), "specificity": tn / (tn + fp),
                     "precision": tp / (tp + fp) if tp + fp else np.nan})
    return pd.DataFrame(rows)


def target_recall_threshold(y, p, target: float) -> float:
    """Highest grid threshold whose recall >= target (fewest false alarms meeting the target)."""
    c = threshold_curve(y, p)
    ok = c[c["recall"] >= target - 1e-12]
    return float(ok["threshold"].max()) if len(ok) else float(c["threshold"].min())


def corrected_ttest(d, n_train: float, n_test: float) -> dict:
    """Nadeau & Bengio (2003) corrected resampled t-test on paired per-fold differences."""
    d = np.asarray(d, float); J = len(d); mean = float(d.mean()); var = float(d.var(ddof=1))
    se = float(np.sqrt((1 / J + n_test / n_train) * var))
    t = mean / se if se > 0 else 0.0
    p = float(2 * sps.t.sf(abs(t), J - 1)) if se > 0 else 1.0
    tc = sps.t.ppf(0.975, J - 1)
    return dict(mean_diff=mean, se=se, t=t, p=p, ci_low=mean - tc * se, ci_high=mean + tc * se, J=J)


def mcnemar_exact(correct_a, correct_b) -> dict:
    a, b_ = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b, c = int((a & ~b_).sum()), int((~a & b_).sum()); n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sps.binom.cdf(min(b, c), n, 0.5))
    return dict(b=b, c=c, n_discordant=n, p=float(p), p_min=1.0 if n == 0 else min(1.0, 2 * 0.5 ** n))


# ------------------------------------------------------------------ the analysis
def _fit_stack_fold(Xtr, ytr, tr, te):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipe = stacking_pipeline().fit(Xtr.iloc[tr], ytr.iloc[tr])
    return te, pipe.predict_proba(Xtr.iloc[te])[:, 1]


def _fit_fold(name, Xtr, ytr, tr, te):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipe = member_pipeline(name).fit(Xtr.iloc[tr], ytr.iloc[tr])
    return name, te, pipe.predict_proba(Xtr.iloc[te])[:, 1]


def run(split) -> dict:
    Xtr, ytr, Xte, yte = split.X_train, split.y_train, split.X_test, split.y_test
    yv, yt = ytr.to_numpy(), yte.to_numpy()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=OOF_SEED)

    # 1) member probabilities: OOF on training (stage-1 folds) and fitted-on-train test scores
    oof, test = {}, {}
    for n in MODEL_ORDER:
        oof[n] = cross_val_predict(member_pipeline(n), Xtr, ytr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        test[n] = member_pipeline(n).fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    # 2) stacking (its own CV — cannot be averaged from members)
    oof[STACK_NAME] = cross_val_predict(stacking_pipeline(), Xtr, ytr, cv=cv, method="predict_proba",
                                        n_jobs=-1)[:, 1]
    test[STACK_NAME] = stacking_pipeline().fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    # 3) stage 1 — every candidate at the default and at its own cost-tuned threshold
    combos = [c for k in (1, 2, 3, 4) for c in itertools.combinations(MODEL_ORDER, k)]
    rows = []
    for c in combos + [(STACK_NAME,)]:
        is_stack = c == (STACK_NAME,)
        p = oof[STACK_NAME] if is_stack else np.mean([oof[m] for m in c], axis=0)
        _, fp0, fn0, _ = counts(yv, p, 0.5)
        t, fn, fp = best_threshold(yv, p)
        rows.append({
            "Candidate": STACK_NAME if is_stack else label(c), "members": c, "Size": len(c),
            "Type": "Stacking" if is_stack else ("Single model" if len(c) == 1 else "Soft vote"),
            "Has linear model": any(m in LINEAR_MODELS for m in c),
            "CV missed @0.5": fn0, "CV false alarms @0.5": fp0, "CV cost @0.5": cost(fn0, fp0),
            "CV accuracy @0.5": 1 - (fn0 + fp0) / len(yv), "CV recall @0.5": 1 - fn0 / yv.sum(),
            "Tuned threshold": t, "CV missed tuned": fn, "CV false alarms tuned": fp, "CV cost tuned": cost(fn, fp),
        })
    cand = pd.DataFrame(rows).sort_values(["CV cost tuned", "Size", "Candidate"]).reset_index(drop=True)
    cand.insert(0, "Stage-1 rank", cand["CV cost tuned"].rank(method="min").astype(int))
    shortlist = cand.head(SHORTLIST).copy()
    if "Random Forest" not in set(shortlist["Candidate"]):
        shortlist = pd.concat([shortlist, cand[cand["Candidate"] == "Random Forest"]])

    # 4) stage 2 — fresh repeated CV for every member (thresholds held fixed)
    fresh_cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=FRESH_REPEATS, random_state=FRESH_SEED)
    splits = list(fresh_cv.split(Xtr, ytr))
    res = Parallel(n_jobs=-1)(delayed(_fit_fold)(n, Xtr, ytr, tr, te) for n in MODEL_ORDER for tr, te in splits)
    fresh = {n: np.zeros((FRESH_REPEATS, len(yv))) for n in MODEL_ORDER}
    for i, (n, te, p) in enumerate(res):
        fresh[n][(i % len(splits)) // N_SPLITS, te] = p
    if STACK_NAME in set(shortlist["Candidate"]):  # stacking cannot be averaged from members: refit it
        res_s = Parallel(n_jobs=-1)(delayed(_fit_stack_fold)(Xtr, ytr, tr, te) for tr, te in splits)
        fresh[STACK_NAME] = np.zeros((FRESH_REPEATS, len(yv)))
        for i, (te, p) in enumerate(res_s):
            fresh[STACK_NAME][i // N_SPLITS, te] = p
    fold_index = [te for _, te in splits]

    def fresh_eval(members, t):
        P = np.mean([fresh[m] for m in members], axis=0)
        per_rep, per_fold = [], []
        for r in range(FRESH_REPEATS):
            tp, fp, fn, tn = counts(yv, P[r], t)
            per_rep.append((fn, fp, cost(fn, fp)))
            for f in range(N_SPLITS):
                te = fold_index[r * N_SPLITS + f]
                _, fpf, fnf, _ = counts(yv[te], P[r, te], t)
                per_fold.append((fnf, fpf, cost(fnf, fpf)))
        a, pf = np.array(per_rep), np.array(per_fold)
        return dict(FN=a[:, 0].mean(), FP=a[:, 1].mean(), cost=a[:, 2].mean(), cost_sd=a[:, 2].std(ddof=1),
                    per_repeat=a[:, 2], per_fold=pf[:, 2], per_fold_fn=pf[:, 0], per_fold_fp=pf[:, 1])

    stage2 = []
    for _, r in shortlist.iterrows():
        fe = fresh_eval(r["members"], r["Tuned threshold"])
        p_test = np.mean([test[m] for m in r["members"]], axis=0)
        tp, fp, fn, tn = counts(yt, p_test, r["Tuned threshold"])  # stacking: members == (STACK_NAME,)
        stage2.append({"Candidate": r["Candidate"], "members": r["members"], "Size": r["Size"], "Type": r["Type"],
                       "Threshold": r["Tuned threshold"], "Stage-1 CV cost": r["CV cost tuned"],
                       "Fresh missed (mean)": fe["FN"], "Fresh false alarms (mean)": fe["FP"],
                       "Fresh cost (mean)": fe["cost"], "Fresh cost (sd)": fe["cost_sd"],
                       "_per_repeat": fe["per_repeat"], "_per_fold": fe["per_fold"],
                       "_per_fold_fn": fe["per_fold_fn"], "_per_fold_fp": fe["per_fold_fp"],
                       "Test missed": fn, "Test false alarms": fp, "Test cost": cost(fn, fp),
                       "Test accuracy": (tp + tn) / len(yt), "Test recall": tp / (tp + fn),
                       "Test precision": tp / (tp + fp) if tp + fp else np.nan})
    s2 = pd.DataFrame(stage2).sort_values(["Fresh cost (mean)", "Size"]).reset_index(drop=True)
    s2.insert(0, "Final rank", range(1, len(s2) + 1))
    winner = s2.iloc[0]
    rf_row = s2[s2["Candidate"] == "Random Forest"].iloc[0]
    n_tr, n_te = len(yv) * (N_SPLITS - 1) / N_SPLITS, len(yv) / N_SPLITS
    # Each shortlisted candidate vs the winner: corrected resampled t-test on the 50 paired fold costs.
    p_vs_w = []
    for _, r in s2.iterrows():
        p_vs_w.append(np.nan if r["Candidate"] == winner["Candidate"] else
                      corrected_ttest(r["_per_fold"] - winner["_per_fold"], n_tr, n_te)["p"])
    s2["p vs winner"] = p_vs_w
    equivalent = s2[(s2["p vs winner"] >= 0.05)]["Candidate"].tolist()
    tt_vs_rf = None
    if winner["Candidate"] != "Random Forest":
        tt_vs_rf = {k: corrected_ttest(winner[f"_per_fold{suf}"] - rf_row[f"_per_fold{suf}"], n_tr, n_te)
                    for k, suf in [("cost", ""), ("missed", "_fn"), ("false_alarms", "_fp")]}
        tt_vs_rf["repeats_better"] = int((winner["_per_repeat"] < rf_row["_per_repeat"]).sum())

    # 5) McNemar on the test set: winner vs Random Forest (tuned and default)
    def correct(members, t):
        p = np.mean([test[m] for m in members], axis=0)
        return (p > t).astype(int) == yt
    w_c = correct(winner["members"], winner["Threshold"])
    mc = {}
    for lab_, mem, t in [("Random Forest @ tuned", ("Random Forest",), rf_row["Threshold"]),
                         ("Random Forest @ 0.5", ("Random Forest",), 0.5)]:
        rf_c = correct(mem, t)
        mc[lab_] = {"all": mcnemar_exact(w_c, rf_c),
                    "malignant": mcnemar_exact(w_c[yt == 1], rf_c[yt == 1]),
                    "benign": mcnemar_exact(w_c[yt == 0], rf_c[yt == 0])}

    # 6) self-check: a real scikit-learn VotingClassifier reproduces the averaged probabilities exactly
    from sklearn.ensemble import VotingClassifier
    wm = winner["members"] if winner["Type"] != "Stacking" else STACK_MEMBERS
    vote = Pipeline([("scale", StandardScaler()),
                     ("model", VotingClassifier([(SHORT[m], prob_model(m)) for m in wm], voting="soft"))])
    real_oof = cross_val_predict(vote, Xtr, ytr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    vote_check = dict(members=wm, max_abs_diff=float(np.abs(real_oof - np.mean([oof[m] for m in wm], 0)).max()))

    E = dict(oof=oof, test=test, fresh=fresh, candidates=cand, shortlist=shortlist, stage2=s2,
             vote_check=vote_check, winner=winner, rf=rf_row, ttest_vs_rf=tt_vs_rf, equivalent=equivalent, mcnemar=mc,
             y_train=yv, y_test=yt)
    E["calibration"] = calibration(E, winner["members"])
    return E


# ------------------------------------------------------------------ calibration
CAL_BINS = 10  # equal-width probability bins for the reliability diagram (empty bins are skipped)


def reliability(y, p, n_bins: int = CAL_BINS) -> pd.DataFrame:
    """Equal-width reliability table: mean predicted vs observed malignant rate per bin, with counts."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)   # p = 1.0 falls in the last bin
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "lo": b / n_bins, "hi": (b + 1) / n_bins, "n": int(m.sum()),
                         "mean_pred": float(p[m].mean()), "frac_pos": float(y[m].mean())})
    return pd.DataFrame(rows)


def calibration_summary(y, p, n_bins: int = CAL_BINS) -> dict:
    """Brier score, Brier skill vs the prevalence-only forecast, and expected calibration error (ECE)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    brier = float(np.mean((p - y) ** 2))
    base = float(np.mean((y.mean() - y) ** 2))                # always predict the training prevalence
    rel = reliability(y, p, n_bins)
    ece = float((rel["n"] * (rel["mean_pred"] - rel["frac_pos"]).abs()).sum() / len(y))
    return dict(brier=brier, brier_skill=1 - brier / base, ece=ece, reliability=rel)


def calibration(E: dict, winner_members: tuple[str, ...]) -> dict:
    """Calibration of every single model, the recommended ensemble and the stack — OOF (600) and test (200)."""
    cfg = {m: (m,) for m in MODEL_ORDER}
    cfg[label(winner_members)] = winner_members
    cfg[STACK_NAME] = (STACK_NAME,)
    out = {}
    for name, mem in cfg.items():
        p_oof = np.mean([E["oof"][m] for m in mem], axis=0)
        p_test = np.mean([E["test"][m] for m in mem], axis=0)
        out[name] = dict(members=mem, oof=calibration_summary(E["y_train"], p_oof),
                         test=calibration_summary(E["y_test"], p_test))
    return out


def evaluate_combo(E: dict, members: tuple[str, ...], t: float) -> dict:
    """Any user-chosen combination at any threshold: OOF, fresh-fold and test counts (exploration)."""
    yv, yt = E["y_train"], E["y_test"]
    p_oof = np.mean([E["oof"][m] for m in members], axis=0)
    p_test = np.mean([E["test"][m] for m in members], axis=0)
    P = np.mean([E["fresh"][m] for m in members], axis=0)
    fr = np.array([[*counts(yv, P[r], t)] for r in range(P.shape[0])])  # TP FP FN TN per repeat
    out = {}
    for key, (y, p) in {"oof": (yv, p_oof), "test": (yt, p_test)}.items():
        tp, fp, fn, tn = counts(y, p, t)
        out[key] = dict(TP=tp, FP=fp, FN=fn, TN=tn, cost=cost(fn, fp), accuracy=(tp + tn) / len(y),
                        recall=tp / (tp + fn), precision=tp / (tp + fp) if tp + fp else np.nan,
                        specificity=tn / (tn + fp))
    out["fresh"] = dict(FN=fr[:, 2].mean(), FP=fr[:, 1].mean(), cost=(COST_FN * fr[:, 2] + fr[:, 1]).mean())
    out["p_test"] = p_test
    return out
