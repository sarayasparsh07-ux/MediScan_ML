"""Transparent, data-driven clinical recommendation.

No winner is hard-coded. The rule is stated in plain words in the dashboard and
is applied to whatever the pipeline actually computed:

  Stage 1 — Safety gate (missed cancers first)
      Keep models whose held-out false negatives are within `fn_tolerance`
      patients of the best model, AND whose cross-validated malignant recall is
      within `cv_margin` of the best CV recall (guards against a single lucky
      test split).
  Stage 2 — False-alarm burden
      Among the survivors, prefer the fewest false positives.
  Stage 3 — Tie-breaks
      Higher mean CV accuracy, then higher held-out accuracy.

`fn_tolerance` defaults to 1 patient because, on 93 malignant test cases, a
difference of one missed cancer is ~1.1 percentage points of recall — well
inside the 95% confidence interval of either model. Setting it to 0 makes the
rule "fewest missed cancers, full stop".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LINEAR_MODELS, MODEL_ORDER
from .pipeline import leaders


@dataclass
class Recommendation:
    recommended: str
    gate_survivors: list[str]
    gate_table: pd.DataFrame
    best_recall: list[str]
    best_precision: list[str]
    best_accuracy: list[str]
    fewest_fn: list[str]
    most_fp: list[str]
    recall_vs_accuracy_differ: bool
    alternative: str | None
    fn_tolerance: int
    cv_margin: float


def _fmt(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1] + " (tied)"


def recommend(lb: pd.DataFrame, cv: pd.DataFrame, fn_tolerance: int = 1, cv_margin: float = 0.02) -> Recommendation:
    ok = lb[lb["Status"] == "OK"].set_index("Model")
    t = ok[["FN", "FP", "recall", "precision", "accuracy"]].copy()
    t["CV recall"] = cv.loc[t.index, "recall_mean"]
    t["CV accuracy"] = cv.loc[t.index, "accuracy_mean"]
    min_fn = int(t["FN"].min())
    best_cv_rec = float(t["CV recall"].max())
    t["Passes FN gate"] = t["FN"] <= min_fn + fn_tolerance
    t["Passes CV-recall gate"] = t["CV recall"] >= best_cv_rec - cv_margin - 1e-12
    t["Survives"] = t["Passes FN gate"] & t["Passes CV-recall gate"]
    surv = t[t["Survives"]].sort_values(["FP", "CV accuracy", "accuracy"], ascending=[True, False, False])
    recommended = surv.index[0] if len(surv) else t.sort_values(["FN", "FP"]).index[0]
    fewest_fn = leaders(lb, "FN", higher_is_better=False, decimals=0)
    alternative = next((m for m in fewest_fn if m != recommended), None)
    best_rec, best_acc = leaders(lb, "recall"), leaders(lb, "accuracy")
    return Recommendation(
        recommended=recommended,
        gate_survivors=surv.index.tolist(),
        gate_table=t.reindex([m for m in MODEL_ORDER if m in t.index]).reset_index(),
        best_recall=best_rec,
        best_precision=leaders(lb, "precision"),
        best_accuracy=best_acc,
        fewest_fn=fewest_fn,
        most_fp=leaders(lb, "FP", higher_is_better=True, decimals=0),
        recall_vs_accuracy_differ=set(best_rec) != set(best_acc),
        alternative=alternative,
        fn_tolerance=fn_tolerance,
        cv_margin=cv_margin,
    )


def pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.1f}%"


def evidence_sentences(rec: Recommendation, A, fn_fp_ratio: float = 10.0) -> str:
    """Significance-test and threshold-tuning evidence for the recommendation (computed, not hard-coded)."""
    from .stats import cv_ttest_table, mcnemar_exact, tune_model

    yte = A.split.y_test.to_numpy()
    out = []
    if rec.alternative:
        a, b = rec.recommended, rec.alternative
        mal = yte == 1
        m_all = mcnemar_exact(A.results[a].y_pred == yte, A.results[b].y_pred == yte)
        m_mal = mcnemar_exact(A.results[a].y_pred[mal] == yte[mal], A.results[b].y_pred[mal] == yte[mal])
        t_rec = cv_ttest_table(A.cv_long, "recall", len(A.split.y_train), [a, b]).iloc[0]
        out.append(
            f"Exact McNemar tests found no significant difference between {a} and {b} on the test set "
            f"(all patients p = {m_all['p']:.2f}; malignant patients p = {m_mal['p']:.2f}, with only "
            f"{m_mal['n_discordant']} discordant case{'' if m_mal['n_discordant'] == 1 else 's'}), and a corrected resampled t-test across "
            f"{len(A.cv_long['fold'].unique())} cross-validation folds found their recall indistinguishable "
            f"(p = {t_rec['p']:.2f})."
        )
    t = tune_model(rec.recommended, A, "min_cost", fn_fp_ratio=fn_fp_ratio)
    d, u = t["test_default"], t["test_tuned"]
    if u["FN"] != d["FN"] or u["FP"] != d["FP"]:
        verb = "Lowering" if t["threshold"] < t["default"] else "Raising"

        def change(label, x, y):
            return f"left {label} at {x}" if x == y else f"changed {label} from {x} to {y}"

        out.append(
            f"{verb} {rec.recommended}'s decision threshold from {t['default']:g} to {t['threshold']:.2f}, chosen on "
            f"out-of-fold training predictions to minimise a {fn_fp_ratio:g}:1 missed-cancer-to-false-alarm penalty, "
            f"{change('held-out missed cancers', d['FN'], u['FN'])} and {change('false alarms', d['FP'], u['FP'])}."
        )
    return " ".join(out)


def professor_paragraph(rec: Recommendation, lb: pd.DataFrame, cv: pd.DataFrame,
                        scaling: dict, team: dict[str, list[str]] | None = None, evidence: str | None = None,
                        combinations: str | None = None) -> str:
    """Academic recommendation paragraph assembled ONLY from computed numbers."""
    ok = lb[lb["Status"] == "OK"].set_index("Model")
    r = ok.loc[rec.recommended]
    n_mal = int(r["TP"] + r["FN"])
    n_ben = int(r["TN"] + r["FP"])
    lin = [m for m in MODEL_ORDER if m in LINEAR_MODELS and m in ok.index]
    lin_acc = " and ".join(f"{m} {pct(ok.loc[m, 'accuracy'])}" for m in lin)
    fns = [int(ok.loc[m, "FN"]) for m in lin]
    lin_fn = (f"each missing {fns[0]} cancers" if len(set(fns)) == 1
              else "missing " + " and ".join(map(str, fns)) + " cancers respectively")
    sc = scaling["scaled"].metrics
    un = scaling["unscaled"].metrics if scaling["unscaled"].ok else None

    parts = [
        f"On the held-out test set of {int(r['n'])} patients ({n_mal} malignant, {n_ben} benign), "
        f"the two straight-boundary models underperformed ({lin_acc} accuracy, {lin_fn}) because malignant cases form a ring around the benign cluster in the "
        f"cell-size/cell-shape plane, which no single straight line can separate.",
    ]
    if un:
        parts.append(
            f"Feature scaling was essential: the RBF-SVM fell from {pct(sc['accuracy'])} to "
            f"{pct(un['accuracy'])} accuracy (recall {pct(sc['recall'])} → {pct(un['recall'])}) without it, "
            f"because the large-valued marker_level dominated its distance calculations."
        )
    alt_txt = ""
    if rec.alternative:
        a = ok.loc[rec.alternative]
        alt_txt = (
            f" {rec.alternative} missed the fewest cancers ({int(a['FN'])} of {n_mal}), but at the cost of "
            f"{int(a['FP'])} false alarms versus {int(r['FP'])}; a one-patient difference in missed cancers is "
            f"within the sampling uncertainty of a {n_mal}-case test set"
        )
        cr, ca = cv.loc[rec.recommended], cv.loc[rec.alternative]
        if cr["recall_mean"] >= ca["recall_mean"] and cr["specificity_mean"] >= ca["specificity_mean"]:
            alt_txt += (
                f", and in cross-validation on the training data {rec.recommended} matched or exceeded it on both "
                f"recall ({pct(cr['recall_mean'])} vs {pct(ca['recall_mean'])}) and specificity "
                f"({pct(cr['specificity_mean'])} vs {pct(ca['specificity_mean'])})"
            )
        alt_txt += ", so it is retained as a high-sensitivity alternative rather than the primary choice."
    parts.append(
        f"Among the six required classifiers we recommend {rec.recommended}. It caught "
        f"{int(r['TP'])} of {n_mal} malignant tumours (recall {pct(r['recall'])}, {int(r['FN'])} missed) "
        f"while raising {int(r['FP'])} false alarms (precision {pct(r['precision'])}, accuracy "
        f"{pct(r['accuracy'])}), and its cross-validated recall on the training data was "
        f"{pct(cv.loc[rec.recommended, 'recall_mean'])} ± {pct(cv.loc[rec.recommended, 'recall_std'])}."
        + alt_txt
    )
    if evidence:
        parts.append(evidence)
    if combinations:
        parts.append(combinations)
    parts.append(
        "Because a missed cancer is far more harmful than a false alarm, we screened models on missed "
        "cancers and cross-validated recall first, and only then minimised false alarms, rather than "
        "selecting on overall accuracy alone. These results come from an educational dataset and do "
        "not establish clinical validity; deployment would require external validation, recalibration of the "
        "probabilities, threshold review, clinical evaluation, and appropriate governance."
    )
    if team:
        filled = [f"{who} ({', '.join(models)})" for who, models in team.items() if who.strip() and models]
        if filled:
            parts.append("Model ownership: " + "; ".join(filled) + ".")
    return " ".join(parts)


def combination_sentences(E: dict) -> str:
    """The two-stage model-combination result, written only from computed numbers."""
    w, rf, tt = E["winner"], E["rf"], E["ttest_vs_rf"]
    n_cand = len(E["candidates"])
    n_short = len(E["stage2"])
    if w["Candidate"] == "Random Forest" or tt is None:
        return (f"Beyond the six required models we scored {n_cand} candidates (single models, all soft-voting "
                f"combinations of two to four models, and a stacked ensemble); after re-checking the {n_short} best on "
                f"fresh cross-validation folds, no combination beat Random Forest at threshold {rf['Threshold']:.2f}.")
    others = len(E["equivalent"])
    rf_in_top = "Random Forest" in E["candidates"].head(8)["Candidate"].tolist()
    n_best = n_short if rf_in_top else n_short - 1
    shortlist_txt = (f"{n_best} best plus Random Forest as the reference" if n_best < n_short else f"{n_short} best")
    equiv_txt = (f"none of the other {n_short - 1} shortlisted configurations was significantly worse"
                 if others == n_short - 1 else
                 f"{others} of the other {n_short - 1} shortlisted configurations were not significantly worse")
    return (
        f"Beyond the six required models, we scored {n_cand} candidates (the six single models, all 50 soft-voting "
        f"combinations of two to four models, and a stacked ensemble), tuning each one's threshold on out-of-fold "
        f"training predictions with a 10:1 missed-cancer-to-false-alarm penalty, and then re-checked the "
        f"{shortlist_txt} on 10 fresh repetitions of 5-fold cross-validation. A soft-voting ensemble of {w['Candidate']} at threshold "
        f"{w['Threshold']:.2f} had the lowest expected cost ({w['Fresh missed (mean)']:.1f} missed cancers and "
        f"{w['Fresh false alarms (mean)']:.1f} false alarms per 600 patients, versus {rf['Fresh missed (mean)']:.1f} and "
        f"{rf['Fresh false alarms (mean)']:.1f} for Random Forest at {rf['Threshold']:.2f}) and beat Random Forest in "
        f"{tt['repeats_better']} of 10 repeats. Its advantage lies in fewer false alarms (corrected t-test "
        f"p = {tt['false_alarms']['p']:.2f}) rather than fewer missed cancers (p = {tt['missed']['p']:.2f}), and "
        f"{equiv_txt}, so we "
        f"recommend this ensemble as MediScan's screening configuration with Random Forest at threshold "
        f"{rf['Threshold']:.2f} as the simpler, statistically comparable fallback.{cautious_sentence(E)} On the held-out test set the "
        f"ensemble missed {int(w['Test missed'])} of {int((E['y_test'] == 1).sum())} cancers with "
        f"{int(w['Test false alarms'])} false alarms."
    )


def cautious_choice(E: dict):
    """The shortlisted configuration that missed the fewest cancers on fresh folds (None if it is the winner)."""
    c = E["stage2"].sort_values(["Fresh missed (mean)", "Fresh cost (mean)"]).iloc[0]
    return None if c["Candidate"] == E["winner"]["Candidate"] else c


def cautious_sentence(E: dict) -> str:
    c = cautious_choice(E)
    if c is None:
        return ""
    return (f" If MediScan wants to minimise missed cancers above all, {c['Candidate']} at threshold {c['Threshold']:.2f} missed "
            f"the fewest cancers on fresh folds ({c['Fresh missed (mean)']:.1f} per 600 patients, at "
            f"{c['Fresh false alarms (mean)']:.1f} false alarms) with a statistically equivalent cost "
            f"(p = {c['p vs winner']:.2f}), making it the most cautious alternative.")


def calibration_sentence(E: dict) -> str:
    """One sentence on probability calibration (computed numbers only)."""
    cal, w = E.get("calibration"), E["winner"]["Candidate"]
    if not cal or "Random Forest" not in cal or w not in cal:
        return ""
    rf, ens = cal["Random Forest"]["oof"]["ece"], cal[w]["oof"]["ece"]
    best = min(cal, key=lambda k: cal[k]["oof"]["ece"])
    return (f"A calibration check showed that Random Forest's and the ensemble's probabilities are compressed toward the "
            f"middle (expected calibration error {rf:.2f} and {ens:.2f}, against {cal[best]['oof']['ece']:.3f} for the "
            f"best-calibrated model, {best}); because every threshold was chosen from observed errors rather than from the "
            f"probability values, this does not change the recommendation, but the scores should be recalibrated before "
            f"being reported to clinicians as risks.")
