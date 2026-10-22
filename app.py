"""MediScan | Diagnostic Intelligence Lab — Streamlit dashboard.

Run:  streamlit run app.py
"""
from __future__ import annotations

import itertools
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import sklearn
import streamlit as st

from mediscan import charts as ch
from mediscan.config import (
    AMBER, CORAL, FEATURE_LABELS, FEATURES, LINEAR_MODELS, MODEL_FAMILY,
    MODEL_ORDER, MODEL_OWNER, MODEL_SPEC, NAVY, TEAL, TEAM,
)
from mediscan.cache import load_or_run
from mediscan.data import DataValidationError, load_and_validate, locate_csv
from mediscan import ensembles as ENS
from mediscan.pipeline import (
    CV_REPEATS, CV_SPLITS, MLP_HIDDEN, cost_scenarios_from_counts,
    cv_expected_errors, leaders, ranking,
)
from mediscan import stats as SIG
from mediscan.qa import run_checks
from mediscan.recommendation import (calibration_sentence, cautious_choice, combination_sentences, evidence_sentences,
                                     pct, professor_paragraph, recommend)

st.set_page_config(page_title="MediScan | Diagnostic Intelligence Lab", page_icon="🩺", layout="wide",
                   initial_sidebar_state="expanded")
PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"], "responsive": True}

# =================================================================== styling
st.markdown(f"""
<style>
  .block-container {{ padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1400px; }}
  h1, h2, h3, h4 {{ color: {NAVY}; letter-spacing: -0.01em; }}
  .ms-header {{ background: linear-gradient(120deg, {NAVY} 0%, #1D4A6E 55%, {TEAL} 130%);
      border-radius: 14px; padding: 22px 28px; color: white; margin-bottom: 18px; }}
  .ms-header .brand {{ font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.2; }}
  .ms-header .brand span {{ color: #9FE0D9; font-weight: 500; }}
  .ms-header .sub {{ font-size: 0.98rem; opacity: 0.88; margin-top: 4px; font-style: italic; }}
  .ms-header .pill {{ display:inline-block; margin-top:12px; margin-right:6px; padding:3px 10px; border-radius:999px;
      background: rgba(255,255,255,0.14); font-size: 0.78rem; border: 1px solid rgba(255,255,255,0.22); }}
  .kicker {{ color: {TEAL}; font-weight: 700; font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase;
      margin-bottom: -6px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 12px; margin: 6px 0 14px; }}
  .kpi {{ background: white; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px 16px;
      border-top: 4px solid {TEAL}; min-height: 112px; }}
  .kpi.coral {{ border-top-color: {CORAL}; }} .kpi.amber {{ border-top-color: {AMBER}; }} .kpi.navy {{ border-top-color: {NAVY}; }}
  .kpi .label {{ font-size: 0.78rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }}
  .kpi .value {{ font-size: 1.65rem; font-weight: 700; color: {NAVY}; margin-top: 2px; line-height: 1.2; }}
  .kpi .sub {{ font-size: 0.83rem; color: #475569; margin-top: 4px; line-height: 1.35; overflow-wrap: anywhere; }}
  .callout {{ border-radius: 10px; padding: 12px 16px; margin: 8px 0 14px; font-size: 0.93rem; line-height: 1.5;
      border-left: 5px solid {TEAL}; background: #DDF3EC; color: #1E3A4C; }}
  .callout.info {{ border-left-color: #2A7DB8; background: #EAF2FA; }}
  .callout.caution {{ border-left-color: {AMBER}; background: #FDF5E3; }}
  .callout.warn {{ border-left-color: {CORAL}; background: #FCEDE8; }}
  .callout b {{ color: {NAVY}; }}
  .card {{ background: white; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px 18px; height: 100%; }}
  .card h4 {{ margin: 0 0 6px 0; font-size: 1.02rem; }}
  .card p, .card li {{ font-size: 0.9rem; line-height: 1.5; color: #334155; }}
  .badge {{ display:inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }}
  .b-pass {{ background:#DDF3EC; color:#0E6E69; }} .b-warn {{ background:#FDF0D5; color:#8A5A00; }}
  .b-fail {{ background:#FCE3DC; color:#A93C22; }}
  .cmsum {{ display:grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }}
  .cmsum > div {{ background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 10px 12px; }}
  .cmsum .fn {{ border: 2px solid {CORAL}; background: #FCEDE8; }}
  .cmsum .fp {{ border: 2px dashed {AMBER}; }}
  .cmsum .t {{ font-size: 0.74rem; color:#64748B; text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }}
  .cmsum .v {{ font-size: 1.4rem; font-weight: 700; color: {NAVY}; }}
  .rec-card {{ background: linear-gradient(135deg, #FFFFFF 0%, #EAF2FA 100%); border: 2px solid {TEAL};
      border-radius: 14px; padding: 18px 22px; }}
  .rec-card .name {{ font-size: 1.6rem; font-weight: 700; color: {NAVY}; }}
  .para {{ background: white; border: 1px solid #CBD5E1; border-radius: 10px; padding: 16px 20px;
      font-family: Georgia, 'Times New Roman', serif; font-size: 1.0rem; line-height: 1.65; color: #1E293B; }}
  section[data-testid="stSidebar"] {{ background: {NAVY}; }}
  section[data-testid="stSidebar"] * {{ color: #E6EEF6 !important; }}
  section[data-testid="stSidebar"] .stRadio label p {{ font-size: 0.96rem; }}
  section[data-testid="stSidebar"] code {{ background: rgba(255,255,255,0.12) !important; color: #9FE0D9 !important; }}
  div[data-testid="stExpander"] details {{ background: white; border-radius: 10px; }}
  .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
  @media (max-width: 760px) {{ .ms-header .brand {{ font-size: 1.3rem; }} .kpi .value {{ font-size: 1.35rem; }} }}
</style>
""", unsafe_allow_html=True)


def callout(text: str, tone: str = "") -> None:
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)  # allow markdown bold inside HTML callouts
    st.markdown(f"<div class='callout {tone}'>{text}</div>", unsafe_allow_html=True)


def kpis(items: list[tuple[str, str, str, str]]) -> None:
    html = "".join(
        f"<div class='kpi {tone}'><div class='label'>{label}</div><div class='value'>{value}</div>"
        f"<div class='sub'>{sub}</div></div>" for label, value, sub, tone in items
    )
    st.markdown(f"<div class='kpi-grid'>{html}</div>", unsafe_allow_html=True)


def section(title: str, kicker: str, intro: str | None = None) -> None:
    st.markdown(f"<div class='kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if intro:
        st.markdown(intro)


_CHART_IDS = itertools.count()  # recreated on every rerun -> stable, unique keys in render order


def plot(fig) -> None:
    # A unique key prevents DuplicateElementId when two charts are identical (e.g. the boundary
    # explorer showing the same model as the SVM highlight above it).
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key=f"chart_{next(_CHART_IDS)}")


def names(lst: list[str]) -> str:
    return lst[0] if len(lst) == 1 else " & ".join(lst) + " (tie)"


# =================================================================== data + models (cached)
@st.cache_resource(show_spinner=False)
def get_analysis(path: str, mtime: float):
    df, report = load_and_validate(path)
    A, source = load_or_run(df, path)
    return df, report, A, source


try:
    csv_path = locate_csv()
    with st.spinner("Validating data and running the full analysis — six classifiers, 57 model combinations and "
                    "the statistical checks (≈ 2 minutes on the very first run, then instant from cache)…"):
        DF, REPORT, A, CACHE_SOURCE = get_analysis(str(csv_path), csv_path.stat().st_mtime)
except FileNotFoundError as e:
    st.error(f"**Dataset not found.** {e}")
    st.stop()
except DataValidationError as e:
    st.error(f"**The dataset failed validation and cannot be modelled safely.** {e}")
    st.stop()

LB = A.lb
OK = LB[LB["Status"] == "OK"].set_index("Model")
FAILED = LB[LB["Status"] != "OK"]
S = A.split
N_TEST = len(S.y_test)
N_MAL_TEST = int((S.y_test == 1).sum())
N_BEN_TEST = N_TEST - N_MAL_TEST
SC = A.scaling["scaled"].metrics
UN = A.scaling["unscaled"].metrics if A.scaling["unscaled"].ok else None

# =================================================================== header + navigation
st.markdown(f"""
<div class='ms-header'>
  <div class='brand'>MediScan <span>| Diagnostic Intelligence Lab</span></div>
  <div class='sub'>Comparative machine learning evaluation for benign and malignant tumour classification.</div>
  <span class='pill'>🧪 {len(DF)} patients · 8 measurements</span>
  <span class='pill'>⚖️ 6 classifiers · one shared test set (n = {N_TEST})</span>
  <span class='pill'>🎯 Positive class = Malignant</span>
  <span class='pill'>👥 {' · '.join(TEAM)}</span>
  <span class='pill'>🎓 Educational exercise — not clinically validated</span>
</div>
""", unsafe_allow_html=True)

PAGES = [
    "Start Here", "Dataset & EDA", "Model Leaderboard", "Confusion Matrix Lab",
    "PCA & Decision Boundaries", "Scaling & Feature Insights", "Significance & Thresholds",
    "Model Combinations", "Clinical Recommendation", "Methodology",
]
ICONS = ["🚀", "🔬", "🏆", "🧮", "🌀", "📏", "📊", "🧩", "🩺", "📘"]
with st.sidebar:
    st.markdown("### 🩺 MediScan")
    st.caption("Diagnostic Intelligence Lab")
    page = st.radio("Navigate", PAGES, format_func=lambda p: f"{ICONS[PAGES.index(p)]}  {p}",
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        f"**Protocol**  \n"
        f"Split: 75/25 stratified, `random_state=1`  \n"
        f"Train {len(S.y_train)} · Test {N_TEST} ({N_MAL_TEST} malignant)  \n"
        f"Scaling: StandardScaler fit on train only  \n"
        f"Positive class: 1 = Malignant"
    )
    st.markdown("---")
    st.markdown("**Team**  \n" + "  \n".join(f"{who} — {', '.join(ENS.SHORT[m] for m in models)}"
                                             for who, models in TEAM.items()))
    st.markdown("---")
    st.caption(f"Data file: `{Path(REPORT.path).name}` · validation "
               f"{'✅ all checks passed' if REPORT.n_fail == 0 and REPORT.n_warn == 0 else f'⚠️ {REPORT.n_warn} warning(s)'}")

if len(FAILED):
    for _, r in FAILED.iterrows():
        st.error(f"**{r['Model']} failed to train or evaluate** — {r['Error']}. It is shown as FAILED, never zero-filled.")


# =================================================================== narrative helpers
def nb_variance_ratio() -> pd.Series:
    nb = A.results["Naive Bayes"].model
    v = pd.DataFrame(nb.var_, columns=FEATURES, index=["Benign", "Malignant"])
    return (v.loc["Malignant"] / v.loc["Benign"])[["cell_size", "cell_shape"]]


def model_story(name: str) -> dict:
    m = OK.loc[name]
    two = A.two_feat["models"][name]["test_acc"]
    obs = (f"Held-out: accuracy {pct(m['accuracy'])}, recall {pct(m['recall'])}, precision {pct(m['precision'])}; "
           f"**{int(m['FN'])} missed cancers**, {int(m['FP'])} false alarms. On the 2-feature ring demo it reaches {pct(two)}.")
    vr = nb_variance_ratio()
    stories = {
        "Logistic Regression": dict(
            how="Models the log-odds of malignancy as a weighted sum of the features, then passes it through the "
                "logistic (sigmoid) function to get a probability.",
            boundary="**Linear.** The sigmoid only converts a score into a probability; the 50% boundary is still "
                     "w·x + b = 0 — a straight line (hyperplane). It cannot wrap around a ring.",
            strengths="Fast, interpretable coefficients, calibrated-ish probabilities.",
            limits="Misses curved structure unless features are engineered (see Scaling & Feature Insights)."),
        "Naive Bayes": dict(
            how="Applies Bayes' rule assuming each feature is Gaussian *within each class* and features are "
                "conditionally independent given the class.",
            boundary=f"**Curved (quadratic).** Gaussian NB learns a separate variance per class. Malignant cell-size "
                     f"variance is {vr['cell_size']:.1f}× and cell-shape variance {vr['cell_shape']:.1f}× the benign "
                     f"variance, so the boundary becomes an ellipse around the benign cluster — almost exactly the ring.",
            strengths="Very high recall here; trains instantly; naturally probabilistic.",
            limits="Independence assumption; the wide malignant variance also produces more false alarms."),
        "SVM (Linear)": dict(
            how="Finds the straight hyperplane with the widest margin between the classes, tolerating some "
                "violations (C = 1).",
            boundary="**Linear.** Maximum-margin, but still a straight line — no margin can separate a ring from its centre.",
            strengths="Robust margin-based fit; good when classes are linearly separable.",
            limits="Same ceiling as logistic regression on this curved problem."),
        "SVM (RBF)": dict(
            how="Uses the radial basis function kernel K(x, x′) = exp(−γ‖x − x′‖²) to measure similarity between "
                "patients, implicitly mapping them to a space where a linear margin exists (C = 3, γ = 'scale').",
            boundary="**Curved.** In the original feature space the boundary bends freely around clusters. "
                     "Because it relies on distances, it needs scaled features.",
            strengths="Flexible, high precision and recall when scaled.",
            limits="Sensitive to feature scale and to C/γ; scores are not probabilities without calibration."),
        "Decision Tree": dict(
            how="Recursively splits the data on single-feature thresholds (max depth 6) to make purer groups.",
            boundary="**Axis-aligned steps.** Many short horizontal/vertical cuts approximate the ring like a "
                     "staircase.",
            strengths="Readable if-then rules; scale-invariant.",
            limits="Unstable (small data changes → different tree); jagged boundary; probabilities are coarse."),
        "Random Forest": dict(
            how="Averages 200 decision trees, each grown on a bootstrap sample with random feature subsets.",
            boundary="**Smoothed steps.** Averaging many staircases gives a smoother approximation of the ring.",
            strengths="Strong, stable, low variance across CV folds; little tuning needed.",
            limits="Less interpretable than one tree; heavier to run; probabilities need calibration for clinical thresholds."),
    }
    return {**stories[name], "observed": obs}


def cm_interpretation(name: str, short: bool = False) -> str:
    m = OK.loc[name]
    med_fp = OK["FP"].median()
    fn, fp = int(m["FN"]), int(m["FP"])
    s = [f"**{fn} of {N_MAL_TEST} malignant patients would be sent home as benign**"
         + (" — the most dangerous outcome, so this count matters more than anything else on the matrix."
            if fn and not short else ".")]
    if fp > med_fp + 2:
        s.append(f"It also raised **{fp} false alarms**, more than the typical model here (median {med_fp:g}), "
                 "meaning extra biopsies and anxiety for healthy patients.")
    elif fp <= med_fp:
        s.append(f"False alarms are low ({fp}), so few healthy patients would be recalled unnecessarily.")
    else:
        s.append(f"It raised {fp} false alarms (median across models {med_fp:g}).")
    if name in LINEAR_MODELS:
        s.append("Its straight-line boundary cannot follow the ring, so errors occur on **both** sides — "
                 "unsuitable for screening support.")
    elif fn <= OK["FN"].min() + 1:
        s.append("As a screening aid this is among the safest options on this test set.")
    else:
        s.append("Usable, but other models miss fewer cancers.")
    return " ".join(s)


# =================================================================== small UI helpers
def takeaway(text: str, tone: str = "") -> None:
    """The one sentence a reader should leave a section with."""
    callout(text, tone)


def details(label: str = "How this works"):
    """Method detail lives in a collapsed expander, so pages stay short."""
    return st.expander(f"ℹ️ {label}", expanded=False)


def owner(m: str) -> str:
    return MODEL_OWNER.get(m, "")


def rec_card(kicker: str, name: str, body: str, accent: str = TEAL) -> None:
    st.markdown(f"<div class='rec-card' style='border-color:{accent}'><div class='kicker'>{kicker}</div>"
                f"<div class='name'>{name}</div>{body}</div>", unsafe_allow_html=True)


# =================================================================== PAGE 1 — start here
def page_start():
    section("Start Here", "The project in five steps",
            "Which model should MediScan trust to flag malignant tumours, when **a missed cancer is far worse "
            "than a false alarm**?")
    rec0, Ew, Erf, E = recommend(LB, A.cv), A.ens["winner"], A.ens["rf"], A.ens
    r0 = OK.loc[rec0.recommended]
    w_owners = ", ".join(f"{ENS.SHORT[m]} ({owner(m)})" for m in Ew["members"])
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        rec_card("Best of the six required models", rec0.recommended,
                 f"<p style='margin:6px 0 0'>Test set: caught <b>{int(r0['TP'])}/{N_MAL_TEST}</b> cancers · "
                 f"<b>{int(r0['FN'])}</b> missed · <b>{int(r0['FP'])}</b> false alarms · accuracy {pct(r0['accuracy'])}"
                 f"<br>Built by {owner(rec0.recommended)} · default threshold 0.5</p>")
    with c2:
        rec_card("Recommended configuration (extension)", f"{Ew['Candidate']} @ {Ew['Threshold']:.2f}",
                 f"<p style='margin:6px 0 0'>Soft vote of {w_owners} · fresh CV "
                 f"<b>{Ew['Fresh missed (mean)']:.1f}</b> missed · <b>{Ew['Fresh false alarms (mean)']:.1f}</b> false "
                 f"alarms per 600 (Random Forest @ {Erf['Threshold']:.2f}: {Erf['Fresh missed (mean)']:.1f} · "
                 f"{Erf['Fresh false alarms (mean)']:.1f})<br>Test set: <b>{int(Ew['Test missed'])}/{N_MAL_TEST}</b> "
                 f"missed · <b>{int(Ew['Test false alarms'])}</b> false alarms</p>", CORAL)

    lin = [m for m in MODEL_ORDER if m in LINEAR_MODELS and m in OK.index]
    cur = [m for m in MODEL_ORDER if m not in LINEAR_MODELS and m in OK.index]
    lin_fn, cur_fn = OK.loc[lin, "FN"].astype(int), OK.loc[cur, "FN"].astype(int)
    st.markdown("")
    kpis([
        ("Patients", f"{len(DF)}", f"{len(S.y_train)} train · {N_TEST} test ({N_MAL_TEST} malignant)", "navy"),
        ("Straight-line models", f"{lin_fn.min()}{'' if lin_fn.min() == lin_fn.max() else '–' + str(lin_fn.max())} missed",
         f"each, of {N_MAL_TEST} cancers", "coral"),
        ("Curved models", f"{cur_fn.min()}–{cur_fn.max()} missed", f"of {N_MAL_TEST} cancers", ""),
        ("RBF-SVM scaling", f"{UN['accuracy'] * 100:.1f} → {pct(SC['accuracy'])}" if UN else "n/a", "accuracy, unscaled → scaled", "amber"),
        ("Combinations tested", f"{len(E['candidates'])}", f"{len(E['stage2'])} re-checked on fresh CV folds", "navy"),
    ])

    st.markdown("### Presentation route")
    st.caption("Five charts that tell the whole story, in order. Each step links to the page with the full analysis.")
    t1, t2, t3, t4, t5 = st.tabs(["① The ring", "② Line vs curve", "③ Scaling", "④ Combinations", "⑤ Recommendation"])
    with t1:
        c1, c2 = st.columns([1.45, 1], gap="large")
        with c1:
            plot(ch.ring_scatter(S.X_train, S.y_train, S.X_test, S.y_test, A.two_feat["centre"]))
        with c2:
            r_size, r_shape = DF["cell_size"].corr(DF["diagnosis"]), DF["cell_shape"].corr(DF["diagnosis"])
            st.markdown(f"""
**What to say**
- Malignant patients **surround** the benign cluster: same centre, much wider spread.
- Correlation calls these two features useless (r = {r_size:+.2f} and {r_shape:+.2f}): it only sees straight-line links.
- Yet they are the strongest predictors, because malignancy depends on **distance from the centre**.
""")
            st.caption("Full analysis → PCA & Decision Boundaries · Dataset & EDA")
    with t2:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            plot(ch.boundary(A.two_feat, "SVM (Linear)", S.X_train, S.y_train, S.X_test, S.y_test, False))
        with c2:
            plot(ch.boundary(A.two_feat, "SVM (RBF)", S.X_train, S.y_train, S.X_test, S.y_test, False))
        tl, tr = A.two_feat["models"]["SVM (Linear)"], A.two_feat["models"]["SVM (RBF)"]
        takeaway(f"On these two features the **linear SVM** finds no useful line and calls everyone benign "
                 f"({tl['test_acc']:.1%} accuracy, recall {tl['test_metrics']['recall']:.0%}); the **RBF kernel** closes a loop "
                 f"around the benign cluster ({tr['test_acc']:.1%}). On all 8 features: "
                 f"{int(OK.loc['SVM (Linear)', 'FN'])} vs {int(OK.loc['SVM (RBF)', 'FN'])} missed cancers "
                 f"({owner('SVM (Linear)')}'s vs {owner('SVM (RBF)')}'s model).")
    with t3:
        c1, c2 = st.columns([1.3, 1], gap="large")
        with c1:
            plot(ch.scaling_compare(SC, UN))
        with c2:
            ds = A.scaling["distance_share_raw"]
            st.markdown(f"""
**What to say**
- Same model, same split; only the scaling changes.
- Unscaled, Marker Level supplies **{ds['marker_level']:.0%}** of every patient-to-patient distance, so the kernel cannot see the ring.
- Missed cancers: **{UN['FN']} → {SC['FN']}**. No other model changes when scaled.
""")
            st.caption("Full analysis → Scaling & Feature Insights")
    with t4:
        plot(ch.fresh_fold_chart(E["stage2"], Ew["Candidate"]))
        tt = E["ttest_vs_rf"]
        takeaway(f"{len(E['candidates'])} candidates were tuned on training folds; the best {len(E['stage2'])} were re-scored on "
                 f"{ENS.FRESH_REPEATS} fresh CV repeats. **{Ew['Candidate']} @ {Ew['Threshold']:.2f}** has the lowest cost "
                 f"and beats Random Forest in {tt['repeats_better']} of {ENS.FRESH_REPEATS} repeats, through fewer false alarms "
                 f"(p = {tt['false_alarms']['p']:.3f}); missed cancers are the same (p = {tt['missed']['p']:.2f}).")
    with t5:
        c1, c2 = st.columns([1.2, 1], gap="large")
        with c1:
            plot(ch.error_bars(LB))
        with c2:
            Ec = cautious_choice(E)
            st.markdown(f"""
**What to say**
- **Six required models → {rec0.recommended}.** Fewest missed cancers within one patient of the best, best cross-validated recall, fewest false alarms among those.
- **Going further → {Ew['Candidate']} @ {Ew['Threshold']:.2f}.** Same cancer-catching rate, fewer false alarms.
- **Fallback:** Random Forest @ {Erf['Threshold']:.2f}, statistically comparable and simpler.
""" + (f"- **Most cautious:** {Ec['Candidate']} @ {Ec['Threshold']:.2f} ({Ec['Fresh missed (mean)']:.1f} missed per 600).\n"
       if Ec is not None else ""))
            st.caption("Full analysis → Clinical Recommendation · Model Combinations · Significance & Thresholds")

    st.markdown("### The six models at a glance")
    t = OK.reindex(MODEL_ORDER)[["accuracy", "recall", "precision", "FN", "FP"]].copy()
    t.insert(0, "Built by", [owner(m) for m in t.index])
    t.insert(1, "Boundary", [MODEL_FAMILY[m] for m in t.index])
    st.dataframe(
        t.style.format({"accuracy": "{:.1%}", "recall": "{:.1%}", "precision": "{:.1%}"})
        .background_gradient(subset=["FN"], cmap="Reds", vmin=0, vmax=max(5, OK["FN"].max())),
        width="stretch",
        column_config={"accuracy": "Accuracy", "recall": "Recall (M)", "precision": "Precision (M)",
                       "FN": "⚠ Missed (FN)", "FP": "False alarms (FP)"})
    st.caption(f"Held-out test set, n = {N_TEST}. 🎓 Educational exercise on a constructed dataset: no model here is "
               "clinically validated.")


# =================================================================== PAGE 2 — EDA
def page_eda():
    section("Dataset & EDA", "Know the data", "The cleaned CSV is re-validated on every start-up.")
    counts = DF["diagnosis"].value_counts()
    kpis([
        ("Total patients", f"{len(DF)}", f"{DF.shape[1]} columns", "navy"),
        ("Benign", f"{counts.get(0, 0)}", f"{counts.get(0, 0) / len(DF):.2%} of patients", ""),
        ("Malignant", f"{counts.get(1, 0)}", f"{counts.get(1, 0) / len(DF):.2%} of patients", "coral"),
        ("Predictor features", f"{len(FEATURES)}", "patient_id & diagnosis excluded", "navy"),
        ("Missing values", f"{int(DF.isna().sum().sum())}", "after EDA-team imputation", ""),
        ("Duplicate rows", f"{int(DF.duplicated().sum())}", f"{int(DF['patient_id'].duplicated().sum())} repeated IDs", ""),
    ])
    with st.expander(f"🔎 Start-up validation report — {len(REPORT.checks)} checks "
                     f"({sum(c.status == 'pass' for c in REPORT.checks)} passed, {REPORT.n_warn} warnings, {REPORT.n_fail} failed)"):
        rep = REPORT.as_frame()
        rep["Status"] = rep["Status"].map({"pass": "✅ pass", "warn": "⚠️ warn", "fail": "❌ fail"})
        st.dataframe(rep, width="stretch", hide_index=True)
        st.caption(f"File: `{REPORT.path}` · dtypes: " + ", ".join(f"{c}={t}" for c, t in DF.dtypes.astype(str).items()))

    c1, c2 = st.columns([1, 1.35], gap="large")
    with c1:
        plot(ch.class_balance(DF))
    with c2:
        st.markdown("#### Summary statistics (raw units)")
        d = DF[FEATURES].describe().T[["mean", "std", "min", "50%", "max"]]
        d.index = [FEATURE_LABELS[f] for f in d.index]
        st.dataframe(d.style.format("{:.2f}"), width="stretch", column_config={"50%": "median"})
        st.caption("Classes are close to balanced (54 / 46), so accuracy is a fair start, but malignant recall matters most.")

    st.markdown("### Feature distributions")
    cc1, cc2, _ = st.columns([1, 1, 2])
    split_by = cc1.toggle("Split by diagnosis", value=True)
    density = cc2.toggle("Density (instead of counts)", value=False)
    for row in range(2):
        cols = st.columns(4, gap="small")
        for i, col in enumerate(cols):
            with col:
                plot(ch.feature_hist(DF, FEATURES[row * 4 + i], split_by, density))
    takeaway("Cell Size and Cell Shape: **same centre, much wider malignant spread**, the first hint of a ring.")

    st.markdown("### Correlation analysis")
    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        plot(ch.corr_heatmap(DF))
    with c2:
        plot(ch.target_corr(DF))
    takeaway(f"Cell Size (r = {DF['cell_size'].corr(DF['diagnosis']):+.3f}) and Cell Shape "
             f"(r = {DF['cell_shape'].corr(DF['diagnosis']):+.3f}) look useless to Pearson correlation, which only sees "
             "straight-line links. They are the strongest predictors (a ring has r ≈ 0), so they stay in every model.",
             "caution")

    st.markdown("### Feature scales")
    view = st.radio("Scale view", ["Raw units", "Standardised (training-fitted)"], horizontal=True,
                    label_visibility="collapsed")
    if view == "Raw units":
        plot(ch.scale_boxes(S.X_train, scaled=False))
    else:
        plot(ch.scale_boxes(pd.DataFrame(S.X_train_s, columns=FEATURES), scaled=True))
    takeaway("Marker Level spans hundreds of units, the scores single digits. Distance-based models need "
             "standardised features (fitted on training rows only).", "info")


# =================================================================== PAGE 3 — leaderboard
def metric_explainer():
    with st.expander("📐 Metric definitions (malignant = positive class)"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Accuracy**: share of all predictions that are correct")
            st.latex(r"\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}")
            st.markdown("**Malignant precision**: of patients flagged malignant, share truly malignant")
            st.latex(r"\text{Precision}=\frac{TP}{TP+FP}")
            st.markdown("**Malignant recall / sensitivity**: of all real cancers, share caught")
            st.latex(r"\text{Recall}=\frac{TP}{TP+FN}")
            st.markdown("**F1-score**: harmonic mean of precision and recall")
            st.latex(r"F_1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}")
        with c2:
            st.markdown("**Benign specificity**: of all benign patients, share correctly cleared")
            st.latex(r"\text{Specificity}=\frac{TN}{TN+FP}")
            st.markdown("**Balanced accuracy**: average of recall and specificity")
            st.latex(r"\text{Balanced acc.}=\tfrac{1}{2}\left(\text{Recall}+\text{Specificity}\right)")
            st.markdown("**ROC-AUC / PR-AUC**: threshold-free ranking quality from *continuous* scores "
                        "(probabilities for LR/NB/trees, decision values for SVMs), never hard labels.")
            st.markdown("**FN count**: malignant patients classified benign. **The most clinically important number.**")
        st.caption("Undefined ratios (e.g. precision when a model never predicts malignant) are shown as n/a, never 0.")


def page_leaderboard():
    section("Model Leaderboard", "Head-to-head",
            f"Same 600 training patients, same scaler, same **{N_TEST} test patients** ({N_MAL_TEST} malignant); "
            "exact Student Guide settings.")
    metric_explainer()
    if len(FAILED):
        st.warning("Failed models are listed with their error rather than omitted.")
    show = LB[LB["Status"] == "OK"].set_index("Model").reindex([m for m in MODEL_ORDER if m in OK.index])
    show = show[["accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy", "roc_auc", "pr_auc",
                 "FN", "FP"]]
    show.insert(0, "Built by", [owner(m) for m in show.index])
    st.markdown("#### Full leaderboard (held-out test set)")
    st.dataframe(
        show.style.format({c: "{:.1%}" for c in ["accuracy", "precision", "recall", "f1", "specificity",
                                                 "balanced_accuracy"]} | {"roc_auc": "{:.3f}", "pr_auc": "{:.3f}"})
        .background_gradient(subset=["FN"], cmap="Reds", vmin=0, vmax=max(5, show["FN"].max()))
        .background_gradient(subset=["recall"], cmap="Greens", vmin=0.6, vmax=1.0),
        width="stretch",
        column_config={
            "accuracy": st.column_config.Column("Accuracy", help="(TP+TN)/all"),
            "precision": st.column_config.Column("Precision (M)", help="TP/(TP+FP)"),
            "recall": st.column_config.Column("Recall (M)", help="TP/(TP+FN): share of cancers caught"),
            "f1": st.column_config.Column("F1 (M)", help="Harmonic mean of precision and recall"),
            "specificity": st.column_config.Column("Specificity (B)", help="TN/(TN+FP)"),
            "balanced_accuracy": st.column_config.Column("Bal. acc.", help="(Recall+Specificity)/2"),
            "roc_auc": st.column_config.Column("ROC-AUC", help="From continuous scores"),
            "pr_auc": st.column_config.Column("PR-AUC", help="Average precision, malignant positive"),
            "FN": st.column_config.Column("⚠ FN", help="Missed cancers"),
            "FP": st.column_config.Column("FP", help="False alarms"),
        },
    )
    if len(FAILED):
        st.dataframe(FAILED[["Model", "Status", "Error"]], hide_index=True, width="stretch")

    c1, c2 = st.columns([1.4, 1], gap="large")
    with c1:
        label_to_key = {v: k for k, v in ch.METRIC_NAMES.items()}
        metric = label_to_key[st.selectbox("Metric to chart", list(label_to_key), index=2)]
        plot(ch.metric_bars(LB, metric))
    with c2:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        best = leaders(LB, metric, higher_is_better=True)
        takeaway(f"<b>{ch.METRIC_NAMES[metric]}</b>, best: <b>{names(best)}</b> ({OK.loc[best[0], metric]:.1%}). "
                 "The four curved models swap places by one or two patients on every metric; the two linear models "
                 "stay about 25 points behind.", "info")
    plot(ch.tradeoff_map(LB))

    st.markdown("### Separate rankings")
    st.caption("Tie-aware ranks on values rounded to 0.1 pt (1, 1, 3 …). FN = missed cancers · FP = false alarms · "
               "Errors = FN + FP, all on the 200 test patients.")
    rA = ranking(LB, "recall", "FN", "FN")
    rB = ranking(LB, "precision", "FP", "FP")
    rC = ranking(LB, "accuracy", "errors", "Errors")
    fmt = lambda df, col: df.assign(**{col: df[col].map(lambda v: f"{v:.1%}")}).assign(
        Rank=df.apply(lambda r: f"{r['Rank']}{' (tie)' if r['Tied'] else ''}", axis=1)).drop(columns="Tied")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("**A · Highest malignant recall**")
        st.dataframe(fmt(rA, "recall").rename(columns={"recall": "Recall"}), hide_index=True, width="stretch")
    with c2:
        st.markdown("**B · Highest malignant precision**")
        st.dataframe(fmt(rB, "precision").rename(columns={"precision": "Precision"}), hide_index=True, width="stretch")
    with c3:
        st.markdown("**C · Highest accuracy**")
        st.dataframe(fmt(rC, "accuracy").rename(columns={"accuracy": "Accuracy"}), hide_index=True, width="stretch")

    st.markdown("**D · Clinical comparison**, sorted by fewest missed cancers, then fewest false alarms")
    d = OK[["recall", "FN", "precision", "FP", "accuracy", "recall_ci_low", "recall_ci_high"]].sort_values(["FN", "FP"])
    d["Recall 95% CI"] = d.apply(lambda r: f"{r['recall_ci_low']:.1%} – {r['recall_ci_high']:.1%}", axis=1)
    st.dataframe(d.drop(columns=["recall_ci_low", "recall_ci_high"]).style.format(
        {"recall": "{:.1%}", "precision": "{:.1%}", "accuracy": "{:.1%}"}), width="stretch",
        column_config={"recall": "Recall", "FN": "⚠ FN", "precision": "Precision", "FP": "FP", "accuracy": "Accuracy"})
    takeaway(f"One patient = {100 / N_MAL_TEST:.1f} pts of recall. The top four's 95% intervals overlap heavily, so a gap "
             "of one or two patients is not conclusive on its own (tested on Significance & Thresholds).", "caution")

    st.markdown("### Threshold-free view")
    t1, t2 = st.tabs(["ROC curves", "Precision–recall curves"])
    with t1:
        plot(ch.roc_pr(A.curves, "roc", LB))
    with t2:
        plot(ch.roc_pr(A.curves, "pr", LB))
    st.caption("AUCs measure how well each model ranks patients; the scores themselves are not calibrated risks "
               "(see the calibration check on Significance & Thresholds).")


# =================================================================== PAGE 4 — confusion matrices
def page_confusion():
    section("Confusion Matrix Lab", "Where the mistakes are",
            "Rows = **actual**, columns = **predicted**. Coral frame = missed cancers (FN); dashed amber = false alarms (FP).")
    c1, c2 = st.columns([1, 2])
    mode = c1.radio("View", ["Counts", "Row percentages"], horizontal=True)
    mode_key = "percent" if mode.startswith("Row") else "count"
    c2.caption(f"Row % = cell ÷ actual patients in that row (benign {N_BEN_TEST}, malignant {N_MAL_TEST}), so FN % = "
               f"1 − recall. One shared colour scale (0 → {max(N_BEN_TEST, N_MAL_TEST) if mode_key == 'count' else 100}).")

    zmax = max(N_BEN_TEST, N_MAL_TEST)
    for row in range(2):
        cols = st.columns(3, gap="medium")
        for i, col in enumerate(cols):
            name = MODEL_ORDER[row * 3 + i]
            with col:
                if name in OK.index:
                    plot(ch.confusion_fig(A.results[name].metrics, name, mode_key, compact=True, zmax_count=zmax))
                else:
                    st.error(f"{name}: failed — no matrix.")

    st.markdown("#### Inspect one model")
    sel = st.selectbox("Classifier", [m for m in MODEL_ORDER if m in OK.index], index=5)
    m = A.results[sel].metrics
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        plot(ch.confusion_fig(m, sel, mode_key, compact=False, zmax_count=zmax))
    with c2:
        st.markdown(f"""
<div class='cmsum'>
  <div><div class='t'>✓ TN · benign cleared</div><div class='v'>{m['TN']}</div></div>
  <div class='fp'><div class='t'>△ FP · false alarms</div><div class='v'>{m['FP']}</div></div>
  <div class='fn'><div class='t'>⚠ FN · missed cancers</div><div class='v'>{m['FN']}</div></div>
  <div><div class='t'>✓ TP · cancers caught</div><div class='v'>{m['TP']}</div></div>
</div>""", unsafe_allow_html=True)
        lb_rec = OK.loc[sel, "recall"]
        mat_rec = m["TP"] / (m["TP"] + m["FN"])
        st.markdown("")
        st.caption(f"Built by {owner(sel)} · recall = {m['TP']} / ({m['TP']} + {m['FN']}) = {mat_rec:.1%} "
                   f"{'✅ matches' if np.isclose(lb_rec, mat_rec) else '❌ does NOT match'} the leaderboard · "
                   f"cells sum to {m['TN'] + m['FP'] + m['FN'] + m['TP']} = {N_TEST}")
        takeaway(cm_interpretation(sel), "warn" if m["FN"] > OK["FN"].min() + 1 else "")
    with details("One-line reading of every model"):
        for name in MODEL_ORDER:
            if name in OK.index:
                st.markdown(f"- **{name}**: {cm_interpretation(name, short=True)}")


# =================================================================== PAGE 5 — PCA & boundaries
def page_pca():
    section("PCA & Decision Boundaries", "See the shape of the problem")
    p = A.pca
    st.markdown("### 1 · PCA of all eight scaled predictors")
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        show_test = st.toggle("Overlay held-out test patients (hollow markers)", value=True)
        plot(ch.pca_scatter(p, S.y_train, S.y_test, show_test))
    with c2:
        kpis([("PC1 + PC2 variance", f"{p['evr'].sum():.1%}", f"PC1 {p['evr'][0]:.1%} · PC2 {p['evr'][1]:.1%}", "navy"),
              ("RBF-SVM on 2 PCs", f"{p['separability']['SVM (RBF)']:.1%}",
               f"linear SVM: {p['separability']['SVM (Linear)']:.1%}", "amber")])
        takeaway("Benign patients sit in the middle, malignant at the edges, but only **partially**: the eight features "
                 "are nearly uncorrelated, so two components keep under a third of the variance. The crisp ring is in "
                 "Cell Size × Cell Shape (next chart).", "caution")
        with details("How to read the PCA plot and its loadings"):
            st.markdown("PCA finds the two directions of greatest variance in the 8 standardised measurements (fitted on "
                        "training patients only; test patients are projected with the same transform). It is a **linear "
                        "projection for visualisation**: the models still use all 8 features.")
            lo = p["loadings"].copy()
            lo.index = [FEATURE_LABELS[f] for f in lo.index]
            st.dataframe(lo.style.format("{:+.2f}").background_gradient(cmap="RdBu_r", vmin=-0.7, vmax=0.7),
                         width="stretch")

    st.markdown("### 2 · The two features that hide the ring")
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        plot(ch.ring_scatter(S.X_train, S.y_train, S.X_test, S.y_test, A.two_feat["centre"]))
    with c2:
        plot(ch.radius_hist(A.two_feat["radius_train"], S.y_train))
        rt = pd.Series(np.asarray(A.two_feat["radius_train"]), index=S.y_train.index)
        takeaway(f"Median distance from the benign centre: benign <b>{rt[S.y_train == 0].median():.2f}</b>, malignant "
                 f"<b>{rt[S.y_train == 1].median():.2f}</b>. One curved quantity separates the classes; no straight "
                 "line can, because malignant cases lie on both sides of any line.")

    st.markdown("### 3 · Straight line vs curved boundary: the SVM highlight")
    show_train = st.toggle("Show training patients (faint) as well as test patients (hollow)", value=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        plot(ch.boundary(A.two_feat, "SVM (Linear)", S.X_train, S.y_train, S.X_test, S.y_test, show_train))
    with c2:
        plot(ch.boundary(A.two_feat, "SVM (RBF)", S.X_train, S.y_train, S.X_test, S.y_test, show_train))
    tl, tr = A.two_feat["models"]["SVM (Linear)"]["test_acc"], A.two_feat["models"]["SVM (RBF)"]["test_acc"]
    lm = A.two_feat["models"]["SVM (Linear)"]["test_metrics"]
    lin_txt = (f"calls every patient benign ({tl:.1%} = the benign share; recall 0%, all {lm['FN']} cancers missed)"
               if lm["TP"] + lm["FP"] == 0 else f"reaches only {tl:.1%} (recall {lm['recall']:.1%})")
    takeaway(f"Trained on just these two features (demonstration only): the <b>linear</b> SVM {lin_txt}; the <b>RBF</b> "
             f"kernel closes a loop around the benign cluster (<b>{tr:.1%}</b>). On all 8 features the gap is "
             f"{pct(OK.loc['SVM (Linear)', 'accuracy'])} vs {pct(OK.loc['SVM (RBF)', 'accuracy'])}.")

    st.markdown("### 4 · Boundary explorer: all six models on the same two features")
    sel = st.selectbox("Model", MODEL_ORDER, index=1)
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        plot(ch.boundary(A.two_feat, sel, S.X_train, S.y_train, S.X_test, S.y_test, show_train))
    with c2:
        st.markdown(f"**Why {sel} behaves this way** *(built by {owner(sel)})*")
        st.markdown(model_story(sel)["boundary"])
        tbl = pd.DataFrame({
            "2 features": {m: A.two_feat["models"][m]["test_acc"] for m in MODEL_ORDER},
            "8 features": {m: OK.loc[m, "accuracy"] if m in OK.index else np.nan for m in MODEL_ORDER},
        })
        st.markdown("**Held-out accuracy**")
        st.dataframe(tbl.style.format("{:.1%}"), width="stretch")
        st.caption("Two separate experiments, shown side by side for comparison. Shaded regions = predicted class.")
    takeaway("<b>Strugglers</b> (LR, linear SVM) draw one straight boundary. <b>Winners</b> bend: the RBF kernel, Naive "
             "Bayes' class-specific variances (an ellipse), and trees' many axis-aligned cuts.", "info")


# =================================================================== PAGE 6 — scaling & features
def page_scaling():
    section("Scaling & Feature Insights", "Preprocessing and what the models rely on")
    st.markdown("### Scaling experiment: RBF-SVM with and without standardisation")
    st.caption("Same split, labels and `SVC(kernel='rbf', C=3)`. An experiment, not a seventh leaderboard entry.")
    if not UN:
        st.error(f"Unscaled run failed: {A.scaling['unscaled'].error}")
        return
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        plot(ch.scaling_compare(SC, UN))
    with c2:
        rows = []
        for k, lab in [("accuracy", "Accuracy"), ("precision", "Precision (M)"), ("recall", "Recall (M)"),
                       ("f1", "F1 (M)"), ("specificity", "Specificity (B)")]:
            rows.append({"Metric": lab, "Unscaled": f"{UN[k]:.1%}", "Scaled": f"{SC[k]:.1%}",
                         "Change": f"{(SC[k] - UN[k]) * 100:+.1f} pts"})
        for k, lab in [("FN", "⚠ Missed cancers (FN)"), ("FP", "False alarms (FP)")]:
            rows.append({"Metric": lab, "Unscaled": str(UN[k]), "Scaled": str(SC[k]), "Change": f"{SC[k] - UN[k]:+d}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        takeaway(f"One preprocessing line: missed cancers <b>{UN['FN']} → {SC['FN']}</b>, accuracy "
                 f"<b>{(SC['accuracy'] - UN['accuracy']) * 100:+.1f} pts</b>.", "warn")

    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        plot(ch.distance_share(A.scaling["distance_share_raw"], A.scaling["distance_share_scaled"]))
    with c2:
        which = st.radio("Confusion matrix", ["Unscaled", "Scaled"], horizontal=True)
        mm = UN if which == "Unscaled" else SC
        plot(ch.confusion_fig(mm, f"RBF-SVM — {which.lower()}", "count", compact=True,
                              zmax_count=max(N_BEN_TEST, N_MAL_TEST)))
    takeaway(f"<b>Why:</b> the RBF kernel uses distances. Raw Marker Level supplies "
             f"<b>{A.scaling['distance_share_raw']['marker_level']:.1%}</b> of the squared distance between patients and "
             f"Cell Size + Cell Shape only <b>{A.scaling['distance_share_raw'][['cell_size', 'cell_shape']].sum():.1%}</b>, so "
             "the kernel is blind to the ring. Scaled, each feature contributes about one-eighth.", "info")

    st.markdown("### Is every model sensitive to scaling?")
    am = A.scaling["all_models"]
    plot(ch.scaling_all_models(am))
    with details("Table: every model, scaled vs unscaled"):
        st.dataframe(am.set_index("Model")[["Unscaled accuracy", "Scaled accuracy", "Accuracy change", "Unscaled FN",
                                            "Scaled FN"]].style.format(
            {"Unscaled accuracy": "{:.1%}", "Scaled accuracy": "{:.1%}", "Accuracy change": "{:+.1%}"}),
            width="stretch", column_config={"Unscaled FN": "FN raw", "Scaled FN": "FN scaled"})
    changed = am[am["Accuracy change"].abs() > 1e-9]["Model"].tolist()
    takeaway(f"Only <b>{', '.join(changed) if changed else 'no model'}</b> changes. Trees split one feature at a time, Naive "
             "Bayes models each feature separately and the linear models rescale their weights, so one common scaled "
             "protocol is fair to every model.", "info")

    st.markdown("### Feature importance")
    c1, c2, _ = st.columns([1, 1, 1])
    model = c1.selectbox("Model", [m for m in MODEL_ORDER if m in OK.index], index=5, key="imp_model")
    scoring = c2.radio("Importance measured on", ["Malignant recall", "Accuracy"], horizontal=True)
    imp = A.importance_recall if scoring.startswith("Malignant") else A.importance_accuracy
    plot(ch.importance_bars(imp, model, "recall" if scoring.startswith("Malignant") else "accuracy"))
    st.caption("Permutation importance: each feature is shuffled 20 times on the test set and the score drop is recorded. "
               "It shows what a model relies on, not what causes cancer. Curved models rely on Cell Size and Cell Shape; "
               "linear models cannot use them.")

    st.markdown("### Optional experiment: engineered features")
    eng = A.engineered
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        plot(ch.engineered_compare(LB, eng))
    with c2:
        st.markdown("")
        takeaway(f"Adding the squares of centred Cell Size and Cell Shape lifts Logistic Regression from "
                 f"{pct(OK.loc['Logistic Regression', 'accuracy'])} to {pct(eng.iloc[0]['Test accuracy'])} "
                 f"(CV {pct(eng.iloc[0]['CV accuracy'])}). The linear model was not broken; it lacked a representation "
                 "in which the boundary is straight. The full 44-feature polynomial does no better.")
        st.caption("Insight only: the leaderboard always uses the 8 original features. All transforms are refit inside "
                   "every CV fold.")
    with details("Table: engineered-feature variants"):
        e = eng.set_index("Variant")[["Input features", "CV accuracy", "CV accuracy sd", "CV recall", "CV recall sd",
                                      "Test accuracy", "Test recall", "Test precision", "FN", "FP"]]
        st.dataframe(e.style.format({c: "{:.1%}" for c in ["CV accuracy", "CV accuracy sd", "CV recall", "CV recall sd",
                                                          "Test accuracy", "Test recall", "Test precision"]}
                                    | {"FN": "{:.0f}", "FP": "{:.0f}"}), width="stretch",
                     column_config={"CV accuracy sd": "± sd", "CV recall sd": "± sd", "Input features": "Inputs"})

    st.markdown("### Robustness: cross-validation on the training set")
    c1, _ = st.columns([1, 2])
    cv_opts = {ch.METRIC_NAMES[k]: k for k in ["recall", "accuracy", "precision", "specificity"]}
    cvm = cv_opts[c1.selectbox("Metric", list(cv_opts))]
    plot(ch.cv_boxes(A.cv_long, cvm, OK.reindex(MODEL_ORDER)[cvm] if cvm in OK.columns else None))
    cvt = A.cv.copy()
    cvt = pd.DataFrame({k: cvt[f"{k}_mean"].map("{:.1%}".format) + " ± " + cvt[f"{k}_std"].map("{:.1%}".format)
                        for k in ["accuracy", "recall", "precision", "specificity"]})
    st.dataframe(cvt, width="stretch", column_config={"accuracy": "CV accuracy", "recall": "CV recall (M)",
                                                       "precision": "CV precision (M)", "specificity": "CV specificity (B)"})
    st.caption(f"{CV_SPLITS}-fold stratified CV × {CV_REPEATS} repeats on the 600 training patients, scaler refit per fold; "
               "the test set is never touched.")


# =================================================================== PAGE 7 — significance, thresholds, calibration
STRONG = [m for m in MODEL_ORDER if m not in LINEAR_MODELS]
PROB_SOURCE = {"Logistic Regression": "native", "Naive Bayes": "native", "SVM (Linear)": "Platt-calibrated",
               "SVM (RBF)": "Platt-calibrated", "Decision Tree": "leaf frequencies",
               "Random Forest": "share of trees voting"}


def p_fmt(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def page_significance():
    section("Significance, Thresholds & Calibration", "Are the differences real? Can we catch more cancers?",
            "Four checks on top of the leaderboard (which stays unchanged): two significance tests, threshold tuning "
            "and probability calibration.")
    yte = S.y_test.to_numpy()

    # ------------------------------------------------------------ McNemar
    st.markdown("### 1 · McNemar's test: paired comparison on the same test patients")
    with details("How McNemar's test works"):
        st.markdown("Both models classify the **same 200 patients**, so only the patients they **disagree** on carry "
                    "information: **b** = A right & B wrong, **c** = A wrong & B right. If the models were equally good, "
                    "disagreements would split 50/50, so b ~ Binomial(b + c, ½). The **exact** test is used because the "
                    "counts are small, and **Holm–Bonferroni** adjusts for the 15 simultaneous comparisons. "
                    "**Min. p** is the smallest p-value the test could produce with that many disagreements.")
    c1, c2 = st.columns([1.4, 1])
    sub_by_label = {v: k for k, v in SIG.SUBSETS.items()}
    subset = sub_by_label[c1.radio("Patients compared", list(sub_by_label), horizontal=True, index=1)]
    adjust = c2.toggle("Holm-adjust for multiple comparisons", value=True)
    tbl = SIG.mcnemar_table(A.results, yte, subset)
    rec = recommend(LB, A.cv)
    key_pairs = []
    if rec.alternative:
        key_pairs.append((rec.recommended, rec.alternative, "Recommended vs high-sensitivity alternative"))
    acc_l, rec_l = leaders(LB, "accuracy"), leaders(LB, "recall")
    for a_ in acc_l:
        for b_ in rec_l:
            if a_ != b_ and {a_, b_} not in [set(k[:2]) for k in key_pairs]:
                key_pairs.append((a_, b_, "Highest accuracy vs highest recall"))
    key_pairs.append(("Random Forest", "Logistic Regression", "Best ensemble vs linear model (reference)"))
    mask = SIG._subset_mask(yte, subset)
    rows = []
    for a_, b_, why in key_pairs:
        m = SIG.mcnemar_exact(A.results[a_].y_pred[mask] == yte[mask], A.results[b_].y_pred[mask] == yte[mask])
        hrow = tbl[((tbl["Model A"] == a_) & (tbl["Model B"] == b_)) | ((tbl["Model A"] == b_) & (tbl["Model B"] == a_))]
        ph = float(hrow["p_holm"].iloc[0]) if len(hrow) else np.nan
        pu = ph if adjust else m["p"]
        rows.append({"Comparison": why, "Model A": a_, "Model B": b_, "b": m["b"], "c": m["c"],
                     "p": p_fmt(m["p"]), "Holm p": p_fmt(ph), "Min. p": p_fmt(m["p_min"]),
                     "Verdict": "✱ significant" if pu < 0.05 else "not significant"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"b": st.column_config.Column("b", help="A right, B wrong"),
                                "c": st.column_config.Column("c", help="A wrong, B right"),
                                "p": st.column_config.Column("Exact p", help="Two-sided exact binomial McNemar"),
                                "Min. p": st.column_config.Column("Min. p", help="Smallest p-value achievable with "
                                                                  "this many discordant patients"),
                                "Verdict": st.column_config.Column("Verdict (α = 0.05)")})

    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        M = SIG.pvalue_matrix(tbl, [m for m in MODEL_ORDER if m in OK.index], "p_holm" if adjust else "p")
        plot(ch.pvalue_heatmap(M, f"Pairwise McNemar p-values — {SIG.SUBSETS[subset].split(' (')[0].lower()}",
                               ("Holm-adjusted" if adjust else "Unadjusted") + " exact p · ✱ teal = significant at 5%"))
    with c2:
        st.markdown("**Inspect a pair**")
        opts = [m for m in MODEL_ORDER if m in OK.index]
        ca, cb = st.columns(2)
        ma = ca.selectbox("Model A", opts, index=opts.index(rec.recommended), key="mc_a")
        mb = cb.selectbox("Model B", opts, index=opts.index(rec.alternative or "SVM (RBF)"), key="mc_b")
        if ma == mb:
            st.info("Choose two different models.")
        else:
            r_ = SIG.mcnemar_exact(A.results[ma].y_pred[mask] == yte[mask], A.results[mb].y_pred[mask] == yte[mask])
            ct = pd.DataFrame([[r_["both_right"], r_["b"]], [r_["c"], r_["both_wrong"]]],
                              index=[f"{ma} right", f"{ma} wrong"], columns=[f"{mb} right", f"{mb} wrong"])
            st.dataframe(ct, width="stretch")
            impossible = r_["p_min"] >= 0.05
            takeaway(f"Exact p = <b>{p_fmt(r_['p'])}</b> (unadjusted) from <b>{r_['n_discordant']}</b> disagreement"
                     f"{'' if r_['n_discordant'] == 1 else 's'}; the smallest possible p here is "
                     f"<b>{p_fmt(r_['p_min'])}</b>" + (", so significance is impossible." if impossible else "."),
                     "warn" if impossible else "")
    strong_tbl = SIG.mcnemar_table(A.results, yte, subset, STRONG)
    n_sig = int((strong_tbl["p"] < 0.05).sum())
    lc = tbl[tbl["Model A"].isin(LINEAR_MODELS) != tbl["Model B"].isin(LINEAR_MODELS)]
    lc_sig = int((lc["p_holm"] < 0.05).sum())
    takeaway(f"The test set separates the families (<b>{lc_sig} of {len(lc)}</b> linear-vs-curved pairs significant after "
             f"Holm) but not the curved models from each other (<b>{n_sig} of {len(strong_tbl)}</b>, unadjusted). "
             "'Not significant' means too few disagreements to tell, not proven equal.", "caution")

    # ------------------------------------------------------------ corrected t-test
    st.markdown("### 2 · Corrected resampled t-test: paired comparison across CV folds")
    with details("How the corrected t-test works"):
        st.markdown(f"Every model is scored on the **same {CV_SPLITS * CV_REPEATS} cross-validation folds** of the 600 "
                    "training patients, so per-fold differences are paired. Folds share training data, which makes a "
                    "naive t-test over-confident; the **Nadeau–Bengio correction** inflates the variance by "
                    "(1/J + n_test/n_train) = (1/15 + 120/480). Holm adjusts for the multiple pairs.")
    c1, c2, _ = st.columns([1, 1, 1])
    opts = {"Malignant recall": "recall", "Accuracy": "accuracy", "Benign specificity": "specificity",
            "Malignant precision": "precision"}
    mlab = c1.selectbox("Metric", list(opts), key="tt_metric")
    include_lin = c2.toggle("Include linear models", value=False)
    models = MODEL_ORDER if include_lin else STRONG
    tt = SIG.cv_ttest_table(A.cv_long, opts[mlab], len(S.y_train), models)
    c1, c2 = st.columns([1.35, 1], gap="large")
    with c1:
        plot(ch.cv_diff_forest(tt, mlab.lower()))
    with c2:
        show = tt[["Model A", "Model B", "mean_diff", "ci_low", "ci_high", "p", "p_holm"]].copy()
        st.dataframe(show.style.format({"mean_diff": "{:+.1%}", "ci_low": "{:+.1%}", "ci_high": "{:+.1%}",
                                        "p": "{:.4f}", "p_holm": "{:.4f}"}),
                     hide_index=True, width="stretch",
                     column_config={"mean_diff": "Mean A − B", "ci_low": "CI low", "ci_high": "CI high",
                                    "p_holm": "Holm p"})
        sig = tt[tt["p_holm"] < 0.05]
        if len(sig):
            txt = "; ".join(f"{r['Model A'] if r['mean_diff'] > 0 else r['Model B']} > "
                            f"{r['Model B'] if r['mean_diff'] > 0 else r['Model A']} ({abs(r['mean_diff']):.1%}, "
                            f"Holm p {p_fmt(r['p_holm'])})" for _, r in sig.iterrows())
            takeaway(f"<b>Significant after correction:</b> {txt}. All other pairs: no reliable difference.")
        else:
            takeaway("No pair differs significantly after correction on this metric.")

    # ------------------------------------------------------------ threshold tuning
    st.markdown("### 3 · Threshold tuning: fewer missed cancers for more false alarms")
    with details("How thresholds are chosen"):
        st.markdown("A model labels a patient malignant when its probability exceeds a cut-off (default 0.5). The cut-off "
                    "is chosen **only from out-of-fold predictions on the 600 training patients** (5-fold, scaler refit "
                    "per fold) on the grid 0.05 … 0.95; the test set is then used **once** to report the effect. SVMs "
                    "use Platt-calibrated probabilities, so their 0.5 starting point can differ slightly from the "
                    "leaderboard. Same protocol as Model Combinations and the notebook.")
    c1, c2, c3 = st.columns([1, 1.2, 1.2])
    opts_m = [m for m in MODEL_ORDER if m in OK.index]
    model = c1.selectbox("Model", opts_m, index=opts_m.index(rec.recommended), key="thr_model")
    rule_lab = c2.radio("Selection rule (on training OOF data)", ["Minimise penalty", "Reach a target recall"],
                        horizontal=False)
    if rule_lab.startswith("Minimise"):
        ratio = c3.slider("FN : FP penalty ratio", 1, 30, 10, key="thr_ratio")
        rule, kw = "min_cost", dict(fn_fp_ratio=float(ratio))
    else:
        tgt = c3.select_slider("Target OOF recall", options=[0.95, 0.96, 0.97, 0.98, 0.99, 1.0], value=0.99,
                               format_func=lambda v: f"{v:.0%}", key="thr_target")
        rule, kw = "target_recall", dict(target_recall=float(tgt))
    t = SIG.tune_model(model, A, rule, **kw)
    td, tt_ = t["test_default"], t["test_tuned"]
    degenerate = t["oof_tuned"]["specificity"] < 0.5
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        plot(ch.threshold_curve_fig(t))
    with c2:
        comp = pd.DataFrame({
            "OOF default": t["oof_default"], "OOF tuned": t["oof_tuned"],
            "Test default": td, "Test tuned": tt_,
        }).loc[["FN", "FP", "recall", "specificity", "precision", "accuracy"]]
        comp.index = ["⚠ FN", "FP", "Recall", "Specificity", "Precision", "Accuracy"]
        fmt = comp.copy().astype(object)
        for i in comp.index:
            fmt.loc[i] = [f"{v:.0f}" if i in ("⚠ FN", "FP") else ("n/a" if pd.isna(v) else f"{v:.1%}")
                          for v in comp.loc[i]]
        st.markdown(f"**Threshold:** default **{t['default']:g}** → tuned **{t['threshold']:.2f}**  \n"
                    f"<span style='color:#64748B;font-size:0.85rem'>{t['why']}</span>", unsafe_allow_html=True)
        st.dataframe(fmt, width="stretch")
        st.caption("OOF = out-of-fold on 600 training patients (chooses). Test = 200 held-out patients (reports).")
        if degenerate:
            takeaway(f"⚠ <b>Not usable.</b> {model} must flag almost everyone to reach this "
                     f"(training specificity {t['oof_tuned']['specificity']:.0%}): its scores cannot rank patients finely.",
                     "warn")
        else:
            d_fn = tt_["FN"] - td["FN"]
            takeaway(f"Test set: missed cancers <b>{td['FN']} → {tt_['FN']}</b>, false alarms <b>{td['FP']} → {tt_['FP']}</b>, "
                     f"accuracy {td['accuracy']:.1%} → {tt_['accuracy']:.1%}.", "" if d_fn < 0 else "caution")
    c1, c2 = st.columns(2, gap="large")
    zmax = max(N_BEN_TEST, N_MAL_TEST)
    with c1:
        plot(ch.confusion_fig(td, f"{model} — default threshold {t['default']:g}", "count", compact=True, zmax_count=zmax))
    with c2:
        plot(ch.confusion_fig(tt_, f"{model} — tuned threshold {t['threshold']:.2f}", "count", compact=True,
                              zmax_count=zmax))

    st.markdown("**Same rule applied to every model**")
    allt = SIG.tune_all(A, rule, **kw)
    compact = pd.DataFrame({
        "Model": allt["Model"],
        "Threshold": [f"{d:g} → {t_:.2f}" for d, t_ in zip(allt["Default t"], allt["Tuned t"])],
        "OOF recall": allt["OOF recall (tuned)"].map("{:.1%}".format),
        "Test FN": [f"{a_} → {b_}" for a_, b_ in zip(allt["Test FN default"], allt["Test FN tuned"])],
        "Test FP": [f"{a_} → {b_}" for a_, b_ in zip(allt["Test FP default"], allt["Test FP tuned"])],
        "Test recall": allt["Test recall tuned"].map("{:.1%}".format),
        "Test accuracy": allt["Test accuracy tuned"].map("{:.1%}".format),
        "Usable?": ["⚠ flags ~everyone" if r["OOF specificity (tuned)"] < 0.5 else "✓"
                    for _, r in allt.iterrows()],
    })
    st.dataframe(compact, hide_index=True, width="stretch",
                 column_config={"Threshold": st.column_config.Column("Threshold", help="default → tuned"),
                                "OOF recall": st.column_config.Column("OOF recall", help="training out-of-fold, tuned"),
                                "Test FN": st.column_config.Column("⚠ Test FN", help="missed cancers, default → tuned"),
                                "Test FP": st.column_config.Column("Test FP", help="false alarms, default → tuned"),
                                "Test recall": st.column_config.Column("Test recall", help="at tuned threshold"),
                                "Test accuracy": st.column_config.Column("Test acc.", help="at tuned threshold"),
                                "Usable?": st.column_config.Column("Usable?", width="medium")})
    st.caption("Arrows read default → tuned. '⚠ flags ~everyone' = training specificity below 50% at the tuned threshold.")

    # ------------------------------------------------------------ calibration
    st.markdown("### 4 · Calibration: can the probabilities be read as risks?")
    with details("What calibration means and how it is measured"):
        st.markdown("A model is **well calibrated** if, among patients it scores at about 0.3, about 30% really are "
                    "malignant. A model can **rank** patients perfectly (high ROC-AUC) and still be badly calibrated.  \n"
                    "- **Reliability diagram:** mean predicted probability vs observed malignant share in 10 equal-width "
                    "bins; the diagonal is perfect.  \n"
                    "- **Brier score:** mean squared error of the probabilities (0 = perfect); *skill* = improvement over "
                    "always predicting the base rate.  \n"
                    "- **ECE:** average gap between predicted and observed rates, weighted by patients per bin.  \n"
                    "Main view: 600 out-of-fold training predictions (more data, no test-set reuse).")
    cal = A.ens["calibration"]
    wname = A.ens["winner"]["Candidate"]
    src = {**PROB_SOURCE, wname: "average of members", ENS.STACK_NAME: "logistic meta-model"}
    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        parts = {"Out-of-fold (600 training)": "oof", "Held-out test (200)": "test"}
        part = parts[st.radio("Patients", list(parts), horizontal=True, key="cal_part")]
        pick = st.multiselect("Show", list(cal), default=["Random Forest", wname, "SVM (RBF)", ENS.STACK_NAME],
                              key="cal_pick")
        if pick:
            plot(ch.reliability_fig(cal, pick, part, wname))
        else:
            st.info("Select at least one model.")
    with c2:
        ct = pd.DataFrame([{"Model / configuration": n.replace("Stack: NB + SVM-RBF + RF → LR", "Stack (NB, SVM-RBF, RF → LR)"),
                            "Probability": src.get(n, ""), "ECE": v[part]["ece"], "Brier": v[part]["brier"]}
                           for n, v in cal.items()]).sort_values("ECE")
        st.dataframe(ct.style.format({"ECE": "{:.3f}", "Brier": "{:.3f}"})
                     .background_gradient(subset=["ECE"], cmap="Reds", vmin=0, vmax=0.15),
                     hide_index=True, width="stretch",
                     column_config={"ECE": st.column_config.Column("ECE", help="Expected calibration error, lower = better"),
                                    "Brier": st.column_config.Column("Brier", help="Mean squared error of probabilities "
                                                                      "(0 = perfect)")})
        rr = cal["Random Forest"]["oof"]["reliability"]
        low = rr[(rr["lo"] >= 0.1 - 1e-9) & (rr["hi"] <= 0.4 + 1e-9)]
        k_low = int(round((low["n"] * low["frac_pos"]).sum()))
        rf_auc = OK.loc["Random Forest", "roc_auc"]
        takeaway(f"Random Forest ranks almost perfectly (ROC-AUC {rf_auc:.3f}) but its probabilities are <b>squeezed toward "
                 f"the middle</b> (ECE {cal['Random Forest']['oof']['ece']:.2f}): of the {int(low['n'].sum())} training "
                 f"patients it scored 0.1–0.4, {k_low} {'was' if k_low == 1 else 'were'} malignant. The ensemble is better "
                 f"(ECE {cal[wname]['oof']['ece']:.2f}). The recommendation stands, because thresholds were set from "
                 "observed errors; recalibrate before showing a score as a % risk.", "caution")


# =================================================================== PAGE 8 — model combinations
def page_combinations():
    E = A.ens
    w, rf, tt = E["winner"], E["rf"], E["ttest_vs_rf"]
    cand, s2 = E["candidates"], E["stage2"]
    section("Model Combinations", "Can several models beat the best single one?",
            "An extension beyond the six required models. **Soft voting** averages the members' probabilities; "
            "**stacking** learns a logistic-regression meta-model. Each candidate gets its own cost-tuned threshold.")
    kpis([
        ("Candidates scored", f"{len(cand)}", "6 single · 50 soft votes (2–4 models) · 1 stack", "navy"),
        ("Recommended configuration", f"{w['Candidate']}", f"soft vote @ threshold {w['Threshold']:.2f} · "
         f"{', '.join(owner(m) for m in dict.fromkeys(w['members']))}", "coral"),
        ("Fresh-fold expected errors", f"{w['Fresh missed (mean)']:.1f} · {w['Fresh false alarms (mean)']:.1f}",
         f"missed · false alarms per 600 (RF @ {rf['Threshold']:.2f}: {rf['Fresh missed (mean)']:.1f} · "
         f"{rf['Fresh false alarms (mean)']:.1f})", ""),
        ("Held-out test (once)", f"{int(w['Test missed'])} missed · {int(w['Test false alarms'])} alarms",
         f"of {N_MAL_TEST} malignant / {N_BEN_TEST} benign · accuracy {w['Test accuracy']:.1%}", ""),
    ])
    with details("Protocol, fixed before any result was read"):
        st.markdown(f"""
1. **Leakage-safe members.** Each model is `StandardScaler → model`, refit inside every fold; SVMs are Platt-calibrated
   (`CalibratedClassifierCV`, 5-fold) so they give probabilities.
2. **Soft vote = average of member probabilities**, verified against scikit-learn's `VotingClassifier`
   (max difference {E['vote_check']['max_abs_diff']:.1e}).
3. **Stage 1, tune & shortlist.** 5-fold out-of-fold predictions (seed {ENS.OOF_SEED}) on the 600 training patients;
   each candidate's threshold minimises **{ENS.COST_FN} × missed + {ENS.COST_FP} × false alarm** on the grid 0.05…0.95
   (ties → higher threshold). The {ENS.SHORTLIST} cheapest are shortlisted, plus Random Forest as the reference.
4. **Stage 2, fresh check.** The shortlist is re-scored on **{ENS.FRESH_REPEATS} new repeats of 5-fold CV (seed
   {ENS.FRESH_SEED})** with thresholds fixed; lowest mean cost wins; a corrected resampled t-test compares each with the winner.
5. **Test set once**, to report the chosen configurations.

*The 10:1 penalty ratio is an illustrative assumption. Fresh folds reuse the same 600 patients: they remove the luck of
one split but are not external validation.*""")

    st.markdown("### Stage 1: all 57 candidates")
    plot(ch.combos_landscape(cand, s2["Candidate"].tolist(), w["Candidate"]))
    lin_best = cand[cand["Has linear model"]]["CV cost @0.5"].min()
    nonlin_best = cand[~cand["Has linear model"] & (cand["Size"] > 1)]["CV cost @0.5"].min()
    rf_c = cand[cand["Candidate"] == "Random Forest"].iloc[0]
    better_05 = cand[(cand["CV cost @0.5"] < rf_c["CV cost @0.5"])]
    takeaway(f"At the default 0.5 threshold <b>{'no candidate beats' if better_05.empty else str(len(better_05)) + ' candidate(s) beat'} "
             f"Random Forest</b> (cost {rf_c['CV cost @0.5']:.0f}). Combinations win only once thresholds are tuned: mixing "
             "different boundary shapes (ellipse, kernel curve, tree boxes) separates the probabilities better. Any linear "
             f"member hurts (best cost {lin_best:.0f} vs {nonlin_best:.0f}).", "info")
    with details("Table: top 12 of 57 by tuned out-of-fold cost"):
        show = cand.drop(columns=["members", "Has linear model"]).head(12)
        st.dataframe(show.style.format({"CV accuracy @0.5": "{:.1%}", "CV recall @0.5": "{:.1%}", "Tuned threshold": "{:.2f}"}),
                     hide_index=True, width="stretch",
                     column_config={"CV cost tuned": st.column_config.Column(
                         "CV cost tuned", help="10 × missed + false alarms at the candidate's own threshold")})
        st.caption("Costs are counts on 600 training patients (277 malignant, 323 benign).")

    st.markdown("### Stage 2: fresh cross-validation check")
    plot(ch.fresh_fold_chart(s2, w["Candidate"]))
    tbl = s2[["Final rank", "Candidate", "Threshold", "Fresh missed (mean)", "Fresh false alarms (mean)",
              "Fresh cost (mean)", "p vs winner", "Stage-1 CV cost"]]
    tbl = tbl.assign(**{"p vs winner": tbl["p vs winner"].map(lambda v: "— (winner)" if pd.isna(v) else f"{v:.2f}")})
    st.dataframe(tbl.style.format({"Threshold": "{:.2f}", "Fresh missed (mean)": "{:.1f}",
                                   "Fresh false alarms (mean)": "{:.1f}", "Fresh cost (mean)": "{:.1f}"}),
                 hide_index=True, width="stretch",
                 column_config={"Fresh missed (mean)": "Missed", "Fresh false alarms (mean)": "False alarms",
                                "Fresh cost (mean)": "Cost", "Stage-1 CV cost": "Stage-1 cost"})
    top_s1 = cand.iloc[0]
    takeaway(f"<b>Why stage 2:</b> the stage-1 leader {top_s1['Candidate']} (cost {top_s1['CV cost tuned']:.0f}) scores "
             f"{s2.set_index('Candidate').loc[top_s1['Candidate'], 'Fresh cost (mean)']:.1f} on fresh folds. Tuning many "
             "candidates on one set of folds flatters whichever fits them best.", "caution")
    if tt:
        Ec = cautious_choice(E)
        takeaway(f"<b>{w['Candidate']} @ {w['Threshold']:.2f} vs Random Forest @ {rf['Threshold']:.2f}:</b> cheaper in "
                 f"<b>{tt['repeats_better']} of {ENS.FRESH_REPEATS}</b> repeats; false alarms {tt['false_alarms']['mean_diff']:+.2f} "
                 f"per fold (p = {tt['false_alarms']['p']:.3f}), missed cancers {tt['missed']['mean_diff']:+.2f} "
                 f"(p = {tt['missed']['p']:.2f}). A borderline edge in false alarms, so Random Forest at its tuned threshold "
                 "is the simpler fallback."
                 + (f" Most cautious: <b>{Ec['Candidate']} @ {Ec['Threshold']:.2f}</b> ({Ec['Fresh missed (mean)']:.1f} missed, "
                    f"{Ec['Fresh false alarms (mean)']:.1f} false alarms; cost p = {Ec['p vs winner']:.2f})." if Ec is not None else ""))

    st.markdown("### Held-out test set: the finalists, evaluated once")
    zmax = max(N_BEN_TEST, N_MAL_TEST)
    fin = [("Random Forest @ 0.50 (leaderboard)", ("Random Forest",), 0.5),
           (f"Random Forest @ {rf['Threshold']:.2f}", ("Random Forest",), rf["Threshold"]),
           (f"{w['Candidate']} @ {w['Threshold']:.2f}", w["members"], w["Threshold"])]
    cols = st.columns(3, gap="medium")
    for col, (lab_, mem, t_) in zip(cols, fin):
        ev = ENS.evaluate_combo(E, mem, t_)["test"]
        with col:
            plot(ch.confusion_fig(ev, lab_, "count", compact=True, zmax_count=zmax))
    mc = E["mcnemar"]["Random Forest @ tuned"]
    takeaway(f"McNemar, ensemble vs Random Forest @ {rf['Threshold']:.2f}: all patients p = {mc['all']['p']:.2f} "
             f"(b = {mc['all']['b']}, c = {mc['all']['c']}), malignant p = {mc['malignant']['p']:.2f}, benign "
             f"p = {mc['benign']['p']:.2f}. Too few disagreements for the test set to separate them.", "caution")

    st.markdown("### Combination explorer")
    st.caption("Pick any members and threshold. Exploration only: the recommendation follows the fixed protocol.")
    c1, c2 = st.columns([1.6, 1])
    members = c1.multiselect("Members (soft vote)", MODEL_ORDER, default=list(w["members"]), key="cx_members")
    thr = c2.select_slider("Threshold", options=[float(t) for t in ENS.THRESHOLDS], value=float(w["Threshold"]),
                           key="cx_thr")
    if not members:
        st.info("Select at least one model.")
        return
    ev = ENS.evaluate_combo(E, tuple(m for m in MODEL_ORDER if m in members), thr)
    o, te, fr = ev["oof"], ev["test"], ev["fresh"]
    kpis([
        ("Stage-1 CV (600)", f"{o['FN']} missed · {o['FP']} alarms", f"cost {o['cost']:.0f} · accuracy {o['accuracy']:.1%}", ""),
        ("Fresh folds (mean of 10)", f"{fr['FN']:.1f} missed · {fr['FP']:.1f} alarms", f"cost {fr['cost']:.1f}", ""),
        ("Held-out test (200)", f"{te['FN']} missed · {te['FP']} alarms",
         f"cost {te['cost']:.0f} · recall {te['recall']:.1%} · accuracy {te['accuracy']:.1%}", "coral" if te["FN"] else ""),
    ])
    c1, c2 = st.columns([1.3, 1], gap="large")
    p_oof = np.mean([E["oof"][m] for m in members], axis=0)
    curve = ENS.threshold_curve(E["y_train"], p_oof)
    with c1:
        plot(ch.threshold_curve_fig(dict(model=" + ".join(ENS.SHORT[m] for m in MODEL_ORDER if m in members),
                                         source="probability (soft-vote average)", default=0.5, threshold=thr,
                                         curve=curve, label="selected")))
    with c2:
        plot(ch.confusion_fig(te, "Selected configuration — test set", "count", compact=True, zmax_count=zmax))


# =================================================================== PAGE 9 — recommendation
def page_recommendation():
    section("Clinical Recommendation", "Decision",
            "Best-by-metric and recommended-overall are **different questions**; this page answers both from computed results.")
    rA, rB, rC = leaders(LB, "recall"), leaders(LB, "precision"), leaders(LB, "accuracy")
    fn_lead, fp_most = leaders(LB, "FN", False, 0), leaders(LB, "FP", True, 0)

    st.markdown("### Best by individual metric")
    kpis([
        ("Highest malignant recall", pct(OK.loc[rA[0], "recall"]), names(rA), ""),
        ("Highest malignant precision", pct(OK.loc[rB[0], "precision"]), names(rB), ""),
        ("Highest accuracy", pct(OK.loc[rC[0], "accuracy"]), names(rC), ""),
        ("Fewest false negatives", f"{int(OK.loc[fn_lead[0], 'FN'])}", names(fn_lead), "coral"),
        ("Most false positives", f"{int(OK.loc[fp_most[0], 'FP'])}", names(fp_most), "amber"),
    ])
    if set(rA) != set(rC):
        takeaway(f"<b>Highest recall ≠ highest accuracy.</b> {names(rA)} misses only {int(OK.loc[rA[0], 'FN'])} cancer(s) "
                 f"but raises {int(OK.loc[rA[0], 'FP'])} false alarms; {names(rC)} make{'' if len(rC) > 1 else 's'} the fewest errors overall. "
                 "Accuracy alone would pick a model that misses more cancers.", "caution")

    st.markdown("### Illustrative cost analysis")
    c1, c2, c3 = st.columns([1, 1, 1.4])
    fn_cost = c1.slider("Penalty per missed cancer (FN)", 1, 50, 10, help="Should exceed the FP penalty: the brief "
                                                                          "states a missed cancer is far worse.")
    fp_cost = c2.slider("Penalty per false alarm (FP)", 1, 10, 1)
    evidence = c3.radio("Evidence used for error counts", ["Held-out test set (n = 200)",
                                                           "Cross-validation (expected per 1,000)"],
                        help="Test set = the one official split. Cross-validation = average error rates over "
                             "15 folds of the 600 training patients, scaled to 1,000 patients at the test-set "
                             "malignancy rate.")
    st.caption("Teaching assumptions, not real hospital costs; only the ratio between the two penalties matters.")
    if fn_cost <= fp_cost:
        st.warning("The FN penalty should be larger than the FP penalty to reflect the clinical priority.")
    ratio = fn_cost / fp_cost
    prevalence = N_MAL_TEST / N_TEST
    cv_counts = cv_expected_errors(A.cv, prevalence)
    use_cv = evidence.startswith("Cross")
    counts = cv_counts if use_cv else OK.reset_index()[["Model", "FN", "FP"]]
    ct = counts.assign(**{"Illustrative cost": counts["FN"] * fn_cost + counts["FP"] * fp_cost})
    ct["Cost rank"] = ct["Illustrative cost"].round(6).rank(method="min").astype(int)
    c1, c2 = st.columns([1, 1.2], gap="large")
    with c1:
        plot(ch.cost_bars(ct.round({"FN": 1, "FP": 1, "Illustrative cost": 1}), fn_cost, fp_cost))
    with c2:
        top = [m for m in MODEL_ORDER if m not in LINEAR_MODELS and m in OK.index]
        scen = cost_scenarios_from_counts(counts)
        plot(ch.cost_scenario_lines(scen, ratio, top, "cv" if use_cv else "test"))

    def winners(cnt):
        sc = cost_scenarios_from_counts(cnt).drop_duplicates("ratio")[["ratio", "best"]]
        ch_ = sc[sc["best"] != sc["best"].shift()]
        return " → ".join(f"<b>{r.best}</b> (from {r.ratio:g}:1)" for r in ch_.itertuples())

    test_counts = OK.reset_index()[["Model", "FN", "FP"]]
    best_test = test_counts.loc[(test_counts["FN"] * ratio + test_counts["FP"]).idxmin(), "Model"]
    best_cv = cv_counts.loc[(cv_counts["FN"] * ratio + cv_counts["FP"]).idxmin(), "Model"]
    takeaway(f"At <b>{ratio:g}:1</b> the lowest-penalty model is <b>{best_test}</b> on the test split and <b>{best_cv}</b> on "
             f"cross-validated rates. As the ratio rises from 1:1 to 30:1: test {winners(test_counts)}; CV {winners(cv_counts)}.",
             "info")
    if best_test != best_cv:
        tf = OK.loc[[best_test, best_cv], ["FN", "FP"]]
        cf = cv_counts.set_index("Model").loc[[best_test, best_cv]]
        with details("Why the test split and cross-validation disagree"):
            st.markdown(f"On the test split {best_test} misses {int(tf.loc[best_test, 'FN'])} cancer(s) vs "
                        f"{int(tf.loc[best_cv, 'FN'])} for {best_cv}: one patient ({100 / N_MAL_TEST:.1f} pts of recall). "
                        f"Across 15 CV folds, {best_cv} is expected to miss {cf.loc[best_cv, 'FN']:.1f} and {best_test} "
                        f"{cf.loc[best_test, 'FN']:.1f} per 1,000 patients, with {cf.loc[best_cv, 'FP']:.0f} vs "
                        f"{cf.loc[best_test, 'FP']:.0f} false alarms. The CV rates are the more stable estimate.")
    st.caption("Linear models are hidden from the line chart (never competitive); they stay in the bar chart.")

    st.markdown("### Recommended: six required models")
    c1, c2 = st.columns([1, 1.4], gap="large")
    with c1:
        tol = st.slider("FN tolerance (patients)", 0, 3, 1,
                        help="How many extra missed cancers vs the best model count as 'equivalent' given test-set "
                             "uncertainty. 0 = strictly fewest missed cancers.")
        margin = st.slider("CV-recall margin (pts)", 0.0, 5.0, 2.0, 0.5) / 100
        rec = recommend(LB, A.cv, tol, margin)
        r = OK.loc[rec.recommended]
        rec_card("Recommended for MediScan screening support — six required models", rec.recommended,
                 f"<p style='margin:6px 0 0'>Caught <b>{int(r['TP'])}/{N_MAL_TEST}</b> cancers · <b>{int(r['FN'])}</b> missed · "
                 f"<b>{int(r['FP'])}</b> false alarms · recall {pct(r['recall'])} · precision {pct(r['precision'])} · "
                 f"accuracy {pct(r['accuracy'])} · CV recall {pct(A.cv.loc[rec.recommended, 'recall_mean'])} ± "
                 f"{pct(A.cv.loc[rec.recommended, 'recall_std'])}<br>Built by {owner(rec.recommended)}</p>"
                 + (f"<p style='margin:8px 0 0'>High-sensitivity alternative: <b>{rec.alternative}</b> "
                    f"({int(OK.loc[rec.alternative, 'FN'])} missed, {int(OK.loc[rec.alternative, 'FP'])} false alarms; "
                    f"{owner(rec.alternative)}).</p>" if rec.alternative else ""))
    with c2:
        st.markdown(f"""**Decision rule (stated before looking at a winner)**
1. **Safety gate:** missed cancers within **{tol}** of the best on the test set *and* CV recall within
   **{margin * 100:.1f} pts** of the best (guards against one lucky split).
2. **False-alarm burden:** among survivors, fewest false positives.
3. **Tie-breaks:** higher CV accuracy, then higher test accuracy.""")
        g = rec.gate_table.set_index("Model")[["FN", "FP", "recall", "CV recall", "Passes FN gate",
                                               "Passes CV-recall gate", "Survives"]]
        st.dataframe(g.style.format({"recall": "{:.1%}", "CV recall": "{:.1%}"}), width="stretch",
                     column_config={"FN": "⚠ FN", "FP": "FP", "recall": "Recall", "Passes FN gate": "FN gate",
                                    "Passes CV-recall gate": "CV gate", "Survives": "Survives"})
        st.caption(f"Default tolerance 1: one patient = {100 / N_MAL_TEST:.1f} pts of recall, inside every top model's "
                   "95% interval. Set it to 0 for 'fewest missed cancers, full stop'.")

    Ew, Erf = A.ens["winner"], A.ens["rf"]
    Ec = cautious_choice(A.ens)
    st.markdown("### Recommended: going further")
    c1, c2 = st.columns([1, 1.4], gap="large")
    with c1:
        rec_card("Recommended configuration (extension)", f"{Ew['Candidate']} @ {Ew['Threshold']:.2f}",
                 f"<p style='margin:6px 0 0'>Soft vote, malignant if averaged P &gt; {Ew['Threshold']:.2f}.<br>"
                 f"Fresh CV: <b>{Ew['Fresh missed (mean)']:.1f}</b> missed · <b>{Ew['Fresh false alarms (mean)']:.1f}</b> "
                 f"false alarms per 600<br>Test: <b>{int(Ew['Test missed'])}/{N_MAL_TEST}</b> missed · "
                 f"<b>{int(Ew['Test false alarms'])}</b> false alarms · accuracy {pct(Ew['Test accuracy'])}</p>"
                 f"<p style='margin:8px 0 0'>Fallback: <b>Random Forest @ {Erf['Threshold']:.2f}</b> "
                 f"({Erf['Fresh missed (mean)']:.1f} · {Erf['Fresh false alarms (mean)']:.1f}).</p>"
                 + ("" if Ec is None else
                    f"<p style='margin:8px 0 0'>Most cautious: <b>{Ec['Candidate']} @ {Ec['Threshold']:.2f}</b> "
                    f"({Ec['Fresh missed (mean)']:.1f} · {Ec['Fresh false alarms (mean)']:.1f}; cost p = {Ec['p vs winner']:.2f}).</p>"),
                 CORAL)
    with c2:
        st.markdown(f"""**How the two recommendations fit together**
- The brief asks which **required classifier** to deploy → **{rec.recommended}** (default threshold, as on the leaderboard).
- **Combining models and tuning the threshold** lowers the expected cost of errors → **{Ew['Candidate']} @ {Ew['Threshold']:.2f}**
  (two-stage protocol, Model Combinations page).
- They agree: Random Forest is the strongest single model and part of the ensemble; at {Erf['Threshold']:.2f} it is the
  statistically comparable fallback.""")
        ev = evidence_sentences(rec, A, fn_cost / fp_cost)
        if ev:
            with details("Statistical and threshold evidence"):
                st.markdown(ev, unsafe_allow_html=True)

    st.markdown("### Team contributions")
    default = pd.DataFrame({"Member": list(TEAM), "Model(s) built": [", ".join(v) for v in TEAM.values()]})
    team_df = st.data_editor(default, num_rows="dynamic", width="stretch", hide_index=True, key="team",
                             column_config={"Member": st.column_config.TextColumn("Member", required=False),
                                            "Model(s) built": st.column_config.TextColumn("Model(s) built")})
    st.caption("Five members (the brief suggests 3–4; the professor approved five). Editable; the paragraph below updates.")
    team = {str(r["Member"]).strip(): [s.strip() for s in str(r["Model(s) built"]).split(",") if s.strip()]
            for _, r in team_df.iterrows() if str(r["Member"]).strip() and str(r["Member"]).strip() != "nan"}

    st.markdown("### Professor-ready recommendation paragraph")
    extra = " ".join(x for x in [combination_sentences(A.ens), calibration_sentence(A.ens)] if x)
    para = professor_paragraph(rec, LB, A.cv, A.scaling, team, evidence_sentences(rec, A, fn_cost / fp_cost), extra)
    c1, c2 = st.columns([3, 1])
    c1.caption(f"Built only from computed numbers ({len(para.split())} words); it follows the sliders above and names "
               "every team member.")
    c2.download_button("⬇ Download (.txt)", para, file_name="mediscan_recommendation.txt", width="stretch")
    with st.expander("📄 Read the paragraph", expanded=False):
        st.markdown(f"<div class='para'>{para}</div>", unsafe_allow_html=True)

    st.markdown("### The questions the report must answer")
    lin = [m for m in MODEL_ORDER if m in LINEAR_MODELS]
    qa = [
        ("Which models struggled or did well, and what do the strugglers share?",
         f"Struggled: {', '.join(f'{m} ({pct(OK.loc[m, 'accuracy'])})' for m in lin)}, both a single straight boundary. "
         f"Did well: {', '.join(f'{m} ({pct(OK.loc[m, 'accuracy'])})' for m in MODEL_ORDER if m not in lin)}, all able to bend."),
        ("Why does the boundary look the way it does?",
         "Malignant patients form a ring around a central benign cluster in Cell Size × Cell Shape (same centre, larger "
         "spread). PCA shows it only partially because it mixes in six other features."),
        ("Did scaling change performance? Which model, by how much?",
         f"Only the RBF-SVM: accuracy {pct(UN['accuracy'])} → {pct(SC['accuracy'])} "
         f"({(SC['accuracy'] - UN['accuracy']) * 100:+.1f} pts), missed cancers {UN['FN']} → {SC['FN']}."),
        ("Most accurate model, or the one catching most cancers?",
         f"Catching cancers comes first. Highest accuracy: {names(rC)}; fewest missed: {names(fn_lead)}. We recommend "
         f"{rec.recommended} among the six, and {Ew['Candidate']} @ {Ew['Threshold']:.2f} going further."),
    ]
    for q, a in qa:
        st.markdown(f"**{q}**  \n{a}")

    with details("Limitations"):
        st.markdown(f"""- One held-out test set of {N_TEST} patients ({N_MAL_TEST} malignant): one patient ≈ {100 / N_MAL_TEST:.1f} pts of recall.
- Constructed educational dataset; no external validation cohort.
- Thresholds were chosen on training data and checked once on one test set; Random Forest's and the ensemble's
  probabilities are not calibrated risks.
- McNemar tests on 200 patients have little power: 'not significant' means too few disagreements to tell.
- Cost penalties are illustrative. Real deployment needs external validation, recalibration, prospective clinical
  evaluation, fairness review and regulatory governance.""")


# =================================================================== model internals (Methodology tabs)
def model_internals(name: str) -> None:
    m = A.results[name].model
    n_tr = len(S.y_train)
    if name in ("Logistic Regression", "SVM (Linear)"):
        coef = pd.Series(m.coef_[0], index=FEATURES)
        plot(ch.signed_bars(coef, f"{name} — learned weights", "Weight on scaled feature (+ → malignant, − → benign)",
                            "Amber = ring features: a straight line gives them almost no weight"))
        if name == "SVM (Linear)":
            nsv = int(m.n_support_.sum())
            callout(f"<b>{nsv} of {n_tr} training patients ({nsv / n_tr:.0%}) are support vectors</b> — a sign that no "
                    "straight margin separates the classes cleanly.", "caution")
    elif name == "Naive Bayes":
        var = m.var_
        t = pd.DataFrame({"Mean benign": m.theta_[0], "Mean malignant": m.theta_[1], "SD benign": np.sqrt(var[0]),
                          "SD malignant": np.sqrt(var[1])}, index=[FEATURE_LABELS[f] for f in FEATURES])
        t["SD ratio (M/B)"] = t["SD malignant"] / t["SD benign"]
        st.markdown("**What Gaussian Naive Bayes learned (scaled features)**")
        st.dataframe(t.style.format("{:.2f}").background_gradient(subset=["SD ratio (M/B)"], cmap="Reds"),
                     width="stretch")
        callout("Same centre, very different spread for Cell Size and Cell Shape: where two Gaussians with a shared "
                "centre but unequal variances cross, the boundary is an <b>ellipse</b> — the ring.", "info")
    elif name == "SVM (RBF)":
        nsv = int(m.n_support_.sum())
        st.markdown(f"**Support vectors:** {nsv} of {n_tr} training patients ({nsv / n_tr:.0%}) — benign "
                    f"{int(m.n_support_[0])}, malignant {int(m.n_support_[1])}.  \n"
                    f"**Kernel width:** γ = 'scale' = 1 / (n_features × variance) = {1 / (len(FEATURES) * S.X_train_s.var()):.3f}.")
        callout("An RBF SVM has no per-feature weights to show: its decision is a weighted sum of similarities "
                "exp(−γ‖x − x′‖²) to these support vectors. Its boundary shape is on the PCA & Decision Boundaries page.",
                "info")
    elif name == "Decision Tree":
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
        fig, ax = plt.subplots(figsize=(13, 5.2))
        plot_tree(m, max_depth=2, feature_names=[FEATURE_LABELS[f] for f in FEATURES], class_names=["Benign", "Malignant"],
                  filled=True, rounded=True, impurity=False, fontsize=9, ax=ax)
        ax.set_title("Decision Tree — top 3 levels (thresholds on scaled values)", fontsize=12, color=NAVY)
        st.pyplot(fig, clear_figure=True)
        imp = pd.Series(m.feature_importances_, index=FEATURES)
        plot(ch.signed_bars(imp, "Decision Tree — feature importance", "Share of impurity reduction",
                            f"The root split uses {FEATURE_LABELS[FEATURES[m.tree_.feature[0]]]}"))
    elif name == "Random Forest":
        imp = pd.Series(m.feature_importances_, index=FEATURES)
        sd = pd.Series(np.std([t.feature_importances_ for t in m.estimators_], axis=0), index=FEATURES)
        plot(ch.signed_bars(imp, "Random Forest — feature importance", "Mean decrease in impurity (± SD across 200 trees)",
                            "The ranking is the opposite of the correlation ranking: the ring features dominate",
                            errors=sd))


# =================================================================== PAGE 10 — methodology
def page_methodology():
    section("Methodology", "How every number was produced")
    st.markdown("### Pipeline")
    st.markdown(f"""
1. **Load & validate** `{Path(REPORT.path).name}` ({len(REPORT.checks)} checks: schema, types, missing values, duplicates, labels, class counts, negatives).
2. **Split** `train_test_split(test_size=0.25, random_state=1, stratify=y)` → {len(S.y_train)} train / {N_TEST} test; `patient_id` dropped.
3. **Scale** `StandardScaler().fit(X_train)`, applied to train and test (never fitted on test data).
4. **Train** the six Student Guide configurations; **evaluate** on the identical test rows, malignant = positive class.
5. **Separate experiments:** scaled vs unscaled · PCA · 2-feature boundaries · CV robustness · McNemar & corrected t-tests · thresholds · calibration · 57 model combinations with a fresh-fold check · engineered features · MLP.
""")
    st.markdown("### The six required classifiers: how each works and what it learned")
    st.caption("One tab per model, labelled with the team member who built it. Fitted on scaled training data, so "
               "weights and split points are in standard-deviation units.")
    names_ok = [m for m in MODEL_ORDER if m in OK.index]
    tabs = st.tabs([f"{m} · {owner(m)}" for m in names_ok])
    for tab, name in zip(tabs, names_ok):
        s_ = model_story(name)
        with tab:
            c1, c2 = st.columns([1, 1.15], gap="large")
            with c1:
                st.markdown(f"`{MODEL_SPEC[name]}` · built by **{owner(name)}**  \n**How it works.** {s_['how']}  \n"
                            f"**Decision boundary.** {s_['boundary']}  \n**Observed here.** {s_['observed']}  \n"
                            f"**Strengths / limits.** {s_['strengths']} {s_['limits']}")
            with c2:
                model_internals(name)

    st.markdown("### Activation functions, neural networks and backpropagation")
    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        st.markdown("""
**Is backpropagation needed for the six required models? No.** None is a multilayer neural network:

| Model | How it is fitted | Activation / backprop? |
|---|---|---|
| Logistic Regression | Convex log-loss optimisation (L-BFGS) | Sigmoid **link** for probabilities; one output unit, no hidden layers, no backprop |
| Linear SVM | Margin maximisation (quadratic programme) | No |
| RBF SVM | Same QP with kernel exp(−γ‖x−x′‖²) | No: a kernel compares *pairs of patients*; it is not a neuron activation |
| Gaussian Naive Bayes | Closed-form class means and variances | No |
| Decision Tree | Greedy impurity-reducing splits | No |
| Random Forest | Bagged trees + random feature subsets | No |
""")
        with details("Kernel vs activation, and what a neural network would need"):
            st.markdown("Both add non-linearity, which is why the Student Guide links them, but they differ: an activation "
                        "(ReLU, tanh, sigmoid) transforms each neuron's weighted input inside a network trained by gradient "
                        "descent; the RBF kernel compares pairs of data points and the SVM solves a convex problem. In a "
                        "neural network, hidden layers need a non-linear activation (otherwise the layers collapse into "
                        "one linear model), a binary output uses a **sigmoid** for P(malignant), and **backpropagation** "
                        "computes the gradients an optimiser such as Adam uses to update the weights.")
    with c2:
        mlp = A.mlp
        plot(ch.mlp_cv_bars(mlp["cv"], mlp["selected"]))
        h = mlp["held_out"]
        lg = mlp["cv"].set_index("Hidden activation")
        if h.ok:
            hm = h.metrics
            takeaway(f"<b>Optional MLP{MLP_HIDDEN}, not on the leaderboard.</b> Activation chosen by CV recall on training "
                     f"data: <b>{mlp['selected']}</b>. Test: accuracy {pct(hm['accuracy'])}, {hm['FN']} missed, {hm['FP']} "
                     "false alarms, comparable to the best baselines, so no clear gain for the extra complexity."
                     + (f" With sigmoid hidden units it stalls at {pct(lg.loc['logistic', 'CV accuracy'])} CV accuracy "
                        "(this architecture and settings only)." if "logistic" in lg.index else ""), "info")

    st.markdown("### Live quality-assurance checks")
    qa = run_checks(A)
    n_pass = int((qa["Result"] == "PASS").sum())
    (st.success if n_pass == len(qa) else st.error)(f"{n_pass} / {len(qa)} automated checks passed on this run.")
    with details(f"All {len(qa)} checks"):
        qa["Result"] = qa["Result"].map({"PASS": "✅ PASS", "FAIL": "❌ FAIL"})
        st.dataframe(qa, hide_index=True, width="stretch", height=36 * (len(qa) + 1) + 4,
                     column_config={"Detail": st.column_config.TextColumn("Detail", width="large")})

    st.markdown("### Reproducibility")
    st.markdown(f"Python {platform.python_version()} · scikit-learn {sklearn.__version__} · pandas {pd.__version__} · "
                f"NumPy {np.__version__} · Plotly {plotly.__version__} · Streamlit {st.__version__}  \n"
                "Seeds: split `random_state=1`; trees `random_state=0`; CV `42`; fresh folds `2026`; MLP `0`. "
                f"Analysis source this run: **{CACHE_SOURCE}** (the cache is keyed to the CSV and library versions).")


# =================================================================== router
{
    "Start Here": page_start,
    "Dataset & EDA": page_eda,
    "Model Leaderboard": page_leaderboard,
    "Confusion Matrix Lab": page_confusion,
    "PCA & Decision Boundaries": page_pca,
    "Scaling & Feature Insights": page_scaling,
    "Significance & Thresholds": page_significance,
    "Model Combinations": page_combinations,
    "Clinical Recommendation": page_recommendation,
    "Methodology": page_methodology,
}[page]()
