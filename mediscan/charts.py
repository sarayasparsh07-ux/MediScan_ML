"""Plotly figures in the MediScan visual identity.

Conventions used by every figure
  * Title, axis titles (with units where meaningful) and a legend whenever >1 series.
  * Benign = blue circle, Malignant = coral diamond (colour is never the only cue).
  * Models keep ONE colour + dash style everywhere (config.MODEL_COLORS / MODEL_DASH).
  * Coral is reserved for false negatives / malignant warnings; amber for cautions.
  * Generous margins + automargin so titles, ticks and legends never clip or overlap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import (
    AMBER, CLASS_COLORS, CLASS_NAMES, CLASS_SYMBOLS, CORAL, FEATURE_LABELS,
    FEATURES, GRID, LINEAR_MODELS, MODEL_COLORS, MODEL_DASH,
    MODEL_ORDER, MODEL_SYMBOLS, MUTED, NAVY, SLATE, TEAL,
)
from plotly.subplots import make_subplots

FONT = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
PCT_FMT = ".1%"
SHORT_UNITS = {"cell_size": "units", "cell_shape": "score", "clump_thickness": "1–10", "uniformity": "1–10",
               "bland_chromatin": "1–10", "mitoses": "count", "age": "years", "marker_level": "reading"}


def _base(fig: go.Figure, title: str, height: int = 380, legend: bool = True,
          subtitle: str | None = None, top: int | None = None) -> go.Figure:
    text = f"<b>{title}</b>"
    if subtitle:
        text += f"<br><span style='font-size:12px;color:{MUTED}'>{subtitle}</span>"
    fig.update_layout(
        template="plotly_white",
        title=dict(text=text, x=0.0, xanchor="left", y=1.0, yanchor="top", yref="container", pad=dict(t=18),
                   font=dict(size=15, color=NAVY, family=FONT)),
        font=dict(family=FONT, size=12, color=SLATE),
        height=height,
        margin=dict(l=10, r=16, t=top if top is not None else (100 if subtitle else 70), b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                    bgcolor="rgba(255,255,255,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, color=SLATE)),
    )
    fig.update_xaxes(showgrid=False, linecolor="#CBD5E1", ticks="outside", tickcolor="#CBD5E1",
                     automargin=True, title_font=dict(size=12, color=SLATE), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor="#CBD5E1", automargin=True,
                     title_font=dict(size=12, color=SLATE), zeroline=False)
    return fig


def _label(y) -> np.ndarray:
    return np.where(np.asarray(y) == 1, CLASS_NAMES[1], CLASS_NAMES[0])


# =============================================================== EDA
def class_balance(df: pd.DataFrame) -> go.Figure:
    counts = df["diagnosis"].value_counts().reindex([0, 1]).fillna(0).astype(int)
    total = counts.sum()
    fig = go.Figure()
    for k in [0, 1]:
        name = CLASS_NAMES[k]
        fig.add_bar(
            x=[name], y=[counts[k]], name=name, marker_color=CLASS_COLORS[name],
            marker_line=dict(color="white", width=2),
            text=[f"<b>{counts[k]}</b>  ({counts[k] / total:.1%})"], textposition="outside",
            cliponaxis=False, width=0.55,
            hovertemplate=f"{name}: %{{y}} patients ({counts[k] / total:.1%})<extra></extra>",
        )
    fig.update_yaxes(title="Number of patients", range=[0, counts.max() * 1.22])
    fig.update_xaxes(title="Diagnosis")
    return _base(fig, "Class balance", 340, subtitle=f"{total} patients · 0 = Benign, 1 = Malignant")


def feature_hist(df: pd.DataFrame, feature: str, by_diagnosis: bool = True, density: bool = False,
                 nbins: int = 28) -> go.Figure:
    fig = go.Figure()
    histnorm = "probability density" if density else None
    ytitle = "Density" if density else "Number of patients"
    if by_diagnosis:
        for k in [0, 1]:
            name = CLASS_NAMES[k]
            fig.add_histogram(
                x=df.loc[df["diagnosis"] == k, feature], name=name, nbinsx=nbins, histnorm=histnorm,
                marker_color=CLASS_COLORS[name], opacity=0.55, marker_line=dict(color="white", width=0.6),
                hovertemplate=f"{name}<br>%{{x}}<br>{ytitle}: %{{y:.3g}}<extra></extra>",
            )
        fig.update_layout(barmode="overlay")
    else:
        fig.add_histogram(x=df[feature], nbinsx=nbins, histnorm=histnorm, name="All patients",
                          marker_color="#7E93AE", marker_line=dict(color="white", width=0.6),
                          hovertemplate=f"%{{x}}<br>{ytitle}: %{{y:.3g}}<extra></extra>")
    fig.update_xaxes(title=f"{FEATURE_LABELS[feature]} ({SHORT_UNITS[feature]})")
    fig.update_yaxes(title=ytitle)
    s = df[feature]
    sub = f"mean {s.mean():.2f} · sd {s.std():.2f} · skew {s.skew():+.2f}"
    fig = _base(fig, FEATURE_LABELS[feature], 310, legend=by_diagnosis, subtitle=sub)
    fig.update_layout(legend=dict(y=1.0, font=dict(size=10)), margin=dict(t=104))
    if feature == "mitoses":
        fig.update_traces(xbins=dict(start=-0.5, end=float(df[feature].max()) + 0.5, size=1), selector=dict(type="histogram"))
    return fig


def _diverging():
    # two hues + neutral grey midpoint (never a hue at the midpoint)
    return [[0.0, "#1F5F99"], [0.25, "#7FA7CF"], [0.5, "#EEF0F2"], [0.75, "#F0A58F"], [1.0, "#C2462A"]]


def corr_heatmap(df: pd.DataFrame) -> go.Figure:
    c = df[FEATURES].corr()
    labels = [FEATURE_LABELS[f] for f in FEATURES]
    z = c.values.copy()
    np.fill_diagonal(z, np.nan)  # r = 1 with itself is trivial; leave diagonal neutral grey
    txt = np.vectorize(lambda v: f"{v:.2f}")(c.values)
    np.fill_diagonal(txt, "1")
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, zmin=-1, zmax=1, colorscale=_diverging(),
        text=txt, texttemplate="%{text}", textfont=dict(size=11, color=SLATE),
        xgap=2, ygap=2, colorbar=dict(title=dict(text="Pearson r", side="right"), thickness=12, len=0.85),
        hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", title="Predictor", showgrid=False)
    fig.update_xaxes(title="Predictor", tickangle=-35)
    fig.update_layout(plot_bgcolor="#E2E8F0")
    return _base(fig, "Predictor-to-predictor correlation", 470, legend=False,
                 subtitle="Pearson r between the 8 measurements (diagnosis excluded)")


def target_corr(df: pd.DataFrame) -> go.Figure:
    r = df[FEATURES].corrwith(df["diagnosis"]).sort_values()
    highlight = {"cell_size", "cell_shape"}
    colors = [AMBER if f in highlight else ("#C2462A" if v > 0 else "#1F5F99") for f, v in r.items()]
    labels = [FEATURE_LABELS[f] + ("  ⚑" if f in highlight else "") for f in r.index]
    fig = go.Figure(go.Bar(
        x=r.values, y=labels, orientation="h", marker_color=colors,
        marker_line=dict(color="white", width=1.5),
        text=[f"{v:+.3f}" for v in r.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>r with diagnosis = %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#94A3B8", line_width=1)
    fig.update_xaxes(title="Pearson correlation with diagnosis (1 = Malignant)", range=[-0.12, 0.42])
    fig.update_yaxes(title=None, showgrid=False)
    return _base(fig, "Feature-to-diagnosis correlation", 380, legend=False,
                 subtitle="⚑ amber = near-zero r, yet a strong non-linear (ring) signal")


def scale_boxes(X: pd.DataFrame, scaled: bool) -> go.Figure:
    fig = go.Figure()
    for f in FEATURES:
        fig.add_box(
            y=X[f], name=FEATURE_LABELS[f], boxpoints="outliers",
            marker=dict(color=CORAL if f == "marker_level" else TEAL, size=3),
            line=dict(color=CORAL if f == "marker_level" else TEAL, width=1.5),
            fillcolor="rgba(231,111,81,0.18)" if f == "marker_level" else "rgba(22,140,134,0.15)",
            hovertemplate=f"{FEATURE_LABELS[f]}<br>%{{y:.2f}}<extra></extra>",
        )
    if scaled:
        title, ytitle = "Standardised feature values (training-fitted scaler)", "z-score (standard deviations from training mean)"
        sub = "After StandardScaler every feature has mean 0 and SD 1 on the training rows"
    else:
        title, ytitle = "Raw feature values — original units", "Raw value (each feature in its own units)"
        sub = "Marker Level (coral) spans ~5–334 while most features span ~1–10"
    fig.update_yaxes(title=ytitle)
    fig.update_xaxes(title="Feature", tickangle=-25)
    return _base(fig, title, 400, legend=False, subtitle=sub)


# =============================================================== PCA & ring
def _class_scatter(fig, x, y, labels, test: bool, size=7, opacity=0.65, row=None, col=None):
    for name in [CLASS_NAMES[0], CLASS_NAMES[1]]:
        msk = labels == name
        color = CLASS_COLORS[name]
        marker = dict(
            size=size + (1 if test else 0), symbol=CLASS_SYMBOLS[name] + ("-open" if test else ""),
            color=color, opacity=0.95 if test else opacity,
            line=dict(width=1.6 if test else 0.6, color=color if test else "white"),
        )
        fig.add_scatter(
            x=x[msk], y=y[msk], mode="markers", marker=marker,
            name=f"{name} — {'test' if test else 'train'}",
            legendgroup=f"{name}{test}",
            hovertemplate=f"{name} ({'test' if test else 'train'})<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<extra></extra>",
            row=row, col=col,
        )


def pca_scatter(pca: dict, y_train, y_test, show_test: bool = True) -> go.Figure:
    evr = pca["evr"]
    fig = go.Figure()
    _class_scatter(fig, pca["train"][:, 0], pca["train"][:, 1], _label(y_train), test=False)
    if show_test:
        _class_scatter(fig, pca["test"][:, 0], pca["test"][:, 1], _label(y_test), test=True)
    fig.update_xaxes(title=f"Principal Component 1 (PC1) — {evr[0]:.1%} of variance", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title=f"Principal Component 2 (PC2) — {evr[1]:.1%} of variance", scaleanchor="x", scaleratio=1)
    return _base(fig, "MediScan Patients in 2D — PCA Projection", 520,
                 subtitle=f"8 scaled predictors → 2 components · {evr.sum():.1%} of variance kept · hollow = test", top=124)


def ring_scatter(X_train, y_train, X_test, y_test, centre=None, show_test=True) -> go.Figure:
    fig = go.Figure()
    _class_scatter(fig, X_train["cell_size"].to_numpy(), X_train["cell_shape"].to_numpy(), _label(y_train), False)
    if show_test:
        _class_scatter(fig, X_test["cell_size"].to_numpy(), X_test["cell_shape"].to_numpy(), _label(y_test), True)
    if centre is not None:
        fig.add_scatter(x=[centre.iloc[0]], y=[centre.iloc[1]], mode="markers",
                        marker=dict(symbol="x-thin", size=16, line=dict(width=3, color=NAVY)),
                        name="Benign cluster centre (train)",
                        hovertemplate="Training benign centre<br>(%{x:.2f}, %{y:.2f})<extra></extra>")
    fig.update_xaxes(title="Cell Size", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title="Cell Shape", scaleanchor="x", scaleratio=1)
    return _base(fig, "The hidden ring — Cell Size × Cell Shape", 520,
                 subtitle="Raw units · malignant surrounds benign · hollow = test", top=124)


def radius_hist(radius_train, y_train) -> go.Figure:
    fig = go.Figure()
    lab = _label(y_train)
    for name in [CLASS_NAMES[0], CLASS_NAMES[1]]:
        fig.add_histogram(x=np.asarray(radius_train)[lab == name], name=name, nbinsx=30, opacity=0.6,
                          marker_color=CLASS_COLORS[name], marker_line=dict(color="white", width=0.6),
                          hovertemplate=f"{name}<br>distance %{{x}}<br>%{{y}} patients<extra></extra>")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Distance from benign centre (raw units)")
    fig.update_yaxes(title="Number of training patients")
    return _base(fig, "Distance from the centre separates the classes", 330,
                 subtitle="Training patients · one curved quantity separates the classes")


def boundary(two: dict, model: str, X_train, y_train, X_test, y_test, show_train=True) -> go.Figure:
    d = two["models"][model]
    fig = go.Figure()
    fig.add_heatmap(
        x=two["gx"], y=two["gy"], z=d["Z"], zmin=0, zmax=1, showscale=False, hoverinfo="skip",
        colorscale=[[0, "rgba(42,125,184,0.16)"], [0.5, "rgba(42,125,184,0.16)"],
                    [0.5, "rgba(231,111,81,0.18)"], [1, "rgba(231,111,81,0.18)"]],
    )
    fig.add_contour(
        x=two["gx"], y=two["gy"], z=d["Z"], showscale=False, contours=dict(start=0.5, end=0.5, size=1, coloring="none"),
        line=dict(color=NAVY, width=2.5), hoverinfo="skip", name="Decision boundary", showlegend=True,
    )
    if show_train:
        _class_scatter(fig, X_train["cell_size"].to_numpy(), X_train["cell_shape"].to_numpy(), _label(y_train),
                       False, size=5, opacity=0.45)
    _class_scatter(fig, X_test["cell_size"].to_numpy(), X_test["cell_shape"].to_numpy(), _label(y_test), True, size=7)
    fig.update_xaxes(title="Cell Size", range=[two["gx"][0], two["gx"][-1]])
    fig.update_yaxes(title="Cell Shape", range=[two["gy"][0], two["gy"][-1]], scaleanchor="x", scaleratio=1)
    m = d["test_metrics"]
    if len(np.unique(d["Z"])) == 1:
        only = CLASS_NAMES[int(d["Z"].flat[0])]
        fig.add_annotation(x=0.5, y=0.97, xref="paper", yref="paper", showarrow=False,
                           text=f"<b>No boundary in view — predicts {only} for every patient</b>",
                           bgcolor="rgba(255,255,255,0.92)", bordercolor=CORAL, borderwidth=1.5, borderpad=5,
                           font=dict(size=12, color=NAVY))
    return _base(fig, f"{model} — 2 features", 480,
                 subtitle=f"Test acc. {d['test_acc']:.1%} · recall {m['recall']:.1%} · missed {m['FN']}", top=124)


# =============================================================== confusion matrices
def confusion_fig(m: dict, model: str, mode: str = "count", compact: bool = False,
                  zmax_count: int | None = None) -> go.Figure:
    """[[TN, FP], [FN, TP]] with rows = actual, cols = predicted. Same scale for every model."""
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    row_tot = cm.sum(1, keepdims=True)
    rowpct = np.divide(cm, row_tot, out=np.zeros_like(cm, dtype=float), where=row_tot > 0)
    if mode == "percent":
        z, zmax = rowpct * 100, 100
    else:
        z, zmax = cm, (zmax_count or int(row_tot.max()))
    codes = [["TN", "FP"], ["FN", "TP"]]
    names = [["True negative", "False positive (false alarm)"], ["False negative (MISSED CANCER)", "True positive"]]
    xl, yl = ["Benign", "Malignant"], ["Benign", "Malignant"]
    fig = go.Figure(go.Heatmap(
        z=z, x=xl, y=yl, zmin=0, zmax=zmax, showscale=False, xgap=4, ygap=4,
        colorscale=[[0, "#F4F8FB"], [0.5, "#8CC8C3"], [1, "#0E6E69"]],
        customdata=np.dstack([cm, rowpct * 100, np.array(names, dtype=object)]),
        hovertemplate="Actual %{y} · Predicted %{x}<br><b>%{customdata[2]}</b><br>"
                      "%{customdata[0]} patients (%{customdata[1]:.1f}% of actual %{y})<extra></extra>",
    ))
    big = 15 if compact else 22
    for i in range(2):
        for j in range(2):
            frac = z[i, j] / zmax if zmax else 0
            ink = "white" if frac > 0.55 else NAVY
            code = codes[i][j]
            main = f"{rowpct[i, j] * 100:.1f}%" if mode == "percent" else f"{cm[i, j]}"
            sub = f"n = {cm[i, j]}" if mode == "percent" else f"{rowpct[i, j] * 100:.1f}% of row"
            tag = {"FN": "⚠ FN", "FP": "△ FP", "TP": "✓ TP", "TN": "✓ TN"}[code]
            fig.add_annotation(x=xl[j], y=yl[i], text=f"<b>{tag}</b>", showarrow=False, yshift=big + 8,
                               font=dict(size=10 if compact else 12, color=ink))
            fig.add_annotation(x=xl[j], y=yl[i], text=f"<b>{main}</b>", showarrow=False,
                               font=dict(size=big, color=ink))
            if not compact:
                fig.add_annotation(x=xl[j], y=yl[i], text=sub, showarrow=False, yshift=-big - 2,
                                   font=dict(size=11, color=ink))
    # FN highlight: coral frame + label (shape + text, not colour alone)
    fig.add_shape(type="rect", x0=-0.47, x1=0.47, y0=0.53, y1=1.47, xref="x", yref="y",
                  line=dict(color=CORAL, width=4 if not compact else 3), fillcolor="rgba(231,111,81,0.10)")
    fig.add_shape(type="rect", x0=0.53, x1=1.47, y0=-0.47, y1=0.47, xref="x", yref="y",
                  line=dict(color=AMBER, width=2, dash="dash"))
    fig.update_xaxes(title="Predicted Diagnosis", side="bottom", showgrid=False, tickfont=dict(size=12))
    fig.update_yaxes(title="Actual Diagnosis", autorange="reversed", showgrid=False, tickfont=dict(size=12))
    rec = m["recall"]
    sub = f"Missed cancers (FN): {m['FN']} of {m['FN'] + m['TP']} · recall {rec:.1%}"
    return _base(fig, model, 300 if compact else 400, legend=False, subtitle=sub, top=84)


def error_bars(lb: pd.DataFrame) -> go.Figure:
    ok = lb[lb["Status"] == "OK"].set_index("Model").reindex(MODEL_ORDER)
    fig = go.Figure()
    fig.add_bar(y=ok.index, x=ok["FN"], name="⚠ False negatives — missed cancers", orientation="h",
                marker_color=CORAL, marker_line=dict(color="white", width=1.5),
                text=ok["FN"], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>Missed cancers: %{x}<extra></extra>")
    fig.add_bar(y=ok.index, x=ok["FP"], name="△ False positives — false alarms", orientation="h",
                marker_color=AMBER, marker_pattern=dict(shape="/", fgcolor="rgba(255,255,255,0.6)", size=6),
                marker_line=dict(color="white", width=1.5),
                text=ok["FP"], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>False alarms: %{x}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    fig.update_yaxes(autorange="reversed", title=None, showgrid=False)
    fig.update_xaxes(title="Number of test patients (held-out n = %d)" % int(ok["n"].iloc[0]),
                     range=[0, max(ok["FN"].max(), ok["FP"].max()) * 1.15 + 1], showgrid=True, gridcolor=GRID)
    return _base(fig, "Errors on the held-out test set", 400,
                 subtitle="The two error types are not equally costly — coral is the dangerous one", top=108)


# =============================================================== leaderboard charts
METRIC_NAMES = {
    "accuracy": "Accuracy", "precision": "Malignant precision", "recall": "Malignant recall (sensitivity)",
    "f1": "F1-score (malignant)", "specificity": "Benign specificity", "balanced_accuracy": "Balanced accuracy",
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC (average precision)",
}


def metric_bars(lb: pd.DataFrame, metric: str) -> go.Figure:
    ok = lb[lb["Status"] == "OK"].sort_values(metric, ascending=True)
    xmin = max(0.0, np.floor((ok[metric].min() - 0.08) * 10) / 10)
    err = None
    if metric == "recall":
        err = dict(type="data", symmetric=False, array=ok["recall_ci_high"] - ok["recall"],
                   arrayminus=ok["recall"] - ok["recall_ci_low"], color=MUTED, thickness=1.2, width=4)
    fig = go.Figure(go.Bar(
        x=ok[metric], y=ok["Model"], orientation="h",
        marker_color=[MODEL_COLORS[m] for m in ok["Model"]], marker_line=dict(color="white", width=1.5),
        error_x=err, hovertemplate="%{y}<br>" + METRIC_NAMES[metric] + ": %{x:.1%}<extra></extra>",
    ))
    # Value labels sit to the right of the bar AND its whisker, in text ink (never on the series colour)
    right = ok["recall_ci_high"] if metric == "recall" else ok[metric]
    for m, v, r in zip(ok["Model"], ok[metric], right):
        if np.isfinite(v):
            fig.add_annotation(x=r, y=m, text=f"<b>{v:.1%}</b>", showarrow=False, xanchor="left", xshift=6,
                               font=dict(size=12, color=SLATE))
    fig.update_xaxes(title=METRIC_NAMES[metric] + " — held-out test set", range=[xmin, 1.09], tickformat=".0%",
                     tickvals=[t for t in np.arange(0, 1.01, 0.1) if t >= xmin - 1e-9],
                     showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title=None, showgrid=False)
    sub = "Whiskers = 95% Wilson interval on 93 malignant test cases" if metric == "recall" else \
        f"Axis starts at {xmin:.0%} to show differences; bars sorted best → top"
    return _base(fig, METRIC_NAMES[metric] + " by classifier", 360, legend=False, subtitle=sub)


def tradeoff_map(lb: pd.DataFrame) -> go.Figure:
    """Left: all six models. Right: zoom on the crowded top-right corner with call-out labels."""
    ok = lb[lb["Status"] == "OK"].reset_index(drop=True)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58], horizontal_spacing=0.12,
                        subplot_titles=("All six models", "Zoom: the strong models"))
    for _, r in ok.iterrows():
        for col in (1, 2):
            fig.add_scatter(
                x=[r["precision"]], y=[r["recall"]], mode="markers", name=r["Model"], legendgroup=r["Model"],
                showlegend=(col == 1),
                marker=dict(size=13 if col == 2 else 11, color=MODEL_COLORS[r["Model"]],
                            symbol=MODEL_SYMBOLS[r["Model"]], line=dict(color="white", width=1.5)),
                error_y=dict(type="data", symmetric=False, array=[r["recall_ci_high"] - r["recall"]],
                             arrayminus=[r["recall"] - r["recall_ci_low"]], color=MODEL_COLORS[r["Model"]],
                             thickness=1.2, width=3),
                hovertemplate=f"<b>{r['Model']}</b><br>Precision %{{x:.1%}}<br>Recall %{{y:.1%}}<br>"
                              f"FN {int(r['FN'])} · FP {int(r['FP'])}<extra></extra>",
                row=1, col=col,
            )
    lin = ok[ok["Model"].isin(LINEAR_MODELS)]
    if len(lin):
        fig.add_annotation(x=lin["precision"].max() + 0.015, y=lin["recall"].mean(), xref="x", yref="y",
                           xanchor="left", text="Linear models<br>(LR & SVM-Linear)", showarrow=False,
                           font=dict(size=10, color=SLATE))
    fig.add_shape(type="rect", x0=0.9, x1=1.0, y0=0.9, y1=1.0, xref="x", yref="y",
                  line=dict(color=TEAL, width=1, dash="dot"), fillcolor="rgba(221,243,236,0.5)", layer="below")
    strong = ok[~ok["Model"].isin(LINEAR_MODELS)].sort_values("precision").reset_index(drop=True)
    # Label placement: an isolated point is labelled to its right; points that share (almost) the same
    # precision are split above-left / below-left in recall order, so labels never cross each other.
    placed = set()
    for i, r in strong.iterrows():
        if i in placed:
            continue
        grp = strong[(strong["precision"] - r["precision"]).abs() < 0.005].sort_values("recall", ascending=False)
        placed.update(grp.index)
        for k, (_, g) in enumerate(grp.iterrows()):
            if len(grp) == 1:
                ax, ay = (60, 0) if g["precision"] < 0.99 else (-60, 0)
            else:  # stacked pair: upper point labelled above-right, lower ones below-left
                ax, ay = (60, -36) if k == 0 else (-60, 36 * k)
            fig.add_annotation(x=g["precision"], y=g["recall"], xref="x2", yref="y2", text=g["Model"], ax=ax, ay=ay,
                               showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="#94A3B8",
                               font=dict(size=11, color=SLATE), bgcolor="rgba(255,255,255,0.9)")
    if len(strong):
        lo_x = min(strong["precision"].min(), 0.9) - 0.01
        lo_y = min(strong["recall_ci_low"].min(), 0.9) - 0.01
        fig.update_xaxes(range=[lo_x, 1.0], row=1, col=2)
        fig.update_yaxes(range=[lo_y, 1.005], row=1, col=2)
    fig.update_xaxes(range=[0.6, 1.02], row=1, col=1)
    fig.update_yaxes(range=[0.5, 1.02], row=1, col=1)
    fig.update_xaxes(title="Malignant precision", tickformat=".0%", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(tickformat=".0%")
    fig.update_yaxes(title="Malignant recall (cancers caught)", row=1, col=1)
    fig.update_annotations(selector=dict(text="All six models"), font=dict(size=12, color=MUTED))
    fig.update_annotations(selector=dict(text="Zoom: the strong models"), font=dict(size=12, color=MUTED))
    fig = _base(fig, "Clinical trade-off map — catching cancers vs false alarms", 500,
                subtitle="Up = fewer missed cancers · right = fewer false alarms · whiskers = 95% CI on recall", top=132)
    fig.update_layout(legend=dict(y=1.13, font=dict(size=11)))
    return fig


def roc_pr(curves: dict, kind: str = "roc", lb: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    auc_col = "roc_auc" if kind == "roc" else "pr_auc"
    aucs = lb.set_index("Model")[auc_col] if lb is not None else {}
    for name in MODEL_ORDER:
        if name not in curves:
            continue
        c = curves[name]
        x, y = (c["fpr"], c["tpr"]) if kind == "roc" else (c["recall"], c["precision"])
        label = f"{name} ({aucs[name]:.3f})" if name in aucs else name
        fig.add_scatter(x=x, y=y, mode="lines", name=label,
                        line=dict(color=MODEL_COLORS[name], width=2, dash=MODEL_DASH[name], shape="hv" if kind == "pr" else "linear"),
                        hovertemplate=f"{name}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<extra></extra>")
    if kind == "roc":
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color="#94A3B8", dash="dot", width=1))
        fig.update_xaxes(title="False-positive rate (1 − specificity)", range=[-0.01, 1.01])
        fig.update_yaxes(title="True-positive rate (malignant recall)", range=[-0.01, 1.02])
        title = "ROC curves (malignant = positive class)"
    else:
        fig.update_xaxes(title="Malignant recall", range=[-0.01, 1.01])
        fig.update_yaxes(title="Malignant precision", range=[0.4, 1.02])
        title = "Precision–recall curves"
    fig = _base(fig, title, 450, subtitle="Continuous scores (probabilities or SVM decision values) · AUC in legend")
    fig.update_layout(legend=dict(orientation="v", x=1.0, xanchor="right", y=0.02, yanchor="bottom",
                                  bgcolor="rgba(255,255,255,0.85)", bordercolor=GRID, borderwidth=1))
    return fig


def cv_boxes(cv_long: pd.DataFrame, metric: str, test_values: pd.Series | None = None) -> go.Figure:
    fig = go.Figure()
    d = cv_long[cv_long["metric"] == metric]
    for name in MODEL_ORDER:
        v = d.loc[d["Model"] == name, "value"]
        fig.add_box(y=v, name=name, boxpoints="all", jitter=0.35, pointpos=0, marker=dict(size=4, color=MODEL_COLORS[name]),
                    line=dict(color=MODEL_COLORS[name], width=1.5), fillcolor="rgba(0,0,0,0)", showlegend=False,
                    hovertemplate=f"{name}<br>fold {METRIC_NAMES.get(metric, metric)}: %{{y:.1%}}<extra></extra>")
    if test_values is not None:
        fig.add_scatter(x=list(test_values.index), y=test_values.values, mode="markers", name="Held-out test score",
                        marker=dict(symbol="star", size=13, color=NAVY, line=dict(color="white", width=1)),
                        hovertemplate="%{x}<br>Held-out: %{y:.1%}<extra></extra>")
    fig.update_yaxes(title=METRIC_NAMES.get(metric, metric.title()) + " per CV fold", tickformat=".0%")
    fig.update_xaxes(title=None, tickangle=-15)
    return _base(fig, f"Cross-validation spread — {METRIC_NAMES.get(metric, metric)}", 420,
                 subtitle="5-fold × 3 repeats on the 600 training patients only (scaler refit in every fold) · ★ = held-out score")


# =============================================================== scaling experiment
def scaling_compare(scaled: dict, unscaled: dict) -> go.Figure:
    keys = ["accuracy", "precision", "recall", "f1", "specificity"]
    names = ["Accuracy", "Precision", "Recall", "F1", "Specificity"]
    fig = go.Figure()
    fig.add_bar(x=names, y=[unscaled[k] for k in keys], name="Unscaled (raw features)", marker_color="#B8C2CF",
                marker_pattern=dict(shape="x", fgcolor="#7C8A9C", size=6), marker_line=dict(color="white", width=1.5),
                text=[f"{unscaled[k]:.1%}" for k in keys], textposition="outside", cliponaxis=False,
                hovertemplate="Unscaled · %{x}: %{y:.1%}<extra></extra>")
    fig.add_bar(x=names, y=[scaled[k] for k in keys], name="Scaled (StandardScaler)", marker_color=TEAL,
                marker_line=dict(color="white", width=1.5),
                text=[f"{scaled[k]:.1%}" for k in keys], textposition="outside", cliponaxis=False,
                hovertemplate="Scaled · %{x}: %{y:.1%}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_yaxes(title="Score on held-out test set", tickformat=".0%", range=[0, 1.12])
    fig.update_xaxes(title="Metric (malignant = positive class)")
    return _base(fig, "Scaling experiment — RBF-SVM (C = 3), same split", 400,
                 subtitle="Only one thing changes: whether features are standardised", top=108)


def distance_share(raw: pd.Series, scaled: pd.Series) -> go.Figure:
    order = raw.sort_values(ascending=True).index
    fig = go.Figure()
    fig.add_bar(y=[FEATURE_LABELS[f] for f in order], x=raw[order], orientation="h", name="Raw features",
                marker_color=[CORAL if f == "marker_level" else "#B8C2CF" for f in order],
                marker_line=dict(color="white", width=1.5),
                text=[f"{v:.1%}" for v in raw[order]], textposition="outside", cliponaxis=False,
                hovertemplate="%{y} (raw): %{x:.1%} of squared distance<extra></extra>")
    fig.add_bar(y=[FEATURE_LABELS[f] for f in order], x=scaled[order], orientation="h", name="Standardised features",
                marker_color=TEAL, marker_line=dict(color="white", width=1.5),
                text=[f"{v:.1%}" for v in scaled[order]], textposition="outside", cliponaxis=False,
                hovertemplate="%{y} (scaled): %{x:.1%} of squared distance<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_xaxes(title="Share of average squared distance between two patients", tickformat=".0%", range=[0, 1.13],
                     showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title=None, showgrid=False)
    return _base(fig, "Who controls the SVM's distance calculation?", 430,
                 subtitle="4,000 random training-patient pairs · RBF kernel = exp(−γ·‖x − x′‖²)", top=108)


# =============================================================== feature insights
def importance_bars(imp: pd.DataFrame, model: str, scoring_label: str) -> go.Figure:
    d = imp[imp["Model"] == model].sort_values("mean")
    fig = go.Figure(go.Bar(
        x=d["mean"], y=[FEATURE_LABELS[f] for f in d["feature"]], orientation="h",
        marker_color=[AMBER if f in ("cell_size", "cell_shape") else TEAL for f in d["feature"]],
        marker_line=dict(color="white", width=1.5),
        error_x=dict(type="data", array=d["std"], color=MUTED, thickness=1.2, width=4),
        hovertemplate="%{y}<br>Mean drop in " + scoring_label + ": %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#94A3B8", line_width=1)
    fig.update_xaxes(title=f"Mean drop in {scoring_label} when the feature is shuffled", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title=None, showgrid=False)
    return _base(fig, f"Permutation importance — {model}", 380, legend=False,
                 subtitle="Held-out test set · 20 shuffles per feature · whiskers = ±1 SD · amber = cell features")


def engineered_compare(lb: pd.DataFrame, eng: pd.DataFrame) -> go.Figure:
    base = lb.set_index("Model")
    rows = [("Logistic Regression (baseline, 8 features)", base.loc["Logistic Regression", "accuracy"],
             base.loc["Logistic Regression", "recall"], "#9FB4CC"),
            ("SVM (Linear) (baseline, 8 features)", base.loc["SVM (Linear)", "accuracy"],
             base.loc["SVM (Linear)", "recall"], "#9FB4CC")]
    rows += [(r["Variant"], r["Test accuracy"], r["Test recall"], TEAL) for _, r in eng.iterrows()]
    fig = go.Figure()
    fig.add_bar(y=[r[0] for r in rows], x=[r[1] for r in rows], name="Held-out accuracy", orientation="h",
                marker_color=[r[3] for r in rows], marker_line=dict(color="white", width=1.5),
                hovertemplate="%{y}<br>Accuracy %{x:.1%}<extra></extra>")
    for r in rows:  # label at the visible left edge (axis starts at 50%), clear of the recall diamond
        fig.add_annotation(x=0.505, y=r[0], text=f"<b>{r[1]:.1%}</b>", showarrow=False, xanchor="left",
                           font=dict(size=11, color="white" if r[3] == TEAL else NAVY))
    fig.add_scatter(y=[r[0] for r in rows], x=[r[2] for r in rows], mode="markers", name="Held-out malignant recall",
                    marker=dict(symbol="diamond", size=12, color=CORAL, line=dict(color="white", width=1.5)),
                    hovertemplate="%{y}<br>Recall %{x:.1%}<extra></extra>")
    fig.update_xaxes(title="Held-out score", tickformat=".0%", range=[0.5, 1.01], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title=None, autorange="reversed", showgrid=False)
    return _base(fig, "Can feature engineering rescue a straight-line model?", 380,
                 subtitle="Grey = required baseline · teal = optional engineered variant (separate experiment)", top=108)


# =============================================================== cost analysis
def cost_bars(ct: pd.DataFrame, fn_cost: float, fp_cost: float) -> go.Figure:
    d = ct.sort_values("Illustrative cost", ascending=False)
    fig = go.Figure()
    fig.add_bar(y=d["Model"], x=d["FN"] * fn_cost, name=f"⚠ Missed cancers × {fn_cost:g}", orientation="h",
                marker_color=CORAL, marker_line=dict(color="white", width=1.5),
                hovertemplate="%{y}<br>FN penalty: %{x:g}<extra></extra>")
    fig.add_bar(y=d["Model"], x=d["FP"] * fp_cost, name=f"△ False alarms × {fp_cost:g}", orientation="h",
                marker_color=AMBER, marker_pattern=dict(shape="/", fgcolor="rgba(255,255,255,0.6)", size=6),
                marker_line=dict(color="white", width=1.5),
                text=[f"<b>{v:g}</b>" for v in d["Illustrative cost"]], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>FP penalty: %{x:g}<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.3)
    fig.update_xaxes(title="Illustrative penalty points (lower is better)", range=[0, d["Illustrative cost"].max() * 1.15],
                     showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title=None, showgrid=False)
    return _base(fig, "Illustrative error-cost comparison", 380,
                 subtitle="Penalties are teaching assumptions, NOT real hospital costs", top=108)


def cost_scenario_lines(sc: pd.DataFrame, current_ratio: float, show_models: list[str],
                        evidence: str = "test") -> go.Figure:
    fig = go.Figure()
    for name in MODEL_ORDER:
        if name not in show_models:
            continue
        d = sc[sc["Model"] == name]
        fig.add_scatter(x=d["ratio"], y=d["cost"], mode="lines", name=name,
                        line=dict(color=MODEL_COLORS[name], width=2.4, dash=MODEL_DASH[name]),
                        hovertemplate=f"{name}<br>FN:FP = %{{x}}:1<br>penalty %{{y:.1f}}<extra></extra>")
    fig.add_vline(x=current_ratio, line=dict(color=NAVY, width=1.5, dash="dot"))
    fig.add_annotation(x=current_ratio, y=0, yref="paper", text=f"selected {current_ratio:g}:1", showarrow=False,
                       yanchor="bottom", xanchor="left", xshift=4, font=dict(size=10, color=NAVY))
    fig.update_xaxes(title="FN : FP penalty ratio (false alarms one missed cancer is 'worth')")
    if evidence == "test":
        fig.update_yaxes(title="Penalty on the 200-patient test set")
        sub = "Single held-out split · penalty = FN × ratio + FP"
    else:
        fig.update_yaxes(title="Expected penalty per 1,000 patients")
        sub = "Cross-validated error rates (600 patients × 15 folds) · lower is better"
    fig = _base(fig, "Which model wins as missed cancers become costlier?", 440, subtitle=sub, top=124)
    return fig


# =============================================================== ANN extension
def mlp_cv_bars(cvdf: pd.DataFrame, selected: str) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=cvdf["Hidden activation"], y=cvdf["CV accuracy"], name="CV accuracy", marker_color="#9FB4CC",
                error_y=dict(type="data", array=cvdf["CV accuracy sd"], color=MUTED, thickness=1.2),
                marker_line=dict(color="white", width=1.5),
                text=[f"{v:.1%}" for v in cvdf["CV accuracy"]], textposition="inside", insidetextanchor="start",
                textangle=0, textfont=dict(size=11, color=NAVY), hovertemplate="%{x}<br>CV accuracy %{y:.1%}<extra></extra>")
    fig.add_bar(x=cvdf["Hidden activation"], y=cvdf["CV recall"], name="CV malignant recall", marker_color=TEAL,
                error_y=dict(type="data", array=cvdf["CV recall sd"], color=MUTED, thickness=1.2),
                marker_line=dict(color="white", width=1.5),
                text=[f"{v:.1%}" for v in cvdf["CV recall"]], textposition="inside", insidetextanchor="start",
                textangle=0, textfont=dict(size=11, color="white"), hovertemplate="%{x}<br>CV recall %{y:.1%}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_yaxes(title="Mean score across 15 CV folds", tickformat=".0%", range=[0, 1.05])
    fig.update_xaxes(title=f"Hidden-layer activation (selected by CV: {selected})")
    return _base(fig, "Optional ANN — hidden-layer activations", 380,
                 subtitle="MLP (16, 8) · sigmoid output in every variant", top=108)


# =============================================================== significance & thresholds
def pvalue_heatmap(M: pd.DataFrame, title: str, subtitle: str, alpha: float = 0.05) -> go.Figure:
    """Pairwise p-values. Colour = significance class (not a continuous scale), text = exact p."""
    models = list(M.index)
    z = np.where(np.isnan(M.values), np.nan, (M.values < alpha).astype(float))
    txt = np.where(np.isnan(M.values), "—",
                   np.vectorize(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")(np.nan_to_num(M.values, nan=1)))
    sig = np.where(np.isnan(M.values), "", np.where(M.values < alpha, " ✱", ""))
    short = [m.replace("Logistic Regression", "Log. Regression") for m in models]
    fig = go.Figure(go.Heatmap(
        z=z, x=short, y=short, zmin=0, zmax=1, showscale=False, xgap=3, ygap=3,
        colorscale=[[0, "#EEF2F6"], [0.5, "#EEF2F6"], [0.5, "#8CC8C3"], [1, "#8CC8C3"]],
        text=np.char.add(txt.astype(str), sig.astype(str)), texttemplate="%{text}",
        textfont=dict(size=12, color=NAVY),
        hovertemplate="%{y} vs %{x}<br>p = %{text}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", title=None, showgrid=False)
    fig.update_xaxes(title=None, tickangle=-25, showgrid=False)
    fig.update_layout(plot_bgcolor="#F8FAFC")
    return _base(fig, title, 420, legend=False, subtitle=subtitle, top=100)


def cv_diff_forest(tbl: pd.DataFrame, metric_label: str) -> go.Figure:
    """Mean paired CV difference (A − B) with 95% CI from the corrected resampled t-test."""
    d = tbl.iloc[::-1]
    labels = [f"{a}  −  {b}" for a, b in zip(d["Model A"], d["Model B"])]
    sig = d["p_holm"] < 0.05
    fig = go.Figure()
    fig.add_scatter(
        x=d["mean_diff"], y=labels, mode="markers",
        marker=dict(size=11, symbol=np.where(sig, "diamond", "circle"), color=np.where(sig, TEAL, "#94A3B8"),
                    line=dict(color="white", width=1.5)),
        error_x=dict(type="data", symmetric=False, array=d["ci_high"] - d["mean_diff"],
                     arrayminus=d["mean_diff"] - d["ci_low"], color="#64748B", thickness=1.4, width=4),
        customdata=np.c_[d["p"], d["p_holm"]],
        hovertemplate="%{y}<br>mean difference %{x:+.1%}<br>p = %{customdata[0]:.4f} · Holm p = %{customdata[1]:.4f}"
                      "<extra></extra>", showlegend=False,
    )
    for lab, v, ph in zip(labels, d["ci_high"], d["p_holm"]):
        fig.add_annotation(x=v, y=lab, text=("p<sub>Holm</sub> " + ("<0.001" if ph < 0.001 else f"{ph:.3f}")),
                           showarrow=False, xanchor="left", xshift=8, font=dict(size=10, color=SLATE))
    fig.add_vline(x=0, line=dict(color=NAVY, width=1.2, dash="dot"))
    span = max(abs(d["ci_low"]).max(), abs(d["ci_high"]).max())
    fig.update_xaxes(title=f"Mean difference in CV {metric_label} (A − B), 95% corrected CI", tickformat="+.0%",
                     range=[-span * 1.15, span * 1.55], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title=None, showgrid=False)
    return _base(fig, f"Paired CV comparison — {metric_label}", max(300, 46 * len(d) + 140), legend=False,
                 subtitle="◆ teal = significant (Holm) · ● grey = not significant · CI crossing 0 = no evidence",
                 top=100)


def threshold_curve_fig(t: dict) -> go.Figure:
    c = t["curve"].sort_values("threshold")
    fig = go.Figure()
    for col, name, color, dash in [("recall", "OOF malignant recall", CORAL, "solid"),
                                   ("specificity", "OOF benign specificity", TEAL, "dash"),
                                   ("precision", "OOF malignant precision", NAVY, "dot")]:
        fig.add_scatter(x=c["threshold"], y=c[col], mode="lines", name=name, line=dict(color=color, width=2.2, dash=dash),
                        line_shape="hv", hovertemplate=f"threshold %{{x:.3f}}<br>{name}: %{{y:.1%}}<extra></extra>")
    lo, hi = c["threshold"].min(), c["threshold"].max()
    tuned_left = t["threshold"] < t["default"]  # labels point away from each other, at the bottom (clear of legend)
    for x, lab, col, away in [(t["default"], f"default {t['default']:g}", "#475569", not tuned_left),
                              (t["threshold"], f"{t.get('label', 'tuned')} {t['threshold']:.2f}", "#C2462A", tuned_left)]:
        fig.add_vline(x=x, line=dict(color=col, width=1.6, dash="dot" if lab.startswith("default") else "dash"))
        fig.add_annotation(x=x, y=0.06, yref="y", text=f"<b>{lab}</b>", showarrow=False, yanchor="bottom",
                           xanchor="right" if away else "left", xshift=-5 if away else 5,
                           bgcolor="rgba(255,255,255,0.85)", font=dict(size=11, color=col))
    pad = (hi - lo) * 0.02
    fig.update_xaxes(title="Decision threshold (malignant if P(malignant) > threshold)",
                     range=[lo - pad, hi + pad], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title="Out-of-fold metric (training data)", tickformat=".0%", range=[0, 1.08])
    # Two short subtitle lines so the text never runs past a half-width column.
    return _base(fig, f"Threshold trade-off — {t['model']}", 440,
                 subtitle=f"Score: malignant {t['source']}<br>Out-of-fold predictions, 600 training patients",
                 top=128)


# =============================================================== scaling — every model
def scaling_all_models(tbl: pd.DataFrame) -> go.Figure:
    d = tbl.set_index("Model").reindex(MODEL_ORDER)
    fig = go.Figure()
    fig.add_bar(x=d.index, y=d["Unscaled accuracy"], name="Unscaled (raw features)", marker_color="#B8C2CF",
                marker_pattern=dict(shape="x", fgcolor="#7C8A9C", size=6), marker_line=dict(color="white", width=1.5),
                text=[f"{v:.1%}" for v in d["Unscaled accuracy"]], textposition="outside", cliponaxis=False,
                hovertemplate="%{x}<br>Unscaled accuracy %{y:.1%}<extra></extra>")
    fig.add_bar(x=d.index, y=d["Scaled accuracy"], name="Scaled (StandardScaler)", marker_color=TEAL,
                marker_line=dict(color="white", width=1.5),
                text=[f"{v:.1%}" for v in d["Scaled accuracy"]], textposition="outside", cliponaxis=False,
                hovertemplate="%{x}<br>Scaled accuracy %{y:.1%}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.28)
    fig.update_traces(textfont=dict(size=11), selector=dict(type="bar"))
    fig.update_yaxes(title="Held-out test accuracy", tickformat=".0%", range=[0, 1.12])
    fig.update_xaxes(title=None, tickangle=0)
    return _base(fig, "Scaling, model by model — only the distance-based SVM-RBF changes", 400,
                 subtitle="Same split; each model fitted on raw vs standardised features", top=108)


# =============================================================== model combinations
def combos_landscape(cand: pd.DataFrame, highlight: list[str], winner: str) -> go.Figure:
    """(a) every candidate at the default 0.5 threshold; (b) at its own cost-tuned threshold."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.11,
                        subplot_titles=("(a) Default threshold 0.5 — CV accuracy vs recall",
                                        "(b) Cost-tuned thresholds — the error trade-off"))
    groups = [("Single model", "star", 15, None), ("Soft vote", "circle", 9, None), ("Stacking", "square", 11, None)]
    for typ, sym, size, _ in groups:
        for has_lin, col, nm in [(False, TEAL, "non-linear members only"), (True, "#A3ADBA", "includes a linear model")]:
            d = cand[(cand["Type"] == typ) & (cand["Has linear model"] == has_lin)]
            if d.empty:
                continue
            name = f"{typ} — {nm}" if typ == "Soft vote" else typ + ("" if not has_lin else " (linear, off-scale)")
            cd = np.c_[d["Candidate"], d["CV missed tuned"], d["CV false alarms tuned"], d["Tuned threshold"]]
            for col_i, (x, y) in enumerate([("CV accuracy @0.5", "CV recall @0.5"),
                                            ("CV false alarms tuned", "CV missed tuned")], start=1):
                fig.add_scatter(
                    x=d[x], y=d[y], mode="markers", name=name, legendgroup=name, showlegend=col_i == 1,
                    marker=dict(symbol=sym if not (has_lin and typ == "Soft vote") else "x", size=size,
                                color=(col if (typ == "Soft vote" or has_lin) else NAVY), opacity=0.85,
                                line=dict(color="white", width=1)),
                    customdata=cd, row=1, col=col_i,
                    hovertemplate="<b>%{customdata[0]}</b><br>" + (
                        "CV accuracy %{x:.1%} · recall %{y:.1%}" if col_i == 1 else
                        "tuned t = %{customdata[3]:.2f}<br>missed %{y} · false alarms %{x}") + "<extra></extra>")
    # label the single models in panel (a) so each star is identifiable
    sing = cand[(cand["Type"] == "Single model") & ~cand["Has linear model"]]
    for _, r in sing.iterrows():
        short = {"Naive Bayes": "NB", "SVM (RBF)": "SVM-RBF", "Decision Tree": "DT", "Random Forest": "RF"}.get(
            r["Candidate"], r["Candidate"])
        fig.add_annotation(x=r["CV accuracy @0.5"], y=r["CV recall @0.5"], xref="x", yref="y", text=f"<b>{short}</b>",
                           showarrow=False, yshift=14, font=dict(size=10, color=NAVY))
    # iso-cost lines in panel (b)
    xmax = float(cand.loc[~cand["Has linear model"], "CV false alarms tuned"].max()) + 8
    for c in (20, 30, 40, 60):
        xs = np.linspace(0, min(c, xmax), 50)
        fig.add_scatter(x=xs, y=(c - xs) / 10, mode="lines", line=dict(color="#CBD5E1", width=1, dash="dot"),
                        showlegend=False, hoverinfo="skip", row=1, col=2)
        fig.add_annotation(x=min(c, xmax) - 0.5, y=max((c - min(c, xmax)) / 10, 0) + 0.15, text=f"cost {c}",
                           showarrow=False, font=dict(size=9, color=MUTED), xref="x2", yref="y2", xanchor="right")
    hl = cand[cand["Candidate"].isin(highlight)]
    fig.add_scatter(x=hl["CV false alarms tuned"], y=hl["CV missed tuned"], mode="markers", name="Shortlisted for stage 2",
                    marker=dict(size=19, color="rgba(0,0,0,0)", line=dict(color=AMBER, width=2.5)),
                    hoverinfo="skip", row=1, col=2)
    w = cand[cand["Candidate"] == winner].iloc[0]
    fig.add_annotation(x=w["CV false alarms tuned"], y=w["CV missed tuned"], xref="x2", yref="y2",
                       text=f"<b>Recommended</b><br>{winner} @ {w['Tuned threshold']:.2f}", ax=60, ay=-46,
                       showarrow=True, arrowhead=0, arrowcolor=NAVY, bgcolor="rgba(255,255,255,0.92)",
                       bordercolor=CORAL, borderwidth=1.5, font=dict(size=11, color=NAVY))
    lo = cand.loc[~cand["Has linear model"], ["CV accuracy @0.5", "CV recall @0.5"]].min() - 0.01
    fig.update_xaxes(title="CV accuracy (600 training patients)", tickformat=".0%", range=[lo.iloc[0], 1.0],
                     showgrid=True, gridcolor=GRID, row=1, col=1)
    fig.update_yaxes(title="CV malignant recall", tickformat=".0%", range=[lo.iloc[1], 1.005], row=1, col=1)
    fig.update_xaxes(title="CV false alarms (of 323 benign)", range=[-1, xmax], showgrid=True, gridcolor=GRID, row=1, col=2)
    fig.update_yaxes(title="CV missed cancers (of 277 malignant)", range=[-0.5, 8], row=1, col=2)
    fig.update_annotations(selector=dict(text="(a) Default threshold 0.5 — CV accuracy vs recall"), font=dict(size=12, color=MUTED))
    fig.update_annotations(selector=dict(text="(b) Cost-tuned thresholds — the error trade-off"), font=dict(size=12, color=MUTED))
    fig = _base(fig, "57 candidates: 6 single models, 50 soft-vote combinations, 1 stacked ensemble", 520,
                subtitle="Stage 1 · 5-fold out-of-fold predictions · combinations containing a linear model hidden from the zoomed ranges",
                top=140)
    fig.update_layout(legend=dict(y=1.11, font=dict(size=10)))
    return fig


def fresh_fold_chart(s2: pd.DataFrame, winner: str) -> go.Figure:
    """Stage-2 chart. The p-value sits in each row's axis label, so nothing can clip at narrow widths."""
    d = s2.iloc[::-1]
    xmax = float(max(max(r.max() for r in s2["_per_repeat"]), (s2["Fresh cost (mean)"] + s2["Fresh cost (sd)"]).max()))
    fig = go.Figure()
    for _, r in d.iterrows():
        is_w, is_rf = r["Candidate"] == winner, r["Candidate"] == "Random Forest"
        p = r.get("p vs winner")
        tag = "<b>winner</b>" if is_w else (f"p = {p:.2f}" if pd.notna(p) else "")
        name = r["Candidate"].replace("Stack: NB + SVM-RBF + RF → LR", "Stack (NB, SVM-RBF, RF)")
        lab = f"{'<b>' if is_w else ''}{name} @ {r['Threshold']:.2f}{'</b>' if is_w else ''}  ·  {tag}"
        col = CORAL if is_w else (NAVY if is_rf else "#94A3B8")
        fig.add_scatter(x=r["_per_repeat"], y=[lab] * len(r["_per_repeat"]), mode="markers", showlegend=False,
                        marker=dict(size=7, color=col, opacity=0.45), hovertemplate="repeat cost %{x}<extra></extra>")
        fig.add_scatter(x=[r["Fresh cost (mean)"]], y=[lab], mode="markers", showlegend=False,
                        marker=dict(size=15, symbol="diamond", color=col, line=dict(color="white", width=1.5)),
                        error_x=dict(type="data", array=[r["Fresh cost (sd)"]], color=col, thickness=1.5, width=5),
                        hovertemplate=f"<b>{r['Candidate']} @ {r['Threshold']:.2f}</b><br>mean cost "
                                      f"{r['Fresh cost (mean)']:.1f} ± {r['Fresh cost (sd)']:.1f}<br>missed "
                                      f"{r['Fresh missed (mean)']:.1f} · false alarms {r['Fresh false alarms (mean)']:.1f}"
                                      "<extra></extra>")
    fig.update_xaxes(title="Cost per repeat = 10 × missed + false alarms (lower is better)",
                     showgrid=True, gridcolor=GRID, range=[0, xmax * 1.08])
    fig.update_yaxes(title=None, showgrid=False, automargin=True)
    return _base(fig, "Stage 2 — fresh CV check of the shortlist", max(360, 44 * len(s2) + 140),
                 legend=False, subtitle="10 repeats × 5 folds, thresholds fixed · dots = repeats, ◆ = mean ± sd · "
                                        "p = corrected t-test vs winner", top=100)


def signed_bars(values: pd.Series, title: str, xlabel: str, subtitle: str | None = None,
                errors: pd.Series | None = None) -> go.Figure:
    s = values.sort_values()
    fig = go.Figure(go.Bar(
        x=s.values, y=[FEATURE_LABELS.get(i, i) for i in s.index], orientation="h",
        marker_color=[AMBER if i in ("cell_size", "cell_shape") else (CORAL if v > 0 else "#2A7DB8")
                      for i, v in s.items()],
        marker_line=dict(color="white", width=1.5),
        error_x=None if errors is None else dict(type="data", array=errors.reindex(s.index).values, color=MUTED,
                                                 thickness=1.2, width=4),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color="#94A3B8", line_width=1)
    fig.update_xaxes(title=xlabel, showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title=None, showgrid=False)
    return _base(fig, title, 340, legend=False, subtitle=subtitle)


# =============================================================== calibration
CAL_STYLE = {  # name -> (colour, marker, dash)
    "Random Forest": (MODEL_COLORS["Random Forest"], "circle", "solid"),
    "SVM (RBF)": (MODEL_COLORS["SVM (RBF)"], "square", "longdash"),
    "Naive Bayes": (MODEL_COLORS["Naive Bayes"], "triangle-up", "dash"),
    "Decision Tree": (MODEL_COLORS["Decision Tree"], "pentagon", "longdashdot"),
    "Logistic Regression": (MODEL_COLORS["Logistic Regression"], "square", "dot"),
    "SVM (Linear)": (MODEL_COLORS["SVM (Linear)"], "x", "dashdot"),
}


def reliability_fig(cal: dict, names: list[str], part: str = "oof", winner: str | None = None) -> go.Figure:
    """Reliability diagram: mean predicted probability vs observed malignant share per bin.

    Marker area grows with the number of patients in the bin, so sparse bins read as less certain.
    """
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                    line=dict(color="#94A3B8", dash="dot", width=1.5), hoverinfo="skip")
    for n in names:
        col, sym, dash = CAL_STYLE.get(n, (CORAL if n == winner else NAVY, "diamond" if n == winner else "triangle-up",
                                           "solid" if n == winner else "dash"))
        s = cal[n][part]
        r = s["reliability"]
        fig.add_scatter(
            x=r["mean_pred"], y=r["frac_pos"], mode="lines+markers", name=f"{n} (ECE {s['ece']:.3f})",
            line=dict(color=col, dash=dash, width=2),
            marker=dict(symbol=sym, color=col, size=np.clip(6 + np.sqrt(r["n"]) * 1.6, 7, 26),
                        line=dict(color="white", width=1)),
            customdata=np.c_[r["n"], r["lo"], r["hi"]],
            hovertemplate=f"<b>{n}</b><br>bin %{{customdata[1]:.1f}}–%{{customdata[2]:.1f}} · %{{customdata[0]}} patients"
                          "<br>mean predicted %{x:.2f}<br>observed malignant %{y:.2f}<extra></extra>")
    fig.update_xaxes(title="Mean predicted probability of malignancy (bin)", range=[-0.02, 1.02], tickformat=".1f",
                     showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title="Observed share malignant", range=[-0.03, 1.03], tickformat=".1f",
                     showgrid=True, gridcolor=GRID)
    src = "Out-of-fold, 600 training patients" if part == "oof" else "Held-out test, 200 patients"
    fig = _base(fig, "Reliability diagram (diagonal = perfect)", 500 + 22 * len(names),
                subtitle=f"{src} · 10 bins · marker size = patients in bin", top=92)
    # Legend BELOW the plot area so it can never collide with the title, subtitle or the curves.
    fig.update_layout(legend=dict(orientation="v", yanchor="top", y=-0.2, xanchor="left", x=0.0,
                                  bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                      margin=dict(b=40 + 22 * (len(names) + 1)))
    return fig
