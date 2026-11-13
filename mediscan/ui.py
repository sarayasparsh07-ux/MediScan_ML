"""Presentation layer for the Streamlit app: stylesheet, the 'field of view' specimen graphic, colour maps.

Design idea: the dashboard borrows from the H&E-stained biopsy slide. Haematoxylin violet and eosin rose on
pale slide glass; the signature graphic is the dataset itself seen through a microscope's circular field,
where the malignant ring around the benign core is visible at a glance.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .config import AMBER, BENIGN_COLOR, EOSIN, GLASS, HAEM, INK, MUTED, OK_GREEN, SLATE, WASH

# ------------------------------------------------------------------ colour maps for pandas Styler tables
CMAP_EOSIN = LinearSegmentedColormap.from_list("ms_eosin", ["#FFFFFF", "#F6CBD5", "#E67F98", EOSIN])
CMAP_HAEM = LinearSegmentedColormap.from_list("ms_haem", ["#FFFFFF", "#DCD5EE", "#9F92D0", HAEM])
CMAP_DIVERGING = LinearSegmentedColormap.from_list("ms_div", [BENIGN_COLOR, "#B7C8E6", "#F7F5FB", "#F0B3C1", EOSIN])


# ------------------------------------------------------------------ the specimen field
def field_svg(x, y, label, centre, radius: float | None = None, size: int = 400, animate: bool = True,
              max_points: int | None = None, title: str = "") -> str:
    """Patients drawn inside a circular microscope field, centred on the benign cluster.

    x, y    raw cell_size and cell_shape values; label 0 = benign, 1 = malignant
    centre  (cx, cy) of the benign cluster in raw units
    radius  optional decision radius (raw units) drawn as a dashed circle
    """
    x, y, label = np.asarray(x, float), np.asarray(y, float), np.asarray(label, int)
    if max_points and len(x) > max_points:
        idx = np.random.default_rng(0).choice(len(x), max_points, replace=False)
        x, y, label = x[idx], y[idx], label[idx]
    cx, cy = float(centre[0]), float(centre[1])
    half = size / 2
    field_r = half * 0.92
    reach = np.percentile(np.hypot(x - cx, y - cy), 99.5)
    s = field_r * 0.97 / reach

    def px(vx, vy):
        return half + (vx - cx) * s, half - (vy - cy) * s

    small = size < 120
    dot = 1.9 if small else 3.0
    benign, malig = [], []
    for xi, yi, li in zip(x, y, label):
        X, Y = px(xi, yi)
        if li == 0:
            benign.append(f"<circle cx='{X:.1f}' cy='{Y:.1f}' r='{dot:.1f}'/>")
        else:
            d = dot * 1.25
            malig.append(f"<path d='M{X:.1f} {Y - d:.1f}L{X + d:.1f} {Y:.1f}L{X:.1f} {Y + d:.1f}L{X - d:.1f} {Y:.1f}Z'/>")
    uid = f"f{size}"
    anim = " ms-anim" if animate else ""
    parts = [
        f"<svg class='ms-field{anim}' viewBox='0 0 {size} {size}' role='img' aria-label='{title}' "
        f"xmlns='http://www.w3.org/2000/svg'>",
        f"<title>{title}</title>" if title else "",
        "<defs>",
        f"<radialGradient id='{uid}g' cx='50%' cy='46%' r='55%'>"
        "<stop offset='0%' stop-color='#FFFFFF'/><stop offset='70%' stop-color='#FBF7FB'/>"
        "<stop offset='100%' stop-color='#EFE3EE'/></radialGradient>",
        f"<clipPath id='{uid}c'><circle cx='{half}' cy='{half}' r='{field_r:.1f}'/></clipPath>",
        "</defs>",
        f"<circle cx='{half}' cy='{half}' r='{half - 1:.1f}' fill='{INK}'/>",
        f"<circle cx='{half}' cy='{half}' r='{field_r:.1f}' fill='url(#{uid}g)'/>",
        f"<g clip-path='url(#{uid}c)'>",
    ]
    if not small:  # reticle and scale ticks: one tick per raw unit along both axes
        sw = 0.8
        parts.append(f"<g stroke='{INK}' stroke-opacity='0.18' stroke-width='{sw}'>"
                     f"<line x1='0' y1='{half}' x2='{size}' y2='{half}'/>"
                     f"<line x1='{half}' y1='0' x2='{half}' y2='{size}'/></g>")
        ticks = []
        for k in range(-12, 13):
            if k == 0:
                continue
            t = half + k * s
            h = 7 if k % 5 == 0 else 4
            ticks.append(f"<line x1='{t:.1f}' y1='{half - h}' x2='{t:.1f}' y2='{half + h}'/>")
            ticks.append(f"<line x1='{half - h}' y1='{t:.1f}' x2='{half + h}' y2='{t:.1f}'/>")
        parts.append(f"<g stroke='{INK}' stroke-opacity='0.28' stroke-width='{sw}'>{''.join(ticks)}</g>")
    parts.append(f"<g class='ms-b' fill='{BENIGN_COLOR}' fill-opacity='0.78'>{''.join(benign)}</g>")
    parts.append(f"<g class='ms-m' fill='{EOSIN}' fill-opacity='0.82'>{''.join(malig)}</g>")
    if radius is not None:
        parts.append(f"<circle class='ms-r' cx='{half}' cy='{half}' r='{radius * s:.1f}' fill='none' stroke='{INK}' "
                     f"stroke-width='{1.6 if not small else 1}' stroke-dasharray='6 5'/>")
    parts.append("</g>")
    parts.append(f"<circle cx='{half}' cy='{half}' r='{field_r:.1f}' fill='none' stroke='{INK}' "
                 f"stroke-opacity='0.35' stroke-width='{3 if not small else 1.5}'/>")
    parts.append("</svg>")
    return "".join(parts)


def best_radius(r_train, y_train) -> tuple[float, float]:
    """The single radius (benign inside, malignant outside) with the fewest training errors."""
    r, yv = np.asarray(r_train, float), np.asarray(y_train, int)
    cands = np.unique(np.round(r, 3))
    errs = [int(((r > c) != (yv == 1)).sum()) for c in cands]
    i = int(np.argmin(errs))
    return float(cands[i]), 1 - errs[i] / len(r)


# ------------------------------------------------------------------ stylesheet
CSS = f"""
<style>
:root {{
  --ink: {INK}; --haem: {HAEM}; --eosin: {EOSIN}; --glass: {GLASS}; --wash: {WASH};
  --slate: {SLATE}; --muted: {MUTED}; --amber: {AMBER}; --benign: {BENIGN_COLOR}; --ok: {OK_GREEN};
  --rule: #DDD6EA; --display: 'MediScan Display', 'MediScan Sans', sans-serif;
}}
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1380px; }}
h1, h2, h3, h4 {{ color: var(--ink); font-family: var(--display); }}
h2 {{ letter-spacing: -0.02em; }}
h3 {{ letter-spacing: -0.012em; font-weight: 650 !important; margin-top: 1.6rem !important; }}
p, li {{ line-height: 1.55; }}
[data-testid="stMarkdownContainer"] strong {{ color: var(--ink); }}

/* ---------- page header */
.ms-page {{ margin: 0 0 1.4rem; padding-bottom: 1.1rem; border-bottom: 1px solid var(--rule); }}
.ms-page h1 {{ font-size: clamp(2rem, 3.4vw, 2.75rem); font-weight: 720; letter-spacing: -0.03em; line-height: 1.05;
  margin: 0; padding: 0; }}
.ms-page .deck {{ font-family: var(--display); font-size: 1.18rem; font-weight: 400; color: var(--muted);
  margin-top: 0.35rem; letter-spacing: -0.005em; }}
.ms-page .lede {{ max-width: 72ch; margin-top: 0.7rem; font-size: 1.02rem; color: var(--slate); }}

/* ---------- hero (Start Here) */
.ms-hero [data-testid="stHeaderActionElements"], .ms-page [data-testid="stHeaderActionElements"] {{ display: none; }}
.ms-hero h1 {{ font-size: clamp(2.1rem, 4vw, 3.3rem); font-weight: 760; letter-spacing: -0.035em; line-height: 1.0;
  color: var(--ink); margin: 0.2rem 0 0; padding: 0; }}
.ms-hero h1 .soft {{ display: block; font-weight: 380; color: var(--haem); letter-spacing: -0.03em; }}
.ms-hero .lede {{ font-size: 1.1rem; max-width: 56ch; margin: 1.1rem 0 1.2rem; color: var(--slate); }}
.ms-hero .lede b {{ color: var(--eosin); }}
.ms-figure {{ margin: 0; text-align: center; }}
.ms-figure svg.ms-field {{ width: 100%; max-width: 430px; height: auto; display: block; margin: 0 auto;
  filter: drop-shadow(0 18px 30px rgba(35, 26, 59, 0.18)); }}
.ms-figure figcaption {{ font-size: 0.86rem; color: var(--muted); max-width: 44ch; margin: 0.8rem auto 0; line-height: 1.45; }}
.ms-key {{ display: inline-flex; gap: 1rem; margin-top: 0.5rem; font-size: 0.84rem; color: var(--slate); }}
.ms-key i {{ display: inline-block; width: 9px; height: 9px; margin-right: 6px; vertical-align: 0; }}
.ms-key .kb {{ background: var(--benign); border-radius: 50%; }}
.ms-key .km {{ background: var(--eosin); transform: rotate(45deg) scale(0.9); }}
.ms-key .kr {{ border: 1.5px dashed var(--ink); border-radius: 50%; width: 10px; height: 10px; }}
.ms-anim .ms-b {{ animation: ms-stain 0.9s ease-out 0.1s both; }}
.ms-anim .ms-m {{ animation: ms-stain 1.1s ease-out 0.7s both; }}
.ms-anim .ms-r {{ animation: ms-stain 0.8s ease-out 1.6s both; }}
@keyframes ms-stain {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
@media (prefers-reduced-motion: reduce) {{ .ms-anim .ms-b, .ms-anim .ms-m, .ms-anim .ms-r {{ animation: none; }} }}

/* ---------- verdicts (recommendation blocks) */
.rec-card {{ background: white; border: 1px solid var(--rule); border-left: 5px solid var(--haem);
  border-radius: 4px 10px 10px 4px; padding: 14px 18px 14px 18px; margin-bottom: 12px; }}
.rec-card .kicker {{ font-size: 0.86rem; color: var(--muted); font-weight: 700; }}
.rec-card .name {{ font-family: var(--display); font-size: 1.62rem; font-weight: 700; color: var(--ink);
  letter-spacing: -0.02em; line-height: 1.15; margin-top: 2px; }}
.rec-card p {{ font-size: 0.93rem; color: var(--slate); line-height: 1.5; }}
.rec-card b {{ color: var(--ink); }}

/* ---------- figure strip (key numbers) */
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); margin: 0.6rem 0 1.3rem;
  background: white; border: 1px solid var(--rule); border-radius: 10px; overflow: hidden; }}
.kpi {{ padding: 14px 18px 15px; border-left: 1px solid var(--rule); margin-left: -1px; }}
.kpi .label {{ font-size: 0.86rem; color: var(--muted); font-weight: 700; }}
.kpi .value {{ font-family: var(--display); font-size: 1.7rem; font-weight: 680; color: var(--ink);
  letter-spacing: -0.025em; line-height: 1.12; margin-top: 3px; font-variant-numeric: tabular-nums; }}
.kpi.coral .value {{ color: var(--eosin); }}
.kpi.amber .value {{ color: #A8730F; }}
.kpi .sub {{ font-size: 0.84rem; color: var(--slate); margin-top: 4px; line-height: 1.38; overflow-wrap: anywhere; }}

/* ---------- findings */
.callout {{ position: relative; background: white; border: 1px solid var(--rule); border-left: 4px solid var(--haem);
  border-radius: 3px 8px 8px 3px; padding: 12px 16px; margin: 10px 0 16px; font-size: 0.97rem; line-height: 1.55;
  color: var(--slate); max-width: 110ch; }}
.callout.info {{ border-left-color: var(--benign); }}
.callout.caution {{ border-left-color: var(--amber); background: #FFFCF5; }}
.callout.warn {{ border-left-color: var(--eosin); background: #FFF8FA; }}
.callout b {{ color: var(--ink); }}

.para {{ background: white; border: 1px solid var(--rule); border-radius: 8px; padding: 22px 28px;
  font-family: Georgia, 'Times New Roman', serif; font-size: 1.04rem; line-height: 1.72; color: #2A2440; max-width: 82ch; }}
.badge {{ display:inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.78rem; font-weight: 700; }}
.cmsum {{ display:grid; grid-template-columns: 1fr 1fr; gap: 0; margin-top: 8px; border: 1px solid var(--rule);
  border-radius: 10px; overflow: hidden; background: white; }}
.cmsum > div {{ padding: 12px 14px; border-right: 1px solid var(--rule); border-bottom: 1px solid var(--rule); }}
.cmsum > div:nth-child(2n) {{ border-right: none; }}
.cmsum > div:nth-child(n+3) {{ border-bottom: none; }}
.cmsum .fn {{ background: #FFF1F4; box-shadow: inset 0 0 0 2px var(--eosin); }}
.cmsum .fp {{ background: #FFF9EE; box-shadow: inset 0 0 0 2px var(--amber); }}
.cmsum .t {{ font-size: 0.84rem; color: var(--muted); font-weight: 700; }}
.cmsum .v {{ font-family: var(--display); font-size: 1.7rem; font-weight: 700; color: var(--ink); }}
.cmsum .fn .v {{ color: var(--eosin); }}
.ms-say {{ border-top: 2px solid var(--ink); padding-top: 0.6rem; }}
.ms-say h4 {{ font-size: 1rem; margin: 0 0 0.3rem; padding: 0; }}
.ms-say ul {{ padding-left: 1.1rem; margin: 0; }}
.ms-say li {{ margin-bottom: 0.4rem; font-size: 0.97rem; }}
.ms-next {{ font-size: 0.86rem; color: var(--muted); margin-top: 0.4rem; }}

/* ---------- sidebar */
section[data-testid="stSidebar"] {{ border-right: none; }}
section[data-testid="stSidebar"] .ms-brand {{ display: flex; align-items: center; gap: 12px; margin: 0.2rem 0 1.4rem; }}
section[data-testid="stSidebar"] .ms-brand svg {{ width: 46px; height: 46px; flex: none; }}
section[data-testid="stSidebar"] .ms-brand .w {{ font-family: var(--display); font-size: 1.42rem; font-weight: 720;
  letter-spacing: -0.02em; color: #FFFFFF; line-height: 1; }}
section[data-testid="stSidebar"] .ms-brand .s {{ font-size: 0.8rem; color: #B9AFD3; margin-top: 4px; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] {{
  padding: 6px 10px 6px 12px; margin: 0; border-radius: 0 6px 6px 0; width: 100%; border-left: 3px solid transparent;
  transition: background 120ms ease; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child {{ display: none; }}
section[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 0; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] p {{ font-size: 0.97rem; color: #D9D2EC; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover {{ background: rgba(255,255,255,0.06); }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] {{
  background: rgba(240,138,163,0.13); border-left-color: #F08AA3; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] p {{
  color: #FFFFFF; font-weight: 700; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:focus-visible) {{
  outline: none; box-shadow: inset 0 0 0 2px rgba(240,138,163,0.55); }}
section[data-testid="stSidebar"] .ms-group {{ font-size: 0.8rem; color: #9D93BC; margin: 1.3rem 0 0.4rem 2px; font-weight: 700; }}
section[data-testid="stSidebar"] dl {{ display: grid; grid-template-columns: auto 1fr; gap: 3px 12px; font-size: 0.86rem;
  margin: 0; }}
section[data-testid="stSidebar"] dt {{ color: #9D93BC; }}
section[data-testid="stSidebar"] dd {{ color: #E4DEF2; margin: 0; }}
section[data-testid="stSidebar"] .ms-note {{ font-size: 0.8rem; color: #9D93BC; line-height: 1.45; margin-top: 1.4rem;
  border-top: 1px solid #3B3160; padding-top: 0.9rem; }}
section[data-testid="stSidebar"] hr {{ border-color: #3B3160; }}

/* ---------- widgets */
div[data-testid="stExpander"] details {{ background: white; border-radius: 8px; }}
div[data-testid="stExpander"] summary p {{ font-weight: 700; color: var(--ink); }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--rule); }}
.stTabs [data-baseweb="tab"] {{ font-weight: 700; padding: 8px 12px; }}
.stTabs [data-baseweb="tab"] p {{ font-size: 0.98rem; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: var(--muted) !important; opacity: 1; }}
h4 {{ font-size: 1.12rem !important; font-weight: 650 !important; letter-spacing: -0.01em; }}
[data-testid="stPlotlyChart"] {{ margin-bottom: 0.4rem; }}
:focus-visible {{ outline-color: var(--haem); }}

@media (max-width: 760px) {{
  .block-container, [data-testid="stMainBlockContainer"] {{ padding-left: 1rem; padding-right: 1rem; }}
  .kpi .value {{ font-size: 1.45rem; }}
  .kpi {{ border-top: 1px solid var(--rule); margin-top: -1px; }}
  .ms-figure svg.ms-field {{ max-width: 320px; }}
}}
</style>
"""
