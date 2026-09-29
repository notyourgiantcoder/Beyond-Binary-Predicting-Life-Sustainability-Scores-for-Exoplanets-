"""
Beyond Binary: Estimating life sustainability scores in Exoplanets — Streamlit dashboard.

Run from the project root:
    .venv/bin/streamlit run app.py
"""

import difflib
import os
import random
import re
import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import RobustScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_CSV        = os.path.join(ROOT, "data/PSCompPars_2026.03.05_02.02.54.csv")
COMPARISON_CSV = os.path.join(ROOT, "outputs/model_comparison_with_ensemble.csv")
WATCHLIST_CSV  = os.path.join(ROOT, "outputs/full_planet_watchlist.csv")
LEADERBOARD_PNG = os.path.join(ROOT, "outputs/step10_leaderboard.png")
ACT_VS_PRED_PNG = os.path.join(ROOT, "outputs/step9_actual_vs_predicted.png")
X_TRAIN_CSV    = os.path.join(ROOT, "TRAIN_TEST/X_train.csv")
XGB_MODEL      = os.path.join(ROOT, "model_xgboost.pkl")

FEATURE_COLS = ["pl_rade", "pl_dens", "pl_orbeccen", "pl_eqt",
                "pl_orbsmax", "st_teff", "st_mass", "st_age"]

LSS_COMPONENTS = [
    # key,              name,                   weight, what it measures
    ("hz_score",        "Habitable Zone",       0.35, "Position in the habitable zone"),
    ("temp_score",      "Temperature",          0.25, "Closeness to liquid-water range"),
    ("retention_score", "Atmosphere Retention", 0.20, "Can hold an atmosphere"),
    ("orbit_score",     "Orbital Stability",    0.10, "Low eccentricity → stable climate"),
    ("stellar_score",   "Stellar Suitability",  0.10, "Host star temperature, mass, age"),
]

EARTH_LSS = 0.9340

# The models were trained on log1p of these columns (sliders are in physical units)
LOG_COLS = ["pl_rade", "pl_dens", "pl_orbeccen", "pl_eqt", "pl_orbsmax", "st_age"]

# What-If sliders: key, label, unit, min, max, default, step, display format
WHATIF_SLIDERS = [
    ("pl_rade",     "Planet Radius",           "R⊕",    0.51,  23.2,  1.0,   0.01,  "%.2f"),
    ("pl_dens",     "Planet Density",          "g/cm³", 0.01,  24.4,  5.51,  0.01,  "%.2f"),
    ("pl_orbeccen", "Orbital Eccentricity",    "",      0.0,   0.58,  0.0,   0.001, "%.3f"),
    ("pl_eqt",      "Equilibrium Temperature", "K",     50,    2386,  255,   1,     "%d"),
    ("pl_orbsmax",  "Semi-Major Axis",         "AU",    0.005, 4.6,   1.0,   0.001, "%.3f"),
    ("st_teff",     "Stellar Temperature",     "K",     2960,  8720,  5778,  1,     "%d"),
    ("st_mass",     "Stellar Mass",            "M☉",    0.09,  2.28,  1.0,   0.01,  "%.2f"),
    ("st_age",      "Stellar Age",             "Gyr",   0.0,   14.0,  4.6,   0.1,   "%.1f"),
]
SLIDER = {s[0]: s for s in WHATIF_SLIDERS}
EARTH_PRESET = {"pl_rade": 1.0, "pl_dens": 5.51, "pl_orbeccen": 0.017, "pl_eqt": 255,
                "pl_orbsmax": 1.0, "st_teff": 5778, "st_mass": 1.0, "st_age": 4.6}
KEPLER442B_PRESET = {"pl_rade": 1.34, "pl_dens": 5.39, "pl_orbeccen": 0.04, "pl_eqt": 241,
                     "pl_orbsmax": 0.409, "st_teff": 5200, "st_mass": 0.61, "st_age": 5.0}
MARS_PRESET = {"pl_rade": 0.53, "pl_dens": 3.93, "pl_orbeccen": 0.093, "pl_eqt": 210,
               "pl_orbsmax": 1.524, "st_teff": 5778, "st_mass": 1.0, "st_age": 4.6}
HOT_JUPITER_PRESET = {"pl_rade": 11.2, "pl_dens": 0.6, "pl_orbeccen": 0.02, "pl_eqt": 1500,
                      "pl_orbsmax": 0.05, "st_teff": 6100, "st_mass": 1.15, "st_age": 4.0}
DEFAULT_PRESET = {k: d for k, *_, d, _s, _f in WHATIF_SLIDERS}

# Friendly names and plain-English tooltips for the What-If sliders
FRIENDLY = {
    "pl_rade": "Radius", "pl_dens": "Density", "pl_orbeccen": "Eccentricity",
    "pl_eqt": "Temperature", "pl_orbsmax": "Orbit size", "st_teff": "Star temp",
    "st_mass": "Star mass", "st_age": "Star age",
}
HELP = {
    "pl_rade": "How big the planet is, in Earth radii. Above ~1.6 R⊕ planets tend to be "
               "gas-rich mini-Neptunes rather than rocky worlds.",
    "pl_dens": "Average density. Rocky planets are ~4–6 g/cm³; gas giants are closer to 1.",
    "pl_orbeccen": "0 = perfectly circular orbit; higher = more stretched, wilder seasons.",
    "pl_eqt": "Temperature the planet would have with no atmosphere, from starlight alone. "
              "Earth is 255 K (a greenhouse effect warms it to ~288 K).",
    "pl_orbsmax": "Average distance from the star, in AU (1 AU = Earth–Sun distance).",
    "st_teff": "Surface temperature of the host star. The Sun is 5,778 K; red dwarfs are "
               "cooler, blue-white stars hotter.",
    "st_mass": "Mass of the host star in solar masses. Heavier stars burn out faster.",
    "st_age": "Age of the host star. Complex life on Earth took ~4 billion years.",
}

# ── Palette ───────────────────────────────────────────────────────────────────
CYAN, VIOLET, AMBER = "#00e5ff", "#8b5cf6", "#ffb547"
INK, INK_MUTED = "#e6ecff", "#8fa0c8"
GRID, TRACK = "rgba(230,236,255,0.08)", "rgba(230,236,255,0.06)"
RED, ORANGE, YELLOW, GREEN = "#ff5c7a", "#ff8a3d", "#ffd23f", "#2ee59d"
FONT = "Inter, sans-serif"

# One height per chart role, one margin for every chart
CHART_MARGIN = dict(l=20, r=20, t=20, b=20)
BAR_H, PCT_H, GAUGE_H, RADAR_H = 240, 140, 260, 440


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Beyond Binary", page_icon="🪐", layout="wide")


def _stars(n: int, seed: int, color: str) -> str:
    """box-shadow list of n random stars on a 2560×2000 tile."""
    rng = random.Random(seed)
    return ", ".join(f"{rng.randint(0, 2560)}px {rng.randint(0, 2000)}px {color}" for _ in range(n))


STARS_SM = _stars(420, 1, "rgba(230,236,255,0.75)")
STARS_MD = _stars(140, 2, "rgba(170,220,255,0.9)")
STARS_LG = _stars(45, 3, "#ffffff")
STARFIELD_HTML = ('<div class="starfield"><div class="stars s1"></div>'
                  '<div class="stars s2"></div><div class="stars s3"></div></div>')

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@400;500;600;700&display=swap');

/* ── Design tokens ─────────────────────────────────────────────────────── */
:root {{
  --cyan:{CYAN}; --violet:{VIOLET}; --amber:{AMBER};
  --ink:{INK}; --muted:{INK_MUTED};
  --glass:rgba(255,255,255,0.04); --glass-border:rgba(255,255,255,0.08); --glass-hover:rgba(0,229,255,0.28);
  --glow: 0 0 0 1px rgba(0,229,255,0.08), 0 8px 32px rgba(0,229,255,0.10);
  --space-1: 4px; --space-2: 8px; --space-3: 16px; --space-4: 24px; --space-5: 32px; --space-6: 48px;
  --radius: 16px; --radius-sm: 12px; --radius-pill: 999px;
  --fs-page: clamp(1.75rem, 3.2vw, 2.5rem);   /* page title   */
  --fs-section: 1.05rem;                       /* section title */
  --fs-card: .95rem;                           /* card title    */
  --fs-body: .9375rem;                         /* body          */
  --fs-caption: .8125rem;                      /* caption       */
  --fs-kpi: 1.75rem;
  --t: 200ms ease;
}}

/* ── Space background + starfield ─────────────────────────────────────── */
.stApp {{
  background: radial-gradient(ellipse at 20% 0%, rgba(139,92,246,0.18), transparent 55%),
              radial-gradient(ellipse at 90% 100%, rgba(0,229,255,0.10), transparent 50%),
              linear-gradient(160deg, #05070f 0%, #0b1026 55%, #150d2e 100%) fixed;
  isolation: isolate; color: var(--ink);
}}
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"],
[data-testid="stBottomBlockContainer"] {{ background: transparent !important; }}
.starfield {{ position: fixed; inset: 0; z-index: -1; pointer-events: none; overflow: hidden; }}
.stars {{ position: absolute; top: 0; left: 0; border-radius: 50%; background: transparent; }}
.stars::after {{ content: ""; position: absolute; top: 2000px; left: 0; border-radius: 50%;
                 width: inherit; height: inherit; box-shadow: inherit; }}
.s1 {{ width: 1px; height: 1px; box-shadow: {STARS_SM};
       animation: drift 240s linear infinite, twinkle 5s ease-in-out infinite alternate; }}
.s2 {{ width: 2px; height: 2px; box-shadow: {STARS_MD};
       animation: drift 160s linear infinite, twinkle 3.5s ease-in-out infinite alternate-reverse; }}
.s3 {{ width: 3px; height: 3px; box-shadow: {STARS_LG};
       animation: drift 110s linear infinite, twinkle 2.5s ease-in-out infinite alternate; }}
@keyframes drift {{ from {{ transform: translateY(0); }} to {{ transform: translateY(-2000px); }} }}
@keyframes twinkle {{ from {{ opacity: .35; }} to {{ opacity: 1; }} }}

/* Invisible helper elements (CSS, starfield, markers) take no layout space */
[data-testid="stElementContainer"]:has(.starfield),
[data-testid="stElementContainer"]:has(.boot-fade),
[data-testid="stElementContainer"]:has(style) {{ position: absolute; width: 0; height: 0; overflow: visible; }}

/* ── Chrome + page frame ───────────────────────────────────────────────── */
#MainMenu, footer, [data-testid="stMainMenu"], [data-testid="stToolbarActions"], [data-testid="stDecoration"],
[data-testid="stAppDeployButton"] {{ display: none !important; }}
[data-testid="stHeader"] {{ height: var(--space-6); }}
.block-container {{ max-width: 1200px; margin: 0 auto;
                   padding: var(--space-5) var(--space-5) var(--space-5) var(--space-5) !important; }}
@media (max-width: 640px) {{
  .block-container {{ padding: calc(var(--space-6) + var(--space-2)) var(--space-3) var(--space-5) var(--space-3) !important; }}
}}
html, body, .stApp, p, label, input, textarea {{ font-family: 'Inter', sans-serif; }}
.stApp p, .stApp li {{ font-size: var(--fs-body); }}
h1, h2, h3 {{ font-family: 'Orbitron', sans-serif !important; letter-spacing: .02em; color: var(--ink); }}
a {{ color: var(--cyan); }}
/* Streamlit pulls every markdown block up 16px to cancel <p> margins; our HTML blocks have none */
[data-testid="stMarkdownContainer"]:has(> div:first-child, > ol:first-child) {{ margin-bottom: 0 !important; }}

/* Thin themed scrollbars */
* {{ scrollbar-width: thin; scrollbar-color: rgba(139,92,246,0.5) transparent; }}
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: linear-gradient(var(--cyan), var(--violet)); border-radius: var(--radius-pill); }}

/* One-time fade-in after the splash: opacity only, so nothing shifts */
.boot-fade {{ display: none; }}
.stApp:has(.boot-fade) .block-container,
.stApp:has(.boot-fade) [data-testid="stSidebarContent"] {{ animation: appFade .8s ease-out both; }}
@keyframes appFade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

/* ── Type scale: hero, sections, cards ─────────────────────────────────── */
.hero {{ margin-bottom: var(--space-2); }}
.hero-eyebrow {{ color: var(--cyan); font-size: var(--fs-caption); font-weight: 600; letter-spacing: .22em;
                text-transform: uppercase; margin-bottom: var(--space-2); }}
.hero-title {{ font-family: 'Orbitron', sans-serif; font-weight: 900; font-size: var(--fs-page);
              line-height: 1.15; margin: 0 0 var(--space-2) 0;
              background: linear-gradient(90deg, #e6ecff 0%, var(--cyan) 45%, var(--violet) 100%);
              -webkit-background-clip: text; background-clip: text; color: transparent;
              filter: drop-shadow(0 0 18px rgba(0,229,255,0.18)); }}
.hero-title.sub-page {{ font-size: calc(var(--fs-page) * .75); margin-top: var(--space-2); }}
.hero-sub {{ color: var(--muted); font-size: 1rem; line-height: 1.6; max-width: 780px; }}
/* A section = title + accent bar (+ caption). 16px margin + 16px block gap = 32px between sections */
.section-title {{ font-family: 'Orbitron', sans-serif; font-size: var(--fs-section); font-weight: 700;
                 letter-spacing: .04em; color: var(--ink); margin-top: var(--space-3); }}
.section-title::after {{ content: ""; display: block; width: 36px; height: 2px; margin-top: var(--space-2);
                        border-radius: 2px; background: linear-gradient(90deg, var(--cyan), var(--violet));
                        box-shadow: 0 0 8px rgba(0,229,255,0.5); }}
.section-sub {{ color: var(--muted); font-size: var(--fs-caption); margin-top: var(--space-2); }}
.grp-head {{ display: flex; align-items: center; gap: var(--space-2); }}
.grp-icon {{ font-size: 1.2rem; line-height: 1; width: 1.4rem; text-align: center; }}
.grp-title {{ font-family: 'Orbitron', sans-serif; font-weight: 700; font-size: var(--fs-card); color: var(--ink); }}
.grp-cap {{ color: var(--muted); font-size: var(--fs-caption); margin-top: var(--space-1); }}

/* ── Glass cards ───────────────────────────────────────────────────────── */
.glass, [class*="st-key-card_"] {{
  background: var(--glass); border: 1px solid var(--glass-border) !important; border-radius: var(--radius);
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  transition: box-shadow var(--t), border-color var(--t);
}}
.glass:hover, [class*="st-key-card_"]:hover {{ border-color: var(--glass-hover) !important; box-shadow: var(--glow); }}
[class*="st-key-card_"] {{ padding: var(--space-4); gap: var(--space-3); }}
/* HTML surfaces inside a card don't get a second glass layer */
[class*="st-key-card_"] .glass {{ background: transparent; border: 0 !important; backdrop-filter: none; box-shadow: none; padding: 0; }}

/* Equal-height cards: columns stretch, the last card in a column fills the rest */
[data-testid="stHorizontalBlock"]:has([class*="st-key-card_"]) {{ align-items: stretch; }}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] {{ height: 100%; }}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:last-child:has(> [class*="st-key-card_"]) {{
  flex: 1 1 auto; display: flex; flex-direction: column;
}}
[data-testid="stLayoutWrapper"] > [class*="st-key-card_"] {{ flex: 1 1 auto; }}
/* In paired table+chart cards the chart sits on the bottom edge, so both charts line up */
:is(.st-key-card_lss, .st-key-card_models) > [data-testid="stElementContainer"]:last-child {{ margin-top: auto; }}
/* Results card is width-bound (gauge, radar); spread any spare height over its gaps instead of leaving a hole */
.st-key-card_results {{ justify-content: space-between; }}
[class*="st-key-card_fig"] > [data-testid="stElementContainer"]:has([data-testid="stImage"]) {{
  flex: 1 1 auto; display: flex; align-items: center; justify-content: center; }}

/* KPI cards: identical size, same slot positions */
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 150px), 1fr)); gap: var(--space-5); }}
@media (max-width: 640px) {{ .kpi-grid {{ gap: var(--space-3); }} .kpi {{ padding: var(--space-3); }} }}
.kpi {{ padding: var(--space-4); display: grid; grid-template-rows: auto auto 1fr; row-gap: var(--space-2); }}
.kpi-label {{ color: var(--muted); font-size: var(--fs-caption); font-weight: 500; }}
.kpi-value {{ font-family: 'Orbitron', sans-serif; font-size: var(--fs-kpi); font-weight: 700; color: var(--ink);
             line-height: 1.1; text-shadow: 0 0 18px rgba(0,229,255,0.25); }}
.kpi-desc {{ color: var(--muted); font-size: var(--fs-caption); align-self: end; }}

.pills {{ display: flex; flex-wrap: wrap; gap: var(--space-2); }}
.pill {{ font-size: var(--fs-caption); font-weight: 600; padding: var(--space-1) var(--space-3); border-radius: var(--radius-pill);
         border: 1px solid var(--glass-border); background: rgba(255,255,255,0.05); color: var(--ink); }}
.pill.cyan {{ border-color: rgba(0,229,255,0.4); color: var(--cyan); background: rgba(0,229,255,0.08); }}
.pill.violet {{ border-color: rgba(139,92,246,0.45); color: #c4b5fd; background: rgba(139,92,246,0.10); }}
.pill.amber {{ border-color: rgba(255,181,71,0.45); color: var(--amber); background: rgba(255,181,71,0.08); }}

.formula {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: var(--fs-caption); color: var(--cyan);
           padding: var(--space-3) var(--space-4); line-height: 1.6; }}

/* Tables: full width, fixed row rhythm, numbers right / text left */
.gtable-wrap {{ padding: var(--space-2) var(--space-3); overflow-x: auto; }}
.gtable {{ width: 100%; border-collapse: collapse; font-size: var(--fs-body); }}
.gtable th {{ color: var(--muted); font-weight: 500; text-align: left; padding: var(--space-2) var(--space-3);
              border-bottom: 1px solid var(--glass-border); font-size: var(--fs-caption); white-space: nowrap; }}
.gtable td {{ padding: var(--space-2) var(--space-3); height: 40px; border-bottom: 1px solid rgba(255,255,255,0.05);
              color: var(--ink); }}
.gtable th:first-child, .gtable td:first-child {{ padding-left: 0; }}
.gtable th:last-child, .gtable td:last-child {{ padding-right: 0; }}
.gtable tr:last-child td {{ border-bottom: none; }}
.gtable tr.best td {{ color: var(--cyan); font-weight: 600; background: rgba(0,229,255,0.06); }}
.gtable tr.best td:first-child {{ box-shadow: inset 3px 0 0 var(--cyan); padding-left: var(--space-3); }}
.gtable .r {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}

.notice {{ padding: var(--space-3) var(--space-4); border-left: 3px solid var(--cyan) !important; }}
.notice.warn {{ border-left-color: var(--amber) !important; }}
.notice.error {{ border-left-color: {RED} !important; }}
.notice.center {{ text-align: center; max-width: 720px; margin: 0 auto; padding: var(--space-4); border-left-width: 1px !important; }}
.notice.center.warn {{ border-color: rgba(255,181,71,0.45) !important; }}
.notice.center.error {{ border-color: rgba(255,92,122,0.45) !important; }}
.notice-title {{ font-weight: 600; color: var(--ink); font-size: var(--fs-body); }}
.notice-desc {{ color: var(--muted); font-size: var(--fs-caption); margin-top: var(--space-1); }}

/* ── Pipeline stage nodes ──────────────────────────────────────────────── */
.flow {{ display: flex; align-items: stretch; }}
.node {{ flex: 1 1 0; min-width: 0; padding: var(--space-4) var(--space-3); text-align: center; position: relative;
         display: flex; flex-direction: column; align-items: center; gap: var(--space-1);
         box-shadow: 0 0 24px rgba(0,229,255,0.06), inset 0 0 18px rgba(139,92,246,0.06); }}
.node::before {{ content: ""; position: absolute; top: -1px; left: 20%; right: 20%; height: 2px; border-radius: 2px;
                 background: linear-gradient(90deg, transparent, var(--cyan), transparent); opacity: .8; }}
.node-icon {{ font-size: 1.5rem; line-height: 1; margin-bottom: var(--space-1); }}
.node-idx {{ font-family: 'Orbitron', sans-serif; font-size: .66rem; color: var(--cyan); letter-spacing: .18em; }}
.node-title {{ font-weight: 600; color: var(--ink); font-size: var(--fs-body); }}
.node-val {{ font-family: 'Orbitron', sans-serif; font-weight: 700; font-size: 1.05rem; margin: var(--space-1) 0;
            color: var(--ink); text-shadow: 0 0 14px rgba(0,229,255,0.35); }}
.node-desc {{ color: var(--muted); font-size: .75rem; line-height: 1.4; text-wrap: balance; }}
.link {{ flex: 0 0 var(--space-5); position: relative; align-self: center; height: 2px;
         background: linear-gradient(90deg, rgba(0,229,255,0.2), var(--cyan), rgba(139,92,246,0.6));
         background-size: 200% 100%; animation: flowPulse 2.4s linear infinite; }}
.link::after {{ content: ""; position: absolute; right: -2px; top: -4px; border: 5px solid transparent;
               border-left: 7px solid var(--violet); border-right: 0; }}
@keyframes flowPulse {{ from {{ background-position: 200% 0; }} to {{ background-position: 0 0; }} }}
/* Wraps to 3 + 2 on mid widths, a vertical chain on phones */
@media (max-width: 1100px) and (min-width: 641px) {{
  .flow {{ flex-wrap: wrap; row-gap: var(--space-4); }}
  .node {{ flex: 1 1 calc((100% - 2 * var(--space-5)) / 3); }}
  .flow > .link:nth-child(6) {{ display: none; }}
}}
@media (max-width: 640px) {{
  .flow {{ flex-direction: column; }}
  .link {{ flex: 0 0 var(--space-4); width: 2px; height: auto; align-self: center;
           background: linear-gradient(180deg, rgba(0,229,255,0.2), var(--cyan), rgba(139,92,246,0.6)); }}
  .link::after {{ right: -4px; top: auto; bottom: -2px; border: 5px solid transparent;
                 border-top: 7px solid var(--violet); border-bottom: 0; }}
}}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{ background: rgba(7,10,24,0.72) !important; backdrop-filter: blur(14px);
                            border-right: 1px solid var(--glass-border); }}
[data-testid="stSidebarUserContent"] {{ padding-bottom: var(--space-4) !important;
                                       min-height: calc(100dvh - 76px); display: flex; flex-direction: column; }}
[data-testid="stSidebarUserContent"] > div {{ flex: 1 1 auto; display: flex; flex-direction: column; }}
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {{ flex: 1 1 auto; gap: var(--space-4); }}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.side-footer) {{ margin-top: auto; }}
.side-brand {{ display: flex; align-items: center; gap: var(--space-3); }}
.side-title {{ font-family: 'Orbitron', sans-serif; font-weight: 900; font-size: 1.12rem; line-height: 1.1;
              background: linear-gradient(90deg, var(--ink), var(--cyan)); -webkit-background-clip: text;
              background-clip: text; color: transparent; }}
.side-tag {{ color: var(--muted); font-size: .75rem; margin-top: var(--space-1); }}
.mini-orbit {{ position: relative; width: 38px; height: 38px; flex: 0 0 38px; }}
.mini-orbit .core {{ position: absolute; inset: 11px; border-radius: 50%;
                    background: radial-gradient(circle at 35% 30%, #9ff5ff, var(--cyan) 40%, #2b2a7a 100%);
                    box-shadow: 0 0 12px rgba(0,229,255,0.6); }}
.mini-orbit .ring {{ position: absolute; inset: 0; border-radius: 50%; border: 1px solid rgba(139,92,246,0.55);
                    animation: spin 6s linear infinite; }}
.mini-orbit .ring::after {{ content: ""; position: absolute; top: -3px; left: 50%; width: 6px; height: 6px;
                           border-radius: 50%; background: var(--amber); box-shadow: 0 0 8px var(--amber); }}
.side-muted {{ color: var(--muted); font-size: .78rem; line-height: 1.7; padding-top: var(--space-4);
              border-top: 1px solid var(--glass-border); }}
.side-footer {{ padding-top: var(--space-3); border-top: 1px solid var(--glass-border);
               font-family: 'Orbitron', sans-serif; font-size: .66rem; letter-spacing: .12em; color: var(--muted);
               white-space: nowrap; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: var(--space-2); width: 100%; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  width: 100%; min-height: 44px; display: flex; align-items: center; padding: 0 var(--space-3); margin: 0;
  border-radius: var(--radius-sm); border: 1px solid transparent; cursor: pointer; transition: all var(--t);
}}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) {{ display: none; }}
[data-testid="stSidebar"] [role="radiogroup"] label p {{ font-size: var(--fs-body); font-weight: 500; color: var(--muted);
                                                        transition: color var(--t); }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: rgba(255,255,255,0.04); }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover p {{ color: var(--ink); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
  background: linear-gradient(90deg, rgba(0,229,255,0.14), rgba(139,92,246,0.10));
  border-color: rgba(0,229,255,0.35); box-shadow: 0 0 18px rgba(0,229,255,0.12), inset 3px 0 0 var(--cyan);
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{ color: var(--ink); font-weight: 600; }}

/* ── Widgets ───────────────────────────────────────────────────────────── */
.stButton > button {{
  width: 100%; min-height: 44px; border-radius: var(--radius-sm); border: 1px solid rgba(0,229,255,0.35);
  background: linear-gradient(135deg, rgba(0,229,255,0.16), rgba(139,92,246,0.22));
  color: var(--ink); font-weight: 600; white-space: nowrap;
  transition: transform var(--t), box-shadow var(--t), border-color var(--t);
}}
.stButton > button p {{ overflow: hidden; text-overflow: ellipsis; }}
.stButton > button:hover {{ transform: translateY(-2px); border-color: var(--cyan); color: #fff;
                           box-shadow: 0 8px 22px rgba(0,229,255,0.22); }}
.stButton > button:active {{ transform: translateY(0); }}
.stButton > button:focus:not(:active) {{ border-color: var(--cyan); color: var(--ink); }}
.stButton > button:disabled {{ opacity: .45; transform: none; box-shadow: none; }}
.st-key-preset_reset .stButton > button {{ background: rgba(255,255,255,0.03); border-color: var(--glass-border); }}

/* Slider (react-aria DOM): track = first child, thumb = second child of the group's inner div */
[data-testid="stSlider"] [role="group"] > div {{ anchor-scope: --wi-thumb, --wi-track; }}
[data-testid="stSlider"] [role="group"] > div > div:first-child {{ height: 6px; border-radius: 6px; anchor-name: --wi-track; }}
[data-testid="stSlider"] [role="group"] > div > div:nth-child(2) {{
  anchor-name: --wi-thumb; z-index: 2; width: 18px; height: 18px;
  background: #e6ecff !important; border: 2px solid var(--cyan);
  box-shadow: 0 0 0 4px rgba(0,229,255,0.15), 0 0 14px rgba(0,229,255,0.7);
}}
@supports (anchor-name: --a) {{
  [data-testid="stSlider"] [role="group"] > div > div:first-child {{ background: rgba(126,146,215,0.22) !important; }}
  [data-testid="stSlider"] [role="group"] > div::after {{
    content: ""; position: absolute; z-index: 1; pointer-events: none; border-radius: 6px;
    left: anchor(--wi-track left); top: anchor(--wi-track top); bottom: anchor(--wi-track bottom);
    right: anchor(--wi-thumb center);
    background: linear-gradient(90deg, var(--cyan), var(--violet)); box-shadow: 0 0 10px rgba(0,229,255,0.45);
  }}
}}
[data-testid="stSliderThumbValue"] {{ color: var(--cyan) !important; font-family: 'Orbitron', sans-serif; font-size: .78rem; }}
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{ color: var(--muted); }}
[data-testid="stSlider"] label p {{ font-weight: 600; color: var(--ink); }}
/* Earth caption sits below the slider's min/max tick row (shown on hover), never on top of it */
[data-testid="stElementContainer"]:has(.earth-ref) {{ margin-top: calc(-1 * var(--space-1)); }}
.earth-ref {{ color: var(--muted); font-size: var(--fs-caption); }}
.earth-ref b {{ color: var(--amber); font-weight: 600; }}

[data-testid="stProgressBarTrack"] {{ background: rgba(255,255,255,0.07) !important; border-radius: var(--radius-pill); height: 10px; }}
[data-testid="stProgressBarTrack"] > div {{
  background: linear-gradient(90deg, var(--violet), var(--cyan)) !important; border-radius: var(--radius-pill);
  box-shadow: 0 0 12px rgba(0,229,255,0.6);
}}
[data-testid="stProgress"] p {{ color: var(--muted); font-size: var(--fs-caption); }}

[data-testid="stTextInput"] input {{ background: transparent !important; color: var(--ink) !important; font-size: 1rem; }}
[data-testid="stTextInput"] [data-baseweb="input"], [data-testid="stTextInput"] > div > div {{
  border-radius: var(--radius-sm); border-color: var(--glass-border) !important; background: rgba(255,255,255,0.04) !important;
  transition: border-color var(--t), box-shadow var(--t);
}}
[data-testid="stTextInput"] [data-baseweb="input"]:focus-within {{ border-color: var(--cyan) !important;
                                                                  box-shadow: 0 0 0 3px rgba(0,229,255,0.15); }}
[data-testid="stTextInput"] label p, [data-testid="stFileUploader"] label p {{ color: var(--muted); font-size: var(--fs-caption); }}

[data-testid="stTabs"] [role="tablist"] {{ gap: var(--space-2); border-bottom: 1px solid var(--glass-border); }}
[data-testid="stTabs"] button[role="tab"] {{ color: var(--muted); padding: var(--space-2) var(--space-3); border-radius: 10px 10px 0 0; }}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{ color: var(--cyan); background: rgba(0,229,255,0.06); }}

[data-testid="stDataFrame"] {{ border: 1px solid var(--glass-border); border-radius: var(--radius); overflow: hidden;
                              background: var(--glass); }}
/* Figures: same max height, centred, never upscaled or stretched */
[data-testid="stImage"] {{ display: flex; flex-direction: column; align-items: center; gap: var(--space-2); }}
[data-testid="stImage"] img {{ border-radius: var(--radius); border: 1px solid var(--glass-border);
                              width: auto !important; max-width: 100%; height: auto; }}
[class*="st-key-card_fig"] [data-testid="stImage"] img {{ max-height: 340px; }}
[data-testid="stImageCaption"], [data-testid="stCaptionContainer"] {{ color: var(--muted) !important; font-size: var(--fs-caption) !important;
                                                                   text-align: center; }}
[data-testid="stFileUploaderDropzone"] {{ background: rgba(255,255,255,0.03) !important; border: 1px dashed rgba(0,229,255,0.35) !important;
                                          border-radius: var(--radius-sm); padding: var(--space-3) var(--space-4); }}
[data-testid="stFileUploaderDropzone"] button {{ border-color: rgba(0,229,255,0.35); color: var(--ink); background: rgba(0,229,255,0.08); }}
[data-testid="stFileUploaderFile"] {{ color: var(--ink); }}
[data-testid="stExpander"] details {{ background: var(--glass); border: 1px solid var(--glass-border) !important;
                                     border-radius: var(--radius-sm); transition: border-color var(--t); }}
[data-testid="stExpander"] details:hover {{ border-color: var(--glass-hover) !important; }}
[data-testid="stExpander"] summary {{ padding: var(--space-3) var(--space-4); }}
[data-testid="stExpander"] summary p {{ font-weight: 600; }}
[data-testid="stPlotlyChart"] {{ background: transparent; }}
[data-testid="stAlert"] {{ background: rgba(255,92,122,0.08); border: 1px solid rgba(255,92,122,0.3); border-radius: var(--radius-sm); }}
/* Help tooltips + popovers */
[data-testid="stTooltipContent"], div[role="tooltip"] {{
  background: #0b1026 !important; color: var(--ink) !important; border: 1px solid rgba(0,229,255,0.25);
  border-radius: var(--radius-sm); box-shadow: var(--glow); font-size: var(--fs-caption);
}}
.card-divider {{ height: 1px; background: var(--glass-border); }}

/* ── Design Your Planet ────────────────────────────────────────────────── */
.verdict {{ display: block; text-align: center; font-weight: 700; font-size: 1.05rem; padding: var(--space-3) var(--space-4);
            border-radius: var(--radius-pill); letter-spacing: .01em; }}
.insights {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: var(--space-3); }}
.insight {{ display: flex; gap: var(--space-3); align-items: flex-start; padding: var(--space-3);
            border-radius: var(--radius-sm); background: rgba(255,255,255,0.03); border: 1px solid var(--glass-border);
            font-size: var(--fs-body); color: var(--ink); line-height: 1.5; }}
.insight .dot {{ width: 8px; height: 8px; border-radius: 50%; margin-top: 7px; flex: 0 0 8px; }}
.hz-hint {{ padding: var(--space-3); border-radius: var(--radius-sm); font-size: var(--fs-body); line-height: 1.5; }}
.delta {{ font-weight: 600; white-space: nowrap; }}
/* Container queries: stack side-by-side groups when *their* width is too small (sidebar-aware) */
.st-key-whatif_main, .st-key-presets, .st-key-rag_top, .st-key-rag_results {{ container-type: inline-size; }}
@container (max-width: 860px) {{
  .st-key-whatif_main [data-testid="stHorizontalBlock"], .st-key-rag_top [data-testid="stHorizontalBlock"],
  .st-key-rag_results [data-testid="stHorizontalBlock"] {{ flex-direction: column; }}
  .st-key-whatif_main [data-testid="stColumn"], .st-key-rag_top [data-testid="stColumn"],
  .st-key-rag_results [data-testid="stColumn"] {{
    width: 100% !important; flex: 1 1 auto !important; min-width: 100% !important; }}
}}
@container (max-width: 700px) {{
  .st-key-presets [data-testid="stHorizontalBlock"] {{ flex-flow: row wrap; gap: var(--space-2) !important; }}
  .st-key-presets [data-testid="stColumn"] {{ flex: 1 1 calc(50% - var(--space-2)) !important;
                                              min-width: calc(50% - var(--space-2)) !important; width: auto !important; }}
  .st-key-presets [data-testid="stColumn"]:last-child {{ flex-basis: 100% !important; }}
}}

/* ── Vision RAG ────────────────────────────────────────────────────────── */
.rag-answer {{ line-height: 1.7; font-size: var(--fs-body); color: var(--ink); }}
.steps {{ margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: var(--space-3); }}
.steps li {{ display: flex; gap: var(--space-3); align-items: flex-start; color: var(--muted); line-height: 1.5; }}
.steps b {{ color: var(--ink); }}
.step-no {{ flex: 0 0 28px; height: 28px; border-radius: 50%; display: grid; place-items: center;
           font-family: 'Orbitron', sans-serif; font-size: .75rem; color: var(--cyan);
           border: 1px solid rgba(0,229,255,0.4); background: rgba(0,229,255,0.08); }}

/* ── Splash ────────────────────────────────────────────────────────────── */
.splash {{ position: fixed; inset: 0; z-index: 1000000; display: flex; align-items: center; justify-content: center;
           background: radial-gradient(ellipse at 50% 40%, rgba(139,92,246,0.25), transparent 60%),
                       linear-gradient(160deg, #05070f 0%, #0b1026 55%, #150d2e 100%);
           overflow: hidden; }}
.splash .starfield {{ position: absolute; z-index: 0; }}
.splash-inner {{ position: relative; z-index: 1; text-align: center; padding: 0 16px; width: min(560px, 100%); }}
.orbit-wrap {{ position: relative; width: 200px; height: 200px; margin: 0 auto 1.8rem auto; }}
.planet {{ position: absolute; inset: 45px; border-radius: 50%;
           background: radial-gradient(circle at 32% 28%, rgba(255,255,255,0.55), transparent 32%),
                       repeating-linear-gradient(170deg, #1b3a8f 0 10px, #2c5fd6 10px 18px, #00b8d4 18px 24px, #6d3fd8 24px 34px);
           background-size: 100% 100%, 220px 220px;
           box-shadow: inset -22px -16px 40px rgba(0,0,0,0.75), 0 0 40px rgba(0,229,255,0.45), 0 0 90px rgba(139,92,246,0.35);
           animation: planetSpin 9s linear infinite; }}
@keyframes planetSpin {{ from {{ background-position: 0 0, 0 0; }} to {{ background-position: 0 0, 220px 0; }} }}
.moon-orbit {{ position: absolute; inset: 0; border-radius: 50%; border: 1px dashed rgba(230,236,255,0.18);
               animation: spin 4.5s linear infinite; }}
.moon {{ position: absolute; top: -8px; left: calc(50% - 8px); width: 16px; height: 16px; border-radius: 50%;
         background: radial-gradient(circle at 35% 30%, #fff, #c9cfe6 45%, #6b7394 100%); box-shadow: 0 0 14px rgba(255,181,71,0.7); }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.splash-title {{ font-family: 'Orbitron', sans-serif; font-weight: 900; font-size: clamp(2.2rem, 8vw, 3.8rem); line-height: 1.05;
                background: linear-gradient(90deg, var(--cyan), #e6ecff 50%, var(--violet)); -webkit-background-clip: text;
                background-clip: text; color: transparent; filter: drop-shadow(0 0 24px rgba(0,229,255,0.45)); }}
.splash-sub {{ color: var(--muted); font-size: 1rem; margin: .7rem 0 1.8rem 0; }}
.splash-bar {{ height: 6px; border-radius: 99px; background: rgba(255,255,255,0.08); overflow: hidden; width: min(380px, 100%); margin: 0 auto; }}
.splash-fill {{ height: 100%; width: 0; border-radius: 99px; background: linear-gradient(90deg, var(--violet), var(--cyan));
               box-shadow: 0 0 14px var(--cyan); animation: fill 3s cubic-bezier(.4,.1,.3,1) forwards; }}
@keyframes fill {{ to {{ width: 100%; }} }}
.splash-status {{ position: relative; height: 1.6rem; margin-top: 1rem; color: var(--ink); font-size: .92rem; }}
.splash-status span {{ position: absolute; left: 0; right: 0; opacity: 0; animation: statusLine .75s ease both; }}
.splash-status span:nth-child(1) {{ animation-delay: 0s; }}
.splash-status span:nth-child(2) {{ animation-delay: .75s; }}
.splash-status span:nth-child(3) {{ animation-delay: 1.5s; }}
.splash-status span:nth-child(4) {{ animation: statusLast .5s ease 2.25s both; color: var(--cyan); }}
@keyframes statusLine {{ 0% {{ opacity: 0; transform: translateY(6px); }} 20%, 80% {{ opacity: 1; transform: none; }}
                         100% {{ opacity: 0; transform: translateY(-6px); }} }}
@keyframes statusLast {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: none; }} }}
"""


def html(markup: str):
    """Render raw HTML; strips indentation so Markdown never turns it into a code block."""
    st.markdown(" ".join(line.strip() for line in markup.strip().splitlines()), unsafe_allow_html=True)


def inject_css():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    st.markdown(STARFIELD_HTML, unsafe_allow_html=True)


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_watchlist() -> pd.DataFrame | None:
    """Read the Step 13 watchlist (XGBoost, all planets, ranked by predicted LSS)."""
    if not os.path.exists(WATCHLIST_CSV):
        return None
    wl = pd.read_csv(WATCHLIST_CSV).rename(columns={
        "Rank": "rank", "Planet": "pl_name", "Host": "hostname",
        "ML_Predicted_LSS": "predicted_lss", "Actual_LSS": "actual_lss",
        "Difference": "difference", "Actual_Rank": "actual_rank",
        "Percentile": "percentile", "Split": "split"})
    return wl.sort_values("rank").reset_index(drop=True)


def require_watchlist() -> pd.DataFrame:
    wl = load_watchlist()
    if wl is None:
        notice("Watchlist not found",
               "Run Step 13 in exoplanet_preprocessing.ipynb to generate "
               "outputs/full_planet_watchlist.csv, then refresh this page.", "error")
        st.stop()
    return wl


@st.cache_data
def load_comparison() -> pd.DataFrame:
    df = pd.read_csv(COMPARISON_CSV).rename(columns={"Unnamed: 0": "Model"})
    return df


@st.cache_data
def raw_planet_names() -> pd.Series:
    return pd.read_csv(RAW_CSV, comment="#", usecols=["pl_name"])["pl_name"]


def raw_planet_count() -> int:
    return len(raw_planet_names())


@st.cache_data
def load_image(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def normalize(name: str) -> str:
    return re.sub(r"[\s\-_]+", "", str(name)).lower()


@st.cache_resource
def load_whatif_model():
    """XGBoost model + a RobustScaler fitted on the exact train split it was trained on.

    models/robust_scaler.pkl was fitted on an older feature set and does not
    reproduce TRAIN_TEST/X_*_scaled.csv, so it is not used here.
    """
    model  = joblib.load(XGB_MODEL)
    scaler = RobustScaler().fit(pd.read_csv(X_TRAIN_CSV)[FEATURE_COLS])
    return model, scaler


def predict_lss(values: dict) -> float:
    model, scaler = load_whatif_model()
    X = pd.DataFrame([values], columns=FEATURE_COLS).astype(float)
    X[LOG_COLS] = np.log1p(X[LOG_COLS])
    X = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLS)
    return float(np.clip(model.predict(X)[0], 0, 1))


# ── Splash screen ─────────────────────────────────────────────────────────────
SPLASH_HTML = f"""
<div class="splash">{STARFIELD_HTML}
  <div class="splash-inner">
    <div class="orbit-wrap"><div class="planet"></div><div class="moon-orbit"><div class="moon"></div></div></div>
    <div class="splash-title">Beyond Binary</div>
    <div class="splash-sub">Estimating life sustainability scores in Exoplanets</div>
    <div class="splash-bar"><div class="splash-fill"></div></div>
    <div class="splash-status">
      <span>Calibrating telescopes…</span><span>Loading 5,456 planets…</span>
      <span>Waking up XGBoost…</span><span>Ready for launch 🚀</span>
    </div>
  </div>
</div>
"""
SPLASH_MIN_SECONDS = 3.0


def show_splash():
    """Full-screen loader shown once per session while the heavy resources warm the caches."""
    if st.session_state.get("booted"):
        return
    placeholder = st.empty()
    with placeholder:
        html(SPLASH_HTML)
    t0 = time.time()
    # Pages report missing files themselves, so a failure here must not block the app
    for warm in (load_watchlist, load_comparison, raw_planet_names, load_whatif_model,
                 lambda: load_image(LEADERBOARD_PNG), lambda: load_image(ACT_VS_PRED_PNG)):
        try:
            warm()
        except Exception:
            pass
    time.sleep(max(0.0, SPLASH_MIN_SECONDS - (time.time() - t0)))
    st.session_state["booted"] = True
    with placeholder:
        html('<div class="boot-fade"></div>')   # triggers the one-time fade-in


# ── UI helpers ────────────────────────────────────────────────────────────────
def style_fig(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, margin=CHART_MARGIN,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK_MUTED, size=12, family=FONT), showlegend=False,
        hoverlabel=dict(bgcolor="#0b1026", bordercolor=CYAN, font_color=INK, font_family=FONT),
        bargap=0.35,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", zeroline=False, linecolor=GRID, tickfont_color=INK)
    return fig


def hbar(labels, values, colors, fmt, hover, height, x_range=None):
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker=dict(color=colors, cornerradius=4),
        text=[fmt.format(v) for v in values], textposition="outside",
        textfont=dict(color=INK), cliponaxis=False,
        hovertemplate=hover, name="",
    ))
    fig = style_fig(fig, height)
    fig.update_yaxes(autorange="reversed")
    if x_range:
        fig.update_xaxes(range=x_range)
    return fig


PLOTLY_CFG = {"displayModeBar": False}


def hero(eyebrow: str, title: str, sub: str):
    html(f'<div class="hero"><div class="hero-eyebrow">{eyebrow}</div>'
         f'<div class="hero-title">{title}</div><div class="hero-sub">{sub}</div></div>')


def section(title: str, sub: str = ""):
    html(f'<div class="section-title">{title}</div>'
         + (f'<div class="section-sub">{sub}</div>' if sub else ""))


def pills(items: list[tuple[str, str]]):
    html('<div class="pills">' + "".join(f'<span class="pill {tone}">{text}</span>'
                                         for text, tone in items) + "</div>")


def kpis(items: list[tuple[str, str, str]]):
    html('<div class="kpi-grid">' + "".join(
        f'<div class="glass kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-desc">{desc}</div></div>'
        for label, value, desc in items) + "</div>")


def notice(title: str, desc: str, kind: str = "", center: bool = False):
    html(f'<div class="glass notice {kind}{" center" if center else ""}"><div class="notice-title">{title}</div>'
         f'<div class="notice-desc">{desc}</div></div>')


def card_head(icon: str, title: str, sub: str = ""):
    """Title row used at the top of every glass card."""
    html(f'<div><div class="grp-head"><span class="grp-icon">{icon}</span><span class="grp-title">{title}</span></div>'
         + (f'<div class="grp-cap">{sub}</div>' if sub else "") + "</div>")


def glass_table(rows: list[dict], columns: list[tuple[str, str, bool]], highlight=None):
    """columns: (key, label, right_aligned). highlight: predicate on a row → accent it."""
    head = "".join(f'<th class="{"r" if right else ""}">{label}</th>' for _, label, right in columns)
    body = "".join(
        f'<tr class="{"best" if highlight and highlight(r) else ""}">'
        + "".join(f'<td class="{"r" if right else ""}">{r[k]}</td>' for k, _, right in columns) + "</tr>"
        for r in rows)
    html(f'<div class="glass gtable-wrap"><table class="gtable"><thead><tr>{head}</tr></thead>'
         f'<tbody>{body}</tbody></table></div>')


# ── Sidebar navigation ────────────────────────────────────────────────────────
PAGES = ["🛰️ Pipeline Overview", "🪐 Planet Explorer", "🧪 Design Your Planet", "🔭 Vision RAG"]


def sidebar() -> str:
    with st.sidebar:
        html('<div class="side-brand"><div class="mini-orbit"><div class="core"></div><div class="ring"></div></div>'
             '<div><div class="side-title">Beyond Binary</div>'
             '<div class="side-tag">Life sustainability scores in exoplanets</div></div></div>')
        page = st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
        html('<div class="side-muted">Data: NASA Exoplanet Archive (PSCompPars, Mar 2026)<br>'
             'Best model: XGBoost · R² 0.9500<br>Watchlist: XGBoost<br>Explainability: SHAP<br>'
             'Vision RAG: Cohere Embed-4 + Gemini</div>')
        html('<div class="side-footer">BEYOND BINARY • ML PIPELINE</div>')
    return page


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PIPELINE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def page_overview():
    comp = load_comparison()
    best = comp.loc[comp["R² Score"].idxmax()]
    wl = require_watchlist()

    hero("🛰️ ML Pipeline", "Beyond Binary",
         "From the raw NASA archive to a ranked list of potentially habitable worlds: an 8-step "
         "preprocessing pipeline, a physics-informed habitability target, and gradient-boosted "
         "models explained with SHAP.")
    pills([("NASA Exoplanet Archive", ""), ("XGBoost", "cyan"),
           ("Voting Ensemble", "violet"), ("SHAP", "amber")])

    kpis([("Raw planets", f"{raw_planet_count():,}", "54 columns from PSCompPars"),
          ("After cleaning", f"{len(wl):,}", "Imputed + outlier-filtered"),
          ("Final features", str(len(FEATURE_COLS)), "After VIF reduction"),
          (f"{best['Model']} R²", f"{best['R² Score']:.4f}",
           f"MAE {best['MAE']:.4f} · RMSE {best['RMSE']:.4f}")])

    # Pipeline flow as connected stage nodes
    section("Pipeline", "Each stage writes a checkpoint the next stage reads.")
    stages = [
        ("🗄️", "01", "Raw Data",      f"{raw_planet_count():,} × 54", "NASA composite parameters"),
        ("🧹", "02", "Preprocessing", "8 steps",  "Impute · filter · log · VIF"),
        ("🌡️", "03", "LSS Target",    "5 parts",  "Weighted habitability, 0–1"),
        ("🤖", "04", "ML Models",     "5 models", "Ridge · RF · XGB · MLP · Vote"),
        ("🏆", "05", "Rankings",      f"{len(wl):,}", "Every planet scored by XGBoost"),
    ]
    nodes = [f'<div class="glass node"><div class="node-icon">{icon}</div><div class="node-idx">STAGE {idx}</div>'
             f'<div class="node-title">{title}</div><div class="node-val">{val}</div>'
             f'<div class="node-desc">{desc}</div></div>' for icon, idx, title, val, desc in stages]
    html('<div class="flow">' + '<div class="link"></div>'.join(nodes) + "</div>")

    # LSS components + model comparison: formula strip, then two equal cards (table + chart each)
    section("Scoring and models", "Five physics-informed sub-scores (each 0–1) are weighted into the LSS; "
            "five models learn to predict it.")
    html('<div class="glass formula">LSS = 0.35·HZ + 0.25·Temp + 0.20·Retention + 0.10·Orbit + 0.10·Stellar</div>')
    left, right = st.columns([1.25, 1], gap="medium")   # the LSS table has a text column
    with left, st.container(key="card_lss"):
        card_head("🌡️", "LSS components", "Weight of each sub-score in the final LSS.")
        glass_table([{"c": n, "w": f"{w:.2f}", "m": d} for _, n, w, d in LSS_COMPONENTS],
                    [("c", "Component", False), ("w", "Weight", True), ("m", "What it measures", False)])
        fig = hbar([n for _, n, _, _ in LSS_COMPONENTS], [w for _, _, w, _ in LSS_COMPONENTS],
                   [CYAN, VIOLET, AMBER, "#5b8cff", "#c084fc"], "{:.0%}",
                   "%{y}: weight %{x:.2f}<extra></extra>", BAR_H, [0, 0.42])
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, config=PLOTLY_CFG, key="lss_chart")

    with right, st.container(key="card_models"):
        card_head("🤖", "Model comparison", "Held-out test set · 1,092 planets · best model highlighted.")
        comp_sorted = comp.sort_values("R² Score", ascending=False)
        glass_table([{"model": r.Model, "r2": f"{r['R² Score']:.4f}", "mae": f"{r.MAE:.4f}",
                      "rmse": f"{r.RMSE:.4f}"} for _, r in comp_sorted.iterrows()],
                    [("model", "Model", False), ("r2", "R²", True), ("mae", "MAE", True), ("rmse", "RMSE", True)],
                    highlight=lambda r: r["model"] == best["Model"])
        colors = [CYAN if m == best["Model"] else "rgba(139,92,246,0.55)" for m in comp_sorted["Model"]]
        fig = hbar(comp_sorted["Model"].tolist(), comp_sorted["R² Score"].tolist(), colors,
                   "{:.4f}", "%{y}: R² %{x:.4f}<extra></extra>", BAR_H, [0, 1.08])
        st.plotly_chart(fig, config=PLOTLY_CFG, key="model_chart")

    # Figures from the notebook: a row of two equal cards
    section("Evaluation figures", "Generated in notebook Steps 9 and 10.")
    figures = [("card_fig_lb", "🏆", "Model R² leaderboard", LEADERBOARD_PNG, "Step 10 · R² of every model on the test set"),
               ("card_fig_avp", "🎯", "Actual vs predicted", ACT_VS_PRED_PNG, "Step 9 · predicted vs actual LSS, test set")]
    # Column widths follow the images' aspect ratios (2.0 vs 1.2) so both render at the same height
    for col, (key, icon, title, path, caption) in zip(st.columns([1.7, 1], gap="medium"), figures):
        with col, st.container(key=key):
            card_head(icon, title)
            if os.path.exists(path):
                st.image(load_image(path), caption=caption)
            else:
                notice("Figure not found", f"{os.path.relpath(path, ROOT)} is missing — re-run the notebook.", "warn")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PLANET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
def resolve_planet(query: str, wl: pd.DataFrame):
    """Return (row | None, suggestions)."""
    q = normalize(query)
    if not q:
        return None, []
    keys = wl["pl_name"].map(normalize)
    exact = wl[keys == q]
    if len(exact):
        return exact.iloc[0], []
    partial = wl[keys.str.contains(q, regex=False)]
    if len(partial) == 1:
        return partial.iloc[0], []
    if len(partial) > 1:
        return None, partial["pl_name"].head(8).tolist()
    close = difflib.get_close_matches(query, wl["pl_name"].tolist(), n=6, cutoff=0.6)
    return None, close


def earth_percentile(total: int) -> float:
    """Where Earth's LSS would rank on the same 0–100 scale."""
    above = int((load_watchlist()["predicted_lss"] > EARTH_LSS).sum())
    return (total - (above + 1)) / (total - 1) * 100


def percentile_chart(p: pd.Series, total: int) -> go.Figure:
    pct = float(p["percentile"])
    fig = go.Figure()
    fig.add_bar(x=[100], y=[""], orientation="h", marker=dict(color=TRACK, cornerradius=6),
                hoverinfo="skip")
    fig.add_bar(x=[pct], y=[""], orientation="h",
                marker=dict(color=CYAN, cornerradius=6, line=dict(color=VIOLET, width=1)),
                hovertemplate=f"{p['pl_name']}: %{{x:.2f}}th percentile<extra></extra>")
    fig = style_fig(fig, PCT_H)
    fig.update_layout(barmode="overlay", bargap=0.45, margin={**CHART_MARGIN, "t": 40})
    fig.update_xaxes(range=[0, 100], ticksuffix="%", dtick=25, gridcolor=GRID)
    fig.add_annotation(x=pct, y=0.5, yref="paper", yanchor="bottom", yshift=26,
                       text=f"<b>{pct:.1f}%</b>", showarrow=False, font=dict(color=INK, size=13))
    earth = earth_percentile(total)
    fig.add_vline(x=earth, line=dict(color=AMBER, width=2, dash="dot"))
    return fig


def components_chart(p: pd.Series) -> go.Figure:
    names  = [n for _, n, _, _ in LSS_COMPONENTS]
    scores = [float(p[k]) for k, *_ in LSS_COMPONENTS]
    return hbar(names, scores, CYAN, "{:.2f}", "%{y}: %{x:.3f}<extra></extra>", BAR_H, [0, 1.12])


def top20_table(wl: pd.DataFrame, planet: str | None):
    top = wl.head(20)
    if planet is not None and planet not in set(top["pl_name"]):
        top = pd.concat([top, wl[wl["pl_name"] == planet]])
    view = pd.DataFrame({
        "Rank":         top["rank"].values,
        "Planet":       top["pl_name"].values,
        "Host star":    top["hostname"].values,
        "ML Predicted LSS": top["predicted_lss"].values,
        "Actual LSS":   top["actual_lss"].values,
        "Difference":   top["difference"].values,
        "Split":        top["split"].values,
    })

    def highlight(row):
        on = row["Planet"] == planet
        return ["background-color: rgba(0,229,255,0.16); color: #e6ecff; font-weight: 600" if on else ""] * len(row)

    styled = (view.style.apply(highlight, axis=1)
              .format({"Actual LSS": "{:.4f}", "ML Predicted LSS": "{:.4f}", "Difference": "{:+.4f}"}))
    st.dataframe(
        styled, hide_index=True, width="stretch", height=38 + 35 * len(view), row_height=35,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small", format="#%d"),
            "ML Predicted LSS": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.4f"),
        })


def page_explorer():
    wl = require_watchlist()
    total = len(wl)

    hero("🪐 Planet Explorer", "Find any planet",
         f"Search {total:,} confirmed exoplanets. See where each one ranks by ML-predicted "
         "Life Sustainability Score (XGBoost) next to its actual LSS.")

    st.session_state.setdefault("planet_query", "Kepler-442 b")
    with st.container(key="card_search"):
        query = st.text_input("Planet name", key="planet_query",
                              placeholder="e.g. Kepler-442 b, TOI-700 d, LHS 1140 b")
    p, suggestions = resolve_planet(query, wl)

    if p is None:
        if not normalize(query):
            notice("Start typing a planet name", "Try Kepler-442 b, TOI-700 d or LHS 1140 b.", center=True)
        elif (raw_planet_names().map(normalize) == normalize(query)).any():
            notice(f"“{query}” was filtered out during preprocessing",
                   "It is in the NASA archive, but the Step 4 outlier filters removed it, "
                   "so it has no LSS score.", "warn", center=True)
        elif suggestions:
            notice(f"No exact match for “{query}”", "Did you mean: " + ", ".join(suggestions) + "?", "warn", center=True)
        else:
            notice(f"We couldn't find “{query}”",
                   "Check the spelling — names follow the NASA archive "
                   "(e.g. “Kepler-442 b”, with a space before the letter).", "error", center=True)
        section("Top 20 planets by ML-predicted LSS")
        top20_table(wl, None)
        return

    name = p["pl_name"]
    badges = [(f"Host: {p['hostname']}", ""),
              ("Held-out test planet" if p["split"] == "test" else "Training planet", "violet")]
    if p["actual_lss"] >= EARTH_LSS:
        badges.append(("Scores above Earth", "amber"))
    html(f'<div class="hero-title sub-page">{name}</div>')
    pills(badges)

    kpis([("Rank", f"#{int(p['rank']):,}", f"of {total:,} · actual-LSS rank #{int(p['actual_rank']):,}"),
          ("ML predicted LSS", f"{p['predicted_lss']:.4f}", f"XGBoost · {p['difference']:+.4f} vs actual"),
          ("Actual LSS", f"{p['actual_lss']:.4f}", f"Earth = {EARTH_LSS:.4f}")])

    with st.container(key="card_pct"):
        card_head("📈", f"Percentile · {p['percentile']:.1f}",
                  f"Scores above {total - int(p['rank']):,} of {total - 1:,} other planets. "
                  f"<span style='color:{AMBER}'>┆ Earth ({earth_percentile(total):.2f}%)</span>")
        st.plotly_chart(percentile_chart(p, total), config=PLOTLY_CFG, key="pct_chart")

    left, right = st.columns(2, gap="medium")
    with left, st.container(key="card_breakdown"):
        card_head("🧩", "LSS breakdown", "Component sub-scores (0–1) that make up the LSS.")
        st.plotly_chart(components_chart(p), config=PLOTLY_CFG, key="comp_chart")
    with right, st.container(key="card_phys"):
        card_head("🔭", "Physical parameters", "Measured values from the NASA archive.")
        glass_table(
            [{"p": "Planet radius", "v": f"{p['pl_rade']:.2f} R⊕"},
             {"p": "Equilibrium temperature", "v": f"{p['pl_eqt']:.0f} K"},
             {"p": "Semi-major axis", "v": f"{p['pl_orbsmax']:.3f} AU"},
             {"p": "Orbital eccentricity", "v": f"{p['pl_orbeccen']:.3f}"},
             {"p": "Stellar temperature", "v": f"{p['st_teff']:.0f} K"},
             {"p": "Stellar mass", "v": f"{p['st_mass']:.3f} M☉"},
             {"p": "Stellar age", "v": f"{p['st_age']:.2f} Gyr"}],
            [("p", "Parameter", False), ("v", "Value", True)])

    section("Top 20 planets by ML-predicted LSS",
            "Searched planet highlighted" + ("" if p["rank"] <= 20 else f" — shown below the top 20 at rank #{int(p['rank']):,}"))
    top20_table(wl, name)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DESIGN YOUR PLANET
# ══════════════════════════════════════════════════════════════════════════════
VERDICTS = [
    # threshold (LSS > t), label, accent colour
    (0.80, "🟢 Highly Promising Candidate",       GREEN),
    (0.60, "🟡 Moderate Habitability Potential",  YELLOW),
    (0.40, "🟠 Low Habitability Potential",       ORANGE),
    (-1.0, "🔴 Not Habitable",                    RED),
]
SLIDER_GROUPS = [
    ("card_planet", "🪐", "The Planet", "Size and bulk — is it rocky like Earth?", ["pl_rade", "pl_dens"]),
    ("card_orbit",  "🛤️", "The Orbit", "Where it travels and how much heat it gets",
     ["pl_orbsmax", "pl_orbeccen", "pl_eqt"]),
    ("card_star",   "⭐", "The Star", "The sun it circles", ["st_teff", "st_mass", "st_age"]),
]
PRESETS = [
    ("preset_earth",  "🌍 Earth",        EARTH_PRESET),
    ("preset_k442",   "🪐 Kepler-442b",  KEPLER442B_PRESET),
    ("preset_mars",   "🔴 Mars-like",    MARS_PRESET),
    ("preset_hotjup", "🔥 Hot Jupiter",  HOT_JUPITER_PRESET),
    ("preset_reset",  "↺ Reset",         DEFAULT_PRESET),
]
LIQUID_WATER_K = (200, 330)


def apply_preset(preset: dict):
    """on_click callback: runs before the sliders are instantiated on the next rerun."""
    for k, v in preset.items():
        _, _, _, lo, hi, *_ = SLIDER[k]
        st.session_state[f"wi_{k}"] = type(lo)(min(max(v, lo), hi))


def fmt_value(k: str, v: float) -> str:
    _, _, unit, *_, fmt = SLIDER[k]
    return (fmt % v) + (f" {unit}" if unit else "")


def gauge_chart(lss: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=lss,
        number=dict(valueformat=".4f", font=dict(family="Orbitron, sans-serif", size=36, color=INK)),
        delta=dict(reference=EARTH_LSS, valueformat="+.4f", suffix=" vs Earth",
                   increasing=dict(color=GREEN), decreasing=dict(color=RED), font=dict(size=15)),
        gauge=dict(
            axis=dict(range=[0, 1], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1], tickcolor=INK_MUTED,
                      tickfont=dict(color=INK_MUTED, size=11)),
            bar=dict(color=CYAN, thickness=0.28),
            bgcolor="rgba(255,255,255,0.03)", borderwidth=0,
            steps=[dict(range=[0, 0.4], color="rgba(255,92,122,0.35)"),
                   dict(range=[0.4, 0.6], color="rgba(255,138,61,0.35)"),
                   dict(range=[0.6, 0.8], color="rgba(255,210,63,0.30)"),
                   dict(range=[0.8, 1.0], color="rgba(46,229,157,0.32)")],
            threshold=dict(line=dict(color=AMBER, width=3), thickness=0.85, value=EARTH_LSS),
        ),
    ))
    fig.update_layout(template="plotly_dark", height=GAUGE_H, margin={**CHART_MARGIN, "t": 40},
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=INK_MUTED))
    return fig


def radar_chart(values: dict) -> go.Figure:
    """Each feature scaled 0–1 across its slider range; your planet vs Earth."""
    def scaled(vals):
        return [(vals[k] - SLIDER[k][3]) / (SLIDER[k][4] - SLIDER[k][3]) for k in FEATURE_COLS]
    labels = [FRIENDLY[k] for k in FEATURE_COLS]
    yours, earth = scaled(values), scaled(EARTH_PRESET)
    yours_txt = [fmt_value(k, values[k]) for k in FEATURE_COLS]
    earth_txt = [fmt_value(k, EARTH_PRESET[k]) for k in FEATURE_COLS]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=yours + yours[:1], theta=labels + labels[:1], name="Your planet",
        customdata=yours_txt + yours_txt[:1],
        line=dict(color=CYAN, width=2), fill="toself", fillcolor="rgba(0,229,255,0.20)",
        marker=dict(size=7, color=CYAN, line=dict(color="#05070f", width=1.5)),
        hovertemplate="Your planet · %{theta}: %{customdata}<extra></extra>"))
    fig.add_trace(go.Scatterpolar(
        r=earth + earth[:1], theta=labels + labels[:1], name="Earth",
        customdata=earth_txt + earth_txt[:1],
        line=dict(color=AMBER, width=2, dash="dash"), fill="none", mode="lines",
        hovertemplate="Earth · %{theta}: %{customdata}<extra></extra>"))
    fig.update_layout(
        template="plotly_dark", height=RADAR_H, margin=dict(l=64, r=64, t=40, b=64),  # room for axis labels
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=INK_MUTED, size=12, family=FONT),
        hoverlabel=dict(bgcolor="#0b1026", bordercolor=CYAN, font_color=INK),
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5,
                    font=dict(color=INK)),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1], showticklabels=False,
                            gridcolor=GRID, linecolor=GRID),
            angularaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=INK, size=11)),
        ),
    )
    return fig


def comparison_table(values: dict):
    """Each feature vs Earth, with a coloured arrow for how far it strays."""
    rows = []
    for k in FEATURE_COLS:
        v, e = values[k], EARTH_PRESET[k]
        pct = (v - e) / e * 100
        if abs(pct) < 5:
            arrow, color, delta = "≈", GREEN, f"{pct:+.1f}%"
        else:
            arrow = "▲" if pct > 0 else "▼"
            color = GREEN if abs(pct) <= 15 else AMBER if abs(pct) <= 50 else RED
            delta = f"{pct:+,.0f}%"
        rows.append({"f": FRIENDLY[k], "v": fmt_value(k, v), "e": fmt_value(k, e),
                     "d": f'<span class="delta" style="color:{color}">{arrow} {delta}</span>'})
    glass_table(rows, [("f", "Feature", False), ("v", "Your planet", True),
                       ("e", "Earth", True), ("d", "vs Earth", True)])


def insights(v: dict) -> list[tuple[str, str]]:
    """Plain-English threshold rules (not the model). Returns (colour, text), worst first."""
    out = []   # (severity, colour, text)
    dt = v["pl_eqt"] - EARTH_PRESET["pl_eqt"]
    if v["pl_eqt"] > LIQUID_WATER_K[1]:
        out.append((3, RED, f"Equilibrium temperature is {dt:,.0f} K hotter than Earth — likely too hot "
                            "for liquid water."))
    elif v["pl_eqt"] < LIQUID_WATER_K[0]:
        out.append((3, RED, f"Equilibrium temperature is {-dt:,.0f} K colder than Earth — surface water "
                            "would probably freeze."))
    if v["pl_rade"] > 6:
        out.append((3, RED, f"A radius of {v['pl_rade']:.1f} R⊕ puts it in gas-giant territory — "
                            "no solid surface to live on."))
    elif v["pl_rade"] > 1.6:
        out.append((2, AMBER, f"At {v['pl_rade']:.2f} R⊕ it is probably a gas-rich mini-Neptune rather "
                              "than a rocky world."))
    elif v["pl_rade"] < 0.7:
        out.append((1, AMBER, f"At {v['pl_rade']:.2f} R⊕ it may be too small to hold on to a thick "
                              "atmosphere."))
    if v["pl_dens"] < 3:
        out.append((2, AMBER, f"Density of {v['pl_dens']:.2f} g/cm³ is low for a rocky planet — "
                              "likely a thick gas or ice envelope."))
    if v["st_age"] < 1:
        out.append((2, AMBER, f"Stellar age of {v['st_age']:.1f} Gyr may be too young for complex life "
                              "to evolve."))
    elif v["st_age"] > 10:
        out.append((1, AMBER, f"A {v['st_age']:.1f} Gyr-old star is nearing the end of its stable life."))
    if v["pl_orbeccen"] > 0.3:
        out.append((2, AMBER, f"Eccentricity of {v['pl_orbeccen']:.2f} means a very stretched orbit — "
                              "extreme seasonal swings."))
    if v["st_teff"] < 3900:
        out.append((1, AMBER, f"A {v['st_teff']:,.0f} K red dwarf host brings flares and likely "
                              "tidal locking."))
    elif v["st_teff"] > 7200:
        out.append((1, AMBER, f"A {v['st_teff']:,.0f} K star is hot and short-lived, with strong UV."))
    if v["st_mass"] > 1.5:
        out.append((1, AMBER, f"A {v['st_mass']:.2f} M☉ star burns out quickly — little time for life."))
    if not out:
        out.append((0, GREEN, "Every parameter is within an Earth-like range — a promising setup."))
    out.sort(key=lambda t: -t[0])
    return [(c, t) for _, c, t in out[:3]]


def hz_hint(eqt: float) -> str:
    lo, hi = LIQUID_WATER_K
    if lo <= eqt <= hi:
        return (f'<div class="hz-hint" style="background:rgba(46,229,157,0.08);border:1px solid rgba(46,229,157,0.3)">'
                f'💧 <b>Habitable zone hint:</b> {eqt:,.0f} K sits inside the {lo}–{hi} K liquid-water range.</div>')
    off = eqt - hi if eqt > hi else lo - eqt
    side = "above" if eqt > hi else "below"
    return (f'<div class="hz-hint" style="background:rgba(255,181,71,0.08);border:1px solid rgba(255,181,71,0.3)">'
            f'💧 <b>Habitable zone hint:</b> {eqt:,.0f} K is {off:,.0f} K {side} the {lo}–{hi} K '
            f'liquid-water range.</div>')


def page_whatif():
    for k, *_, default, _step, _fmt in WHATIF_SLIDERS:
        st.session_state.setdefault(f"wi_{k}", default)

    hero("🧪 Design Your Planet", "Design Your Planet", "Tweak the knobs, watch habitability change.")

    with st.container(key="presets"):
        for col, (key, label, preset) in zip(st.columns(len(PRESETS), gap="small"), PRESETS):
            with col:
                st.button(label, key=key, on_click=apply_preset, args=(preset,), width="stretch")

    values = {}
    with st.container(key="whatif_main"):
        left, right = st.columns(2, gap="medium")
        with left:
            for gkey, icon, title, caption, keys in SLIDER_GROUPS:
                with st.container(key=gkey):
                    card_head(icon, title, caption)
                    for k in keys:
                        _, label, unit, lo, hi, _default, step, fmt = SLIDER[k]
                        values[k] = st.slider(f"{label} ({unit})" if unit else label, min_value=lo, max_value=hi,
                                              step=step, format=fmt, key=f"wi_{k}", help=HELP[k])
                        html(f'<div class="earth-ref">🌍 Earth = <b>{fmt_value(k, EARTH_PRESET[k])}</b></div>')

        with right, st.container(key="card_results"):
            try:
                lss = predict_lss(values)
            except Exception as exc:
                st.error(f"Couldn't run the model — check that model_xgboost.pkl and "
                         f"TRAIN_TEST/X_train.csv exist. ({type(exc).__name__}: {exc})")
                return
            verdict, color = next((v, c) for t, v, c in VERDICTS if lss > t)

            card_head("🎯", "Predicted habitability", "XGBoost (R² 0.9500) on your 8 parameters · "
                      f"<span style='color:{AMBER}'>┃ Earth {EARTH_LSS:.4f}</span>")
            st.plotly_chart(gauge_chart(lss), config=PLOTLY_CFG, key="gauge_chart")
            html(f'<div class="verdict" style="color:{color};background:{color}1f;border:1px solid {color}66;'
                 f'box-shadow:0 0 22px {color}33">{verdict}</div>')
            st.progress(lss, text=f"{lss:.0%} of the maximum score")
            html('<div class="card-divider"></div>')
            card_head("🕸️", "Planet profile", "Each feature scaled 0–1 across its slider range · "
                      f"<span style='color:{CYAN}'>■ your planet</span> vs "
                      f"<span style='color:{AMBER}'>┅ Earth</span>")
            st.plotly_chart(radar_chart(values), config=PLOTLY_CFG, key="radar_chart")

    section("Why this score?", "How your planet compares with Earth, plus rule-of-thumb notes.")
    with st.container(key="card_why"):
        card_head("💡", "What stands out", "Simple threshold rules, not the model.")
        html('<div class="insights">' + "".join(
            f'<div class="insight"><span class="dot" style="background:{c};box-shadow:0 0 8px {c}"></span>'
            f'<span>{t}</span></div>' for c, t in insights(values)) + hz_hint(values["pl_eqt"]) + "</div>")
        html('<div class="card-divider"></div>')
        card_head("⚖️", "Your planet vs Earth", "Arrow colour shows how far each value strays.")
        comparison_table(values)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — VISION RAG (logic in Vision_RAG/rag_core.py)
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_FIGURES = sorted(
    os.path.join(ROOT, "outputs", f) for f in os.listdir(os.path.join(ROOT, "outputs")) if f.endswith(".png")
) if os.path.isdir(os.path.join(ROOT, "outputs")) else []


@st.cache_resource(show_spinner=False)
def rag_clients(cohere_key: str, google_key: str):
    from Vision_RAG import rag_core
    return rag_core.make_clients(cohere_key, google_key)


@st.cache_data(ttl=3600, show_spinner=False)
def rag_embed(base64_img: str, _co) -> np.ndarray:
    from Vision_RAG import rag_core
    return rag_core.embed_image(base64_img, _co)


def short_error(exc: Exception) -> str:
    """The API's own message when there is one, instead of the full response dump."""
    body = getattr(exc, "body", None)
    msg = body.get("message") if isinstance(body, dict) else getattr(exc, "message", None)
    return f"{type(exc).__name__}: {msg or str(exc)[:240]}"


def is_auth_error(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) in (401, 403) or getattr(exc, "code", None) in (401, 403)


def rag_add(paths: list[str], co) -> int:
    """Embeds any new images and appends them to the session library. Returns how many were added."""
    from Vision_RAG import rag_core
    new = [p for p in paths if p not in st.session_state.rag_paths]
    if not new:
        return 0
    bar = st.progress(0.0, text="Embedding with Cohere Embed-4…")
    added_paths, added_embs = [], []
    for i, path in enumerate(new, start=1):
        try:
            added_embs.append(rag_embed(rag_core.base64_from_image(path), co))
            added_paths.append(path)
        except Exception as exc:
            if is_auth_error(exc):
                notice("Cohere rejected the API key", short_error(exc) + " Check the key and try again.", "error")
                break
            notice(f"Skipped {os.path.basename(path)}", short_error(exc), "warn")
        bar.progress(i / len(new), text=f"Embedding with Cohere Embed-4… {i}/{len(new)}")
    bar.empty()
    if added_embs:
        stack = np.vstack(added_embs)
        emb = st.session_state.rag_emb
        st.session_state.rag_emb = stack if emb is None else np.vstack((emb, stack))
        st.session_state.rag_paths.extend(added_paths)
    return len(added_paths)


def rag_source_label(path: str) -> str:
    parts = path.split(os.sep)
    if "pdf_pages" in parts:
        return f"{parts[-2]}.pdf · {parts[-1].removesuffix('.png').replace('_', ' ')}"
    return os.path.basename(path)


def similarity_chart(scores: np.ndarray, paths: list[str]) -> go.Figure:
    top = np.argsort(scores)[::-1][:5]
    labels = [rag_source_label(paths[i]) for i in top]
    colors = [CYAN] + ["rgba(139,92,246,0.55)"] * (len(top) - 1)
    lo = float(scores[top].min())
    return hbar(labels, [float(scores[i]) for i in top], colors, "{:.3f}",
                "%{y}: similarity %{x:.3f}<extra></extra>", BAR_H, [lo * 0.9, float(scores[top[0]]) * 1.08])


def page_vision_rag():
    st.session_state.setdefault("rag_paths", [])
    st.session_state.setdefault("rag_emb", None)
    st.session_state.setdefault("rag_seen", set())
    st.session_state.setdefault("rag_cohere_key", os.environ.get("COHERE_API_KEY", ""))
    st.session_state.setdefault("rag_google_key", os.environ.get("GOOGLE_API_KEY", ""))

    hero("🔭 Vision RAG", "Ask your figures",
         "Retrieve the most relevant chart or PDF page with Cohere Embed-4, then let Gemini 2.5 Flash "
         "answer your question from it.")
    pills([("Cohere Embed-4", "cyan"), ("Gemini 2.5 Flash", "violet"), ("Images · PDFs", "amber")])

    try:
        from Vision_RAG import rag_core  # noqa: F401  (fails if the RAG packages are missing)
    except ImportError as exc:
        notice("Vision RAG packages missing",
               f"{exc}. Install them with <code>.venv/bin/pip install -r Vision_RAG/requirements.txt</code>.", "error")
        return

    # ── Keys + status ─────────────────────────────────────────────────────────
    co = gemini = None
    status = st.empty()   # filled once the clients are known, but rendered above the cards
    with st.container(key="rag_top"):
        left, right = st.columns(2, gap="medium")
        with left, st.container(key="card_keys"):
            card_head("🔑", "API keys", "Kept in this browser session only. "
                      '<a href="https://dashboard.cohere.com/api-keys" target="_blank">Get a Cohere key</a> · '
                      '<a href="https://aistudio.google.com/app/apikey" target="_blank">Get a Google key</a>')
            cohere_key = st.text_input("Cohere API key", type="password", key="rag_cohere_key")
            google_key = st.text_input("Google API key (Gemini)", type="password", key="rag_google_key")
            if cohere_key and google_key:
                try:
                    co, gemini = rag_clients(cohere_key, google_key)
                except Exception as exc:
                    notice("Couldn't start the API clients", short_error(exc), "error")
        with right, st.container(key="card_how"):
            card_head("ℹ️", "How it works")
            html('<ol class="steps">'
                 '<li><span class="step-no">1</span><span><b>Embed</b> — every image or PDF page is embedded with '
                 '<b>Cohere Embed-4</b>, a multimodal model that reads charts without OCR.</span></li>'
                 '<li><span class="step-no">2</span><span><b>Retrieve</b> — your question is embedded the same way; '
                 'the closest image by cosine similarity wins.</span></li>'
                 '<li><span class="step-no">3</span><span><b>Answer</b> — <b>Gemini 2.5 Flash</b> reads that image '
                 'and answers your question.</span></li></ol>')
    with status:
        kpis([("API keys", "Entered" if co else "Missing", "Checked on first request" if co else "Enter both keys"),
              ("Library", f"{len(st.session_state.rag_paths)}", "images & PDF pages indexed")])

    # ── Library ───────────────────────────────────────────────────────────────
    section("1 · Build the library", "Load the project's own figures, the original demo charts, or upload images and PDFs.")
    with st.container(key="card_library"):
        b = st.columns(3, gap="small")
        load_project = b[0].button("🪐 Load project figures", key="rag_load_project", disabled=co is None, width="stretch")
        load_demo = b[1].button("🌐 Load demo charts", key="rag_load_demo", disabled=co is None, width="stretch")
        if b[2].button("🗑️ Clear library", key="rag_clear", disabled=not st.session_state.rag_paths, width="stretch"):
            st.session_state.update(rag_paths=[], rag_emb=None, rag_seen=set(), rag_result=None)
            st.rerun()

        if load_project:
            added = rag_add(PROJECT_FIGURES, co)
            if added or all(p in st.session_state.rag_paths for p in PROJECT_FIGURES):
                notice(f"Added {added} project figures" if added else "Project figures already loaded",
                       "SHAP plots, leaderboard, watchlist and evaluation charts from outputs/.")
        if load_demo:
            try:
                demo_paths = rag_core.download_demo_images()
                added = rag_add(demo_paths, co)
                if added or all(p in st.session_state.rag_paths for p in demo_paths):
                    notice(f"Added {added} demo charts" if added else "Demo charts already loaded",
                           "Company earnings infographics from appeconomyinsights.com.")
            except Exception as exc:
                notice("Couldn't download the demo charts", short_error(exc), "error")

        uploads = st.file_uploader("Upload images or PDFs", type=["png", "jpg", "jpeg", "pdf"],
                                   accept_multiple_files=True, key="rag_uploader", disabled=co is None)
        for up in uploads or []:
            sig = f"{up.name}:{up.size}"
            if sig in st.session_state.rag_seen:
                continue
            try:
                if up.type == "application/pdf":
                    paths = rag_core.pdf_to_page_images(up.getvalue(), up.name)
                else:
                    paths = [rag_core.save_upload(up.getvalue(), up.name)]
                added = rag_add(paths, co)
            except Exception as exc:
                notice(f"Couldn't process {up.name}", short_error(exc), "error")
                continue
            if added:   # only mark as done once it really made it into the library
                notice(f"Indexed {up.name}", f"{added} new {'page' if added == 1 else 'pages'} added to the library.")
                st.session_state.rag_seen.add(sig)

        if st.session_state.rag_paths:
            with st.expander(f"🖼️ View library ({len(st.session_state.rag_paths)})"):
                cols = st.columns(5, gap="small")
                for i, path in enumerate(st.session_state.rag_paths):
                    with cols[i % 5]:
                        if os.path.exists(path):
                            st.image(path, caption=rag_source_label(path))

    # ── Ask ───────────────────────────────────────────────────────────────────
    section("2 · Ask a question", "The best-matching image is retrieved, then Gemini answers from it.")
    ready = co is not None and gemini is not None and bool(st.session_state.rag_paths)
    with st.container(key="card_ask"):
        if not ready:
            notice("Not ready yet",
                   "Enter both API keys" if co is None else "Load or upload at least one image first.", "warn")
        q_col, b_col = st.columns([3, 1], gap="small", vertical_alignment="bottom")
        question = q_col.text_input("Question", key="rag_question", disabled=not ready,
                                    placeholder="e.g. Which feature matters most for the XGBoost model?")
        run = b_col.button("🚀 Run Vision RAG", key="rag_run", disabled=not (ready and question), width="stretch")

        if run:
            try:
                with st.spinner("Finding the most relevant image…"):
                    hit, scores = rag_core.search(question, co, st.session_state.rag_emb, st.session_state.rag_paths)
                with st.spinner("Gemini is reading the image…"):
                    reply = rag_core.answer(question, hit, gemini)
                st.session_state.rag_result = {"q": question, "hit": hit, "scores": scores,
                                               "paths": list(st.session_state.rag_paths), "answer": reply}
            except Exception as exc:
                notice("Vision RAG failed", short_error(exc), "error")

    res = st.session_state.get("rag_result")
    if res:
        section("Results", f"“{res['q']}”")
        with st.container(key="rag_results"):
            img_col, ans_col = st.columns(2, gap="medium")
            with img_col, st.container(key="card_hit"):
                card_head("🖼️", "Retrieved image", rag_source_label(res["hit"]))
                st.image(res["hit"])
                html('<div class="card-divider"></div>')
                card_head("📊", "Top matches", "Cosine similarity between your question and each image.")
                st.plotly_chart(similarity_chart(res["scores"], res["paths"]), config=PLOTLY_CFG, key="rag_sim")
            with ans_col, st.container(key="card_answer"):
                card_head("✨", "Answer", "Gemini 2.5 Flash, grounded in the retrieved image.")
                body = "<br>".join(line for line in res["answer"].replace("<", "&lt;").splitlines() if line.strip())
                html(f'<div class="rag-answer">{body}</div>')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
# Keep widget values alive while another page is shown (Streamlit drops state of unrendered widgets)
for _k in [f"wi_{k}" for k in FEATURE_COLS] + ["planet_query", "rag_cohere_key", "rag_google_key", "rag_question"]:
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]

inject_css()
show_splash()
page = sidebar()
if page == PAGES[0]:
    page_overview()
elif page == PAGES[1]:
    page_explorer()
elif page == PAGES[2]:
    page_whatif()
else:
    page_vision_rag()
