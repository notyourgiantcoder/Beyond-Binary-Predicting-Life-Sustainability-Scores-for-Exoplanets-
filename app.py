"""
Exoplanet Life Sustainability Score — Streamlit dashboard.

Run from the project root:
    .venv/bin/streamlit run app.py
"""

import difflib
import os
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit_shadcn_ui as ui

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_CSV        = os.path.join(ROOT, "data/PSCompPars_2026.03.05_02.02.54.csv")
COMPARISON_CSV = os.path.join(ROOT, "outputs/model_comparison_with_ensemble.csv")
WATCHLIST_CSV  = os.path.join(ROOT, "outputs/full_planet_watchlist.csv")
LEADERBOARD_PNG = os.path.join(ROOT, "outputs/step10_leaderboard.png")
ACT_VS_PRED_PNG = os.path.join(ROOT, "outputs/step9_actual_vs_predicted.png")

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

# Chart colours (dark-mode steps of the validated reference palette)
BLUE, ORANGE = "#3987e5", "#d95926"
INK, INK_MUTED, GRID, TRACK = "#fafafa", "#a1a1aa", "#27272a", "#1f1f23"


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Exoplanet LSS", page_icon="🪐", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; max-width: 1280px; }
  h1, h2, h3 { letter-spacing: -0.02em; }
  .hero-eyebrow { color:#3987e5; font-size:.78rem; font-weight:600;
                  letter-spacing:.12em; text-transform:uppercase; margin-bottom:.35rem; }
  .hero-title { font-size:2.35rem; font-weight:700; line-height:1.15; margin:0 0 .5rem 0; }
  .hero-sub { color:#a1a1aa; font-size:1.02rem; max-width:760px; margin-bottom:.9rem; }
  .section-title { font-size:1.15rem; font-weight:600; margin:1.6rem 0 .15rem 0; }
  .section-sub { color:#a1a1aa; font-size:.88rem; margin-bottom:.6rem; }
  .flow-arrow { display:flex; align-items:center; justify-content:center;
                height:100%; min-height:120px; color:#52525b; font-size:1.4rem; }
  .formula { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.85rem;
             background:#18181b; border:1px solid #27272a; border-radius:.6rem;
             padding:.7rem .9rem; color:#d4d4d8; margin:.4rem 0 .8rem 0; }
  .side-brand { font-weight:700; font-size:1.05rem; margin-bottom:.1rem; }
  .side-muted { color:#71717a; font-size:.8rem; }
  [data-testid="stImage"] img { border-radius:.6rem; border:1px solid #27272a; }
  footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_watchlist() -> pd.DataFrame | None:
    """Read the Step 13 watchlist (Voting Ensemble, all planets, ranked by predicted LSS)."""
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
        ui.alert("Watchlist not found",
                 description="Run Step 13 in exoplanet_preprocessing.ipynb to generate "
                             "outputs/full_planet_watchlist.csv, then refresh this page.",
                 variant="destructive", key="wl_missing")
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


def normalize(name: str) -> str:
    return re.sub(r"[\s\-_]+", "", str(name)).lower()


# ── Chart helpers ─────────────────────────────────────────────────────────────
def style_fig(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=24, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK_MUTED, size=12), showlegend=False,
        hoverlabel=dict(bgcolor="#18181b", bordercolor=GRID, font_color=INK),
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


def section(title: str, sub: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>'
                + (f'<div class="section-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


# ── Sidebar navigation ────────────────────────────────────────────────────────
PAGES = ["Pipeline Overview", "Planet Explorer"]
with st.sidebar:
    st.markdown('<div class="side-brand">Exoplanet LSS</div>'
                '<div class="side-muted">Life Sustainability Score · ML pipeline</div>',
                unsafe_allow_html=True)
    st.write("")
    page = ui.tabs(PAGES, orientation="vertical", key="nav", label="Navigation")
    st.write("")
    ui.separator(key="side_sep")
    st.markdown('<div class="side-muted">Data: NASA Exoplanet Archive (PSCompPars, Mar 2026)<br>'
                'Best model: XGBoost · R² 0.9500<br>Watchlist: Voting Ensemble<br>Explainability: SHAP</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PIPELINE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def page_overview():
    comp = load_comparison()
    best = comp.loc[comp["R² Score"].idxmax()]
    wl = require_watchlist()

    st.markdown(
        '<div class="hero-eyebrow">ML Pipeline</div>'
        '<div class="hero-title">Exoplanet Life Sustainability Score — ML Pipeline</div>'
        '<div class="hero-sub">From the raw NASA archive to a ranked list of potentially habitable '
        'worlds: an 8-step preprocessing pipeline, a physics-informed habitability target, '
        'and gradient-boosted models explained with SHAP.</div>',
        unsafe_allow_html=True)
    ui.badges([("NASA Exoplanet Archive", "secondary"), ("XGBoost", "secondary"),
               ("Voting Ensemble", "secondary"), ("SHAP", "secondary")], key="hero_badges")

    # KPIs
    c = st.columns(4)
    with c[0]:
        ui.metric_card("Raw planets", f"{raw_planet_count():,}",
                       description="54 columns from PSCompPars", key="kpi_raw")
    with c[1]:
        ui.metric_card("After cleaning", f"{len(wl):,}",
                       description="Imputed + outlier-filtered", key="kpi_clean")
    with c[2]:
        ui.metric_card("Final features", str(len(FEATURE_COLS)),
                       description="After VIF reduction", key="kpi_feat")
    with c[3]:
        ui.metric_card(f"{best['Model']} R²", f"{best['R² Score']:.4f}",
                       description=f"MAE {best['MAE']:.4f} · RMSE {best['RMSE']:.4f}", key="kpi_r2")

    # Pipeline flow
    section("Pipeline", "Each stage writes a checkpoint the next stage reads.")
    stages = [
        ("01 · Raw Data",      f"{raw_planet_count():,} × 54", "NASA composite parameters"),
        ("02 · Preprocessing", "8 steps",  "Impute · filter · log · VIF"),
        ("03 · LSS Target",    "5 parts",  "Weighted habitability, 0–1"),
        ("04 · ML Models",     "5 models", "Ridge · RF · XGB · MLP · Vote"),
        ("05 · Rankings",      f"{len(wl):,}", "Every planet scored by the ensemble"),
    ]
    cols = st.columns([1, 0.12, 1, 0.12, 1, 0.12, 1, 0.12, 1])
    for i, (title, content, desc) in enumerate(stages):
        with cols[i * 2]:
            ui.card(title=title, content=content, description=desc, key=f"stage_{i}", size="sm")
        if i < len(stages) - 1:
            cols[i * 2 + 1].markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)

    # LSS components + model comparison
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        section("LSS components", "Five physics-informed sub-scores, each on a 0–1 scale.")
        st.markdown('<div class="formula">LSS = 0.35·HZ + 0.25·Temp + 0.20·Retention '
                    '+ 0.10·Orbit + 0.10·Stellar</div>', unsafe_allow_html=True)
        ui.table(
            [{"component": n, "weight": f"{w:.2f}", "measures": d}
             for _, n, w, d in LSS_COMPONENTS],
            columns=[{"key": "component", "label": "Component"},
                     {"key": "weight", "label": "Weight", "align": "right"},
                     {"key": "measures", "label": "What it measures"}],
            key="lss_table")
        fig = hbar([n for _, n, _, _ in LSS_COMPONENTS], [w for _, _, w, _ in LSS_COMPONENTS],
                   BLUE, "{:.0%}", "%{y}: weight %{x:.2f}<extra></extra>", 210, [0, 0.42])
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, config=PLOTLY_CFG, key="lss_chart")

    with right:
        section("Model comparison", "Held-out test set · 1,092 planets.")
        comp_sorted = comp.sort_values("R² Score", ascending=False)
        ui.table(
            [{"model": r.Model, "r2": f"{r['R² Score']:.4f}", "mae": f"{r.MAE:.4f}",
              "rmse": f"{r.RMSE:.4f}"} for _, r in comp_sorted.iterrows()],
            columns=[{"key": "model", "label": "Model"},
                     {"key": "r2", "label": "R²", "align": "right"},
                     {"key": "mae", "label": "MAE", "align": "right"},
                     {"key": "rmse", "label": "RMSE", "align": "right"}],
            key="model_table")
        colors = [BLUE if m == best["Model"] else "#52525b" for m in comp_sorted["Model"]]
        fig = hbar(comp_sorted["Model"].tolist(), comp_sorted["R² Score"].tolist(), colors,
                   "{:.4f}", "%{y}: R² %{x:.4f}<extra></extra>", 250, [0, 1.08])
        st.plotly_chart(fig, config=PLOTLY_CFG, key="model_chart")

    # Figures from the notebook
    section("Evaluation figures", "Generated in notebook Steps 9 and 10.")
    tab = ui.tabs(["Leaderboard", "Actual vs Predicted"], key="fig_tabs", variant="line")
    if tab == "Leaderboard":
        st.image(LEADERBOARD_PNG, width="stretch")
    else:
        st.image(ACT_VS_PRED_PNG, width="stretch")


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
    fig.add_bar(x=[pct], y=[""], orientation="h", marker=dict(color=BLUE, cornerradius=6),
                hovertemplate=f"{p['pl_name']}: %{{x:.2f}}th percentile<extra></extra>")
    fig = style_fig(fig, 150)
    fig.update_layout(barmode="overlay", bargap=0.45, margin=dict(l=8, r=24, t=34, b=8))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", dtick=25, gridcolor=GRID)
    fig.add_annotation(x=pct, y=0.5, yref="paper", yanchor="bottom", yshift=26,
                       text=f"<b>{pct:.1f}%</b>", showarrow=False, font=dict(color=INK, size=13))
    earth = earth_percentile(total)
    fig.add_vline(x=earth, line=dict(color=ORANGE, width=2, dash="dot"))
    return fig


def components_chart(p: pd.Series) -> go.Figure:
    names  = [n for _, n, _, _ in LSS_COMPONENTS]
    scores = [float(p[k]) for k, *_ in LSS_COMPONENTS]
    return hbar(names, scores, BLUE, "{:.2f}", "%{y}: %{x:.3f}<extra></extra>", 230, [0, 1.12])


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
        return ["background-color: rgba(57,135,229,0.22); font-weight: 600" if on else ""] * len(row)

    styled = (view.style.apply(highlight, axis=1)
              .format({"Actual LSS": "{:.4f}", "ML Predicted LSS": "{:.4f}", "Difference": "{:+.4f}"}))
    st.dataframe(
        styled, hide_index=True, width="stretch", height=38 + 35 * len(view),
        column_config={
            "Rank": st.column_config.NumberColumn(width="small", format="#%d"),
            "ML Predicted LSS": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.4f"),
        })


def page_explorer():
    wl = require_watchlist()
    total = len(wl)

    st.markdown(
        '<div class="hero-eyebrow">Planet Explorer</div>'
        '<div class="hero-title">Find any planet</div>'
        f'<div class="hero-sub">Search {total:,} confirmed exoplanets. See where each one ranks '
        'by ML-predicted Life Sustainability Score (Voting Ensemble) next to its actual LSS.</div>',
        unsafe_allow_html=True)

    query = ui.input("Planet name", value="Kepler-442 b", key="planet_query",
                     placeholder="e.g. Kepler-442 b, TOI-700 d, LHS 1140 b")
    p, suggestions = resolve_planet(query, wl)

    if p is None:
        if not normalize(query):
            ui.alert("Start typing a planet name",
                     description="Try Kepler-442 b, TOI-700 d or LHS 1140 b.", key="empty_alert")
        elif (raw_planet_names().map(normalize) == normalize(query)).any():
            ui.alert(f"“{query}” was filtered out during preprocessing",
                     description="It is in the NASA archive, but the Step 4 outlier filters removed it, "
                                 "so it has no LSS score.", key="nf_alert")
        elif suggestions:
            ui.alert(f"No exact match for “{query}”",
                     description="Did you mean: " + ", ".join(suggestions) + "?", key="nf_alert")
        else:
            ui.alert(f"We couldn't find “{query}”",
                     description="Check the spelling — names follow the NASA archive "
                                 "(e.g. “Kepler-442 b”, with a space before the letter).",
                     variant="destructive", key="nf_alert")
        section("Top 20 planets by ML-predicted LSS")
        top20_table(wl, None)
        return

    name = p["pl_name"]
    badges = [(f"Host: {p['hostname']}", "secondary"),
              ("Held-out test planet" if p["split"] == "test" else "Training planet", "outline")]
    if p["actual_lss"] >= EARTH_LSS:
        badges.append(("Scores above Earth", "default"))
    st.markdown(f"<h2 style='margin:.8rem 0 .3rem 0'>{name}</h2>", unsafe_allow_html=True)
    ui.badges(badges, key="planet_badges")

    c = st.columns(4)
    with c[0]:
        ui.metric_card("Rank", f"#{int(p['rank']):,}", description=f"of {total:,} · actual-LSS rank #{int(p['actual_rank']):,}", key="m_rank")
    with c[1]:
        ui.metric_card("Actual LSS", f"{p['actual_lss']:.4f}",
                       description=f"Earth = {EARTH_LSS:.4f}", key="m_actual")
    with c[2]:
        ui.metric_card("ML predicted LSS", f"{p['predicted_lss']:.4f}",
                       description=f"Ensemble · {p['difference']:+.4f} vs actual", key="m_pred")
    with c[3]:
        ui.metric_card("Percentile", f"{p['percentile']:.1f}",
                       description=f"Scores above {total - int(p['rank']):,} of {total - 1:,} others", key="m_pct")

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        section("Percentile", "Where this planet sits among all scored planets. "
                f"<span style='color:{ORANGE}'>┆ Earth ({earth_percentile(total):.2f}%)</span>")
        st.plotly_chart(percentile_chart(p, total), config=PLOTLY_CFG, key="pct_chart")
        section("LSS breakdown", "Component sub-scores (0–1) that make up the LSS.")
        st.plotly_chart(components_chart(p), config=PLOTLY_CFG, key="comp_chart")
    with right:
        section("Physical parameters")
        ui.table(
            [{"p": "Planet radius", "v": f"{p['pl_rade']:.2f} R⊕"},
             {"p": "Equilibrium temperature", "v": f"{p['pl_eqt']:.0f} K"},
             {"p": "Semi-major axis", "v": f"{p['pl_orbsmax']:.3f} AU"},
             {"p": "Orbital eccentricity", "v": f"{p['pl_orbeccen']:.3f}"},
             {"p": "Stellar temperature", "v": f"{p['st_teff']:.0f} K"},
             {"p": "Stellar mass", "v": f"{p['st_mass']:.3f} M☉"},
             {"p": "Stellar age", "v": f"{p['st_age']:.2f} Gyr"}],
            columns=[{"key": "p", "label": "Parameter"}, {"key": "v", "label": "Value", "align": "right"}],
            key="phys_table")

    section("Top 20 planets by ML-predicted LSS",
            "Searched planet highlighted" + ("" if p["rank"] <= 20 else f" — shown below the top 20 at rank #{int(p['rank']):,}"))
    top20_table(wl, name)


if page == "Pipeline Overview":
    page_overview()
else:
    page_explorer()
