"""
5-fold cross-validation for the two main models (XGBoost, Random Forest).

The notebook evaluates every model on a single 80/20 split; this script checks
that the scores hold across all data partitions. Each fold fits its own
RobustScaler inside a Pipeline, so no test-fold statistics leak into training.

Run from the project root:
    .venv/bin/python cv_analysis.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.abspath(__file__))
# The copy in outputs/checkpoints/ is from an older run and does not match TRAIN_TEST/
DATA_CSV = os.path.join(ROOT, "EXTRAS/exoplanets_step7_model_ready.csv")
OUT_DIR = os.path.join(ROOT, "outputs")
RESULTS_CSV = os.path.join(OUT_DIR, "cv_results.csv")
STABILITY_PNG = os.path.join(OUT_DIR, "cv_r2_stability.png")
SUMMARY_PNG = os.path.join(OUT_DIR, "cv_metrics_summary.png")

FEATURE_COLS = ["pl_rade", "pl_dens", "pl_orbeccen", "pl_eqt",
                "pl_orbsmax", "st_teff", "st_mass", "st_age"]
TARGET = "LSS"
N_SPLITS = 5

# Same hyperparameters as notebook Step 9
MODELS = {
    "XGBoost": lambda: XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                    subsample=0.8, colsample_bytree=0.8,
                                    random_state=42, verbosity=0),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=300, max_depth=10,
                                                   min_samples_leaf=4, random_state=42, n_jobs=-1),
}

# Dashboard palette (app.py)
COLORS = {"XGBoost": "#c6f432", "Random Forest": "#5eead4"}
BG, INK, INK_MUTED, GRID = "#0b0d10", "#f4f5f7", "#8d929c", "#23262b"


def run_cv(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    rows = []
    for name, make_model in MODELS.items():
        for fold, (tr, te) in enumerate(kf.split(X), start=1):
            pipe = Pipeline([("scaler", RobustScaler()), ("model", make_model())])
            pipe.fit(X.iloc[tr], y.iloc[tr])
            pred = pipe.predict(X.iloc[te])
            rows.append({"Model": name, "Fold": fold,
                         "R2": r2_score(y.iloc[te], pred),
                         "MAE": mean_absolute_error(y.iloc[te], pred),
                         "RMSE": np.sqrt(mean_squared_error(y.iloc[te], pred))})
            print(f"  {name:<14} fold {fold}  R²={rows[-1]['R2']:.4f}")
    return pd.DataFrame(rows)


def print_table(res: pd.DataFrame):
    for name, g in res.groupby("Model", sort=False):
        print(f"\n{name}")
        print(f"{'Fold':<12} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
        print("-" * 38)
        for _, r in g.iterrows():
            print(f"{int(r.Fold):<12} {r.R2:>8.4f} {r.MAE:>8.4f} {r.RMSE:>8.4f}")
        print("-" * 38)
        print(f"{'Mean ± std':<12} " + " ".join(
            f"{g[m].mean():.4f}±{g[m].std():.4f}" for m in ("R2", "MAE", "RMSE")))


def style_axes(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=INK_MUTED, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot_stability(res: pd.DataFrame):
    names = list(MODELS)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    style_axes(ax)
    for i, name in enumerate(names):
        r2 = res.loc[res.Model == name, "R2"].to_numpy()
        jitter = np.linspace(-0.12, 0.12, len(r2))
        ax.scatter(i + jitter, r2, s=70, color=COLORS[name], edgecolor=BG, linewidth=2, zorder=3)
        ax.hlines(r2.mean(), i - 0.25, i + 0.25, color=INK, linewidth=2, zorder=2)
        ax.annotate(f"mean {r2.mean():.4f}", (i + 0.28, r2.mean()), color=INK,
                    va="center", fontsize=10)
    ax.set_xticks(range(len(names)), names, color=INK, fontsize=11)
    ax.set_xlim(-0.6, len(names) - 0.2)
    ax.set_ylabel("R² per fold", color=INK_MUTED)
    ax.set_title(f"{N_SPLITS}-fold CV — R² stability across folds", color=INK, loc="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(STABILITY_PNG, dpi=150, facecolor=BG)
    plt.close(fig)


def plot_summary(res: pd.DataFrame):
    names = list(MODELS)
    stats = res.groupby("Model")["R2"].agg(["mean", "std"]).loc[names]
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
    style_axes(ax)
    ax.bar(names, stats["mean"], width=0.5, color=[COLORS[n] for n in names],
           yerr=stats["std"], capsize=8, error_kw=dict(ecolor=INK, elinewidth=1.5))
    for i, (m, s) in enumerate(zip(stats["mean"], stats["std"])):
        ax.text(i, m + s + 0.025, f"{m:.4f} ± {s:.4f}", ha="center", color=INK, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(range(len(names)), names, color=INK, fontsize=11)
    ax.set_ylabel("Mean R² (error bars ±1 std)", color=INK_MUTED)
    ax.set_title(f"{N_SPLITS}-fold CV — mean R²", color=INK, loc="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(SUMMARY_PNG, dpi=150, facecolor=BG)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_CSV)
    X, y = df[FEATURE_COLS], df[TARGET]
    print(f"Loaded {len(df)} planets × {len(FEATURE_COLS)} features from {os.path.relpath(DATA_CSV, ROOT)}")
    print(f"Running {N_SPLITS}-fold CV (scaler fit inside each training fold)")

    res = run_cv(X, y)
    print_table(res)

    os.makedirs(OUT_DIR, exist_ok=True)
    res.to_csv(RESULTS_CSV, index=False)
    plot_stability(res)
    plot_summary(res)
    print(f"\nSaved {os.path.relpath(RESULTS_CSV, ROOT)}, "
          f"{os.path.relpath(STABILITY_PNG, ROOT)}, {os.path.relpath(SUMMARY_PNG, ROOT)}")


if __name__ == "__main__":
    main()
