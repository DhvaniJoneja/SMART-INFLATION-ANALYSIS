"""
eda.py - exploratory data analysis: statistics, correlations and plots.

Place at:  <project root>/src/eda/eda.py
Run from the project root:  python -m src.eda.eda

Reads   data/processed/features.csv  and  data/interim/cpi_long.csv
Writes  outputs/figures/*.png        (8 figures, paste them into the report)
        outputs/tables/*.csv         (summary stats, correlations, lead-lag tables)

Questions it answers (from the project plan)
  1. How has India's inflation changed over time?
  2. Which CPI categories are the most volatile?
  3. How do USD/INR, crude oil, WPI and the repo rate move with inflation?
  4. Does crude oil movement coincide with fuel inflation, and which indicator leads?
"""
import matplotlib
matplotlib.use("Agg")                      # draw to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as cfg

LEVEL_COLORS = {"LOW": "#2a9d8f", "MEDIUM": "#e9c46a", "HIGH": "#e76f51"}


def save(fig, name):
    path = cfg.FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.name}")


def table(df, name, index=True):
    df.to_csv(cfg.TABLE_DIR / name, index=index)


# ------------------------------------------------------------------ figures
def fig_inflation_trend(df):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df["month"], df["infl_yoy"], color="#264653", lw=2, label="Headline CPI inflation (YoY %)")
    ax.axhline(cfg.LOW_MAX, color="#2a9d8f", ls="--", lw=1, label="RBI target (4%)")
    ax.axhline(cfg.HIGH_MIN, color="#e76f51", ls="--", lw=1, label="Upper tolerance limit (6%)")
    ax.set_title("India headline CPI inflation, 2014-2025")
    ax.set_ylabel("% year-on-year")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "01_inflation_trend.png")


def fig_cpi_groups(df):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    series = [("infl_yoy", "Headline", "#264653", 2.2), ("cfpi_yoy", "Food (CFPI)", "#e76f51", 1.3),
              ("fuel_yoy", "Fuel and light", "#2a9d8f", 1.3), ("transport_yoy", "Transport & comm.", "#8d99ae", 1.3)]
    for col, name, color, lw in series:
        ax.plot(df["month"], df[col], label=name, color=color, lw=lw)
    ax.set_title("Year-on-year inflation: headline vs main CPI groups")
    ax.set_ylabel("% year-on-year")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(alpha=0.3)
    save(fig, "02_cpi_groups_yoy.png")


def fig_usdinr(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["month"], df["usdinr_avg"], color="#6a4c93", lw=2)
    ax.set_title("USD/INR monthly average (RBI reference rate)")
    ax.set_ylabel("Rupees per US dollar")
    ax.grid(alpha=0.3)
    save(fig, "03_usdinr_trend.png")


def fig_oil_vs_fuel(df):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(df["month"], df["brent_inr_yoy"], color="#e76f51", lw=1.6, label="Brent in rupees (YoY %)")
    axes[0].plot(df["month"], df["wpi_fuel_yoy"], color="#264653", lw=1.6, label="WPI fuel & power (YoY %)")
    axes[0].axhline(0, color="grey", lw=0.8)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("% YoY")
    axes[0].set_title("Crude oil and wholesale fuel prices vs CPI fuel inflation")
    axes[1].plot(df["month"], df["fuel_yoy"], color="#2a9d8f", lw=1.8, label="CPI fuel and light (YoY %)")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_ylabel("% YoY")
    for a in axes:
        a.grid(alpha=0.3)
    save(fig, "04_oil_vs_fuel_inflation.png")


def fig_repo_vs_inflation(df):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df["month"], df["infl_yoy"], color="#264653", lw=2, label="Headline inflation (YoY %)")
    ax.step(df["month"], df["repo"], where="post", color="#e76f51", lw=2, label="Policy repo rate (%)")
    ax.set_title("RBI repo rate and headline inflation")
    ax.set_ylabel("%")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    save(fig, "05_repo_vs_inflation.png")


def fig_corr(df, cols):
    corr = df[cols].corr()
    table(corr.round(3), "correlation_matrix.csv")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation between inflation and its possible drivers")
    save(fig, "06_correlation_heatmap.png")
    return corr


def fig_class_by_year(df):
    ct = pd.crosstab(df["year"], df["infl_level"]).reindex(columns=cfg.LEVEL_LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bottom = np.zeros(len(ct))
    for lvl in cfg.LEVEL_LABELS:
        ax.bar(ct.index, ct[lvl], bottom=bottom, color=LEVEL_COLORS[lvl], label=lvl)
        bottom += ct[lvl].values
    ax.set_xticks(ct.index)
    ax.set_title("Months per year in each inflation level (LOW <4%, MEDIUM 4-6%, HIGH >6%)")
    ax.set_ylabel("Months")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    save(fig, "07_levels_by_year.png")
    table(ct, "levels_by_year.csv")


def fig_subgroup_volatility(cpi):
    c = cpi[(cpi["sector"] == cfg.CPI_MODEL_SECTOR) & (cpi["month"] >= "2014-01-01")
            & cpi["infl_yoy_calc"].notna()]
    vol = (c.groupby("subgroup")["infl_yoy_calc"]
           .agg(mean="mean", std="std", min="min", max="max").round(2)
           .sort_values("std", ascending=False))
    table(vol, "cpi_subgroup_volatility.csv")
    fig, ax = plt.subplots(figsize=(9, 8))
    v = vol.sort_values("std")
    ax.barh(v.index, v["std"], color="#264653")
    ax.set_title("Volatility of year-on-year inflation by CPI subgroup, 2014-2025\n(standard deviation, percentage points)")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3)
    save(fig, "08_subgroup_volatility.png")
    return vol


# ------------------------------------------------------------------ tables
def lead_lag(df):
    """corr(indicator at t-k, inflation at t): which outside indicator LEADS inflation?"""
    ind = ["usdinr_yoy", "brent_inr_yoy", "wpi_fuel_yoy", "wpi_food_yoy", "repo_chg_6m"]
    a = pd.DataFrame({x: {f"lag{k}": df[x].shift(k).corr(df["infl_yoy"]) for k in range(7)}
                      for x in ind}).T.round(2)
    b = pd.DataFrame({x: {f"lag{k}": df[x].shift(k).corr(df["fuel_yoy"]) for k in range(7)}
                      for x in ["brent_inr_yoy", "wpi_fuel_yoy"]}).T.round(2)
    return a, b


def main():
    for p, runner in ((cfg.FEATURES_CSV, "features.build_features"), (cfg.CPI_LONG_CSV, "ingest.load_cpi")):
        if not p.exists():
            raise SystemExit(f"{p} not found. Run first:  python -m src.{runner}")
    df = pd.read_csv(cfg.FEATURES_CSV, parse_dates=["month"])
    cpi = pd.read_csv(cfg.CPI_LONG_CSV, parse_dates=["month"])
    print(f"Loaded {len(df)} monthly rows ({df['month'].min():%Y-%m} to {df['month'].max():%Y-%m})")

    # ---- descriptive statistics
    keys = ["infl_yoy", "cfpi_yoy", "fuel_yoy", "usdinr_avg", "brent_usd", "brent_inr",
            "wpi_all", "wpi_food_articles", "wpi_fuel", "repo"]
    stats = df[keys].describe().T[["mean", "50%", "std", "min", "max"]].rename(columns={"50%": "median"}).round(2)
    table(stats, "summary_stats.csv")
    print("\nSummary statistics:")
    print(stats.to_string())

    hi, lo = df.loc[df["infl_yoy"].idxmax()], df.loc[df["infl_yoy"].idxmin()]
    print(f"\nHighest inflation: {hi['infl_yoy']:.2f}% in {hi['month']:%b %Y}; "
          f"lowest: {lo['infl_yoy']:.2f}% in {lo['month']:%b %Y}")

    # ---- profile of each inflation level
    prof_cols = ["cfpi_yoy", "fuel_yoy", "usdinr_yoy", "brent_inr_yoy", "wpi_all_yoy", "wpi_food_yoy", "repo"]
    prof = df.groupby("infl_level")[prof_cols].mean().reindex(cfg.LEVEL_LABELS).round(2)
    table(prof, "level_profile.csv")
    print("\nAverage of each indicator in LOW / MEDIUM / HIGH inflation months:")
    print(prof.to_string())

    # ---- figures
    print("\nFigures:")
    fig_inflation_trend(df)
    fig_cpi_groups(df)
    fig_usdinr(df)
    fig_oil_vs_fuel(df)
    fig_repo_vs_inflation(df)
    corr = fig_corr(df, ["infl_yoy", "cfpi_yoy", "fuel_yoy", "housing_yoy", "transport_yoy",
                         "usdinr_yoy", "brent_inr_yoy", "wpi_all_yoy", "wpi_food_yoy",
                         "wpi_fuel_yoy", "repo"])
    fig_class_by_year(df)
    vol = fig_subgroup_volatility(cpi)

    print("\nMost volatile CPI subgroups (std of YoY inflation, points):")
    print(vol.head(6)[["mean", "std"]].to_string())
    print("Least volatile:")
    print(vol.tail(3)[["mean", "std"]].to_string())

    # ---- lead-lag
    a, b = lead_lag(df)
    table(a, "leadlag_vs_inflation.csv")
    table(b, "leadlag_vs_fuel_inflation.csv")
    print("\nCorrelation of indicator at t-k with headline inflation at t (lag0 = same month):")
    print(a.to_string())
    print("\nCorrelation of crude/WPI fuel at t-k with CPI fuel inflation at t:")
    print(b.to_string())

    top = corr["infl_yoy"].drop("infl_yoy")
    print("\nStrongest links to headline inflation:", ", ".join(
        f"{k} ({v:+.2f})" for k, v in top.reindex(top.abs().sort_values(ascending=False).index).head(4).items()))
    print(f"\nFigures in {cfg.FIG_DIR}\nTables  in {cfg.TABLE_DIR}")


if __name__ == "__main__":
    main()
