"""
build_features.py - create the modelling table: changes, lags, rolling means, labels.

Place at:  <project root>/src/features/build_features.py
Run from the project root:  python -m src.features.build_features

Input   data/processed/master_monthly.csv
Output  data/processed/features.csv           144 modelling rows (Jan 2014 - Dec 2025)
        data/processed/feature_sets.json      which columns each task may use
        data/processed/features.arff          numeric features + class, for WEKA
        data/processed/features_nominal.arff  same, every feature binned low/mid/high
                                              (WEKA's ID3 needs nominal attributes)

Targets
  infl_level        LOW (<4%), MEDIUM (4-6%), HIGH (>6%) from headline YoY inflation
  target_next_yoy   next month's YoY inflation (for the neural network); the last row
                    (Dec 2025) has none because CPI ends there

Feature sets (saved in feature_sets.json)
  CLF_FEATURES        for the classifiers. NO CPI-based columns (they define the label)
  REG_EXOG            for next-month prediction, outside indicators only
  REG_WITH_CPI        REG_EXOG + current/lagged inflation (allowed: the target is t+1)
  CLUSTER_FEATURES    for K-Means (descriptive, so CPI columns are fine)
  ARM_COLUMNS         columns discretised for Apriori in R
"""
import json

import pandas as pd

from src import config as cfg

FEATURE_SETS_JSON = cfg.PROCESSED_DIR / "feature_sets.json"

# master column -> short name used in the feature names
SHORT = {
    "usdinr_avg": "usdinr", "brent_usd": "brent_usd", "brent_inr": "brent_inr",
    "wpi_all": "wpi_all", "wpi_fuel": "wpi_fuel", "wpi_food_articles": "wpi_food",
    "wpi_manuf": "wpi_manuf", "wpi_primary": "wpi_primary",
}
ROLL_BASES = ["usdinr", "brent_inr", "wpi_fuel"]
LAG_BASES = ["usdinr_yoy", "brent_inr_yoy", "wpi_fuel_yoy", "wpi_food_yoy"]

CLF_FEATURES = [
    "usdinr_yoy", "usdinr_mom", "usdinr_vol",
    "brent_inr_yoy", "brent_inr_mom", "brent_usd_yoy",
    "wpi_all_yoy", "wpi_fuel_yoy", "wpi_food_yoy", "wpi_manuf_yoy",
    "repo", "repo_chg_6m",
    "usdinr_mom_r3", "brent_inr_mom_r3",
]
REG_EXOG = CLF_FEATURES + [f"{b}_lag{k}" for b in LAG_BASES for k in (1, 3)]
REG_WITH_CPI = REG_EXOG + ["infl_yoy", "infl_yoy_lag1", "infl_yoy_lag3", "cfpi_yoy", "fuel_yoy"]
CLUSTER_FEATURES = ["infl_yoy", "cfpi_yoy", "fuel_yoy", "usdinr_yoy",
                    "brent_inr_yoy", "wpi_all_yoy", "repo"]
ARM_COLUMNS = ["infl_yoy", "cfpi_yoy", "fuel_yoy", "usdinr_yoy",
               "brent_inr_yoy", "wpi_all_yoy", "wpi_fuel_yoy", "repo"]


def label(v):
    if pd.isna(v):
        return pd.NA
    if v < cfg.LOW_MAX:
        return "LOW"
    return "HIGH" if v > cfg.HIGH_MIN else "MEDIUM"


def build(master: pd.DataFrame) -> pd.DataFrame:
    df = master.sort_values("month").reset_index(drop=True).copy()

    # ---- month-on-month and year-on-year % changes of outside indicators
    for col, short in SHORT.items():
        df[f"{short}_mom"] = df[col].pct_change(1, fill_method=None) * 100
        df[f"{short}_yoy"] = df[col].pct_change(12, fill_method=None) * 100

    # ---- repo: change over 6 months (percentage points) and a real-rate version
    df["repo_chg_6m"] = df["repo"] - df["repo"].shift(6)
    df["repo_real"] = df["repo"] - df["infl_yoy"]          # descriptive use only

    # ---- rolling means of month-on-month changes
    for base in ROLL_BASES:
        for w in cfg.ROLL_WINDOWS:
            df[f"{base}_mom_r{w}"] = df[f"{base}_mom"].rolling(w).mean()

    # ---- lags
    for base in LAG_BASES:
        for k in cfg.LAGS:
            df[f"{base}_lag{k}"] = df[base].shift(k)
    for k in cfg.LAGS:
        df[f"infl_yoy_lag{k}"] = df["infl_yoy"].shift(k)

    # ---- calendar + targets
    df["year"] = df["month"].dt.year
    df["month_num"] = df["month"].dt.month
    df["infl_level"] = df["infl_yoy"].apply(label)
    df["target_next_yoy"] = df["infl_yoy"].shift(-1)
    return df


def to_arff(df, features, path, relation, nominal=False):
    lines = [f"@relation {relation}", ""]
    for f in features:
        lines.append(f"@attribute {f} " + ("{low,mid,high}" if nominal else "numeric"))
    lines.append("@attribute infl_level {LOW,MEDIUM,HIGH}")
    lines += ["", "@data"]
    for _, r in df.iterrows():
        vals = [str(r[f]) if nominal else f"{r[f]:.4f}" for f in features]
        lines.append(",".join(vals + [str(r["infl_level"])]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if not cfg.MASTER_CSV.exists():
        raise SystemExit(f"{cfg.MASTER_CSV} not found. Run first:  python -m src.preprocess.merge")
    master = pd.read_csv(cfg.MASTER_CSV, parse_dates=["month"])
    df = build(master)

    # ---- choose the modelling rows
    in_study = df["in_study"] == 1
    has_label = df["infl_yoy"].notna()
    complete = df[CLF_FEATURES].notna().all(axis=1)
    rows = df[in_study & has_label & complete].copy()
    dropped = df[in_study & has_label & ~complete]
    print(f"Study months: {int(in_study.sum())}; with inflation: {int((in_study & has_label).sum())}; "
          f"modelling rows: {len(rows)}")
    if len(dropped):
        print("Dropped because a classifier feature is blank:", list(dropped["month"].dt.strftime("%Y-%m")))
    print(f"Modelling period: {rows['month'].min():%Y-%m} to {rows['month'].max():%Y-%m}")

    # ---- class balance, overall / by split / by year
    print("\nClass counts:")
    print(rows["infl_level"].value_counts().reindex(cfg.LEVEL_LABELS).fillna(0).astype(int).to_string())
    rows["split"] = rows["month"].apply(
        lambda m: "train" if m <= pd.Timestamp(cfg.TRAIN_END)
        else ("val" if m <= pd.Timestamp(cfg.VAL_END) else "test"))
    print(f"\nBy time split (train <= {cfg.TRAIN_END[:7]}, val <= {cfg.VAL_END[:7]}, test after):")
    print(pd.crosstab(rows["split"], rows["infl_level"]).reindex(
        index=["train", "val", "test"], columns=cfg.LEVEL_LABELS, fill_value=0).to_string())
    print("\nBy year:")
    print(pd.crosstab(rows["year"], rows["infl_level"]).reindex(
        columns=cfg.LEVEL_LABELS, fill_value=0).to_string())

    # ---- feature sets (check they all exist)
    sets = {
        "CLF_FEATURES": CLF_FEATURES, "REG_EXOG": REG_EXOG, "REG_WITH_CPI": REG_WITH_CPI,
        "CLUSTER_FEATURES": CLUSTER_FEATURES, "ARM_COLUMNS": ARM_COLUMNS,
    }
    for name, cols in sets.items():
        missing = [c for c in cols if c not in rows.columns]
        if missing:
            raise ValueError(f"{name} refers to missing columns: {missing}")
        n_blank = int(rows[cols].isna().any(axis=1).sum())
        print(f"  {name:17s} {len(cols):2d} columns; rows with a blank value: {n_blank}")
    meta = dict(sets, levels=cfg.LEVEL_LABELS, low_max=cfg.LOW_MAX, high_min=cfg.HIGH_MIN,
                train_end=cfg.TRAIN_END, val_end=cfg.VAL_END)
    FEATURE_SETS_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- save
    rows.to_csv(cfg.FEATURES_CSV, index=False)
    to_arff(rows, CLF_FEATURES, cfg.FEATURES_ARFF, "inflation_level")
    nominal = rows.copy()
    for f in CLF_FEATURES:   # equal-size bins; rank() makes ties safe
        nominal[f] = pd.qcut(nominal[f].rank(method="first"), 3, labels=["low", "mid", "high"])
    to_arff(nominal, CLF_FEATURES, cfg.PROCESSED_DIR / "features_nominal.arff",
            "inflation_level_nominal", nominal=True)
    print(f"\nSaved {cfg.FEATURES_CSV}  ({len(rows)} rows x {rows.shape[1]} columns)")
    print(f"Saved {FEATURE_SETS_JSON}")
    print(f"Saved {cfg.FEATURES_ARFF} and features_nominal.arff (for WEKA)")

    corr = rows[CLF_FEATURES + ["infl_yoy"]].corr()["infl_yoy"].drop("infl_yoy")
    print("\nCorrelation of each classifier feature with headline inflation (strongest first):")
    print(corr.reindex(corr.abs().sort_values(ascending=False).index).round(2).to_string())


if __name__ == "__main__":
    main()
