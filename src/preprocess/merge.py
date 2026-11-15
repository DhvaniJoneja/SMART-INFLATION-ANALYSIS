"""
merge.py - join the five monthly tables into ONE master table.

Place at:  <project root>/src/preprocess/merge.py
Run from the project root:  python -m src.preprocess.merge
(run the five loaders in src/ingest first)

Output  data/processed/master_monthly.csv

Rows    Apr 2012 to Dec 2025 (165 months). The study period is Jan 2013 to Dec 2025;
        the extra 9 months before it are kept ONLY so that year-on-year changes and
        lags of USD/INR, oil, WPI and repo can be computed for 2013-2014 without
        losing rows. They are flagged with in_study = 0 and are never used as
        modelling rows. CPI columns are blank before Jan 2013 (CPI starts then).

Columns
  month, in_study
  CPI (Combined)   cpi_general, cpi_cfpi, cpi_food_bev, cpi_fuel, cpi_housing,
                   cpi_transport, cpi_clothing, cpi_misc, and their *_yoy columns
                   (infl_yoy = headline YoY inflation), infl_yoy_reported, cpi_status
  USD/INR          usdinr_avg, usdinr_end, usdinr_vol
  Crude oil        brent_usd, dubai_usd, wti_usd, crude_avg_usd, brent_inr
  WPI              wpi_all, wpi_primary, wpi_food_articles, wpi_fuel, wpi_manuf,
                   wpi_food_index
  Policy           repo (rate in force at month end), repo_avg
"""
import pandas as pd

from src import config as cfg

MASTER_START = "2012-04-01"   # first month of WPI data


def load(path, name, runner):
    if not path.exists():
        raise SystemExit(f"{name} not found: {path}\nRun first:  python -m src.ingest.{runner}")
    df = pd.read_csv(path, parse_dates=["month"])
    if df["month"].duplicated().any():
        raise ValueError(f"{name} has duplicate months")
    print(f"  {name:12s} {len(df):4d} months  {df['month'].min():%Y-%m} to {df['month'].max():%Y-%m}")
    return df.set_index("month")


def main():
    print("Loading interim tables:")
    cpi = load(cfg.CPI_MONTHLY_CSV, "CPI", "load_cpi")
    fx = load(cfg.FOREX_MONTHLY_CSV, "USD/INR", "load_usdinr")[["usdinr_avg", "usdinr_end", "usdinr_vol"]]
    crude = load(cfg.CRUDE_MONTHLY_CSV, "Crude oil", "load_crude")
    wpi = load(cfg.WPI_MONTHLY_CSV, "WPI", "load_wpi")
    repo = load(cfg.REPO_MONTHLY_CSV, "Repo", "load_repo")[["repo_end", "repo_avg"]]
    repo = repo.rename(columns={"repo_end": "repo"})

    months = pd.date_range(MASTER_START, cfg.END_MONTH, freq="MS")
    master = pd.DataFrame(index=pd.Index(months, name="month"))
    for part in (cpi, fx, crude, wpi, repo):
        master = master.join(part, how="left")

    # Brent in rupees: monthly average Brent x monthly average USD/INR (approximation)
    master["brent_inr"] = (master["brent_usd"] * master["usdinr_avg"]).round(2)
    master.insert(0, "in_study", (master.index >= cfg.START_MONTH).astype(int))
    master = master.reset_index()

    # ---------------- checks ----------------
    study = master[master["in_study"] == 1]
    pre = master[master["in_study"] == 0]
    print(f"\nMaster table: {len(master)} rows x {master.shape[1]} columns "
          f"({len(study)} in study period, {len(pre)} earlier months for look-back)")

    nulls = study.isna().sum()
    nulls = nulls[nulls > 0]
    print("\nMissing values INSIDE the study period (Jan 2013 - Dec 2025):")
    if nulls.empty:
        print("  none")
    else:
        print(nulls.to_string())
        print("  (expected: the *_yoy columns have 12 blanks for 2013, infl_yoy_reported 14)")

    unexpected = [c for c in nulls.index
                  if not (c.endswith("_yoy") or c in ("infl_yoy", "infl_yoy_reported"))]
    if unexpected:
        print(f"\n!! UNEXPECTED blanks in: {unexpected} - tell me before continuing")

    pre_nulls = pre.drop(columns=["month", "in_study"]).isna().sum()
    pre_cpi = [c for c in pre_nulls.index if c.startswith(("cpi_", "infl_"))
               or c.endswith("_yoy")]
    non_cpi_pre = pre_nulls.drop(pre_cpi)
    non_cpi_pre = non_cpi_pre[non_cpi_pre > 0]
    if len(non_cpi_pre):
        print("\nBlanks before 2013 outside the CPI columns (should be none):")
        print(non_cpi_pre.to_string())

    master.to_csv(cfg.MASTER_CSV, index=False)
    print(f"\nSaved {cfg.MASTER_CSV}")

    show = ["month", "infl_yoy", "usdinr_avg", "brent_usd", "brent_inr", "wpi_all", "wpi_fuel", "repo"]
    print("\nFirst 3 and last 3 rows (selected columns):")
    print(pd.concat([master[show].head(3), master[show].tail(3)]).to_string(index=False))

    print("\nCorrelation with headline inflation (study period, levels and YoY):")
    corr_cols = ["usdinr_avg", "brent_inr", "wpi_all", "repo", "fuel_yoy", "cfpi_yoy"]
    print(study[["infl_yoy"] + corr_cols].corr()["infl_yoy"].drop("infl_yoy").round(2).to_string())


if __name__ == "__main__":
    main()
