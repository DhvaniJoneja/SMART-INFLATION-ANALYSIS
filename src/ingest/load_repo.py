"""
load_repo.py - turn the RBI policy-rate history into a clean MONTHLY repo-rate table.

Place at:  <project root>/src/ingest/load_repo.py
Run from the project root:  python -m src.ingest.load_repo

Input   rates-policy.csv  (a row only on dates when some rate changed; column
        "Policy repo rate" is blank before the repo rate existed)
Output  data/interim/repo_monthly.csv   one row per month

Method  the rate stays in force until the next change, so we forward-fill it to a
        daily series and then take, for each month,
          repo_end = rate in force on the last day of the month
          repo_avg = average of the daily rate over the month (time-weighted)
        Forward-filling from before 2013 gives January 2013 its correct value.
"""
import pandas as pd

from src import config as cfg


def read_changes() -> pd.DataFrame:
    raw = pd.read_csv(cfg.REPO_FILE)
    raw.columns = [str(c).strip() for c in raw.columns]
    for c in (cfg.REPO_DATE_COL, cfg.REPO_RATE_COL):
        if c not in raw.columns:
            raise ValueError(f"Column '{c}' not found. Got {list(raw.columns)}")

    df = raw[[cfg.REPO_DATE_COL, cfg.REPO_RATE_COL]].copy()
    df.columns = ["date", "repo"]
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), errors="coerce")
    df["repo"] = pd.to_numeric(df["repo"], errors="coerce")

    bad_dates = df["date"].isna().sum()
    if bad_dates:
        print(f"WARNING: {bad_dates} rows with unreadable dates were dropped")
    df = df.dropna(subset=["date", "repo"]).sort_values("date")
    df = df.drop_duplicates("date", keep="last").reset_index(drop=True)
    print(f"Rows with a repo rate: {len(df)}  "
          f"({df['date'].min():%d %b %Y} to {df['date'].max():%d %b %Y})")
    return df


def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    first = df["date"].min()
    last_month_end = df["date"].max() + pd.offsets.MonthEnd(0)
    days = pd.date_range(first, last_month_end, freq="D")
    daily = df.set_index("date")["repo"].reindex(days).ffill()

    month = daily.index.to_period("M").to_timestamp()
    g = daily.groupby(month)
    out = pd.DataFrame({"repo_end": g.last(), "repo_avg": g.mean().round(4)})
    out.index.name = "month"
    return out.reset_index()


def main():
    changes = read_changes()

    # rows where the rate actually moved
    moved = changes[changes["repo"].ne(changes["repo"].shift())]
    window = moved[(moved["date"] >= "2012-06-01") & (moved["date"] <= "2026-12-31")]
    print(f"\nRepo rate CHANGES from mid-2012 on ({len(window)} changes) - compare with the RBI:")
    print(window.assign(date=window["date"].dt.strftime("%Y-%m-%d")).to_string(index=False))

    monthly = to_monthly(changes)

    expected = pd.date_range(cfg.START_MONTH, cfg.END_MONTH, freq="MS")
    study = monthly[(monthly["month"] >= cfg.START_MONTH) & (monthly["month"] <= cfg.END_MONTH)]
    absent = expected.difference(study["month"])
    print(f"\nMonths in study period: {len(study)} of {len(expected)}"
          + (f"  MISSING: {list(absent.strftime('%Y-%m'))}" if len(absent) else "  (complete)")
          + f";  missing values: {int(study['repo_end'].isna().sum())}")

    monthly.to_csv(cfg.REPO_MONTHLY_CSV, index=False)
    print(f"Saved {cfg.REPO_MONTHLY_CSV}  ({len(monthly)} months, "
          f"{monthly['month'].min():%Y-%m} to {monthly['month'].max():%Y-%m})")

    spot = pd.to_datetime(["2012-12-01", "2013-01-01", "2020-05-01", "2023-02-01", "2025-12-01"])
    print("\nSpot-check (rough expectations: 8.00 Dec 2012, 7.75 Jan 2013, 4.00 May 2020, "
          "6.50 Feb 2023, 5.25 Dec 2025):")
    print(monthly[monthly["month"].isin(spot)].to_string(index=False))


if __name__ == "__main__":
    main()
