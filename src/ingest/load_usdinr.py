"""
load_usdinr.py - turn the RBI daily USD/INR file into a clean MONTHLY table.

Place at:  <project root>/src/ingest/load_usdinr.py
Run from the project root:  python -m src.ingest.load_usdinr

Input   r691_daily_interbank_forex_rates_usd_vis_a_vis_inr.csv
        (wrapper columns src_file, tab, period, row_no, c1..c6; real header is on a
         data row, dates look like "30 June 2026")
Output  data/interim/forex_monthly.csv   one row per month, Aug 2000 onward

Columns written
  usdinr_avg    monthly mean of the RBI reference rate (main series)
  usdinr_end    last reference rate of the month
  usdinr_min / usdinr_max
  usdinr_vol    std of DAILY % changes within the month (volatility)
  usdinr_close_avg  monthly mean of the interbank close (cross-check only)
  fx_days       number of daily observations in the month

The table is NOT cut to 2013-2025 here, so later steps can compute changes for
Jan 2013 (needs earlier months). The merge step slices to the study period.
"""
import pandas as pd

from src import config as cfg


def read_daily() -> pd.DataFrame:
    # pandas handles the quoted multi-line cell ("Figures in Rs.\n") on row 8
    # index_col=False stops pandas shifting columns if any row has an extra comma
    raw = pd.read_csv(cfg.FOREX_FILE, dtype=str, encoding_errors="replace",
                      index_col=False)
    print(f"Raw file: {len(raw)} data rows, columns: {list(raw.columns)}")

    missing = [c for c in cfg.FOREX_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"Expected columns {missing} not found. Got {list(raw.columns)}")

    df = raw[list(cfg.FOREX_COLS)].rename(columns=cfg.FOREX_COLS)
    df["date"] = pd.to_datetime(
        df["date"].str.strip(), format=cfg.FOREX_DATE_FORMAT, errors="coerce"
    )
    junk = df["date"].isna().sum()
    df = df.dropna(subset=["date"]).copy()
    print(f"Dropped {junk} non-data rows (titles, blanks); {len(df)} daily rows remain")

    for c in ["open", "high", "low", "close", "ref_rate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    dup = df.duplicated("date", keep="first").sum()
    if dup:
        print(f"WARNING: dropping {dup} duplicate dates")
        df = df.drop_duplicates("date", keep="first")

    df = df.sort_values("date").reset_index(drop=True)
    print(f"Date range: {df['date'].min():%d %b %Y} to {df['date'].max():%d %b %Y}")
    return df


def add_rate_and_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n_missing = df[cfg.FOREX_VALUE_COL].isna().sum()
    # use the RBI reference rate; fall back to the interbank close if it is blank
    df["rate"] = df[cfg.FOREX_VALUE_COL].fillna(df["close"])
    if n_missing:
        print(f"{n_missing} days had no reference rate; used the close instead "
              f"({df['rate'].isna().sum()} still missing)")
    df["ret"] = df["rate"].pct_change() * 100

    big = df[df["ret"].abs() > 2][["date", "rate", "ret"]]
    if len(big):
        print(f"\nDaily moves larger than 2% ({len(big)} days, check they look real):")
        print(big.head(10).round(3).to_string(index=False))
    return df


def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    g = df.groupby("month")
    m = pd.DataFrame({
        "usdinr_avg": g["rate"].mean(),
        "usdinr_end": g["rate"].last(),
        "usdinr_min": g["rate"].min(),
        "usdinr_max": g["rate"].max(),
        "usdinr_vol": g["ret"].std(),
        "usdinr_close_avg": g["close"].mean(),
        "fx_days": g["rate"].count(),
    }).round(4)
    return m.reset_index()


def main():
    daily = read_daily()
    daily = add_rate_and_returns(daily)
    monthly = to_monthly(daily)

    # ---- checks inside the study period
    study = monthly[(monthly["month"] >= cfg.START_MONTH) & (monthly["month"] <= cfg.END_MONTH)]
    expected = pd.date_range(cfg.START_MONTH, cfg.END_MONTH, freq="MS")
    absent = expected.difference(study["month"])
    print(f"\nMonths in study period: {len(study)} of {len(expected)}"
          + (f"  MISSING: {list(absent.strftime('%Y-%m'))}" if len(absent) else "  (complete)"))
    thin = study[study["fx_days"] < 15]
    if len(thin):
        print("Months with fewer than 15 daily observations:")
        print(thin[["month", "fx_days"]].to_string(index=False))

    gap = (monthly["usdinr_avg"] - monthly["usdinr_close_avg"]).abs()
    print(f"Reference-rate avg vs close avg: mean diff {gap.mean():.3f}, max {gap.max():.3f} rupees")

    monthly.to_csv(cfg.FOREX_MONTHLY_CSV, index=False)
    print(f"\nSaved {cfg.FOREX_MONTHLY_CSV}  ({len(monthly)} months, "
          f"{monthly['month'].min():%Y-%m} to {monthly['month'].max():%Y-%m})")

    show = monthly[monthly["month"].isin(pd.to_datetime(
        ["2013-01-01", "2013-09-01", "2020-03-01", "2025-12-01"]))]
    print("\nSpot-check months (rough expectations: ~54 in Jan 2013, ~63-64 in Sep 2013, "
          "~74 in Mar 2020):")
    print(show[["month", "usdinr_avg", "usdinr_end", "usdinr_vol", "fx_days"]]
          .to_string(index=False))
    print("\nNulls per column:")
    print(monthly.isna().sum().to_string())


if __name__ == "__main__":
    main()
