"""
load_crude.py - extract monthly crude oil prices from the World Bank Pink Sheet.

Place at:  <project root>/src/ingest/load_crude.py
Run from the project root:  python -m src.ingest.load_crude

Input   CMO-Historical-Data-Monthly.xlsx, sheet "Monthly Prices"
        (4 title rows, commodity names on row 5, units on row 6, data from row 7;
         periods look like "1960M01"; missing values are written as "…")
Output  data/interim/crude_monthly.csv   one row per month, 2000 onward

Columns written (all in nominal US$ per barrel)
  brent_usd, dubai_usd, wti_usd, crude_avg_usd

Brent in rupees is built later (needs USD/INR), in the merge step.
"""
import pandas as pd

from src import config as cfg

KEEP_FROM = "2000-01-01"   # earlier months are not needed


def read_prices() -> pd.DataFrame:
    raw = pd.read_excel(cfg.PINK_FILE, sheet_name=cfg.PINK_PRICES_SHEET, header=None)
    print(f"Sheet '{cfg.PINK_PRICES_SHEET}': {raw.shape[0]} rows x {raw.shape[1]} columns")

    names = raw.iloc[cfg.PINK_HEADER_ROW].astype(str).str.strip()
    units = raw.iloc[cfg.PINK_HEADER_ROW + 1].astype(str).str.strip()

    cols = {}
    for sheet_name, clean_name in cfg.PINK_COLS.items():
        hits = names[names == sheet_name].index.tolist()
        if not hits:
            crude_like = names[names.str.contains("rude", na=False)].tolist()
            raise ValueError(f"Column '{sheet_name}' not found. Crude-like columns: {crude_like}")
        cols[clean_name] = hits[0]
        print(f"  {clean_name:14s} <- '{sheet_name}'  unit {units[hits[0]]}  (column {hits[0]})")

    data = raw.iloc[cfg.PINK_DATA_START_ROW:].copy()
    period = data.iloc[:, 0].astype(str).str.strip()
    ok = period.str.match(r"^\d{4}M\d{2}$")
    print(f"Data rows with a valid period (e.g. 1960M01): {ok.sum()} "
          f"(skipped {(~ok).sum()} other rows)")
    data, period = data[ok], period[ok]

    out = pd.DataFrame({"month": pd.to_datetime(period.str[:4] + "-" + period.str[5:7] + "-01")})
    for clean_name, idx in cols.items():
        # "…" and "..." (not available) become NaN
        out[clean_name] = pd.to_numeric(data.iloc[:, idx], errors="coerce").values
    return out.sort_values("month").reset_index(drop=True)


def main():
    df = read_prices()
    print(f"Full range in file: {df['month'].min():%Y-%m} to {df['month'].max():%Y-%m}")

    df = df[df["month"] >= KEEP_FROM].reset_index(drop=True)
    value_cols = list(cfg.PINK_COLS.values())

    study = df[(df["month"] >= cfg.START_MONTH) & (df["month"] <= cfg.END_MONTH)]
    expected = pd.date_range(cfg.START_MONTH, cfg.END_MONTH, freq="MS")
    absent = expected.difference(study["month"])
    print(f"\nMonths in study period: {len(study)} of {len(expected)}"
          + (f"  MISSING: {list(absent.strftime('%Y-%m'))}" if len(absent) else "  (complete)"))
    print("Missing values per column in study period:")
    print(study[value_cols].isna().sum().to_string())

    gap = (study["brent_usd"] - study["dubai_usd"]).abs()
    print(f"\nBrent vs Dubai: mean abs gap {gap.mean():.2f} $/bbl, max {gap.max():.2f}, "
          f"correlation {study['brent_usd'].corr(study['dubai_usd']):.3f}")

    df.to_csv(cfg.CRUDE_MONTHLY_CSV, index=False)
    print(f"\nSaved {cfg.CRUDE_MONTHLY_CSV}  ({len(df)} months, "
          f"{df['month'].min():%Y-%m} to {df['month'].max():%Y-%m})")

    spot = pd.to_datetime(["2013-01-01", "2015-12-01", "2020-04-01", "2022-06-01", "2025-12-01"])
    print("\nSpot-check Brent (rough expectations: ~113 Jan 2013, ~38 Dec 2015, "
          "~18 Apr 2020, ~123 Jun 2022):")
    print(df[df["month"].isin(spot)][["month"] + value_cols].to_string(index=False))
    print("\nLast 3 months available:")
    print(df.tail(3)[["month"] + value_cols].to_string(index=False))


if __name__ == "__main__":
    main()
