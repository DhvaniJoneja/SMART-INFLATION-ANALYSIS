"""
load_wpi.py - turn the wide WPI sheet into a clean MONTHLY table of headline indices.

Place at:  <project root>/src/ingest/load_wpi.py
Run from the project root:  python -m src.ingest.load_wpi

Input   monthly_index_202606.xls, sheet "MONTHLY_INDEX"
        (one row per commodity, one column per month: INDX042012 ... INDX042026,
         base 2011-12 = 100; reading .xls needs:  pip install xlrd)
Output  data/interim/wpi_monthly.csv   one row per month, Apr 2012 onward

Only the six top-level rows are kept (the ~860 sub-item rows are not used):
  wpi_all, wpi_primary, wpi_food_articles, wpi_fuel, wpi_manuf, wpi_food_index
"""
import re

import pandas as pd

from src import config as cfg

MONTH_COL = re.compile(r"^INDX(\d{2})(\d{4})$")


def read_sheet() -> pd.DataFrame:
    try:
        df = pd.read_excel(cfg.WPI_FILE, sheet_name=cfg.WPI_SHEET)
    except ImportError as e:
        raise SystemExit(f"{e}\nInstall the .xls reader first:  pip install xlrd")
    df.columns = [str(c).strip() for c in df.columns]
    print(f"Sheet '{cfg.WPI_SHEET}': {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["COMM_CODE"] = pd.to_numeric(df["COMM_CODE"], errors="coerce")

    month_cols = [c for c in df.columns if MONTH_COL.match(c)]
    print(f"Month columns found: {len(month_cols)} ({month_cols[0]} ... {month_cols[-1]})")

    code_to_name = {code: name for name, code in cfg.WPI_CODES.items()}
    top = df[df["COMM_CODE"].isin(code_to_name)].copy()

    found = top["COMM_CODE"].astype("int64").value_counts()
    for code, name in code_to_name.items():
        n = int(found.get(code, 0))
        if n != 1:
            raise ValueError(f"Expected exactly 1 row for {name} (code {code}), found {n}")

    print("\nRows kept (code, name, weight):")
    print(top[["COMM_CODE", "COMM_NAME", "COMM_WT"]].to_string(index=False))

    w = top.set_index(top["COMM_CODE"].astype("int64").map(code_to_name))["COMM_WT"].astype(float)
    parts = w["wpi_primary"] + w["wpi_fuel"] + w["wpi_manuf"]
    print(f"\nWeight check: primary + fuel + manufactured = {parts:.3f} (should be 100)")

    long = top.melt(id_vars=["COMM_CODE"], value_vars=month_cols,
                    var_name="col", value_name="value")
    mm = long["col"].str.extract(MONTH_COL).astype(int)
    long["month"] = pd.to_datetime(dict(year=mm[1], month=mm[0], day=1))
    long["name"] = long["COMM_CODE"].astype("int64").map(code_to_name)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    zeros = (long["value"] <= 0).sum()
    if zeros:
        print(f"NOTE: {zeros} zero/negative index values treated as missing")
        long.loc[long["value"] <= 0, "value"] = float("nan")

    wide = long.pivot(index="month", columns="name", values="value")
    wide = wide[list(cfg.WPI_CODES)]          # keep the config order
    return wide.sort_index().reset_index()


def main():
    wpi = build_monthly(read_sheet())

    expected = pd.date_range(cfg.START_MONTH, cfg.END_MONTH, freq="MS")
    study = wpi[(wpi["month"] >= cfg.START_MONTH) & (wpi["month"] <= cfg.END_MONTH)]
    absent = expected.difference(study["month"])
    print(f"\nMonths in study period: {len(study)} of {len(expected)}"
          + (f"  MISSING: {list(absent.strftime('%Y-%m'))}" if len(absent) else "  (complete)"))
    cols = list(cfg.WPI_CODES)
    print("Missing values per column in study period:")
    print(study[cols].isna().sum().to_string())

    jumps = wpi.set_index("month")[cols].pct_change(fill_method=None).abs().stack()
    jumps = jumps[jumps > 0.08]
    if len(jumps):
        print("\nMonth-on-month moves above 8% (check they look real, e.g. 2020 or 2022):")
        print((jumps * 100).round(1).to_string())

    wpi.to_csv(cfg.WPI_MONTHLY_CSV, index=False)
    print(f"\nSaved {cfg.WPI_MONTHLY_CSV}  ({len(wpi)} months, "
          f"{wpi['month'].min():%Y-%m} to {wpi['month'].max():%Y-%m})")

    spot = pd.to_datetime(["2012-04-01", "2013-01-01", "2020-04-01", "2022-06-01", "2025-12-01"])
    print("\nSpot-check (Apr 2012 all-commodities should be 104.7):")
    print(wpi[wpi["month"].isin(spot)].to_string(index=False))
    print("\nLast 3 months (the latest may be provisional and later revised):")
    print(wpi.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
