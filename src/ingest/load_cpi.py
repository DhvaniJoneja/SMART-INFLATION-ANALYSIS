"""
load_cpi.py - read the 13 yearly CPI Excel files and build two clean tables.

Place at:  <project root>/src/ingest/load_cpi.py
Run from the project root:  python -m src.ingest.load_cpi

Outputs
  data/interim/cpi_long.csv     all sectors x all subgroups x 156 months (for the warehouse)
  data/interim/cpi_monthly.csv  Combined sector, one row per month (for modelling)

Notes
  * The year is taken from the 'year' column inside each file (file names are ignored).
  * Missing index values are filled by linear interpolation within each series.
  * Year-on-year inflation is RECOMPUTED from the index, because the reported
    'inflation' column is blank for all of 2013 and has gaps in 2020.
"""
import pandas as pd

from src import config as cfg

KEY = ["sector", "subgroup"]


def read_all() -> pd.DataFrame:
    files = sorted(cfg.CPI_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {cfg.CPI_DIR}")
    frames = []
    for f in files:
        df = pd.read_excel(f, sheet_name=0)
        df["source_file"] = f.name
        frames.append(df)
    print(f"Read {len(files)} files, {sum(len(x) for x in frames)} rows")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for c in ["state", "sector", "group", "subgroup", "status"]:
        df[c] = df[c].astype("string").str.strip()

    df = df[df["state"] == cfg.CPI_STATE].copy()
    df["month"] = pd.to_datetime(
        dict(year=df["year"], month=df["month_code"], day=1)
    )
    df["index"] = pd.to_numeric(df["index"], errors="coerce")
    df["inflation_reported"] = pd.to_numeric(df["inflation"], errors="coerce")

    dup = df.duplicated(subset=["month"] + KEY, keep="first").sum()
    if dup:
        print(f"WARNING: dropping {dup} duplicate (month, sector, subgroup) rows")
        df = df.drop_duplicates(subset=["month"] + KEY, keep="first")

    df = df.sort_values(KEY + ["month"]).reset_index(drop=True)
    return df


def fill_and_compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["index_was_missing"] = df["index"].isna()

    # ---- report missing index values before filling
    miss = df[df["index_was_missing"]]
    if len(miss):
        print(f"\nMissing index values: {len(miss)} rows. By sector / subgroup / year:")
        print(miss.groupby(["sector", "subgroup", miss["month"].dt.year])
              .size().to_string())
    else:
        print("\nNo missing index values.")

    # ---- fill gaps inside each series (linear in time)
    # Series that are mostly blank (e.g. Rural Housing, which is not published)
    # are left untouched, and gaps at the very start/end are never extrapolated.
    def _fill(s):
        if s.isna().mean() > 0.5:
            return s
        return s.interpolate(method="linear", limit_area="inside")

    blank = df.groupby(KEY)["index"].apply(lambda s: s.isna().mean() > 0.5)
    for (sec, sub), is_blank in blank.items():
        if is_blank:
            print(f"NOTE: '{sec} / {sub}' is mostly blank - left as missing, not filled")
    df["index"] = df.groupby(KEY)["index"].transform(_fill)

    # ---- recompute YoY inflation from the index
    df["infl_yoy_calc"] = (
        df.groupby(KEY)["index"].pct_change(periods=12, fill_method=None) * 100
    ).round(2)

    # ---- compare with the reported inflation (General, Combined)
    g = df[(df["sector"] == cfg.CPI_MODEL_SECTOR)
           & (df["subgroup"] == cfg.CPI_SUBGROUPS["cpi_general"])]
    both = g.dropna(subset=["inflation_reported", "infl_yoy_calc"])
    diff = (both["inflation_reported"] - both["infl_yoy_calc"]).abs()
    print(f"\nGeneral (Combined): recomputed vs reported YoY on {len(both)} months")
    print(f"  mean abs diff = {diff.mean():.3f}   max abs diff = {diff.max():.3f} "
          "(small differences are expected: reported values use unrounded indices)")
    big = both[diff > 0.3][["month", "inflation_reported", "infl_yoy_calc"]]
    if len(big):
        print("  Months differing by more than 0.3 pp (inspect these):")
        print(big.to_string(index=False))
    return df


def build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    comb = df[df["sector"] == cfg.CPI_MODEL_SECTOR]
    months = pd.date_range(cfg.START_MONTH, cfg.END_MONTH, freq="MS")
    wide = pd.DataFrame(index=pd.Index(months, name="month"))

    available = sorted(comb["subgroup"].dropna().unique())
    for col, sub in cfg.CPI_SUBGROUPS.items():
        s = comb[comb["subgroup"] == sub].set_index("month")
        if s.empty:
            raise ValueError(f"Subgroup '{sub}' not found. Available: {available}")
        wide[col] = s["index"]
        yoy_name = "infl_yoy" if col == "cpi_general" else col.replace("cpi_", "") + "_yoy"
        wide[yoy_name] = s["infl_yoy_calc"]

    gen = comb[comb["subgroup"] == cfg.CPI_SUBGROUPS["cpi_general"]].set_index("month")
    wide["infl_yoy_reported"] = gen["inflation_reported"]
    wide["cpi_status"] = gen["status"]
    return wide.reset_index()


def main():
    raw = read_all()
    df = clean(raw)
    expected = 156 * 3 * df["subgroup"].nunique()
    print(f"After cleaning: {len(df)} rows "
          f"({df['month'].min():%Y-%m} to {df['month'].max():%Y-%m}), "
          f"{df['subgroup'].nunique()} subgroups, sectors: {sorted(df['sector'].unique())}")
    print(f"Expected rows if complete: {expected}")

    df = fill_and_compute(df)

    long_cols = ["month", "year", "month_code", "sector", "group", "subgroup",
                 "index", "index_was_missing", "inflation_reported",
                 "infl_yoy_calc", "status"]
    df[long_cols].to_csv(cfg.CPI_LONG_CSV, index=False)

    monthly = build_monthly(df)
    monthly.to_csv(cfg.CPI_MONTHLY_CSV, index=False)

    print(f"\nSaved {cfg.CPI_LONG_CSV}  ({len(df)} rows)")
    print(f"Saved {cfg.CPI_MONTHLY_CSV}  ({len(monthly)} rows x {monthly.shape[1]} cols)")
    print("\nMonthly table, first 3 and last 3 rows (selected columns):")
    show = ["month", "cpi_general", "infl_yoy", "infl_yoy_reported", "cpi_status"]
    print(pd.concat([monthly[show].head(3), monthly[show].tail(3)]).to_string(index=False))
    print("\nNulls per column in monthly table:")
    print(monthly.isna().sum().to_string())
    print("\nInflation class counts (LOW <4, MEDIUM 4-6, HIGH >6):")
    yoy = monthly["infl_yoy"].dropna()
    lab = yoy.apply(lambda v: "LOW" if v < cfg.LOW_MAX
                    else ("HIGH" if v > cfg.HIGH_MIN else "MEDIUM"))
    print(lab.value_counts().reindex(cfg.LEVEL_LABELS).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
