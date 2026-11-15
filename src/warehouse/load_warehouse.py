"""
load_warehouse.py - build the SQLite data warehouse and load it from the cleaned CSVs.

Place at:  <project root>/src/warehouse/load_warehouse.py
Run from the project root:  python -m src.warehouse.load_warehouse
(needs src/warehouse/schema.sql next to it, and the earlier steps already run)

Reads   data/interim/cpi_long.csv        -> Dim_Sector, Dim_Group, Dim_Item, Fact_CPI
        data/processed/master_monthly.csv -> Dim_Date, Fact_Macro
Writes  data/warehouse.db                 (deleted and rebuilt on every run)

Open the .db file with "DB Browser for SQLite" (free) or from Power BI / Python.
"""
import sqlite3
from pathlib import Path

import pandas as pd

from src import config as cfg
from src.features.build_features import label

SCHEMA = Path(__file__).with_name("schema.sql")


def build_dim_date(months: pd.Series) -> pd.DataFrame:
    m = pd.DatetimeIndex(sorted(months.unique()))
    fy = [f"FY{d.year}-{(d.year + 1) % 100:02d}" if d.month >= 4
          else f"FY{d.year - 1}-{d.year % 100:02d}" for d in m]
    return pd.DataFrame({
        "date_id": m.year * 100 + m.month,
        "month_start": m.strftime("%Y-%m-%d"),
        "year": m.year,
        "quarter": m.quarter,
        "month_num": m.month,
        "month_name": m.strftime("%B"),
        "fiscal_year": fy,
    })


def build_cpi_dims(cpi: pd.DataFrame):
    sectors = pd.DataFrame({"sector_name": sorted(cpi["sector"].unique())})
    sectors.insert(0, "sector_id", range(1, len(sectors) + 1))

    pairs = cpi[["group", "subgroup"]].drop_duplicates()
    clash = pairs.groupby("subgroup")["group"].nunique()
    if (clash > 1).any():
        raise ValueError(f"These subgroups belong to more than one group: {list(clash[clash > 1].index)}")

    groups = pd.DataFrame({"group_name": sorted(pairs["group"].unique())})
    groups.insert(0, "group_id", range(1, len(groups) + 1))

    items = pairs.merge(groups, left_on="group", right_on="group_name")
    items = items.sort_values(["group_id", "subgroup"]).reset_index(drop=True)
    items.insert(0, "item_id", range(1, len(items) + 1))
    items = items.rename(columns={"subgroup": "subgroup_name"})[
        ["item_id", "subgroup_name", "group_id"]]
    return sectors, groups, items


def build_fact_cpi(cpi, sectors, items) -> pd.DataFrame:
    sec = dict(zip(sectors["sector_name"], sectors["sector_id"]))
    itm = dict(zip(items["subgroup_name"], items["item_id"]))
    f = pd.DataFrame({
        "date_id": cpi["month"].dt.year * 100 + cpi["month"].dt.month,
        "sector_id": cpi["sector"].map(sec),
        "item_id": cpi["subgroup"].map(itm),
        "cpi_index": cpi["index"],
        "inflation_yoy": cpi["infl_yoy_calc"],
        "inflation_reported": cpi["inflation_reported"],
        "status": cpi["status"],
        "index_imputed": (cpi["index_was_missing"].astype(bool) & cpi["index"].notna()).astype(int),
    })
    return f


def build_fact_macro(master: pd.DataFrame) -> pd.DataFrame:
    m = master[master["in_study"] == 1].copy()
    return pd.DataFrame({
        "date_id": m["month"].dt.year * 100 + m["month"].dt.month,
        "cpi_general": m["cpi_general"],
        "infl_yoy": m["infl_yoy"],
        "food_cpi_yoy": m["cfpi_yoy"],
        "fuel_cpi_yoy": m["fuel_yoy"],
        "usdinr_avg": m["usdinr_avg"],
        "usdinr_end": m["usdinr_end"],
        "usdinr_vol": m["usdinr_vol"],
        "brent_usd": m["brent_usd"],
        "dubai_usd": m["dubai_usd"],
        "brent_inr": m["brent_inr"],
        "wpi_all": m["wpi_all"],
        "wpi_primary": m["wpi_primary"],
        "wpi_food_articles": m["wpi_food_articles"],
        "wpi_fuel": m["wpi_fuel"],
        "wpi_manuf": m["wpi_manuf"],
        "repo_rate": m["repo"],
        "infl_level": m["infl_yoy"].apply(label).astype(object).where(m["infl_yoy"].notna(), None),
        "cluster_id": None,
    })


def main():
    for p, runner in ((cfg.CPI_LONG_CSV, "ingest.load_cpi"), (cfg.MASTER_CSV, "preprocess.merge")):
        if not p.exists():
            raise SystemExit(f"{p} not found. Run first:  python -m src.{runner}")

    cpi = pd.read_csv(cfg.CPI_LONG_CSV, parse_dates=["month"])
    master = pd.read_csv(cfg.MASTER_CSV, parse_dates=["month"])
    study_master = master[master["in_study"] == 1]

    dim_date = build_dim_date(study_master["month"])
    sectors, groups, items = build_cpi_dims(cpi)
    fact_cpi = build_fact_cpi(cpi, sectors, items)
    fact_macro = build_fact_macro(master)

    if fact_cpi[["date_id", "sector_id", "item_id"]].isna().any().any():
        raise ValueError("Some CPI rows did not match a dimension key")
    if fact_cpi.duplicated(["date_id", "sector_id", "item_id"]).any():
        raise ValueError("Duplicate keys in Fact_CPI")

    try:
        cfg.WAREHOUSE_DB.unlink(missing_ok=True)
    except PermissionError:
        raise SystemExit(f"Close {cfg.WAREHOUSE_DB} in any other program (DB Browser, Power BI) and retry")

    con = sqlite3.connect(cfg.WAREHOUSE_DB)
    con.executescript(SCHEMA.read_text(encoding="utf-8"))
    con.execute("PRAGMA foreign_keys = ON")

    # parents before children
    dim_date.to_sql("Dim_Date", con, if_exists="append", index=False)
    sectors.to_sql("Dim_Sector", con, if_exists="append", index=False)
    groups.to_sql("Dim_Group", con, if_exists="append", index=False)
    items.to_sql("Dim_Item", con, if_exists="append", index=False)
    fact_cpi.to_sql("Fact_CPI", con, if_exists="append", index=False)
    fact_macro.to_sql("Fact_Macro", con, if_exists="append", index=False)
    con.commit()

    # ---------------- verification ----------------
    print(f"Built {cfg.WAREHOUSE_DB}\n\nRow counts:")
    for t in ["Dim_Date", "Dim_Sector", "Dim_Group", "Dim_Item", "Fact_CPI", "Fact_Macro"]:
        print(f"  {t:11s} {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>6}")
    bad = con.execute("PRAGMA foreign_key_check").fetchall()
    print("\nForeign-key check:", "all keys valid" if not bad else f"{len(bad)} BROKEN LINKS")

    print("\nExample OLAP roll-up (Combined sector, headline index, average inflation by year):")
    q = """
    SELECT year, ROUND(AVG(inflation_yoy), 2) AS avg_inflation_pct,
           ROUND(MAX(inflation_yoy), 2) AS peak_pct, COUNT(*) AS months
    FROM vw_cpi_detail
    WHERE sector_name = 'Combined' AND subgroup_name = 'General-Overall'
          AND inflation_yoy IS NOT NULL
    GROUP BY year ORDER BY year"""
    print(pd.read_sql_query(q, con).to_string(index=False))

    print("\nLevel counts stored in Fact_Macro:")
    print(pd.read_sql_query(
        "SELECT infl_level, COUNT(*) AS months FROM Fact_Macro GROUP BY infl_level ORDER BY months DESC",
        con).to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
