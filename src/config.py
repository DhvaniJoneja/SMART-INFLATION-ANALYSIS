"""
config.py - central settings for the Smart Inflation Analysis project.

Place this file at:  <project root>/src/config.py
Every other script imports it with:  from src import config as cfg
Run from the project root, e.g.:     python -m src.ingest.load_cpi
Check your setup with:               python -m src.config
"""
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# 1. PATHS  (raw files stay where you downloaded them, in project root)
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

CPI_DIR = ROOT / "cpi_data"
FOREX_FILE = ROOT / "r691_daily_interbank_forex_rates_usd_vis_a_vis_inr.csv"
PINK_FILE = ROOT / "CMO-Historical-Data-Monthly.xlsx"
WPI_FILE = ROOT / "monthly_index_202606.xls"
REPO_FILE = ROOT / "rates-policy.csv"

DATA_DIR = ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
WAREHOUSE_DB = DATA_DIR / "warehouse.db"

for _d in (INTERIM_DIR, PROCESSED_DIR, FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Intermediate / output files written by the pipeline
CPI_LONG_CSV = INTERIM_DIR / "cpi_long.csv"          # all sectors, all subgroups (for warehouse)
CPI_MONTHLY_CSV = INTERIM_DIR / "cpi_monthly.csv"    # Combined sector, one row per month
FOREX_MONTHLY_CSV = INTERIM_DIR / "forex_monthly.csv"
CRUDE_MONTHLY_CSV = INTERIM_DIR / "crude_monthly.csv"
WPI_MONTHLY_CSV = INTERIM_DIR / "wpi_monthly.csv"
REPO_MONTHLY_CSV = INTERIM_DIR / "repo_monthly.csv"
MASTER_CSV = PROCESSED_DIR / "master_monthly.csv"
FEATURES_CSV = PROCESSED_DIR / "features.csv"
FEATURES_ARFF = PROCESSED_DIR / "features.arff"

# ------------------------------------------------------------------
# 2. STUDY PERIOD  (month key = first day of the month)
# ------------------------------------------------------------------
START_MONTH = "2013-01-01"   # CPI index starts here
END_MONTH = "2025-12-01"     # CPI index ends here

# ------------------------------------------------------------------
# 3. CPI SETTINGS
# ------------------------------------------------------------------
CPI_STATE = "All India"
CPI_MODEL_SECTOR = "Combined"   # Rural / Urban are kept only for the warehouse

# column name in the master table -> subgroup name in the CPI files
CPI_SUBGROUPS = {
    "cpi_general": "General-Overall",
    "cpi_cfpi": "Consumer Food Price-Overall",
    "cpi_food_bev": "Food and Beverages-Overall",
    "cpi_fuel": "Fuel and Light-Overall",
    "cpi_housing": "Housing-Overall",
    "cpi_transport": "Transport and Communication",
    "cpi_clothing": "Clothing and Footwear-Overall",
    "cpi_misc": "Miscellaneous-Overall",
}

# ------------------------------------------------------------------
# 4. USD/INR (RBI daily file)
# ------------------------------------------------------------------
FOREX_COLS = {            # raw column -> clean name
    "c1": "date",
    "c2": "open",
    "c3": "high",
    "c4": "low",
    "c5": "close",
    "c6": "ref_rate",
}
FOREX_DATE_FORMAT = "%d %B %Y"     # e.g. "30 June 2026"
FOREX_VALUE_COL = "ref_rate"       # RBI reference rate = most reliable column

# ------------------------------------------------------------------
# 5. CRUDE OIL (World Bank Pink Sheet)
# ------------------------------------------------------------------
PINK_PRICES_SHEET = "Monthly Prices"
PINK_HEADER_ROW = 4        # 0-based row holding commodity names
PINK_DATA_START_ROW = 6    # 0-based row where data starts (row 5 holds units)
PINK_COLS = {              # sheet column name -> clean name
    "Crude oil, Brent": "brent_usd",
    "Crude oil, Dubai": "dubai_usd",
    "Crude oil, WTI": "wti_usd",
    "Crude oil, average": "crude_avg_usd",
}

# ------------------------------------------------------------------
# 6. WPI (Office of Economic Adviser, wide format)
# ------------------------------------------------------------------
WPI_SHEET = "MONTHLY_INDEX"
WPI_CODES = {              # clean name -> COMM_CODE
    "wpi_all": 1000000000,
    "wpi_primary": 1100000000,
    "wpi_food_articles": 1101000000,
    "wpi_fuel": 1200000000,
    "wpi_manuf": 1300000000,
    "wpi_food_index": 2000000000,
}

# ------------------------------------------------------------------
# 7. REPO RATE (RBI)
# ------------------------------------------------------------------
REPO_DATE_COL = "date"
REPO_RATE_COL = "Policy repo rate"

# ------------------------------------------------------------------
# 8. LABELS, LAGS, SPLITS
# ------------------------------------------------------------------
# RBI inflation target band: 4% +/- 2%
LOW_MAX = 4.0     # infl_yoy <  4          -> LOW
HIGH_MIN = 6.0    # infl_yoy >  6          -> HIGH ; otherwise MEDIUM
LEVEL_LABELS = ["LOW", "MEDIUM", "HIGH"]

LAGS = [1, 2, 3, 6]
ROLL_WINDOWS = [3, 6]

TRAIN_END = "2022-12-01"   # neural net / time-ordered split
VAL_END = "2023-12-01"     # test = after this date
RANDOM_SEED = 42
K_RANGE = range(2, 9)      # K-Means search range


# ------------------------------------------------------------------
# 9. SMALL SHARED HELPER
# ------------------------------------------------------------------
def to_month_start(series):
    """Convert any date-like Series to the first day of its month."""
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp()


if __name__ == "__main__":
    print("Project root:", ROOT)
    checks = {
        "CPI folder": CPI_DIR,
        "Forex file": FOREX_FILE,
        "Pink Sheet": PINK_FILE,
        "WPI file": WPI_FILE,
        "Repo file": REPO_FILE,
    }
    for name, path in checks.items():
        print(f"{name:12s} {'OK     ' if path.exists() else 'MISSING'} {path}")
    n_cpi = len(list(CPI_DIR.glob("*.xlsx"))) if CPI_DIR.exists() else 0
    print("CPI .xlsx files found:", n_cpi, "(expected 13)")
