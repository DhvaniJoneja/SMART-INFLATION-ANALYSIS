-- schema.sql - data warehouse for the Smart Inflation Analysis project (SQLite).
-- Place at:  <project root>/src/warehouse/schema.sql
-- It is executed by load_warehouse.py (next file); nothing to run on its own.
--
-- DESIGN: a "fact constellation" - two fact tables that share conformed dimensions.
--
--   Dim_Date   Dim_Sector     Dim_Group
--      |           |             |   (snowflaked: item -> group)
--      |           |          Dim_Item
--      |           |             |
--      +-----+-----+-------------+
--            |
--        Fact_CPI        grain: one CPI index value per month x sector x item (subgroup)
--
--   Dim_Date --- Fact_Macro   grain: one row per month, all outside indicators together
--
--   * Star part:      Fact_CPI -> Dim_Date, Dim_Sector, Dim_Item
--   * Snowflake part: Dim_Item -> Dim_Group (the CPI group / subgroup hierarchy)
--   * Fact_Macro holds monthly USD/INR, crude oil, WPI, repo rate, headline inflation,
--     the LOW/MEDIUM/HIGH label and (later) the K-Means cluster of each month.
--
-- Dates are stored as text 'YYYY-MM-DD' (first day of the month); date_id is YYYYMM.
-- For MySQL/PostgreSQL: the same DDL works with REAL -> DOUBLE / DOUBLE PRECISION.

PRAGMA foreign_keys = ON;

DROP VIEW  IF EXISTS vw_cpi_detail;
DROP VIEW  IF EXISTS vw_macro_monthly;
DROP TABLE IF EXISTS Fact_CPI;
DROP TABLE IF EXISTS Fact_Macro;
DROP TABLE IF EXISTS Dim_Item;
DROP TABLE IF EXISTS Dim_Group;
DROP TABLE IF EXISTS Dim_Sector;
DROP TABLE IF EXISTS Dim_Date;

-- ---------------------------------------------------------------- dimensions
CREATE TABLE Dim_Date (
    date_id      INTEGER PRIMARY KEY,          -- e.g. 201301
    month_start  TEXT    NOT NULL UNIQUE,      -- '2013-01-01'
    year         INTEGER NOT NULL,
    quarter      INTEGER NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    month_num    INTEGER NOT NULL CHECK (month_num BETWEEN 1 AND 12),
    month_name   TEXT    NOT NULL,
    fiscal_year  TEXT    NOT NULL              -- India's April-March year, e.g. 'FY2012-13'
);

CREATE TABLE Dim_Sector (
    sector_id    INTEGER PRIMARY KEY,
    sector_name  TEXT NOT NULL UNIQUE          -- Rural / Urban / Combined
);

CREATE TABLE Dim_Group (
    group_id     INTEGER PRIMARY KEY,
    group_name   TEXT NOT NULL UNIQUE          -- e.g. 'Food and Beverages', 'Housing'
);

CREATE TABLE Dim_Item (
    item_id        INTEGER PRIMARY KEY,
    subgroup_name  TEXT    NOT NULL UNIQUE,    -- e.g. 'Cereals and Products'
    group_id       INTEGER NOT NULL,
    FOREIGN KEY (group_id) REFERENCES Dim_Group (group_id)
);

-- ---------------------------------------------------------------- facts
CREATE TABLE Fact_CPI (
    date_id             INTEGER NOT NULL,
    sector_id           INTEGER NOT NULL,
    item_id             INTEGER NOT NULL,
    cpi_index           REAL,                  -- base 2012 = 100; blank where not published
    inflation_yoy       REAL,                  -- year-on-year %, recomputed from the index
    inflation_reported  REAL,                  -- as published by MoSPI (blank in 2013)
    status              TEXT,                  -- F, F*, F+ as in the source files
    index_imputed       INTEGER NOT NULL DEFAULT 0 CHECK (index_imputed IN (0, 1)),
    PRIMARY KEY (date_id, sector_id, item_id),
    FOREIGN KEY (date_id)   REFERENCES Dim_Date   (date_id),
    FOREIGN KEY (sector_id) REFERENCES Dim_Sector (sector_id),
    FOREIGN KEY (item_id)   REFERENCES Dim_Item   (item_id)
);

CREATE TABLE Fact_Macro (
    date_id            INTEGER PRIMARY KEY,
    -- headline CPI (Combined sector)
    cpi_general        REAL,
    infl_yoy           REAL,                   -- headline YoY inflation %
    food_cpi_yoy       REAL,                   -- Consumer Food Price index YoY %
    fuel_cpi_yoy       REAL,                   -- Fuel and Light YoY %
    -- exchange rate (RBI reference rate, monthly)
    usdinr_avg         REAL,
    usdinr_end         REAL,
    usdinr_vol         REAL,
    -- crude oil, US$ per barrel, and Brent in rupees
    brent_usd          REAL,
    dubai_usd          REAL,
    brent_inr          REAL,
    -- wholesale price index (base 2011-12 = 100)
    wpi_all            REAL,
    wpi_primary        REAL,
    wpi_food_articles  REAL,
    wpi_fuel           REAL,
    wpi_manuf          REAL,
    -- RBI policy repo rate, % (rate in force at month end)
    repo_rate          REAL,
    -- mining outputs
    infl_level         TEXT CHECK (infl_level IN ('LOW', 'MEDIUM', 'HIGH')),
    cluster_id         INTEGER,                -- filled in after K-Means
    FOREIGN KEY (date_id) REFERENCES Dim_Date (date_id)
);

-- ---------------------------------------------------------------- indexes
CREATE INDEX idx_factcpi_sector ON Fact_CPI (sector_id);
CREATE INDEX idx_factcpi_item   ON Fact_CPI (item_id);
CREATE INDEX idx_item_group     ON Dim_Item (group_id);

-- ---------------------------------------------------------------- convenience views
-- (Power BI and the OLAP queries can read these instead of writing the joins)
CREATE VIEW vw_cpi_detail AS
SELECT d.month_start, d.year, d.quarter, d.month_num, d.month_name, d.fiscal_year,
       s.sector_name, g.group_name, i.subgroup_name,
       f.cpi_index, f.inflation_yoy, f.inflation_reported, f.status, f.index_imputed
FROM Fact_CPI f
JOIN Dim_Date   d ON d.date_id   = f.date_id
JOIN Dim_Sector s ON s.sector_id = f.sector_id
JOIN Dim_Item   i ON i.item_id   = f.item_id
JOIN Dim_Group  g ON g.group_id  = i.group_id;

CREATE VIEW vw_macro_monthly AS
SELECT d.month_start, d.year, d.quarter, d.month_num, d.fiscal_year, m.*
FROM Fact_Macro m
JOIN Dim_Date d ON d.date_id = m.date_id;
