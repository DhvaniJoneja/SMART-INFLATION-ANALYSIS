# Smart Inflation Analysis: Project Plan

**Course:** Data Mining and Data Warehousing
**Working title:** Economic Intelligence System for Inflation Analysis, Regime Discovery and Prediction in India
**Period covered:** January 2013 to December 2025 (156 monthly rows, limited by the CPI files)

---

## 1. Purpose and goal

Indian inflation is driven by several things at once: food prices, fuel, the rupee, crude oil, wholesale prices and RBI policy. This project integrates these into one monthly dataset, stores it in a data warehouse, and applies data mining to answer four questions:

| # | Question | Technique |
|---|---|---|
| Q1 | What kinds of economic periods (regimes) has India had since 2013? | K-Means |
| Q2 | Given economic conditions, is inflation LOW, MEDIUM or HIGH? | ID3, Naive Bayes, kNN, SVM (+ RF/XGBoost as extras) |
| Q3 | What will next month's inflation be? | Backpropagation neural network |
| Q4 | Which conditions occur together (e.g. high oil + weak rupee -> high fuel inflation)? | Apriori (R) |

**Relation to the reference paper (CENTRALIQ, Jangir 2026):** the paper forecasts CPI from its own lags with ARIMA/XGBoost. We extend it with exogenous indicators (USD/INR, crude, WPI, repo), add clustering, classification and rule mining, and add a warehouse. We cite it as related work and reuse its ideas (lag features, time-ordered split, ARIMA baseline).

---

## 2. Data inventory (based on the files you have)

| # | Dataset | Format you have | Coverage | Role |
|---|---|---|---|---|
| 1 | CPI (MoSPI) | One CSV per year (2013-2025). Columns: `baseyear, year, month_code, month, state, sector, group, subgroup, index, inflation, status` | 2013-2025 | Target + CPI components |
| 2 | USD/INR (RBI) | Daily CSV with 10 junk header lines; columns: Date, Open, High, Low, Close, RBI Reference Rate | Aug 2000 to 30 Jun 2026 | Exchange rate |
| 3 | Crude oil (World Bank Pink Sheet) | Excel. Sheets: monthly prices, monthly indices, description, index weights | 1960M01 to 2026 | Oil price |
| 4 | WPI (Office of Economic Adviser) | Wide CSV: one row per commodity, one column per month (`INDX042012` ... `INDX042026`) | Apr 2012 to Apr 2026 | Wholesale prices |
| 5 | Repo rate (RBI) | Event-based CSV (a row only when a rate changes) | 1935 to 2026 | Monetary policy |

**Sheets you do not need:** the Pink Sheet "description" and "index weights" sheets. Skip them. The "monthly indices" sheet has an **Energy index** column that is a useful optional extra.

**No further datasets are required.** The overlap of all five is Jan 2013 to Dec 2025, and the CPI files set that limit. Optional, only if time allows: CPI for 2026 months (if MoSPI publishes it in the same format), but check `baseyear` first, because a base-year change makes the series non-comparable.

### Known quirks per dataset (handle in preprocessing)

**CPI**
- Filter `state == "All India"`. Keep Rural, Urban and Combined (Combined is the modelling series; Rural/Urban feed the warehouse).
- Check the `baseyear` column is 2012 for every row.
- `status` is F (final) or P (provisional). Keep both, record it in the warehouse.
- `inflation` is already year-on-year %. Recompute it from `index` for a sanity check where possible. Early 2013 months may have no value because the base series starts in 2012.
- Look for missing months (April/May 2020 are suspect). Document how each gap is handled.
- The `group`/`subgroup` hierarchy provides Food, Fuel and light, Housing, Transport and communication, etc. for features.

**USD/INR**
- The file has a wrapper (`src_file, tab, period, row_no`) and the real header is on row 10. Parse from there and drop the trailing empty row.
- Dates look like `30 June 2026`. Parse with `format="%d %B %Y"`.
- The sample shows rows where High < Open, so OHLC columns are unreliable. Use **RBI Reference Rate** (official) as the main monthly series, and Close as a cross-check.
- Aggregate to monthly mean, plus monthly end value and monthly std (volatility).

**Crude oil**
- Missing values are `…` or `...`. Convert to NaN.
- Two header rows (name, then unit). Period is `1960M01` format, so convert to a month date.
- Use **Brent** as primary, **Dubai** as a cross-check (India imports closer to the Dubai/Oman basket). The "Crude oil, average" column is also fine.
- Oil is in USD. Also build **oil in INR** = Brent x USD/INR, which is more relevant for Indian fuel prices.

**WPI**
- Melt the wide columns (`INDX042012` -> 2012-04). Column name pattern is `INDX` + MM + YYYY.
- Keep only top-level rows by `COMM_CODE`: All commodities `1000000000`, Primary articles `1100000000`, Food articles `1101000000`, Fuel & power `1200000000`, Manufactured products `1300000000`, Food index `2000000000`.
- Sub-item rows contain `0.0` for "not available". Ignore them, since we do not use sub-items.
- Check whether the latest months are provisional and may be revised.

**Repo rate**
- Use column `Policy repo rate` only. It is blank before the repo rate was introduced (2000).
- Forward-fill from the last known change **before Jan 2013** so January 2013 has a value. Use the rate in force at the end of each month (or month average, but be consistent).
- Drop other columns (SDF, MSF, Bank Rate, CRR, SLR), though CRR is optional.

---

## 3. End-to-end pipeline

```
 RAW DATA (5 sources)
   |
   v
 [1] INGEST          load each source, parse headers, standardise to a `month` key (first day of month)
   |
   v
 [2] CLEAN           missing values, duplicates, outliers, unit checks, document every decision
   |
   v
 [3] INTEGRATE       monthly master table (Jan 2013 to Dec 2025), left-joined on month
   |
   v
 [4] FEATURES        YoY / MoM changes, lags, rolling means, Inflation_Level label
   |
   +--> [5] WAREHOUSE  star schema in SQLite/MySQL/PostgreSQL + OLAP queries
   |
   +--> [6] EDA         stats, correlations, trend plots
   |
   +--> [7] MINING
   |       |-- K-Means             -> regimes
   |       |-- ID3/NB/kNN/SVM      -> Inflation_Level
   |       |-- Neural network      -> next-month inflation
   |       +-- Apriori (R)         -> association rules
   |
   v
 [8] EVALUATION     accuracy/precision/recall/F1/confusion matrix, RMSE/MAE, WEKA cross-check
   |
   v
 [9] PRESENTATION   Power BI dashboards + final report + viva
```

---

## 4. Feature design

### 4.1 Master table (one row per month)

| Column | Meaning | Source |
|---|---|---|
| `month` | First day of month | all |
| `cpi_general` | CPI Combined general index | CPI |
| `cpi_food`, `cpi_fuel`, `cpi_housing`, `cpi_transport` | Group indices | CPI |
| `infl_yoy` | Headline YoY inflation % (target source) | CPI |
| `usdinr_avg`, `usdinr_end`, `usdinr_vol` | Monthly average, end value, std | Forex |
| `brent`, `dubai` | Crude USD/bbl | Pink Sheet |
| `brent_inr` | Brent x USD/INR | derived |
| `wpi_all`, `wpi_fuel`, `wpi_food`, `wpi_manuf` | WPI top-level indices | WPI |
| `repo` | Policy repo rate % | RBI |

### 4.2 Engineered features

- Percentage change MoM and YoY for USD/INR, Brent, Brent in INR, WPI groups.
- Lags (1, 2, 3, 6 months) of the exogenous variables.
- 3 and 6 month rolling means and volatility.
- Real repo rate = repo - infl_yoy (for the regime/association analysis only, not for classifier inputs).
- Time features: month number (seasonality), fiscal-year dummy if useful.

### 4.3 Target variables

- **Classification target `Inflation_Level`** from `infl_yoy`, using the RBI tolerance band:
  - LOW: below 4%
  - MEDIUM: 4% to 6%
  - HIGH: above 6%
- **Regression target:** `infl_yoy` at month t+1 (next-month YoY inflation).

### 4.4 Rules to avoid fooling ourselves (important for viva)

1. **No label leakage.** `Inflation_Level` is derived from CPI inflation, so classifiers must NOT receive CPI general, CPI inflation or lagged CPI inflation as inputs. Use USD/INR, oil, WPI, repo and their changes/lags only. State this explicitly in the report.
2. **Do not predict the raw CPI level.** It trends upward, and neural nets and tree models cannot extrapolate beyond the training range. Predict YoY inflation (or MoM change).
3. **Time-ordered evaluation.** Train on earlier years, test on the latest. For the neural network use a strict chronological split; for classifiers use time-series CV (expanding window) and also report plain k-fold with a clear caveat that it leaks time.
4. **Class imbalance.** HIGH months are probably few (2013-14, 2020, 2022-23). Check class counts first, report per-class precision/recall, and consider adjusting thresholds or using macro-F1 instead of only accuracy.
5. **Small sample.** About 150 rows means simple models, regularisation, and no claims of large improvements.
6. **Correlation is not causation.** Say so when describing association rules and feature importance.

---

## 5. Data warehouse design

Two fact tables sharing conformed dimensions (a fact constellation), since the CPI grain and the macro grain differ.

```
Dim_Date (date_id, date, month, month_name, quarter, year, fiscal_year)
Dim_Sector (sector_id, sector_name)          -- Rural / Urban / Combined
Dim_Item (item_id, group_name, subgroup_name) -- CPI group hierarchy (snowflake if you split group/subgroup)
Dim_Indicator (indicator_id, name, source, unit, frequency)

Fact_CPI    (date_id, sector_id, item_id, cpi_index, inflation_yoy, status)
            grain: month x sector x subgroup (All India)

Fact_Macro  (date_id, usdinr_avg, usdinr_end, brent_usd, brent_inr,
             wpi_all, wpi_fuel, wpi_food, wpi_manuf, repo_rate,
             inflation_level, cluster_id)
            grain: month
```

- Implement in SQLite (easiest), MySQL or PostgreSQL. Include `schema.sql` with primary and foreign keys.
- Write the ETL load script in Python (SQLAlchemy or sqlite3).
- Demonstrate **OLAP-style queries** in `warehouse/queries.sql`: roll-up (month -> quarter -> year), drill-down (group -> subgroup), slice (Rural only), dice (Food group, 2020-2022), and a pivot of average inflation by year x sector.
- Power BI connects to the warehouse tables or the exported CSVs.

---

## 6. Mining and modelling plan

### 6.1 K-Means (regime discovery)
- Features: standardised `infl_yoy`, food/fuel CPI YoY, USD/INR change, Brent change, WPI YoY, repo.
- Choose k with the elbow method and silhouette score (try 2 to 8).
- Profile each cluster (mean of each variable, date ranges) and label them after the fact (e.g. "high food inflation", "currency/commodity shock").
- Show clusters on a timeline and with PCA 2-D scatter.

### 6.2 Classification (Inflation_Level)
| Model | Python | WEKA |
|---|---|---|
| ID3 / decision tree | `DecisionTreeClassifier(criterion="entropy")` (note: scikit-learn is CART-like, not pure ID3; mention this) | `J48` (C4.5) or ID3 plugin (needs nominal attributes) |
| Naive Bayes | `GaussianNB` | `NaiveBayes` |
| kNN | `KNeighborsClassifier` (scaled features, tune k) | `IBk` |
| SVM | `SVC` (RBF, tune C/gamma) | `SMO` |
| **Extra:** Random Forest | `RandomForestClassifier` | `RandomForest` |
| **Extra:** XGBoost | `XGBClassifier` (shallow, few trees) | not standard |

- Same features, same splits for every model.
- Pure ID3 only handles categorical attributes, so discretise features (Low/Med/High) for the ID3 run, and say so.
- Metrics: accuracy, macro precision, recall, F1, confusion matrix. Compare against a majority-class baseline.
- Run the same models in WEKA on the exported ARFF and compare with Python results.

### 6.3 Neural network (next-month inflation)
- `MLPRegressor` (or Keras): 1-2 hidden layers (e.g. 16 and 8 units), ReLU, early stopping, standardised inputs fit on training data only.
- Split: train Jan 2013 to Dec 2022, validation 2023, test 2024 to 2025 (adjust once row counts are known).
- Inputs: lagged exogenous variables plus (for this task only) lagged inflation is allowed, since the target is a forecast, not a label derived from inputs. Run both versions (with and without lagged inflation) and report the difference.
- Baselines to beat: naive (next month = this month) and ARIMA. Metric: RMSE (also MAE).
- Optional extra: XGBoost/Random Forest regressors for comparison with the paper.

### 6.4 Apriori (R, `arules`)
- Discretise variables into Low/Medium/High using terciles (or domain thresholds): USD/INR change, Brent change, food inflation, fuel inflation, WPI inflation, repo, general inflation.
- Choose minimum support (~0.1) and confidence (~0.6) after looking at the item frequencies; keep rules with lift > 1.
- Report the top rules by lift, with support, confidence and lift, and describe them carefully as associations.
- Visualise with `arulesViz`.

---

## 7. Evaluation summary tables (fill with real results)

**Classification**

| Model | Accuracy | Macro-P | Macro-R | Macro-F1 | Notes |
|---|---|---|---|---|---|
| Majority baseline | | | | | |
| ID3 / J48 | | | | | |
| Naive Bayes | | | | | |
| kNN | | | | | |
| SVM | | | | | |
| Random Forest (extra) | | | | | |
| XGBoost (extra) | | | | | |

**Prediction**

| Model | RMSE | MAE | Notes |
|---|---|---|---|
| Naive (last value) | | | |
| ARIMA | | | |
| Neural network | | | |
| XGBoost / RF (extra) | | | |

---

## 8. Project folder structure

```
smart-inflation-analysis/
|-- README.md
|-- planning.md
|-- requirements.txt
|-- data/
|   |-- raw/
|   |   |-- cpi/                 # yearly CSVs 2013..2025
|   |   |-- usdinr/
|   |   |-- crude_oil/
|   |   |-- wpi/
|   |   +-- repo_rate/
|   |-- interim/                 # cleaned, one file per source (monthly)
|   +-- processed/
|       |-- master_monthly.csv
|       |-- features.csv
|       +-- features.arff        # for WEKA
|-- src/
|   |-- config.py                # paths, date range, thresholds
|   |-- ingest/
|   |   |-- load_cpi.py
|   |   |-- load_usdinr.py
|   |   |-- load_crude.py
|   |   |-- load_wpi.py
|   |   +-- load_repo.py
|   |-- preprocess/
|   |   |-- clean.py
|   |   +-- merge.py
|   |-- features/
|   |   +-- build_features.py
|   |-- warehouse/
|   |   |-- schema.sql
|   |   |-- load_warehouse.py
|   |   +-- queries.sql
|   |-- eda/
|   |   +-- eda.py
|   |-- models/
|   |   |-- kmeans.py
|   |   |-- classify.py
|   |   |-- neural_net.py
|   |   +-- baselines.py         # naive, ARIMA, RF, XGBoost extras
|   +-- evaluation/
|       +-- metrics.py
|-- r/
|   +-- apriori_rules.R
|-- weka/
|   |-- features.arff
|   +-- results/                 # saved outputs/screenshots
|-- notebooks/
|   |-- 01_data_audit.ipynb
|   |-- 02_eda.ipynb
|   |-- 03_clustering.ipynb
|   |-- 04_classification.ipynb
|   +-- 05_prediction.ipynb
|-- dashboards/
|   +-- inflation.pbix
|-- outputs/
|   |-- figures/
|   +-- tables/
|-- reports/
|   |-- synopsis.pdf
|   +-- final_report.docx
+-- run_pipeline.py              # one command runs ingest -> features -> models
```

Keep `data/raw/` untouched. Every transformation reads from raw and writes to `interim/` or `processed/`, so the pipeline can be re-run from scratch.

---

## 9. Phases and timeline

Deadline unknown, so timings are relative. Compress or stretch as needed.

| Phase | Work | Output | Done when |
|---|---|---|---|
| 0. Setup (day 1) | Repo, folder structure, requirements, agree on column names and `month` key | Skeleton on GitHub | Everyone can clone and run |
| 1. Ingest and audit (days 2-4) | Loaders for all 5 sources, data audit notebook | `interim/*.csv`, quirk list | Row counts and date ranges verified |
| 2. Integrate and features (days 4-7) | Master table, features, labels | `master_monthly.csv`, `features.csv`, ARFF | 156 rows, no unexplained NaN |
| 3. Warehouse (days 5-8) | Schema, ETL, OLAP queries | `schema.sql`, populated DB | Queries return correct results |
| 4. EDA and K-Means (days 8-11) | Stats, correlations, clusters | Figures, cluster profiles | Clusters interpreted |
| 5. Classification and WEKA (days 9-14) | 4 models (+ extras) in Python and WEKA | Comparison table | Python and WEKA results compared |
| 6. Neural net (days 10-14) | MLP + baselines | RMSE table | Beats or honestly fails to beat baseline |
| 7. Apriori (days 12-15) | R script, rules | Top-rules table | Rules interpreted |
| 8. Power BI (days 14-18) | 3 dashboard pages | `.pbix` | Pages: Trends, Clusters/Rules, Model comparison |
| 9. Report and viva prep (days 17-21) | Final report, README, cleanup | Submission pack | Report matches actual results |

---

## 10. Team split (every member writes code)

| Member | Main responsibility | Code deliverables |
|---|---|---|
| 1. Data engineering | Ingest, clean, merge, features, warehouse | `ingest/*`, `preprocess/*`, `build_features.py`, `schema.sql`, `load_warehouse.py` |
| 2. EDA + clustering | Statistics, correlation, plots, K-Means | `eda.py`, `kmeans.py`, notebooks 02-03 |
| 3. Classification + WEKA | 4 classifiers (+ extras), evaluation, WEKA runs | `classify.py`, `metrics.py`, notebook 04, WEKA results |
| 4. Prediction + rules | Neural net, baselines, Apriori, Power BI | `neural_net.py`, `baselines.py`, `apriori_rules.R`, dashboards |

Everyone also: contributes to `run_pipeline.py` or a utility module, commits under their own name, and writes their own report section. If the team has 3 members, merge member 3 and 4 tasks sensibly (e.g. member 4 takes Power BI, member 3 takes Apriori).

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Only ~156 rows | Simple models, time-series CV, honest reporting, no overclaiming |
| Few HIGH inflation months | Check counts early; use macro-F1; discuss limitation; consider merging thresholds |
| Label leakage | Strict feature lists per task (section 4.4) |
| Missing CPI months (e.g. 2020) | Document; interpolate or drop, and test sensitivity |
| Base-year or definition changes | Check `baseyear`, use only the WPI 2011-12 series, do not mix bases |
| Provisional data revised | Note `status` flags; treat latest months with caution |
| WEKA vs Python results differ | Expected (different defaults); report both and explain |
| Pure ID3 limitations | Discretise for ID3 and also report J48/CART |
| Merge conflicts in Git | One owner per file, small commits, branch per member |

---

## 12. Final deliverables checklist

- [ ] Cleaned and integrated monthly dataset with documented preprocessing
- [ ] Warehouse schema, loaded database and OLAP queries
- [ ] EDA figures and statistics
- [ ] K-Means regimes with interpretation
- [ ] Four classifiers + comparison table (Python and WEKA)
- [ ] Neural network RMSE vs baselines
- [ ] Apriori rules with support/confidence/lift
- [ ] Power BI dashboards (3 pages)
- [ ] Final report (no template leftovers, no "TBD") and consistent tool list across synopsis and report
- [ ] Clean GitHub repo with per-member commits and README
- [ ] Viva prep (see below)

---

## 13. Viva quick answers

- **Why these datasets?** Inflation is driven by food, fuel, currency, global oil, wholesale prices and monetary policy, and each dataset represents one channel.
- **Why monthly?** CPI is monthly, so everything is aligned to that frequency.
- **Why not predict CPI level?** It trends upward, and models cannot extrapolate outside the training range.
- **How did you avoid leakage?** The label comes from CPI inflation, so no CPI-based features are used for classification.
- **How is this different from the paper?** Multi-indicator, multi-task, adds a warehouse; the paper uses CPI lags only.
- **Is clustering "correct"?** K-Means finds groups; the labels are our interpretation afterwards.
- **Do association rules prove causation?** No, only co-occurrence.
