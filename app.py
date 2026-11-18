"""
app.py - Streamlit dashboard for the "Smart Inflation Analysis" project.

Place at:  <project root>/app.py          (next to the src folder)
One-time:  pip install streamlit           (Anaconda often has it already)
Run:       streamlit run app.py            (opens http://localhost:8501)

It only READS what the pipeline already produced, so run  python run_pipeline.py  first:
  data/processed/features.csv, feature_sets.json, data/warehouse.db, outputs/tables/*.csv,
  outputs/figures/*.png
Pages: Overview | Drivers | Regimes (K-Means) | Classifiers | Forecasting | Association rules
       | Warehouse (OLAP) | Try it yourself
"""
import json
import re
import sqlite3
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROC, TAB, FIG = ROOT / "data" / "processed", ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
DB = ROOT / "data" / "warehouse.db"
LEVELS = ["LOW", "MEDIUM", "HIGH"]
LEVEL_COLORS = alt.Scale(domain=LEVELS, range=["#2a9d8f", "#e9c46a", "#e76f51"])
LOW_MAX, HIGH_MIN = 4.0, 6.0

st.set_page_config(page_title="Smart Inflation Analysis", page_icon="📈", layout="wide")


# ------------------------------------------------------------------ loaders
@st.cache_data
def load_csv(path, **kw):
    p = Path(path)
    return pd.read_csv(p, **kw) if p.exists() else None


@st.cache_data
def load_features():
    p = PROC / "features.csv"
    return pd.read_csv(p, parse_dates=["month"]) if p.exists() else None


@st.cache_data
def load_sets():
    p = PROC / "feature_sets.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def sql(query, params=()):
    con = sqlite3.connect(DB)
    try:
        return pd.read_sql_query(query, con, params=params)
    finally:
        con.close()


def need(df, what):
    if df is None:
        st.warning(f"{what} not found. Run `python run_pipeline.py` first.")
        return False
    return True


def show_fig(name, caption=None):
    p = FIG / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)


def level_of(x):
    return "LOW" if x < LOW_MAX else ("HIGH" if x > HIGH_MIN else "MEDIUM")


# ------------------------------------------------------------------ pages
def page_overview(df):
    st.title("Smart Inflation Analysis")
    st.caption("Economic intelligence system for India: inflation regimes, drivers and prediction "
               "(CPI 2013-2025, USD/INR, crude oil, WPI, RBI repo rate)")
    last = df.iloc[-1]
    c = st.columns(4)
    c[0].metric("Latest inflation (YoY)", f"{last['infl_yoy']:.2f}%", help=f"{last['month']:%b %Y}")
    c[1].metric("Level", last["infl_level"])
    c[2].metric("Repo rate", f"{last['repo']:.2f}%")
    c[3].metric("USD/INR (avg)", f"{last['usdinr_avg']:.1f}")

    st.subheader("Headline CPI inflation")
    lines = alt.Chart(df).mark_line(color="#264653", strokeWidth=2).encode(
        x=alt.X("month:T", title=None), y=alt.Y("infl_yoy:Q", title="% year-on-year"),
        tooltip=[alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("infl_yoy:Q", format=".2f"), "infl_level"])
    rules = alt.Chart(pd.DataFrame({"y": [LOW_MAX, HIGH_MIN]})).mark_rule(strokeDash=[5, 4], color="grey").encode(y="y:Q")
    st.altair_chart((lines + rules).properties(height=330), use_container_width=True)
    st.caption("Dashed lines: 4% (RBI target) and 6% (upper tolerance). LOW < 4%, MEDIUM 4-6%, HIGH > 6%.")

    st.subheader("Months in each inflation level, by year")
    ct = pd.crosstab(df["year"], df["infl_level"]).reindex(columns=LEVELS, fill_value=0).reset_index().melt("year", var_name="level", value_name="months")
    st.altair_chart(alt.Chart(ct).mark_bar().encode(
        x=alt.X("year:O", title=None), y=alt.Y("months:Q"), color=alt.Color("level:N", scale=LEVEL_COLORS, sort=LEVELS),
        tooltip=["year", "level", "months"]).properties(height=260), use_container_width=True)


def page_drivers(df):
    st.header("What moves with inflation?")
    opts = {"USD/INR (YoY %)": "usdinr_yoy", "Brent in rupees (YoY %)": "brent_inr_yoy", "WPI all (YoY %)": "wpi_all_yoy",
            "WPI food (YoY %)": "wpi_food_yoy", "WPI fuel (YoY %)": "wpi_fuel_yoy", "WPI manufacturing (YoY %)": "wpi_manuf_yoy",
            "Repo rate (%)": "repo"}
    name = st.selectbox("Indicator", list(opts))
    col = opts[name]
    both = df[["month", "infl_yoy", col]].melt("month", var_name="series", value_name="value")
    both["series"] = both["series"].map({"infl_yoy": "Headline inflation", col: name})
    st.altair_chart(alt.Chart(both).mark_line().encode(
        x=alt.X("month:T", title=None), y=alt.Y("value:Q", title="%"), color=alt.Color("series:N", title=None),
        tooltip=["series", alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("value:Q", format=".2f")]).properties(height=320),
        use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.altair_chart(alt.Chart(df).mark_circle(size=50, opacity=0.7).encode(
            x=alt.X(f"{col}:Q", title=name), y=alt.Y("infl_yoy:Q", title="Inflation (YoY %)"),
            color=alt.Color("infl_level:N", scale=LEVEL_COLORS, sort=LEVELS),
            tooltip=[alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip(f"{col}:Q", format=".2f"), alt.Tooltip("infl_yoy:Q", format=".2f")]
        ).properties(height=300, title=f"Correlation with inflation: {df[col].corr(df['infl_yoy']):+.2f}"), use_container_width=True)
    with right:
        ll = load_csv(TAB / "leadlag_vs_inflation.csv", index_col=0)
        if ll is not None:
            st.markdown("**Does the indicator lead inflation?** Correlation of the indicator *k* months earlier with inflation today.")
            st.dataframe(ll.style.format("{:.2f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1), use_container_width=True)
    prof = load_csv(TAB / "level_profile.csv", index_col=0)
    if prof is not None:
        st.markdown("**Average of each indicator in LOW / MEDIUM / HIGH inflation months**")
        st.dataframe(prof.style.format("{:.2f}"), use_container_width=True)
    with st.expander("More EDA figures"):
        for f in ["02_cpi_groups_yoy.png", "04_oil_vs_fuel_inflation.png", "05_repo_vs_inflation.png", "06_correlation_heatmap.png", "08_subgroup_volatility.png"]:
            show_fig(f)


def page_regimes(df):
    st.header("Economic regimes (K-Means clustering)")
    ca = load_csv(TAB / "cluster_assignments.csv", parse_dates=["month"])
    prof = load_csv(TAB / "kmeans_cluster_profile.csv")
    if not (need(ca, "Cluster assignments") and need(prof, "Cluster profile")):
        return
    ca["cluster"] = "Cluster " + ca["cluster"].astype(str)
    st.altair_chart(alt.Chart(ca).mark_circle(size=60).encode(
        x=alt.X("month:T", title=None), y=alt.Y("infl_yoy:Q", title="Headline inflation (YoY %)"),
        color=alt.Color("cluster:N", title=None), tooltip=[alt.Tooltip("month:T", format="%b %Y"), "cluster", "infl_level",
                                                           alt.Tooltip("infl_yoy:Q", format=".2f")]).properties(height=330),
        use_container_width=True)
    st.markdown("**Cluster profile** (average of each indicator; cluster 0 has the lowest inflation)")
    st.dataframe(prof.set_index("cluster"), use_container_width=True)
    sel = load_csv(TAB / "kmeans_k_selection.csv")
    c1, c2 = st.columns(2)
    if sel is not None:
        c1.markdown("**Choosing k**")
        c1.dataframe(sel.set_index("k"), use_container_width=True)
    with c2:
        show_fig("12_kmeans_profile_heatmap.png")


def page_classifiers():
    st.header("Classifying the inflation level")
    st.caption("Task: LOW / MEDIUM / HIGH from outside indicators only (no CPI columns, so there is no leakage).")
    res = load_csv(TAB / "classification_results.csv")
    if not need(res, "Classification results"):
        return
    schemes = list(res["scheme"].unique())
    tabs = st.tabs(["Compare models", "Confusion matrix", "Feature importance and rules"])
    with tabs[0]:
        piv = res.pivot(index="model", columns="scheme", values="macro_f1")[schemes]
        long = piv.reset_index().melt("model", var_name="scheme", value_name="macro_f1")
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            x=alt.X("model:N", title=None, sort=None), y=alt.Y("macro_f1:Q", title="Macro-F1 (higher is better)"),
            xOffset="scheme:N", color=alt.Color("scheme:N", title=None),
            tooltip=["model", "scheme", alt.Tooltip("macro_f1:Q", format=".3f")]).properties(height=340), use_container_width=True)
        st.caption("Random 10-fold CV (A) is optimistic: months from every year mix. Time-ordered tests (B, C) predict the future "
                   "from the past and are the realistic numbers. Majority (ZeroR) is the do-nothing baseline.")
        s = st.selectbox("Full table for scheme", schemes)
        st.dataframe(res[res["scheme"] == s].drop(columns="scheme").set_index("model").style.format(precision=3), use_container_width=True)
    with tabs[1]:
        cm = load_csv(TAB / "classification_confusion.csv")
        if cm is not None:
            a, b = st.columns(2)
            s = a.selectbox("Scheme", schemes, key="cm_s")
            m = b.selectbox("Model", list(cm["model"].unique()), key="cm_m")
            d = cm[(cm["scheme"] == s) & (cm["model"] == m)]
            base = alt.Chart(d).encode(x=alt.X("predicted:N", sort=LEVELS), y=alt.Y("actual:N", sort=LEVELS))
            st.altair_chart((base.mark_rect().encode(color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None))
                             + base.mark_text(fontSize=18).encode(text="count:Q")).properties(width=380, height=320), use_container_width=False)
    with tabs[2]:
        imp = load_csv(TAB / "rf_feature_importance.csv", index_col=0)
        c1, c2 = st.columns(2)
        if imp is not None:
            c1.markdown("**Random Forest feature importance**")
            c1.altair_chart(alt.Chart(imp.reset_index().rename(columns={"index": "feature"})).mark_bar(color="#264653").encode(
                x="importance:Q", y=alt.Y("feature:N", sort="-x")).properties(height=380), use_container_width=True)
        p = TAB / "id3_rules.txt"
        if p.exists():
            c2.markdown("**ID3 decision rules** (written from scratch)")
            c2.code(p.read_text(), language="text")


def page_forecast():
    st.header("Forecasting next month's inflation")
    res = load_csv(TAB / "prediction_results.csv")
    det = load_csv(TAB / "prediction_details.csv", parse_dates=["month"])
    if not (need(res, "Prediction results") and need(det, "Prediction details")):
        return
    schemes = list(res["scheme"].unique())
    s = st.selectbox("Evaluation", schemes, index=min(1, len(schemes) - 1))
    t = res[res["scheme"] == s].sort_values("rmse")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown("**Error (RMSE, percentage points; lower is better)**")
        st.dataframe(t[["model", "rmse", "mae", "direction_acc"]].set_index("model").style.format(precision=3, na_rep="-"), use_container_width=True)
    with c2:
        st.altair_chart(alt.Chart(t).mark_bar(color="#e76f51").encode(
            x=alt.X("rmse:Q", title="RMSE"), y=alt.Y("model:N", sort="x", title=None)).properties(height=300), use_container_width=True)
    d = det[det["scheme"] == s]
    models = [m for m in d["model"].unique()]
    default = [m for m in ["Naive (last value)", "ARIMA (auto order)", "Neural net - with CPI", "Neural net - outside only"] if m in models]
    pick = st.multiselect("Models to draw", models, default=default)
    act = d[d["model"] == models[0]][["month", "actual"]].assign(series="Actual").rename(columns={"actual": "value"})
    pr = d[d["model"].isin(pick)][["month", "predicted", "model"]].rename(columns={"predicted": "value", "model": "series"})
    all_ = pd.concat([act, pr])
    st.altair_chart(alt.Chart(all_).mark_line().encode(
        x=alt.X("month:T", title=None), y=alt.Y("value:Q", title="YoY inflation (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None), strokeDash=alt.condition(alt.datum.series == "Actual", alt.value([1, 0]), alt.value([4, 3])),
        tooltip=["series", alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("value:Q", format=".2f")]).properties(height=340),
        use_container_width=True)
    st.info("Inflation is very persistent, so 'same as last month' and ARIMA are hard to beat. Outside indicators alone forecast badly; "
            "adding current and lagged CPI helps a lot. The neural net did not beat ARIMA on this small sample.")
    show_fig("18_nn_loss_curve.png", "Neural-network training loss (backpropagation)")


def page_rules():
    st.header("Association rules (Apriori)")
    st.caption("Each month is a basket of items such as `usdinr_yoy=High`. support = share of months; confidence = how often "
               "the right side holds given the left; lift > 1 = more often than chance. Co-occurrence, not cause and effect.")
    a = load_csv(TAB / "apriori_rules_to_level.csv")
    b = load_csv(TAB / "apriori_rules_all.csv")
    if not need(a, "Apriori results"):
        return
    c1, c2 = st.columns(2)
    lvl = c1.selectbox("Rules that predict level", ["level=HIGH", "level=MEDIUM", "level=LOW"])
    min_lift = c2.slider("Minimum lift", 1.0, 4.0, 1.5, 0.1)
    st.dataframe(a[(a["rhs"] == lvl) & (a["lift"] >= min_lift)].reset_index(drop=True), use_container_width=True)
    with st.expander("Rules among all indicators"):
        if b is not None:
            st.dataframe(b[b["lift"] >= min_lift].reset_index(drop=True), use_container_width=True)
    show_fig("19_apriori_scatter.png")


def page_warehouse():
    st.header("Data warehouse and OLAP queries")
    if not DB.exists():
        st.warning("data/warehouse.db not found. Run `python run_pipeline.py` first.")
        return
    st.caption("Fact constellation: Fact_CPI (star: Dim_Date, Dim_Sector, Dim_Item; snowflake: Dim_Item -> Dim_Group) and Fact_Macro.")
    op = st.selectbox("OLAP operation", ["Roll-up: year", "Drill-down: quarter/month of a year", "Slice: one sector",
                                         "Dice: year range + CPI group", "Pivot: level counts by year", "Your own SELECT"])
    if op.startswith("Roll-up"):
        q = ("SELECT year, ROUND(AVG(infl_yoy),2) AS avg_inflation, ROUND(MAX(infl_yoy),2) AS max_inflation, "
             "ROUND(AVG(usdinr_avg),2) AS avg_usdinr, ROUND(AVG(brent_usd),1) AS avg_brent_usd, ROUND(AVG(repo_rate),2) AS avg_repo "
             "FROM vw_macro_monthly WHERE infl_yoy IS NOT NULL GROUP BY year ORDER BY year")
        params = ()
    elif op.startswith("Drill"):
        years = sql("SELECT DISTINCT year FROM vw_macro_monthly ORDER BY year")["year"].tolist()
        y = st.selectbox("Year", years, index=len(years) - 1)
        q = ("SELECT year, quarter, month_num, ROUND(infl_yoy,2) AS inflation, ROUND(usdinr_avg,2) AS usdinr, "
             "ROUND(brent_usd,1) AS brent_usd, repo_rate, infl_level FROM vw_macro_monthly WHERE year = ? ORDER BY month_num")
        params = (int(y),)
    elif op.startswith("Slice"):
        secs = sql("SELECT sector_name FROM Dim_Sector ORDER BY sector_name")["sector_name"].tolist()
        s = st.selectbox("Sector", secs, index=secs.index("Combined") if "Combined" in secs else 0)
        q = ("SELECT year, ROUND(AVG(inflation_yoy),2) AS avg_inflation, COUNT(*) AS rows_used FROM vw_cpi_detail "
             "WHERE sector_name = ? AND subgroup_name = 'General-Overall' AND inflation_yoy IS NOT NULL GROUP BY year ORDER BY year")
        params = (s,)
    elif op.startswith("Dice"):
        groups = sql("SELECT group_name FROM Dim_Group ORDER BY group_name")["group_name"].tolist()
        g = st.multiselect("CPI groups", groups, default=groups[:3])
        y0, y1 = st.slider("Years", 2013, 2025, (2020, 2025))
        if not g:
            st.info("Pick at least one group.")
            return
        marks = ",".join("?" * len(g))
        q = (f"SELECT year, group_name, ROUND(AVG(inflation_yoy),2) AS avg_inflation FROM vw_cpi_detail WHERE sector_name = 'Combined' "
             f"AND group_name IN ({marks}) AND year BETWEEN ? AND ? AND inflation_yoy IS NOT NULL GROUP BY year, group_name ORDER BY year, group_name")
        params = tuple(g) + (y0, y1)
    elif op.startswith("Pivot"):
        d = sql("SELECT year, infl_level, COUNT(*) AS months FROM vw_macro_monthly WHERE infl_level IS NOT NULL GROUP BY year, infl_level")
        out = d.pivot(index="year", columns="infl_level", values="months").reindex(columns=LEVELS).fillna(0).astype(int)
        st.code("SELECT year, infl_level, COUNT(*) FROM vw_macro_monthly GROUP BY year, infl_level   -- then pivoted", language="sql")
        st.dataframe(out, use_container_width=True)
        return
    else:
        q = st.text_area("SQL (SELECT only). Tables: Fact_CPI, Fact_Macro, Dim_*; views: vw_cpi_detail, vw_macro_monthly",
                         "SELECT year, COUNT(*) AS months FROM vw_macro_monthly GROUP BY year", height=120)
        params = ()
        if not re.match(r"^\s*select\b", q, re.I) or ";" in q.strip().rstrip(";"):
            st.error("Only a single SELECT statement is allowed.")
            return
    st.code(q + (f"\n-- parameters: {params}" if params else ""), language="sql")
    try:
        out = sql(q, params)
    except Exception as e:
        st.error(f"Query failed: {e}")
        return
    st.dataframe(out, use_container_width=True)
    if op.startswith(("Roll", "Slice")) and "avg_inflation" in out:
        st.bar_chart(out.set_index("year")["avg_inflation"])


@st.cache_resource
def train_model(name, cols, _df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    model = {"Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, min_samples_leaf=2),
             "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
             "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
             "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))}[name]
    return model.fit(_df[list(cols)], _df["infl_level"])


def page_predict(df):
    st.header("Try it yourself: what inflation level do these conditions suggest?")
    st.caption("Set the outside indicators; a classifier trained on all 144 months predicts LOW / MEDIUM / HIGH. "
               "Defaults are the latest month. This is a teaching demo: held-out time-ordered accuracy is modest (see Classifiers).")
    cols = tuple(load_sets().get("CLF_FEATURES", []))
    if not cols:
        st.warning("feature_sets.json not found. Run the pipeline first.")
        return
    name = st.selectbox("Model", ["Random Forest", "Naive Bayes", "kNN", "SVM"])
    last = df.iloc[-1]
    vals = {}
    grid = st.columns(3)
    for i, c in enumerate(cols):
        lo, hi = float(df[c].quantile(0.0)), float(df[c].quantile(1.0))
        if hi - lo < 1e-9:
            hi = lo + 1.0
        vals[c] = grid[i % 3].slider(c, lo, hi, float(np.clip(last[c], lo, hi)), step=(hi - lo) / 200, key=f"s_{c}")
    model = train_model(name, cols, df)
    x = pd.DataFrame([vals])[list(cols)]
    pred = model.predict(x)[0]
    proba = dict(zip(model.classes_, model.predict_proba(x)[0]))
    st.subheader(f"Predicted level: {pred}")
    pdf = pd.DataFrame({"level": LEVELS, "probability": [proba.get(l, 0.0) for l in LEVELS]})
    st.altair_chart(alt.Chart(pdf).mark_bar().encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("level:N", sort=LEVELS, title=None),
        color=alt.Color("level:N", scale=LEVEL_COLORS, legend=None), tooltip=["level", alt.Tooltip("probability:Q", format=".2f")]).properties(height=140),
        use_container_width=True)
    st.caption(f"Actual level in the latest month ({last['month']:%b %Y}): {last['infl_level']}.")


# ------------------------------------------------------------------ router
PAGES = ["Overview", "Drivers", "Regimes (K-Means)", "Classifiers", "Forecasting", "Association rules",
         "Warehouse (OLAP)", "Try it yourself"]
st.sidebar.title("Inflation Analysis")
page = st.sidebar.radio("Page", PAGES)
st.sidebar.caption("DMDW project: warehouse, K-Means, ID3, Naive Bayes, kNN, SVM, neural network, Apriori.")

data = load_features()
if data is None and page not in ("Warehouse (OLAP)",):
    st.title("Smart Inflation Analysis")
    st.warning("data/processed/features.csv not found. Run `python run_pipeline.py` in the project folder first.")
elif page == "Overview":
    page_overview(data)
elif page == "Drivers":
    page_drivers(data)
elif page == "Regimes (K-Means)":
    page_regimes(data)
elif page == "Classifiers":
    page_classifiers()
elif page == "Forecasting":
    page_forecast()
elif page == "Association rules":
    page_rules()
elif page == "Warehouse (OLAP)":
    page_warehouse()
else:
    page_predict(data)
