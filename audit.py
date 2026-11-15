import glob, os
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

print("FILES IN FOLDER:", sorted(os.listdir(".")))

# ---------- CPI ----------
print("\n===== CPI =====")
files = sorted(glob.glob("cpi_data/*"))
print("number of files:", len(files))
frames = []
for f in files:
    try:
        xl = pd.ExcelFile(f)
        df = pd.read_excel(f, sheet_name=0)
        yrs = sorted(df["year"].unique())
        mths = sorted(df["month_code"].unique())
        print(os.path.basename(f), "| sheets:", xl.sheet_names, "| rows:", len(df),
              "| years:", yrs, "| months:", mths)
        frames.append(df)
    except Exception as e:
        print(os.path.basename(f), "ERROR:", e)

cpi = pd.concat(frames, ignore_index=True)
print("\nshape:", cpi.shape)
print("columns:", list(cpi.columns))
print("baseyear:", cpi["baseyear"].value_counts().to_dict())
print("states:", cpi["state"].unique())
print("sectors:", cpi["sector"].unique())
print("status:", cpi["status"].value_counts().to_dict())

c = cpi[(cpi["state"] == "All India") & (cpi["sector"] == "Combined")]
print("\ngroups:", c["group"].unique())
print("subgroups:", c["subgroup"].unique())
print("\ndistinct months per year (Combined):")
print(c.groupby("year")["month_code"].nunique().to_string())
print("\nnull index per year:")
print(c[c["index"].isna()].groupby("year").size().to_string())
print("\nnull inflation per year:")
print(c[c["inflation"].isna()].groupby("year").size().to_string())
g = c[c["subgroup"].str.contains("general", case=False, na=False)]
print("\nGeneral index sample:")
print(g[["year", "month_code", "index", "inflation", "status"]]
      .sort_values(["year", "month_code"]).head(8).to_string())
print("...")
print(g[["year", "month_code", "index", "inflation", "status"]]
      .sort_values(["year", "month_code"]).tail(5).to_string())

# ---------- PINK SHEET ----------
print("\n===== PINK SHEET =====")
ps = glob.glob("CMO-Historical-Data-Monthly*")[0]
xl = pd.ExcelFile(ps)
print("file:", ps, "| sheets:", xl.sheet_names)
for s in xl.sheet_names[:2]:
    raw = pd.read_excel(ps, sheet_name=s, header=None, nrows=8)
    print("\nsheet:", s)
    print(raw.iloc[:, :6].to_string())

# ---------- monthly_index (WPI?) ----------
print("\n===== monthly_index_202606 =====")
wp = glob.glob("monthly_index_202606*")[0]
xl = pd.ExcelFile(wp)
print("file:", wp, "| sheets:", xl.sheet_names)
raw = pd.read_excel(wp, sheet_name=0, header=None, nrows=8)
print("shape of preview:", raw.shape)
print(raw.iloc[:, :8].to_string())
full = pd.read_excel(wp, sheet_name=0, header=None)
print("total shape:", full.shape)

# ---------- FOREX ----------
print("\n===== FOREX =====")
fx = glob.glob("r691*")[0]
with open(fx, encoding="utf-8", errors="replace") as fh:
    lines = fh.readlines()
print("file:", fx, "| lines:", len(lines))
print("first 13 lines:")
for ln in lines[:13]:
    print("  ", ln.rstrip())
print("last 4 lines:")
for ln in lines[-4:]:
    print("  ", ln.rstrip())

# ---------- REPO ----------
print("\n===== REPO =====")
rp = pd.read_csv(glob.glob("rates-policy*")[0])
print("columns:", list(rp.columns))
r = rp[["date", "Policy repo rate"]].dropna()
print("rows with repo rate:", len(r))
print(r.head(5).to_string())
print(r.tail(8).to_string())