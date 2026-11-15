"""
neural_net.py - predict NEXT MONTH's year-on-year inflation with a backpropagation
neural network, and compare it with simple baselines.

Place at:  <project root>/src/models/neural_net.py
Run from the project root:  python -m src.models.neural_net     (a few minutes)

Target     target_next_yoy = headline YoY inflation (%) one month ahead.
           (We predict inflation, not the raw CPI level: the level trends upward and
            neural nets cannot extrapolate beyond the range they were trained on.)

Models, each run with two input sets
  outside only   the 22 outside indicators (USD/INR, oil, WPI, repo + lags)   REG_EXOG
  with CPI       the same + current/lagged inflation, food and fuel inflation REG_WITH_CPI
  Naive          next month = this month  (the baseline every forecast must beat)
  ARIMA          univariate, order picked by AIC on the training part (needs statsmodels)
  Ridge          regularised linear regression
  Random Forest  EXTRA
  Neural net     scikit-learn MLPRegressor (hidden layers, ReLU, trained by
                 backpropagation with the Adam optimiser). Settings are tuned with
                 time-ordered CV on the training part only, and 5 networks with
                 different random starts are averaged (tiny data = noisy training).

Three evaluations (a row at month t predicts month t+1; splits use the TARGET month)
  V  validation   train: targets up to Dec 2022, test: 2023
  T  test         train: targets up to Dec 2023, test: 2024-2025  (24 months)
  C  expanding    5 time-ordered folds, always train on the past, pooled results
Metrics: RMSE and MAE in percentage points (lower is better), and direction accuracy
(did we get the sign of the change in inflation right).

Writes  outputs/tables/prediction_results.csv, prediction_details.csv
        outputs/figures/16_forecast_vs_actual.png, 17_prediction_rmse.png,
        18_nn_loss_curve.png
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src import config as cfg

warnings.filterwarnings("ignore")
SEED = cfg.RANDOM_SEED
N_NETS = 5


# ============================================================ model fitters
def fit_nn(Xtr, ytr, Xte):
    sx = StandardScaler().fit(Xtr)
    mu, sd = ytr.mean(), (ytr.std() or 1.0)
    A, b = sx.transform(Xtr), (ytr - mu) / sd
    grid = {"hidden_layer_sizes": [(8,), (16, 8)], "alpha": [0.01, 1.0]}
    gs = GridSearchCV(MLPRegressor(solver="adam", max_iter=1500, random_state=SEED), grid,
                      cv=TimeSeriesSplit(3), scoring="neg_root_mean_squared_error", n_jobs=1).fit(A, b)
    preds, curve = [], None
    for s in range(N_NETS):
        m = MLPRegressor(**gs.best_params_, solver="adam", max_iter=1500, random_state=SEED + s).fit(A, b)
        preds.append(m.predict(sx.transform(Xte)))
        if s == 0:
            curve = m.loss_curve_
    return np.mean(preds, axis=0) * sd + mu, curve, gs.best_params_


def fit_ridge(Xtr, ytr, Xte):
    sx = StandardScaler().fit(Xtr)
    gs = GridSearchCV(Ridge(), {"alpha": np.logspace(-1, 3, 9)}, cv=TimeSeriesSplit(3),
                      scoring="neg_root_mean_squared_error").fit(sx.transform(Xtr), ytr)
    return gs.predict(sx.transform(Xte))


def fit_rf(Xtr, ytr, Xte):
    rf = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=3, random_state=SEED)
    return rf.fit(Xtr, ytr).predict(Xte)


def arima_forecasts(infl_full, train_pos, test_pos):
    """One-step-ahead forecasts. Order is chosen by AIC on data up to the last training
    row; then for each test row the model is re-applied to the data known at that month."""
    from statsmodels.tsa.arima.model import ARIMA
    fit_series = infl_full[: max(train_pos) + 2]
    best, best_aic, best_order = None, np.inf, None
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    r = ARIMA(fit_series, order=(p, d, q)).fit()
                except Exception:
                    continue
                if r.aic < best_aic:
                    best, best_aic, best_order = r, r.aic, (p, d, q)
    out = [float(best.apply(infl_full[: p0 + 1]).forecast(1)[0]) for p0 in test_pos]
    return np.array(out), best_order


def metrics(y, p, cur, directional=True):
    err = p - y
    out = {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err)))}
    out["direction_acc"] = float(np.mean(np.sign(p - cur) == np.sign(y - cur))) if directional else np.nan
    return out


# ============================================================ one train/test run
def run_all(tr, te, Xs, y, cur, infl_full, pos, have_arima):
    out, info = {}, {}
    out["Naive (last value)"] = cur[te]
    if have_arima:
        out["ARIMA (auto order)"], info["arima_order"] = arima_forecasts(infl_full, pos[tr], pos[te])
    for tag, X in Xs.items():
        out[f"Ridge - {tag}"] = fit_ridge(X[tr], y[tr], X[te])
        out[f"Random Forest (extra) - {tag}"] = fit_rf(X[tr], y[tr], X[te])
        out[f"Neural net - {tag}"], curve, best = fit_nn(X[tr], y[tr], X[te])
        info[f"nn_{tag}"] = best
        info[f"curve_{tag}"] = curve
    return out, info


def main():
    if not cfg.FEATURES_CSV.exists():
        raise SystemExit("Run first:  python -m src.features.build_features")
    try:
        import statsmodels  # noqa: F401
        have_arima = True
    except ImportError:
        have_arima = False
        print("NOTE: statsmodels not installed - ARIMA skipped (pip install statsmodels)")

    df = pd.read_csv(cfg.FEATURES_CSV, parse_dates=["month"])
    sets = json.loads((cfg.PROCESSED_DIR / "feature_sets.json").read_text())
    cols_exog, cols_cpi = sets["REG_EXOG"], sets["REG_WITH_CPI"]

    ok = df["target_next_yoy"].notna() & df[cols_cpi].notna().all(axis=1)
    pos = np.where(ok)[0]                       # positions in df (consecutive months)
    d = df[ok].reset_index(drop=True)
    y, cur = d["target_next_yoy"].values, d["infl_yoy"].values
    infl_full = df["infl_yoy"].values
    tmonth = d["month"] + pd.offsets.MonthBegin(1)      # month being predicted
    Xs = {"outside only": d[cols_exog].values, "with CPI": d[cols_cpi].values}
    print(f"{len(d)} usable rows: predicting months {tmonth.min():%Y-%m} to {tmonth.max():%Y-%m}; "
          f"inputs: outside only = {len(cols_exog)} features, with CPI = {len(cols_cpi)}")

    tr_end, val_end = pd.Timestamp(cfg.TRAIN_END), pd.Timestamp(cfg.VAL_END)
    idx = np.arange(len(d))
    runs = {
        "V": ("V: validation 2023", idx[tmonth <= tr_end], idx[(tmonth > tr_end) & (tmonth <= val_end)]),
        "T": ("T: test 2024-25", idx[tmonth <= val_end], idx[tmonth > val_end]),
    }
    folds = list(TimeSeriesSplit(n_splits=5).split(idx))

    results, details, infos = [], [], {}
    for key, (label, tr, te) in runs.items():
        print(f"\nRunning {label}: train {len(tr)} rows, test {len(te)} rows ...")
        preds, info = run_all(tr, te, Xs, y, cur, infl_full, pos, have_arima)
        infos[key] = info
        for name, p in preds.items():
            results.append({"scheme": label, "model": name, "n_test": len(te),
                            **metrics(y[te], p, cur[te], directional=not name.startswith("Naive"))})
            details.append(pd.DataFrame({"scheme": label, "month": tmonth.values[te], "actual": y[te],
                                         "model": name, "predicted": p}))

    label_c = "C: expanding-window CV"
    pooled, pooled_te = {}, []
    for i, (tr, te) in enumerate(folds, 1):
        print(f"Running {label_c}, fold {i}/5 (train {len(tr)}, test {len(te)}) ...")
        preds, _ = run_all(tr, te, Xs, y, cur, infl_full, pos, have_arima)
        pooled_te.append(te)
        for name, p in preds.items():
            pooled.setdefault(name, []).append(p)
    te_all = np.concatenate(pooled_te)
    for name, parts in pooled.items():
        p = np.concatenate(parts)
        results.append({"scheme": label_c, "model": name, "n_test": len(te_all),
                        **metrics(y[te_all], p, cur[te_all], directional=not name.startswith("Naive"))})
        details.append(pd.DataFrame({"scheme": label_c, "month": tmonth.values[te_all], "actual": y[te_all],
                                     "model": name, "predicted": p}))

    res = pd.DataFrame(results).round(3)
    det = pd.concat(details, ignore_index=True)
    res.to_csv(cfg.TABLE_DIR / "prediction_results.csv", index=False)
    det.to_csv(cfg.TABLE_DIR / "prediction_details.csv", index=False)

    # ---------------- print
    for label in res["scheme"].unique():
        t = res[res["scheme"] == label].sort_values("rmse")
        print(f"\n=== {label}  ({int(t['n_test'].iloc[0])} months predicted) ===")
        print(t[["model", "rmse", "mae", "direction_acc"]].to_string(index=False))
    for key in ("V", "T"):
        print(f"\nNetwork settings chosen ({runs[key][0]}): outside only {infos[key]['nn_outside only']}, "
              f"with CPI {infos[key]['nn_with CPI']}" + (f"; ARIMA order {infos[key].get('arima_order')}" if have_arima else ""))

    print("\nRMSE change of the neural net (with CPI) relative to the naive baseline and to ARIMA:")
    for label in res["scheme"].unique():
        t = res[res["scheme"] == label].set_index("model")["rmse"]
        nn = t["Neural net - with CPI"]
        s = f"  {label}: vs naive {100 * (nn / t['Naive (last value)'] - 1):+.0f}%"
        if "ARIMA (auto order)" in t:
            s += f", vs ARIMA {100 * (nn / t['ARIMA (auto order)'] - 1):+.0f}%"
        print(s + "   (negative = neural net is better)")

    # ---------------- figures
    print("\nFigures:")
    D = det.copy()
    D["month"] = pd.to_datetime(D["month"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    sty = {"Naive (last value)": ("#8d99ae", "--"), "ARIMA (auto order)": ("#6a4c93", "-"),
           "Neural net - with CPI": ("#e76f51", "-"), "Neural net - outside only": ("#2a9d8f", "-")}
    for ax, label, title in ((axes[0], label_c, "All time-ordered test folds (pooled)"),
                             (axes[1], runs["T"][0], "Test period 2024-25")):
        sub = D[D["scheme"] == label]
        act = sub[sub["model"] == "Naive (last value)"].sort_values("month")
        ax.plot(act["month"], act["actual"], color="black", lw=2.2, label="Actual")
        for name, (color, ls) in sty.items():
            s = sub[sub["model"] == name].sort_values("month")
            if len(s):
                ax.plot(s["month"], s["predicted"], color=color, ls=ls, lw=1.5, label=name)
        ax.set_title(f"Next-month inflation: actual vs predicted - {title}")
        ax.set_ylabel("YoY inflation (%)")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "16_forecast_vs_actual.png", dpi=150)
    plt.close(fig)
    print("  saved 16_forecast_vs_actual.png")

    models = list(res["model"].unique())
    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.26
    for i, label in enumerate(res["scheme"].unique()):
        t = res[res["scheme"] == label].set_index("model").reindex(models)
        ax.bar(np.arange(len(models)) + (i - 1) * w, t["rmse"], w, label=label)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (percentage points, lower is better)")
    ax.set_title("Next-month inflation forecast error")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "17_prediction_rmse.png", dpi=150)
    plt.close(fig)
    print("  saved 17_prediction_rmse.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(infos["T"]["curve_with CPI"], color="#264653")
    ax.set(title="Neural network training: loss falls as backpropagation updates the weights",
           xlabel="Training epoch", ylabel="Loss (squared error, standardised target)")
    ax.title.set_fontsize(9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "18_nn_loss_curve.png", dpi=150)
    plt.close(fig)
    print("  saved 18_nn_loss_curve.png")
    print(f"\nTables in {cfg.TABLE_DIR}")


if __name__ == "__main__":
    main()
