"""
classify.py - classify each month as LOW / MEDIUM / HIGH inflation.

Place at:  <project root>/src/models/classify.py
Run from the project root:  python -m src.models.classify      (takes about a minute)

Models (all use the same 14 outside-indicator features; NO CPI columns, so the label
is never leaked into the inputs)
  Majority (ZeroR)  baseline: always predicts the commonest class
  ID3               written from scratch (information gain); features are first binned
                    into low/mid/high because ID3 only handles categories; depth-limited
  Naive Bayes       Gaussian
  kNN               scaled features, k and weighting tuned
  SVM               RBF kernel, scaled features, C and gamma tuned
  Random Forest     EXTRA (not in the syllabus)
  XGBoost           EXTRA (skipped automatically if the xgboost package is not installed)
Hyper-parameters are tuned inside each training set only (nested), so test data is never
used for tuning.

Three ways of testing (reported separately)
  A  10-fold stratified CV   same idea as WEKA's default; ignores time, so optimistic
  B  time holdout            train Jan 2014-Dec 2023, test 2024-2025 (24 months; only
                             ONE of them is HIGH, so HIGH cannot be judged here)
  C  expanding-window CV     5 folds, always train on the past and test on the next block

Writes  outputs/tables/classification_results.csv   every metric, every model, every scheme
        outputs/tables/classification_confusion.csv confusion matrices (long format)
        outputs/tables/id3_rules.txt                the ID3 decision rules, readable
        outputs/tables/rf_feature_importance.csv
        outputs/figures/13_..15_*.png
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")        # silences the Windows KMeans/MKL warning
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")     # silences the joblib core-count warning

import json
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.svm import SVC

from src import config as cfg

warnings.filterwarnings("ignore")

LEVELS = cfg.LEVEL_LABELS                      # ['LOW', 'MEDIUM', 'HIGH']
TO_INT = {l: i for i, l in enumerate(LEVELS)}
BIN_NAMES = {0: "low", 1: "mid", 2: "high"}
SCHEMES = {"A": "A: 10-fold stratified CV", "B": "B: time holdout 2024-25",
           "C": "C: expanding-window CV"}


# ======================================================================= ID3
class ID3Classifier(ClassifierMixin, BaseEstimator):
    """ID3 decision tree for categorical (integer-coded) features.
    Splits on the attribute with the highest information gain; stops when the node is
    pure, attributes run out, max_depth is reached or fewer than min_samples remain."""

    def __init__(self, max_depth=4, min_samples=4):
        self.max_depth = max_depth
        self.min_samples = min_samples

    @staticmethod
    def _entropy(y):
        p = np.bincount(y) / len(y)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def _gain(self, x, y):
        n, rem = len(y), 0.0
        for v in np.unique(x):
            m = x == v
            rem += m.sum() / n * self._entropy(y[m])
        return self._entropy(y) - rem

    def _build(self, X, y, attrs, depth):
        majority = Counter(y.tolist()).most_common(1)[0][0]
        if len(set(y.tolist())) == 1 or not attrs or depth >= self.max_depth or len(y) < self.min_samples:
            return majority
        gains = [self._gain(X[:, a], y) for a in attrs]
        if max(gains) <= 1e-9:
            return majority
        best = attrs[int(np.argmax(gains))]
        node = {"attr": best, "default": majority, "children": {}}
        rest = [a for a in attrs if a != best]
        for v in np.unique(X[:, best]):
            m = X[:, best] == v
            node["children"][int(v)] = self._build(X[m], y[m], rest, depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X).astype(int)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        self.tree_ = self._build(X, y, list(range(X.shape[1])), 0)
        return self

    def _one(self, row):
        node = self.tree_
        while isinstance(node, dict):
            v = int(row[node["attr"]])
            if v not in node["children"]:
                return node["default"]
            node = node["children"][v]
        return node

    def predict(self, X):
        X = np.asarray(X).astype(int)
        return np.array([self._one(r) for r in X])


def tree_text(node, names, indent=""):
    if not isinstance(node, dict):
        return f"{indent}=> {LEVELS[int(node)]}\n"
    out = ""
    for v, child in sorted(node["children"].items()):
        out += f"{indent}{names[node['attr']]} = {BIN_NAMES.get(v, v)}\n"
        out += tree_text(child, names, indent + "|   ")
    return out


# ===================================================================== models
def tuned(est, grid):
    return GridSearchCV(est, grid, cv=3, scoring="f1_macro", n_jobs=1)


def make_models():
    seed = cfg.RANDOM_SEED
    models = {
        "Majority (ZeroR)": DummyClassifier(strategy="most_frequent"),
        "ID3": tuned(Pipeline([("bin", KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")),
                               ("id3", ID3Classifier())]), {"id3__max_depth": [2, 3, 4, 5]}),
        "Naive Bayes": tuned(GaussianNB(), {"var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2]}),
        "kNN": tuned(Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier())]),
                     {"knn__n_neighbors": [3, 5, 7, 9, 11], "knn__weights": ["uniform", "distance"]}),
        "SVM": tuned(Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="rbf"))]),
                     {"svm__C": [0.5, 1, 5, 10], "svm__gamma": ["scale", 0.02, 0.05, 0.1]}),
        "Random Forest (extra)": RandomForestClassifier(n_estimators=200, max_depth=4,
                                                         min_samples_leaf=2, random_state=seed),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost (extra)"] = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.05,
                                                  subsample=0.8, random_state=seed,
                                                  eval_metric="mlogloss", verbosity=0)
    except ImportError:
        print("NOTE: xgboost is not installed - XGBoost row skipped (pip install xgboost to add it)")
    return models


# ================================================================== evaluation
def scores(yt, yp):
    p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1, 2], average="macro", zero_division=0)
    rec = recall_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
    out = {"accuracy": accuracy_score(yt, yp), "macro_precision": p, "macro_recall": r, "macro_f1": f}
    out.update({f"recall_{l}": rec[i] for i, l in enumerate(LEVELS)})
    return out


def predictions(models, X, y, months):
    """returns {scheme: {model: (y_true, y_pred)}}"""
    res = {k: {} for k in SCHEMES}
    skf = StratifiedKFold(10, shuffle=True, random_state=cfg.RANDOM_SEED)
    train = (months <= pd.Timestamp(cfg.VAL_END)).values
    for name, model in models.items():
        print(f"  {name} ...")
        res["A"][name] = (y, cross_val_predict(clone(model), X, y, cv=skf))
        m = clone(model).fit(X[train], y[train])
        res["B"][name] = (y[~train], m.predict(X[~train]))
        pred = np.full(len(y), -1)
        for tr, te in TimeSeriesSplit(n_splits=5).split(X):
            pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
        ok = pred >= 0
        res["C"][name] = (y[ok], pred[ok])
    return res


def main():
    if not cfg.FEATURES_CSV.exists():
        raise SystemExit("Run first:  python -m src.features.build_features")
    df = pd.read_csv(cfg.FEATURES_CSV, parse_dates=["month"])
    feats = json.loads((cfg.PROCESSED_DIR / "feature_sets.json").read_text())["CLF_FEATURES"]
    X = df[feats].values
    y = df["infl_level"].map(TO_INT).values
    print(f"{len(df)} months, {len(feats)} features (no CPI columns). Class counts: "
          f"{dict(zip(LEVELS, np.bincount(y, minlength=3)))}")

    models = make_models()
    print("\nTraining and testing (nested tuning, 3 schemes):")
    res = predictions(models, X, y, df["month"])

    rows, conf_rows = [], []
    for k, label in SCHEMES.items():
        for name, (yt, yp) in res[k].items():
            rows.append({"scheme": label, "model": name, "n_test": len(yt), **scores(yt, yp)})
            cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
            for i, a in enumerate(LEVELS):
                for j, p in enumerate(LEVELS):
                    conf_rows.append({"scheme": label, "model": name, "actual": a, "predicted": p, "count": int(cm[i, j])})
    out = pd.DataFrame(rows).round(3)
    out.to_csv(cfg.TABLE_DIR / "classification_results.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(cfg.TABLE_DIR / "classification_confusion.csv", index=False)

    show = ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "recall_LOW", "recall_MEDIUM", "recall_HIGH"]
    for k, label in SCHEMES.items():
        t = out[out["scheme"] == label].sort_values("macro_f1", ascending=False)
        print(f"\n=== {label}  ({int(t['n_test'].iloc[0])} test months) ===")
        print(t[show].to_string(index=False))

    pivot = out.pivot(index="model", columns="scheme", values="macro_f1")[list(SCHEMES.values())]
    pivot = pivot.sort_values(SCHEMES["A"], ascending=False)
    print("\nMacro-F1 summary (higher is better):")
    print(pivot.to_string())

    # ---------------- figures
    print("\nFigures:")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    w = 0.26
    for i, (k, label) in enumerate(SCHEMES.items()):
        ax.bar(np.arange(len(pivot)) + (i - 1) * w, pivot[label], w, label=label)
    ax.set_xticks(range(len(pivot)))
    ax.set_xticklabels(pivot.index, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Classifier comparison (higher is better)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "13_classifier_comparison.png", dpi=150)
    plt.close(fig)
    print("  saved 13_classifier_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (k, label) in zip(axes, SCHEMES.items()):
        best = out[(out["scheme"] == label) & ~out["model"].str.contains("Majority")].sort_values("macro_f1").iloc[-1]["model"]
        yt, yp = res[k][best]
        cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels(LEVELS, fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels(LEVELS, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{label}\nbest: {best}", fontsize=9)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "14_confusion_matrices.png", dpi=150)
    plt.close(fig)
    print("  saved 14_confusion_matrices.png")

    # ---------------- which indicators matter (random forest, all rows)
    rf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=2,
                                random_state=cfg.RANDOM_SEED).fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False).round(3)
    imp.to_csv(cfg.TABLE_DIR / "rf_feature_importance.csv", header=["importance"])
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.sort_values().plot.barh(ax=ax, color="#264653")
    ax.set_title("Random Forest feature importance (which indicators separate LOW / MEDIUM / HIGH)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "15_feature_importance.png", dpi=150)
    plt.close(fig)
    print("  saved 15_feature_importance.png")
    print("\nTop 5 features by Random Forest importance:")
    print(imp.head(5).to_string())

    # ---------------- readable ID3 rules (fit on all rows, depth 3)
    pipe = Pipeline([("bin", KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")),
                     ("id3", ID3Classifier(max_depth=3))]).fit(X, y)
    rules = tree_text(pipe.named_steps["id3"].tree_, feats)
    (cfg.TABLE_DIR / "id3_rules.txt").write_text(rules, encoding="utf-8")
    print("\nID3 decision rules (depth 3, fitted on all months; features binned low/mid/high):")
    print(rules)
    print(f"Tables in {cfg.TABLE_DIR}\nTo repeat in WEKA open data/processed/features.arff (J48, NaiveBayes, IBk, SMO, "
          "RandomForest) and features_nominal.arff (Id3) with 10-fold cross-validation.")


if __name__ == "__main__":
    main()
