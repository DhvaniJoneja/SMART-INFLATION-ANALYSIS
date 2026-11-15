"""
kmeans.py - K-Means clustering to discover economic / inflation regimes.

Place at:  <project root>/src/models/kmeans.py
Run from the project root:  python -m src.models.kmeans          (k chosen automatically)
                            python -m src.models.kmeans 4        (force k = 4)

Reads   data/processed/features.csv, data/processed/feature_sets.json (CLUSTER_FEATURES)
Writes  outputs/tables/kmeans_k_selection.csv     inertia / silhouette for k = 2..8
        outputs/tables/kmeans_cluster_profile.csv average of every feature per cluster
        outputs/tables/cluster_assignments.csv    month -> cluster (used by Power BI)
        outputs/figures/09..12_kmeans_*.png
        and fills Fact_Macro.cluster_id in data/warehouse.db if the warehouse exists

Method
  1. Standardise the 7 features (mean 0, std 1) so no variable dominates by its units.
  2. Try k = 2..8; score with silhouette (higher = better separated), inertia (elbow)
     and Davies-Bouldin (lower = better).
  3. Choose k with the best silhouette among k = 3..5 (k = 2 usually wins on score but
     is too coarse to describe regimes; both are reported).
  4. Number clusters 0, 1, 2... from LOWEST to HIGHEST average headline inflation.
  K-Means only finds groups; the names ("food-driven" etc.) are YOUR interpretation
  of the profile table afterwards.
"""
import json
import sqlite3
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config as cfg

COLORS = ["#2a9d8f", "#e9c46a", "#e76f51", "#6a4c93", "#264653", "#8d99ae"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / name, dpi=150)
    plt.close(fig)
    print(f"  saved {name}")


def fit(X, k, seed=None):
    return KMeans(n_clusters=k, n_init=20, random_state=cfg.RANDOM_SEED if seed is None else seed).fit(X)


def select_k(X):
    rows = []
    for k in cfg.K_RANGE:
        m = fit(X, k)
        rows.append({"k": k, "inertia": round(m.inertia_, 1),
                     "silhouette": round(silhouette_score(X, m.labels_), 3),
                     "davies_bouldin": round(davies_bouldin_score(X, m.labels_), 3)})
    return pd.DataFrame(rows)


def reorder_labels(labels, infl):
    """Renumber clusters so 0 = lowest average inflation ... k-1 = highest."""
    order = pd.Series(infl).groupby(labels).mean().sort_values().index
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[l] for l in labels]), mapping


def update_warehouse(assign):
    if not cfg.WAREHOUSE_DB.exists():
        print("  (warehouse.db not found - skipped; run load_warehouse first to store clusters)")
        return
    try:
        con = sqlite3.connect(cfg.WAREHOUSE_DB)
        rows = [(int(c), int(m.year * 100 + m.month)) for m, c in zip(assign["month"], assign["cluster"])]
        con.executemany("UPDATE Fact_Macro SET cluster_id = ? WHERE date_id = ?", rows)
        con.commit()
        con.close()
        print(f"  cluster_id written to Fact_Macro for {len(rows)} months")
    except sqlite3.Error as e:
        print(f"  could not update the warehouse ({e}); close DB Browser/Power BI and rerun")


def main():
    if not cfg.FEATURES_CSV.exists():
        raise SystemExit("Run first:  python -m src.features.build_features")
    df = pd.read_csv(cfg.FEATURES_CSV, parse_dates=["month"])
    feats = json.loads((cfg.PROCESSED_DIR / "feature_sets.json").read_text())["CLUSTER_FEATURES"]
    print(f"{len(df)} months, features: {feats}")

    X = StandardScaler().fit_transform(df[feats])

    # ---------------- choose k
    sel = select_k(X)
    sel.to_csv(cfg.TABLE_DIR / "kmeans_k_selection.csv", index=False)
    print("\nChoosing k:")
    print(sel.to_string(index=False))
    best_all = int(sel.loc[sel["silhouette"].idxmax(), "k"])
    cand = sel[(sel["k"] >= 3) & (sel["k"] <= 5)]
    k = int(cand.loc[cand["silhouette"].idxmax(), "k"])
    if len(sys.argv) > 1:
        k = int(sys.argv[1])
        print(f"\nUsing k = {k} (given on the command line)")
    else:
        print(f"\nBest silhouette overall: k = {best_all};  chosen from k = 3..5: k = {k}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(sel["k"], sel["inertia"], "o-", color="#264653")
    axes[0].set(title="Elbow method", xlabel="k", ylabel="Inertia (within-cluster SSE)")
    axes[1].plot(sel["k"], sel["silhouette"], "o-", color="#e76f51")
    axes[1].axvline(k, color="grey", ls="--", lw=1)
    axes[1].set(title="Silhouette score", xlabel="k", ylabel="Silhouette")
    for a in axes:
        a.grid(alpha=0.3)
    save(fig, "09_kmeans_elbow_silhouette.png")

    # ---------------- final model and stability
    model = fit(X, k)
    labels, mapping = reorder_labels(model.labels_, df["infl_yoy"].values)
    df["cluster"] = labels
    centers = pd.DataFrame(model.cluster_centers_, columns=feats).rename(index=mapping).sort_index()

    ari = [adjusted_rand_score(model.labels_, fit(X, k, seed=s).labels_) for s in range(1, 6)]
    print(f"\nStability across 5 other random starts: adjusted Rand index = {np.mean(ari):.2f} "
          "(1.00 = identical clusters every time)")

    # ---------------- profile
    prof = df.groupby("cluster")[feats].mean().round(2)
    prof.insert(0, "months", df.groupby("cluster").size())
    prof["first_month"] = df.groupby("cluster")["month"].min().dt.strftime("%Y-%m")
    prof["last_month"] = df.groupby("cluster")["month"].max().dt.strftime("%Y-%m")
    prof.to_csv(cfg.TABLE_DIR / "kmeans_cluster_profile.csv")
    print("\nCluster profile (average in original units; cluster 0 = lowest inflation):")
    print(prof.to_string())

    print("\nWhat stands out in each cluster (standardised centre; +/- 1 is one std from average):")
    for c, row in centers.iterrows():
        top = row.reindex(row.abs().sort_values(ascending=False).index).head(3)
        print(f"  cluster {c}: " + ", ".join(f"{n} {v:+.1f}" for n, v in top.items()))

    print("\nWhich years fall in each cluster (months):")
    print(pd.crosstab(df["year"], df["cluster"]).to_string())
    print("\nClusters vs the LOW/MEDIUM/HIGH label:")
    print(pd.crosstab(df["cluster"], df["infl_level"]).reindex(columns=cfg.LEVEL_LABELS, fill_value=0).to_string())

    # ---------------- outputs
    assign = df[["month", "cluster", "infl_yoy", "infl_level"]]
    assign.to_csv(cfg.TABLE_DIR / "cluster_assignments.csv", index=False)
    update_warehouse(assign)

    print("\nFigures:")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df["month"], df["infl_yoy"], color="lightgrey", lw=1)
    for c in range(k):
        s = df[df["cluster"] == c]
        ax.scatter(s["month"], s["infl_yoy"], color=COLORS[c % len(COLORS)], s=30, label=f"Cluster {c}")
    ax.set(title=f"Economic regimes found by K-Means (k = {k})", ylabel="Headline inflation (YoY %)")
    ax.legend(fontsize=8, ncol=k, loc="upper right")
    ax.grid(alpha=0.3)
    save(fig, "10_kmeans_timeline.png")

    pca = PCA(n_components=2, random_state=cfg.RANDOM_SEED)
    P = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for c in range(k):
        m = labels == c
        ax.scatter(P[m, 0], P[m, 1], color=COLORS[c % len(COLORS)], s=35, label=f"Cluster {c}", alpha=0.85)
    ax.set(title="Clusters projected on the first two principal components",
           xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)",
           ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "11_kmeans_pca.png")

    fig, ax = plt.subplots(figsize=(8, 1.0 + 0.7 * k))
    im = ax.imshow(centers.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"Cluster {c}" for c in range(k)], fontsize=8)
    for i in range(k):
        for j in range(len(feats)):
            ax.text(j, i, f"{centers.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("Cluster centres (standardised: red = above average, blue = below)")
    fig.colorbar(im, ax=ax, shrink=0.9)
    save(fig, "12_kmeans_profile_heatmap.png")

    print(f"\nTables in {cfg.TABLE_DIR}\nNow name each cluster from the profile table "
          "(e.g. 'low inflation / stable', 'food-driven', 'cost-push shock') for your report.")


if __name__ == "__main__":
    main()
