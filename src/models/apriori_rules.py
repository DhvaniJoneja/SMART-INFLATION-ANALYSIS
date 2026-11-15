"""
apriori_rules.py - association rule mining with the Apriori algorithm (pure Python,
written from scratch, no extra packages needed).

Place at:  <project root>/src/models/apriori_rules.py
Run from the project root:  python -m src.models.apriori_rules

Reads   data/processed/features.csv
Writes  outputs/tables/apriori_rules_all.csv        rules among the 8 indicators
        outputs/tables/apriori_rules_to_level.csv   rules from outside drivers -> LOW/MEDIUM/HIGH
        outputs/figures/19_apriori_scatter.png

Idea: each month is a "basket" of items such as  usdinr_yoy=High, repo=Low.
Every numeric column is cut into 3 equal-sized groups (Low / Mid / High terciles).
  support    = share of months where the whole rule holds
  confidence = of the months with the left side, share that also have the right side
  lift       = confidence / how common the right side is anyway (>1 = real association)
Apriori principle: if an item set is rare, every bigger set containing it is rare too,
so we grow item sets one item at a time and drop the rare ones early.
Rules show co-occurrence, NOT cause and effect.
"""
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config as cfg

ALL_COLS = ["infl_yoy", "cfpi_yoy", "fuel_yoy", "usdinr_yoy",
            "brent_inr_yoy", "wpi_all_yoy", "wpi_fuel_yoy", "repo"]
DRIVERS = ["usdinr_yoy", "brent_inr_yoy", "wpi_all_yoy", "wpi_fuel_yoy", "repo"]


def tercile(s):
    """Low / Mid / High by terciles. rank() first so ties (e.g. repo) still split evenly."""
    return pd.qcut(s.rank(method="first"), 3, labels=["Low", "Mid", "High"])


def baskets(df, cols, with_level=False):
    d = df.dropna(subset=cols)
    items = pd.DataFrame({c: c + "=" + tercile(d[c]).astype(str) for c in cols})
    if with_level:
        items["level"] = "level=" + d["infl_level"].astype(str)
    return [frozenset(r) for r in items.values.tolist()]


def apriori(trans, min_support, max_len=4):
    n = len(trans)
    support = {}
    # 1-item sets
    counts = {}
    for t in trans:
        for i in t:
            counts[frozenset([i])] = counts.get(frozenset([i]), 0) + 1
    level = {s: c / n for s, c in counts.items() if c / n >= min_support}
    support.update(level)
    k = 2
    while level and k <= max_len:
        prev = list(level)
        cands = set()
        for a, b in combinations(prev, 2):          # join step
            u = a | b
            if len(u) == k and all(frozenset(sub) in level for sub in combinations(u, k - 1)):  # prune step
                cands.add(u)
        counts = {c: 0 for c in cands}
        for t in trans:
            for c in cands:
                if c <= t:
                    counts[c] += 1
        level = {s: v / n for s, v in counts.items() if v / n >= min_support}
        support.update(level)
        k += 1
    return support


def make_rules(support, min_conf, min_lift, rhs_filter=None):
    rows = []
    for itemset, sup in support.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for rhs in combinations(itemset, r):
                rhs = frozenset(rhs)
                lhs = itemset - rhs
                if len(rhs) != 1:                        # keep single-item right sides
                    continue
                if rhs_filter and not rhs_filter(rhs):
                    continue
                conf = sup / support[lhs]
                lift = conf / support[rhs]
                if conf >= min_conf and lift >= min_lift:
                    rows.append({"lhs": " AND ".join(sorted(lhs)), "rhs": next(iter(rhs)),
                                 "support": round(sup, 3), "confidence": round(conf, 3),
                                 "lift": round(lift, 2), "count": round(sup * N_CURRENT)})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    return out


def main():
    global N_CURRENT
    if not cfg.FEATURES_CSV.exists():
        raise SystemExit("Run first:  python -m src.features.build_features")
    df = pd.read_csv(cfg.FEATURES_CSV, parse_dates=["month"])

    # ---------------- 1. rules among all indicators
    tr = baskets(df, ALL_COLS)
    N_CURRENT = len(tr)
    sup = apriori(tr, min_support=0.10, max_len=4)
    print(f"{len(tr)} months (baskets); {sum(len(s) == 1 for s in sup)} frequent items, "
          f"{len(sup)} frequent item sets in total (support >= 0.10)")
    rules = make_rules(sup, min_conf=0.60, min_lift=1.2)
    rules.to_csv(cfg.TABLE_DIR / "apriori_rules_all.csv", index=False)
    print(f"\n{len(rules)} rules with confidence >= 0.60 and lift >= 1.2. Top 10 by lift:")
    print(rules.head(10).to_string(index=False))

    # ---------------- 2. outside drivers -> inflation level
    tr2 = baskets(df, DRIVERS, with_level=True)
    N_CURRENT = len(tr2)
    sup2 = apriori(tr2, min_support=0.05, max_len=4)
    rules2 = make_rules(sup2, min_conf=0.55, min_lift=1.2,
                        rhs_filter=lambda r: next(iter(r)).startswith("level="))
    rules2.to_csv(cfg.TABLE_DIR / "apriori_rules_to_level.csv", index=False)
    print(f"\n{len(rules2)} rules from outside drivers to the inflation level "
          "(support >= 0.05, confidence >= 0.55, lift >= 1.2)")
    for lv in ("level=HIGH", "level=LOW", "level=MEDIUM"):
        s = rules2[rules2["rhs"] == lv] if len(rules2) else rules2
        print(f"\nTop rules for {lv}:")
        print(s.head(5).to_string(index=False) if len(s) else "  (none above the thresholds)")

    # ---------------- figure
    if len(rules):
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        sc = ax.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis", s=30, alpha=0.8)
        fig.colorbar(sc, label="lift")
        ax.set(xlabel="Support", ylabel="Confidence", title="Association rules among the indicators")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(cfg.FIG_DIR / "19_apriori_scatter.png", dpi=150)
        plt.close(fig)
        print("\nsaved 19_apriori_scatter.png")
    print(f"Tables in {cfg.TABLE_DIR}\nReminder: rules show co-occurrence, not cause and effect.")


if __name__ == "__main__":
    main()
