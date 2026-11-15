"""
run_pipeline.py - runs the whole project end to end with one command.

Place at:  <project root>/run_pipeline.py     (next to the src folder)
Run:       python run_pipeline.py              everything (about 5-10 minutes)
           python run_pipeline.py --fast       skip the slow neural-net step
           python run_pipeline.py --from features   start at a step (name or number)
           python run_pipeline.py --list       show the steps

Each step is run as  python -m <module>  exactly as you ran it by hand, in this order:
raw files -> interim CSVs -> master table -> features -> warehouse -> EDA -> models.
The run stops at the first step that fails and tells you which one.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("ingest_cpi",      "src.ingest.load_cpi",           "Load 13 CPI files"),
    ("ingest_usdinr",   "src.ingest.load_usdinr",        "Load USD/INR"),
    ("ingest_crude",    "src.ingest.load_crude",         "Load crude oil prices"),
    ("ingest_wpi",      "src.ingest.load_wpi",           "Load WPI"),
    ("ingest_repo",     "src.ingest.load_repo",          "Load repo rate"),
    ("merge",           "src.preprocess.merge",          "Merge into the master monthly table"),
    ("features",        "src.features.build_features",   "Build features, labels, ARFF files"),
    ("warehouse",       "src.warehouse.load_warehouse",  "Build the SQLite data warehouse"),
    ("eda",             "src.eda.eda",                   "Exploratory analysis and figures"),
    ("kmeans",          "src.models.kmeans",             "K-Means regime discovery"),
    ("classify",        "src.models.classify",           "ID3 / Naive Bayes / kNN / SVM classifiers"),
    ("neural_net",      "src.models.neural_net",         "Neural network inflation forecast (slow)"),
    ("apriori",         "src.models.apriori_rules",      "Apriori association rules"),
]


def main():
    args = sys.argv[1:]
    if "--list" in args:
        for i, (key, mod, desc) in enumerate(STEPS, 1):
            print(f"{i:2d}. {key:14s} {desc}")
        return
    steps = list(STEPS)
    if "--fast" in args:
        steps = [s for s in steps if s[0] != "neural_net"]
    if "--from" in args:
        want = args[args.index("--from") + 1]
        keys = [s[0] for s in steps]
        if want.isdigit():
            start = STEPS[int(want) - 1][0]
        else:
            start = want
        if start not in keys:
            raise SystemExit(f"Unknown step '{want}'. Use --list to see the names.")
        steps = steps[keys.index(start):]

    t0 = time.time()
    done = []
    for i, (key, mod, desc) in enumerate(steps, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(steps)}] {desc}   (python -m {mod})\n{'=' * 70}", flush=True)
        t = time.time()
        rc = subprocess.run([sys.executable, "-m", mod], cwd=ROOT).returncode
        if rc != 0:
            print(f"\nSTOPPED: step '{key}' failed (exit code {rc}). Fix it, then resume with:\n"
                  f"    python run_pipeline.py --from {key}")
            sys.exit(rc)
        done.append((key, time.time() - t))

    print(f"\n{'=' * 70}\nAll {len(done)} steps finished in {(time.time() - t0) / 60:.1f} minutes")
    for key, sec in done:
        print(f"  {key:14s} {sec:6.0f} s")
    print("Results: data/processed/, data/warehouse.db, outputs/tables/, outputs/figures/")


if __name__ == "__main__":
    main()
