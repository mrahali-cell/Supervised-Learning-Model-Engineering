"""Random forest: diagnosis before tuning (Manel).

How a random forest learns: each tree splits the data to reduce impurity
(Gini or entropy), not a global log-loss. A fully grown tree keeps splitting
until its leaves are pure, so it memorises the training data. The forest
averages many such trees, each on a bootstrap sample and a random subset of
features, which reduces that variance. Most RF hyperparameters therefore
control how far each tree may grow (max_depth, min_samples_leaf,
min_samples_split) or how different the trees are (max_features, max_samples).

Stage `baseline`: Dummy, one tree, three forests -> where do we start?
Stage `curves`  : one validation curve per course hyperparameter (S5):
                  train vs validation PR-AUC while ONE value changes.

Development set only, 5-fold CV; the test set is NOT used.
Results: tuning/results/random_forest/ (read by notebooks/03_random_forest.ipynb).

Run from the project root:
    python -m tuning.random_forest baseline
    python -m tuning.random_forest curves                       # all curves
    python -m tuning.random_forest curves --param max_depth     # one curve
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")  # figures are only saved, never shown
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.config import TUNING_RESULTS_DIR
from src.cv import train_val_scores
from src.data import FEATURES, RANDOM_STATE, TARGET, get_cv, load_data
from src.models import build_model
from src.pipeline import build_pipeline

OUT = TUNING_RESULTS_DIR / "random_forest"

# Validation curves: every RF hyperparameter seen in the course, one at a time.
# The other parameters stay at BASE (sklearn defaults, 200 trees).
BASE = dict(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
CURVES = {
    "min_samples_leaf": [1, 2, 5, 10, 20, 50, 100],
    "max_depth": [4, 6, 8, 12, 16, 20, None],
    "min_samples_split": [2, 10, 50, 100, 200],
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.8, None],
    "max_samples": [0.3, 0.5, 0.7, None],
    "criterion": ["gini", "entropy"],
    "n_estimators": [25, 50, 100, 200, 400],
    "class_weight": [None, "balanced", "balanced_subsample"],  # watch the Brier score
}


def show(label, r):
    print(f"{label:<48} train {r['train PR-AUC']:.3f} | val {r['val PR-AUC']:.3f} "
          f"± {r['val PR-AUC std']:.3f} | gap {r['gap PR-AUC']:.3f} | "
          f"ROC val {r['val ROC-AUC']:.3f} | Brier {r['val Brier']:.3f} | "
          f"{r['seconds']:.0f}s", flush=True)


def run_baseline(X, y, cv):
    v1_path = TUNING_RESULTS_DIR / "v1" / "best_params.json"
    best_v1 = json.loads(v1_path.read_text())["Random Forest"]
    experiments = {
        "Dummy (predicts the base rate)": DummyClassifier(strategy="prior"),
        "One decision tree, fully grown": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RF sklearn defaults (100 trees, fully grown)": RandomForestClassifier(
            n_jobs=-1, random_state=RANDOM_STATE),
        "RF baseline (300 trees, min_samples_leaf 5)": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE),
        "RF tuned v1": build_model("Random Forest", best_v1),
    }
    rows = []
    for name, model in experiments.items():
        r = train_val_scores(build_pipeline(model), X, y, cv)
        show(name, r)
        rows.append({"experiment": name, **r})
    result = pd.DataFrame(rows).round(4)
    result.to_csv(OUT / "baseline.csv", index=False)
    print(f"\nSaved {OUT / 'baseline.csv'}")
    return result


def run_curve(param, values, X, y, cv):
    rows = []
    print(f"\nValidation curve: {param}", flush=True)
    for value in values:
        model = RandomForestClassifier(**{**BASE, param: value})
        r = train_val_scores(build_pipeline(model), X, y, cv)
        show(f"  {param} = {value}", r)
        rows.append({"param": param, "value": str(value), **r})
    result = pd.DataFrame(rows).round(4)

    # Figure: train vs validation PR-AUC (left), validation Brier (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    x = range(len(result))
    ax1.plot(x, result["train PR-AUC"], "o-", label="train")
    ax1.errorbar(x, result["val PR-AUC"], yerr=result["val PR-AUC std"],
                 fmt="o-", capsize=3, label="validation")
    ax1.set_title(f"PR-AUC vs {param}")
    ax1.legend()
    ax2.plot(x, result["val Brier"], "o-", color="tab:red")
    ax2.set_title("Validation Brier (lower = better probabilities)")
    for ax in (ax1, ax2):
        ax.set_xticks(list(x), result["value"])
        ax.set_xlabel(param)
    fig.tight_layout()
    (OUT / "curves").mkdir(exist_ok=True)
    fig.savefig(OUT / "curves" / f"{param}.png", dpi=120)
    plt.close(fig)
    return result


def run_curves(params, X, y, cv):
    path = OUT / "curves.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    for param in params:
        result = run_curve(param, CURVES[param], X, y, cv)
        if not old.empty:
            old = old[old["param"] != param]  # replace only this curve
        old = pd.concat([old, result], ignore_index=True)
        old.to_csv(path, index=False)
    print(f"\nSaved {path} and the figures in {OUT / 'curves'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random forest diagnosis")
    parser.add_argument("stage", choices=["baseline", "curves"])
    parser.add_argument("--param", choices=list(CURVES), action="append",
                        help="curves stage: one hyperparameter (repeatable); default all")
    args = parser.parse_args()

    dev_df, _ = load_data()  # the test set is left untouched
    X, y = dev_df[FEATURES], dev_df[TARGET]
    cv = get_cv()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "baseline":
        run_baseline(X, y, cv)
    else:
        run_curves(args.param or list(CURVES), X, y, cv)
