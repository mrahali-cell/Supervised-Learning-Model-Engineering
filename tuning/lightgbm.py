"""LightGBM: diagnosis before tuning (Hadiqa).

How LightGBM learns (gradient boosting, S4): trees are built one after the
other, and each new tree is fitted to the errors (the gradients of the
log-loss) of all the trees before it. Unlike the random forest, the trees are
not averaged but ADDED, each one multiplied by the learning_rate. So:
- learning_rate x n_estimators = how far the model goes in fitting the
  training data (a small rate needs more trees);
- num_leaves, max_depth, min_child_samples = how complex each tree may be
  (LightGBM grows its trees leaf by leaf, so num_leaves is the main limit);
- subsample, colsample_bytree = rows / columns each tree sees (randomness that
  reduces variance, as in the forest);
- reg_alpha, reg_lambda = L1 / L2 penalty on the leaf values (as in S2);
- scale_pos_weight = weight of the high earners in the loss (S6).
Too many steps or too complex trees = overfitting (variance); too few = bias.

Stage `baseline`: Dummy, LightGBM defaults (6 inputs, then + engineered
                  features), tuned v1 (and v2 once it exists)
                  -> train vs validation PR-AUC, gap, ROC-AUC, Brier.
Stage `curves`  : one validation curve per course hyperparameter (S5); for
                  learning_rate, one curve of n_estimators per learning rate.

The class-weighted LightGBM ("balanced") is left out: it gave the same ranking
and worse probabilities in tuning/baseline.py (F16).

Development set only, 5-fold CV; the test set is NOT used.
Results: tuning/results/lightgbm/ (read by notebooks/04_lightgbm.ipynb).

Then: edit the LightGBM search space in suggest_params() in tuning/tune.py and run
    python3 -m tuning.tune --version v2 --model lgbm

Run from the project root (never with the VS Code "Run" button):
    python3 -m tuning.lightgbm baseline
    python3 -m tuning.lightgbm curves                       # all curves
    python3 -m tuning.lightgbm curves --param num_leaves    # one curve
"""

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")  # figures are only saved, never shown
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate

from src.config import TUNING_RESULTS_DIR
from src.cv import SCORING
from src.data import FEATURES, RANDOM_STATE, TARGET, get_cv, load_data
from src.models import build_model, get_models
from src.pipeline import build_pipeline

OUT = TUNING_RESULTS_DIR / "lightgbm"

# Validation curves: ONE setting changes, the others stay at the LightGBM
# defaults (100 trees, learning_rate 0.1, 31 leaves, no depth limit, 20 people
# per leaf, all rows and columns, no penalty, no class weight).
# subsample_freq=1 as in src/models.py: without it, subsample is ignored.
BASE = dict(subsample_freq=1, random_state=RANDOM_STATE, verbose=-1)
CURVES = {
    # learning_rate x n_estimators: one curve of n_estimators per learning rate
    "learning_rate": [0.01, 0.03, 0.1, 0.3],
    "num_leaves": [4, 8, 16, 31, 63, 127, 255],
    "max_depth": [2, 3, 4, 6, 8, 12, -1],  # -1 = no limit (default)
    "min_child_samples": [5, 10, 20, 50, 100, 200, 500],
    "subsample": [0.3, 0.5, 0.7, 0.9, 1.0],
    "colsample_bytree": [0.2, 0.4, 0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.01, 0.1, 1, 10, 100],
    "reg_lambda": [0, 0.01, 0.1, 1, 10, 100],
    "scale_pos_weight": [1, 1.5, 2, 3.2, 5],  # 3.2 = N_negative / N_positive (S6); watch the Brier
}
N_ESTIMATORS = [25, 50, 100, 200, 400, 800, 1600]


def load_best(version):
    """Tuned LightGBM parameters of a version, or None if that version has none yet."""
    path = TUNING_RESULTS_DIR / version / "best_params.json"
    return json.loads(path.read_text()).get("LightGBM") if path.exists() else None


def cv_scores(pipe, X, y, cv):
    """src.cv.train_val_scores, plus the fold std of every validation metric."""
    start = time.time()
    s = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, return_train_score=True)
    return {
        "train PR-AUC": s["train_pr_auc"].mean(),
        "val PR-AUC": s["test_pr_auc"].mean(),
        "val PR-AUC std": s["test_pr_auc"].std(),
        "gap PR-AUC": s["train_pr_auc"].mean() - s["test_pr_auc"].mean(),
        "train ROC-AUC": s["train_roc_auc"].mean(),
        "val ROC-AUC": s["test_roc_auc"].mean(),
        "val ROC-AUC std": s["test_roc_auc"].std(),
        "train Brier": -s["train_brier"].mean(),
        "val Brier": -s["test_brier"].mean(),
        "val Brier std": s["test_brier"].std(),
        "seconds": time.time() - start,
    }


def show(label, r):
    print(f"{label:<52} train {r['train PR-AUC']:.3f} | val {r['val PR-AUC']:.3f} "
          f"± {r['val PR-AUC std']:.3f} | gap {r['gap PR-AUC']:.3f} | "
          f"ROC val {r['val ROC-AUC']:.3f} | Brier {r['val Brier']:.3f} | "
          f"{r['seconds']:.0f}s", flush=True)


def run_baseline(X, y, cv):
    v1 = load_best("v1")
    experiments = {  # name: (model, engineered features?)
        "Dummy (predicts the base rate)": (DummyClassifier(strategy="prior"), False),
        "LightGBM defaults, 6 inputs only": (get_models()["LightGBM"], False),
        "LightGBM defaults + engineered features": (get_models()["LightGBM"], True),
        "LightGBM tuned v1": (build_model("LightGBM", v1), True),
    }
    v2 = load_best("v2")
    if v2:  # add tuning v2 to the comparison once it exists
        experiments["LightGBM tuned v2"] = (build_model("LightGBM", v2), True)
    rows = []
    for name, (model, engineered) in experiments.items():
        r = cv_scores(build_pipeline(model, engineered=engineered), X, y, cv)
        show(name, r)
        rows.append({"experiment": name, **r})
    result = pd.DataFrame(rows).round(4)
    result.to_csv(OUT / "baseline.csv", index=False)
    print(f"\nSaved {OUT / 'baseline.csv'}")
    return result


def plot_curve(result, param, path):
    """Train vs validation PR-AUC (left), validation Brier (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    if param == "learning_rate":  # one line per learning rate, x = number of trees
        for i, (lr, g) in enumerate(result.groupby("learning_rate")):
            color = f"C{i}"
            ax1.plot(g["n_estimators"], g["train PR-AUC"], "--", color=color, alpha=0.6)
            ax1.errorbar(g["n_estimators"], g["val PR-AUC"], yerr=g["val PR-AUC std"],
                         fmt="o-", capsize=3, color=color, label=f"lr {lr}")
            ax2.plot(g["n_estimators"], g["val Brier"], "o-", color=color, label=f"lr {lr}")
        ax1.set_title("PR-AUC vs trees (dashed = train, solid = validation)")
        for ax in (ax1, ax2):
            ax.set_xscale("log")
            ax.set_xlabel("n_estimators (log scale)")
            ax.legend(fontsize=8)
    else:
        x = range(len(result))
        ax1.plot(x, result["train PR-AUC"], "o-", label="train")
        ax1.errorbar(x, result["val PR-AUC"], yerr=result["val PR-AUC std"],
                     fmt="o-", capsize=3, label="validation")
        ax1.set_title(f"PR-AUC vs {param}")
        ax1.legend()
        for ax in (ax1, ax2):
            ax.set_xticks(list(x), result["value"])
            ax.set_xlabel(param)
        ax2.plot(x, result["val Brier"], "o-", color="tab:red")
    ax2.set_title("Validation Brier (lower = better probabilities)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_curve(param, X, y, cv):
    rows = []
    print(f"\nValidation curve: {param}", flush=True)
    if param == "learning_rate":
        grid = [{"learning_rate": lr, "n_estimators": n}
                for lr in CURVES[param] for n in N_ESTIMATORS]
    else:
        grid = [{param: value} for value in CURVES[param]]
    for setting in grid:
        model = LGBMClassifier(**{**BASE, **setting})
        r = cv_scores(build_pipeline(model), X, y, cv)
        label = ", ".join(f"{k} = {v}" for k, v in setting.items())
        show(f"  {label}", r)
        value = label if param == "learning_rate" else str(setting[param])
        rows.append({"param": param, "value": value,
                     "learning_rate": setting.get("learning_rate", 0.1),
                     "n_estimators": setting.get("n_estimators", 100), **r})
    result = pd.DataFrame(rows).round(4)
    (OUT / "curves").mkdir(exist_ok=True)
    plot_curve(result, param, OUT / "curves" / f"{param}.png")
    return result


def run_curves(params, X, y, cv):
    path = OUT / "curves.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    for param in params:
        result = run_curve(param, X, y, cv)
        if not old.empty:
            old = old[old["param"] != param]  # replace only this curve
        old = pd.concat([old, result], ignore_index=True)
        old.to_csv(path, index=False)
    print(f"\nSaved {path} and the figures in {OUT / 'curves'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM diagnosis")
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
