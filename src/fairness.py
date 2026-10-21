"""Fairness audit across sensitive groups (race, sex, native-country), mitigation, before/after comparison (requirement 10).

Uses the cleaned data and split from src/data.py:
    from src.data import load_data, get_cv, FEATURES, SENSITIVE, TARGET

Stage `audit` (development phase, Session 9). The test set is NOT used.
Candidate model: LightGBM with its tuned parameters, probabilities recalibrated
with Platt scaling (see src/evaluate.py). Predictions are out-of-fold.

Fairness definitions, fixed BEFORE looking at any result (S9: pick the definition
from the cost of the decision, then measure):
- primary: equalized odds, i.e. equal TPR and equal FPR for women and men;
- legal screen: the four-fifths rule on selection rates (S9 example of
  disparate-impact guidance);
- also reported: demographic parity, equal opportunity (TPR gap) and predictive
  parity (PPV gap).

Mitigation: ONE technique, post-processing with Fairlearn's ThresholdOptimizer
(a decision threshold per group, S9). Its main constraint is equalized odds; the
demographic-parity and equal-opportunity constraints are run as a sensitivity
analysis, to show that the definitions pull in different directions. The
ThresholdOptimizer is fitted on out-of-fold predictions and scored by a second
cross-validation, so it is never scored on the rows it was fitted on.

Run from the project root (uses the tuned parameters in src/config.BEST_PARAMS):
    python -m src.fairness audit
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")  # figures are only saved, never shown
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from fairlearn.metrics import (MetricFrame, count, false_positive_rate,
                               selection_rate, true_positive_rate)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_curve)
from sklearn.model_selection import StratifiedKFold

from src.data import FEATURES, RANDOM_STATE, TARGET, get_cv, load_data
from src.evaluate import COLORS, FIGURES_DIR, calibrate_out_of_fold
from src.pipeline import build_pipeline
from src.config import BEST_PARAMS, BEST_PARAMS_VERSION, EXPERIMENT, REPORTS_DIR, TRACKING_URI
from src.cv import best_f1_point, cross_validate_pipeline
from src.models import build_model

CANDIDATE = "LightGBM"
SMALL_GROUP = 500  # below this many people, group metrics are unstable (EDA Step 8)
N_BOOTSTRAP = 500
SHORT_NAMES = {  # axis labels of the figure
    "single threshold (best balanced accuracy)": "one threshold\n(before)",
    "equalized odds (ThresholdOptimizer)": "equalized\nodds",
    "equal opportunity (ThresholdOptimizer)": "equal\nopportunity",
    "demographic parity (ThresholdOptimizer)": "demographic\nparity",
}


class ScoreEstimator(BaseEstimator, ClassifierMixin):
    """Wraps a column of already computed probabilities as a fitted classifier,
    so ThresholdOptimizer can work on out-of-fold scores."""

    def fit(self, X=None, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        score = np.asarray(X, dtype=float).reshape(-1)
        return np.column_stack([1 - score, score])


def balanced_accuracy_threshold(y, p):
    """Single, group-blind threshold that maximises balanced accuracy (TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y, p)
    return float(thresholds[np.argmax(tpr - fpr)])


def group_frame(y, pred, groups):
    """Per-group count, base rate, selection rate, TPR, FPR and PPV."""
    frame = MetricFrame(
        metrics={"n": count, "selection_rate": selection_rate,
                 "tpr": true_positive_rate, "fpr": false_positive_rate,
                 "ppv": lambda yt, yp: precision_score(yt, yp, zero_division=0)},
        y_true=y, y_pred=pred, sensitive_features=groups)
    table = frame.by_group.copy()
    table["base_rate"] = pd.Series(y).groupby(np.asarray(groups)).mean()
    return table


def gaps(y, pred, sex):
    """Fairness gaps between women and men (women minus men; ratio = women / men)."""
    t = group_frame(y, pred, sex)
    f, m = t.loc["Female"], t.loc["Male"]
    gap = {"dp_diff": f.selection_rate - m.selection_rate,
           "tpr_diff": f.tpr - m.tpr, "fpr_diff": f.fpr - m.fpr,
           "ppv_diff": f.ppv - m.ppv,
           "four_fifths_ratio": f.selection_rate / m.selection_rate}
    gap["equalized_odds_gap"] = max(abs(gap["tpr_diff"]), abs(gap["fpr_diff"]))
    return gap, t


def performance(y, pred):
    return {"accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "f1": f1_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred)}


def mitigate_cross_fitted(y, p, sex, constraint):
    """ThresholdOptimizer (per-group thresholds), cross-fitted on out-of-fold scores."""
    decision = np.zeros(len(y), dtype=int)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fit_idx, apply_idx in folds.split(p, y):
        optimizer = ThresholdOptimizer(
            estimator=ScoreEstimator().fit(), constraints=constraint,
            objective="balanced_accuracy_score", prefit=True,
            predict_method="predict_proba")
        optimizer.fit(p[fit_idx].reshape(-1, 1), y[fit_idx],
                      sensitive_features=sex[fit_idx])
        decision[apply_idx] = optimizer.predict(
            p[apply_idx].reshape(-1, 1), sensitive_features=sex[apply_idx],
            random_state=RANDOM_STATE)
    return decision


def quick_gaps(y, pred, sex):
    """Same gaps as gaps(), with plain numpy (fast enough for a bootstrap)."""
    rates = {}
    for group in ("Female", "Male"):
        yy, pp = y[sex == group], pred[sex == group]
        rates[group] = (pp.mean(), pp[yy == 1].mean(), pp[yy == 0].mean())
    (sel_f, tpr_f, fpr_f), (sel_m, tpr_m, fpr_m) = rates["Female"], rates["Male"]
    return {"dp_diff": sel_f - sel_m, "tpr_diff": tpr_f - tpr_m,
            "fpr_diff": fpr_f - fpr_m, "four_fifths_ratio": sel_f / sel_m}


def bootstrap_gaps(y, pred, sex, n=N_BOOTSTRAP):
    """95% interval of the sex gaps (resampling people), S8: gaps need uncertainty."""
    rng = np.random.default_rng(RANDOM_STATE)
    draws = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        draws.append(quick_gaps(y[idx], pred[idx], sex[idx]))
    draws = pd.DataFrame(draws)
    return {k: (float(draws[k].quantile(0.025)), float(draws[k].quantile(0.975)))
            for k in ["tpr_diff", "fpr_diff", "dp_diff", "four_fifths_ratio"]}


def plot_groups(results, y, sex, path):
    """Selection rate, TPR, FPR and PPV for women and men under each decision rule."""
    methods = [m for m in results if m != "single threshold (best F1)"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for ax, (key, title) in zip(axes, [("selection_rate", "Selection rate"),
                                       ("tpr", "True positive rate (recall)"),
                                       ("fpr", "False positive rate"),
                                       ("ppv", "Precision (PPV)")]):
        x = np.arange(len(methods))
        for offset, group in ((-0.2, "Female"), (0.2, "Male")):
            values = [group_frame(y, results[m], sex).loc[group, key] for m in methods]
            ax.bar(x + offset, values, width=0.4, color=COLORS[group], label=group)
        ax.set_xticks(x, [SHORT_NAMES[m] for m in methods], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Rate")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Women vs men under each decision rule, {CANDIDATE} {BEST_PARAMS_VERSION} "
                 "+ Platt (out-of-fold, development set)")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_audit():
    dev_df, _ = load_data()  # the test set is left untouched
    X, y = dev_df[FEATURES], dev_df[TARGET].to_numpy()
    sex, race = dev_df["sex"].to_numpy(), dev_df["race"].to_numpy()
    best_params = json.loads(BEST_PARAMS.read_text())

    print(f"Out-of-fold predictions: {CANDIDATE} (tuned) + Platt scaling", flush=True)
    pipe = build_pipeline(build_model(CANDIDATE, best_params[CANDIDATE]))
    raw, _ = cross_validate_pipeline(pipe, X, pd.Series(y), get_cv())
    p = calibrate_out_of_fold(y, raw, "platt")

    # Decision rules: two group-blind thresholds (before) and ThresholdOptimizer (after)
    f1_threshold = best_f1_point(y, p)["threshold"]
    ba_threshold = balanced_accuracy_threshold(y, p)
    results = {
        "single threshold (best F1)": (p >= f1_threshold).astype(int),
        "single threshold (best balanced accuracy)": (p >= ba_threshold).astype(int),
        "equalized odds (ThresholdOptimizer)": mitigate_cross_fitted(y, p, sex, "equalized_odds"),
        "equal opportunity (ThresholdOptimizer)": mitigate_cross_fitted(y, p, sex, "true_positive_rate_parity"),
        "demographic parity (ThresholdOptimizer)": mitigate_cross_fitted(y, p, sex, "demographic_parity"),
    }

    rows, by_group = [], {}
    for name, pred in results.items():
        gap, table = gaps(y, pred, sex)
        rows.append({"rule": name, **performance(y, pred), **gap})
        by_group[name] = table
    summary = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    print(f"\nThresholds: best F1 = {f1_threshold:.3f}, best balanced accuracy = {ba_threshold:.3f}")
    print("\nBefore the mitigation: women vs men at the best-F1 threshold\n")
    print(by_group["single threshold (best F1)"].round(3).to_string())
    print("\nPerformance of each decision rule (all people)\n")
    print(summary[["rule", "accuracy", "balanced_accuracy", "f1", "precision", "recall"]]
          .round(3).to_string(index=False))
    print("\nFairness gaps (women minus men; four-fifths ratio = women / men)\n")
    print(summary[["rule", "dp_diff", "tpr_diff", "fpr_diff", "equalized_odds_gap",
                   "ppv_diff", "four_fifths_ratio"]].round(3).to_string(index=False))

    base_pred = results["single threshold (best F1)"]
    intervals = bootstrap_gaps(y, base_pred, sex)
    print("\n95% bootstrap intervals, best-F1 threshold\n")
    for key, (low, high) in intervals.items():
        print(f"  {key:>18}: [{low:+.3f}, {high:+.3f}]")

    race_table = group_frame(y, base_pred, race)
    race_table["small_group"] = race_table.n < SMALL_GROUP
    print("\nBy race at the best-F1 threshold (small groups are unstable)\n")
    print(race_table.round(3).to_string())

    # One file name per version; the v1 files keep their original names
    suffix = "" if BEST_PARAMS_VERSION == "v1" else f"_{BEST_PARAMS_VERSION}"
    figure_path = FIGURES_DIR / f"fairness_by_sex_dev{suffix}.png"
    details_path = REPORTS_DIR / f"fairness_details_dev{suffix}.json"
    plot_groups(results, y, sex, figure_path)
    REPORTS_DIR.mkdir(exist_ok=True)

    # fairness_dev.json keeps every version: only this version's rows are replaced
    summary_path = REPORTS_DIR / "fairness_dev.json"
    old = pd.read_json(summary_path) if summary_path.exists() else pd.DataFrame()
    if not old.empty:
        if "version" not in old:  # rows written before the version was recorded
            old.insert(0, "version", "v1")
        old = old[old["version"] != BEST_PARAMS_VERSION]
    summary.insert(0, "version", BEST_PARAMS_VERSION)
    pd.concat([old, summary], ignore_index=True).to_json(summary_path, orient="records", indent=2)

    # Per-group rates, thresholds and intervals, so the notebooks only read files
    records = lambda t: t.rename_axis("group").reset_index().to_dict("records")
    details = {"version": BEST_PARAMS_VERSION,
               "thresholds": {"best_f1": f1_threshold, "best_balanced_accuracy": ba_threshold},
               "by_sex": {name: records(table) for name, table in by_group.items()},
               "by_race_best_f1": records(race_table),
               "bootstrap_95_best_f1": intervals}
    details_path.write_text(json.dumps(details, indent=2, default=float))

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="fairness audit (development, out-of-fold)"):
        mlflow.log_params({"model": f"{CANDIDATE} + Platt", "version": BEST_PARAMS_VERSION,
                           "sensitive_feature": "sex",
                           "primary_definition": "equalized odds",
                           "legal_screen": "four-fifths rule",
                           "mitigation": "ThresholdOptimizer (post-processing)"})
        for row in rows:
            key = row["rule"].split(" (")[0].replace(" ", "_")
            mlflow.log_metrics({f"{key}_{k}": float(v) for k, v in row.items()
                                if k != "rule"})
        mlflow.log_artifact(str(figure_path), "figures")
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(details_path))
    print(f"\nFigure in {figure_path}; metrics in {summary_path} and {details_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["audit"])
    parser.parse_args()
    run_audit()
