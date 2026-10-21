"""Final evaluation on the test set: metrics, calibration, SHAP explanations (requirements 8, 9).

Uses the cleaned data and split from src/data.py:
    from src.data import load_data, get_cv, FEATURES, SENSITIVE, TARGET

Stage `calibration` (development phase, Session 7): can we trust the predicted
probabilities of the three tuned models?
- Predictions are out-of-fold: each person is scored by a model that never saw
  them, so the diagnosis is honest without touching the test set.
- Diagnose: reliability diagram (equal-frequency bins), Brier score, ECE, and the
  same checks by sex (S7: "check calibration by segment").
- Recalibrate: Platt scaling and isotonic regression, each evaluated with a second
  cross-validation on the out-of-fold predictions, so a calibrator is never scored
  on the rows it was fitted on (S7: calibrate on predictions unseen by the base model).

Run from the project root (uses the tuned parameters in src/config.BEST_PARAMS):
    python -m src.evaluate calibration
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")  # figures are only saved, never shown
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data import FEATURES, RANDOM_STATE, TARGET, get_cv, load_data
from src.pipeline import build_pipeline
from src.config import BEST_PARAMS, EXPERIMENT, REPORTS_DIR, TRACKING_URI
from src.cv import cross_validate_pipeline
from src.models import TUNED_MODELS, build_model

N_BINS = 10
FIGURES_DIR = REPORTS_DIR / "figures"
VARIANTS = ["raw", "platt", "isotonic"]
COLORS = {"raw": "#1f77b4", "platt": "#d95f02", "isotonic": "#1b9e77",
          "Female": "#7570b3", "Male": "#e6ab02"}


def probability_metrics(y, p, n_bins=N_BINS):
    """Brier score, ECE (equal-frequency bins, S7 formula) and ranking metrics.

    Also returns the mean prediction vs the observed rate overall and in the
    highest-probability bin, which show the direction of the error.
    """
    bins = pd.qcut(p, q=n_bins, labels=False, duplicates="drop")
    table = (pd.DataFrame({"y": y, "p": p, "bin": bins}).groupby("bin")
             .agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean")))
    ece = float((table.n / len(y) * (table.observed - table.predicted).abs()).sum())
    top = table.iloc[-1]
    return {"brier": brier_score_loss(y, p), "ece": ece,
            "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "mean_predicted": float(p.mean()), "observed_rate": float(y.mean()),
            "top_bin_predicted": float(top.predicted), "top_bin_observed": float(top.observed)}


def calibrate_out_of_fold(y, p, method):
    """Platt / isotonic calibrated probabilities, scored by a second CV.

    Each value is produced by a calibrator fitted on the OTHER rows' out-of-fold
    predictions, so the calibrated numbers are not fitted on themselves.
    """
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    score = np.log(clipped / (1 - clipped))  # Platt works on the raw score (logit)
    calibrated = np.zeros(len(p))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fit_idx, apply_idx in folds.split(p, y):
        if method == "platt":  # sigma(A * s + B): two parameters
            model = LogisticRegression(C=1e6).fit(score[fit_idx].reshape(-1, 1), y[fit_idx])
            calibrated[apply_idx] = model.predict_proba(score[apply_idx].reshape(-1, 1))[:, 1]
        else:  # isotonic: free-form non-decreasing step function
            model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            calibrated[apply_idx] = model.fit(p[fit_idx], y[fit_idx]).predict(p[apply_idx])
    return calibrated


def plot_curve(ax, y, p, label, color):
    observed, predicted = calibration_curve(y, p, n_bins=N_BINS, strategy="quantile")
    ax.plot(predicted, observed, marker="o", ms=4, lw=1.6, color=color, label=label)


def style_reliability(ax, title, show_ylabel):
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#888888", label="perfect")
    ax.set(title=title, xlabel="Mean predicted probability", xlim=(0, 1), ylim=(0, 1))
    if show_ylabel:
        ax.set_ylabel("Observed frequency")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")


def save_figures(y, sex, preds):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    names = list(preds)

    # A. Diagnosis: reliability curve + histogram of the predictions (S7: show the mass)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), height_ratios=[3, 1.3], sharex="col")
    for i, name in enumerate(names):
        plot_curve(axes[0, i], y, preds[name]["raw"], "model", COLORS["raw"])
        style_reliability(axes[0, i], name, i == 0)
        axes[1, i].hist(preds[name]["raw"], bins=30, color=COLORS["raw"], alpha=0.8)
        axes[1, i].set(xlabel="Predicted probability", xlim=(0, 1))
        axes[1, 0].set_ylabel("People")
    fig.suptitle("Reliability of the tuned models (out-of-fold, development set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_diagnosis.png", dpi=150)
    plt.close(fig)

    # B. Before / after recalibration
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for i, name in enumerate(names):
        for variant in VARIANTS:
            plot_curve(axes[i], y, preds[name][variant], variant, COLORS[variant])
        style_reliability(axes[i], name, i == 0)
    fig.suptitle("Before (raw) and after Platt / isotonic recalibration")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_before_after.png", dpi=150)
    plt.close(fig)

    # C. By sex (raw probabilities)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for i, name in enumerate(names):
        for group in ("Female", "Male"):
            mask = sex == group
            plot_curve(axes[i], y[mask], preds[name]["raw"][mask], group, COLORS[group])
        style_reliability(axes[i], name, i == 0)
    fig.suptitle("Calibration by sex (raw probabilities)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_by_sex.png", dpi=150)
    plt.close(fig)

    # D. Candidate model only: raw (baseline) vs Platt vs isotonic, same layout as A
    titles = {"raw": "raw (no recalibration)", "platt": "Platt scaling", "isotonic": "isotonic"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), height_ratios=[3, 1.3], sharex="col")
    for i, variant in enumerate(VARIANTS):
        p = preds["LightGBM"][variant]
        m = probability_metrics(y, p)
        if variant != "raw":  # the raw curve, faint, as the reference
            plot_curve(axes[0, i], y, preds["LightGBM"]["raw"], "raw", "#bbbbbb")
        plot_curve(axes[0, i], y, p, variant, COLORS[variant])
        style_reliability(axes[0, i], f"{titles[variant]}\nECE {m['ece']:.4f} | "
                          f"Brier {m['brier']:.4f} | PR-AUC {m['pr_auc']:.4f}", i == 0)
        axes[1, i].hist(p, bins=30, color=COLORS[variant], alpha=0.8)
        axes[1, i].set(xlabel="Predicted probability", xlim=(0, 1))
    axes[1, 0].set_ylabel("People")
    fig.suptitle("LightGBM (tuned): before and after recalibration (out-of-fold, development set)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "calibration_lightgbm_comparison.png", dpi=150)
    plt.close(fig)


def run_calibration():
    dev_df, _ = load_data()  # the test set is left untouched
    X, y = dev_df[FEATURES], dev_df[TARGET].to_numpy()
    sex = dev_df["sex"].to_numpy()
    best_params = json.loads(BEST_PARAMS.read_text())
    cv = get_cv()

    preds, rows = {}, []
    for name in TUNED_MODELS:
        print(f"Out-of-fold predictions: {name}", flush=True)
        pipe = build_pipeline(build_model(name, best_params[name]))
        raw, _ = cross_validate_pipeline(pipe, X, pd.Series(y), cv)
        preds[name] = {"raw": raw,
                       "platt": calibrate_out_of_fold(y, raw, "platt"),
                       "isotonic": calibrate_out_of_fold(y, raw, "isotonic")}
        for variant in VARIANTS:
            p = preds[name][variant]
            rows.append({"model": name, "variant": variant, "group": "all",
                         **probability_metrics(y, p)})
            for group in ("Female", "Male"):
                mask = sex == group
                rows.append({"model": name, "variant": variant, "group": group,
                             **probability_metrics(y[mask], p[mask])})

    save_figures(y, sex, preds)
    result = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    result.to_json(REPORTS_DIR / "calibration_dev.json", orient="records", indent=2)

    pd.set_option("display.width", 200)
    cols = ["model", "variant", "brier", "ece", "pr_auc", "roc_auc",
            "top_bin_predicted", "top_bin_observed"]
    print("\nCalibration, all people (out-of-fold, development set)\n")
    print(result[result.group == "all"][cols].round(4).to_string(index=False))
    cols = ["model", "variant", "group", "brier", "ece", "mean_predicted", "observed_rate"]
    print("\nCalibration by sex\n")
    print(result[result.group != "all"][cols].round(4).to_string(index=False))

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="calibration (development, out-of-fold)"):
        mlflow.log_params({"bins": f"{N_BINS} equal-frequency", "calibrators": "platt, isotonic",
                           "calibrator_evaluation": "second 5-fold CV on out-of-fold predictions"})
        for r in rows:
            if r["group"] == "all":
                key = f"{r['model']}_{r['variant']}".replace(" ", "_")
                mlflow.log_metrics({f"{key}_brier": r["brier"], f"{key}_ece": r["ece"]})
        mlflow.log_artifacts(str(FIGURES_DIR), artifact_path="figures")
        mlflow.log_artifact(str(REPORTS_DIR / "calibration_lightgbm_comparison.png"))
        mlflow.log_artifact(str(REPORTS_DIR / "calibration_dev.json"))
    print(f"\nFigures in {FIGURES_DIR}; metrics in {REPORTS_DIR / 'calibration_dev.json'}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["calibration"])
    args = parser.parse_args()
    run_calibration()
