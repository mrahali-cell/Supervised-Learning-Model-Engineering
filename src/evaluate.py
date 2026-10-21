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

Stage `shap` (development phase, Session 8): which inputs drive the predictions
of the candidate model (LightGBM, tuned), overall and for women vs men?
- Out-of-fold, like the calibration: in each of the 5 folds the Pipeline is
  fitted on the other folds and TreeSHAP explains the held-out people. The test
  set is not used.
- SHAP values are in log-odds (LightGBM's raw score) and add up exactly to it:
  score = base value + sum of the SHAP values of the person's columns. So the
  columns made from one input (its one-hot categories, its engineered columns)
  can be added back together, and the mean score gap between women and men
  splits exactly into one part per input (the proxy question of F30-F31).
- Platt scaling (F41) is a monotone map applied after the score, so it changes
  neither the ranking nor these explanations.

Stage `test` (step 11, run ONCE): the final evaluation on the locked test set.
- Every choice is fixed on the development set before the test set is scored:
  the tuned parameters, the Platt calibrator (fitted on the development
  out-of-fold predictions), the decision thresholds (best F1 and best balanced
  accuracy, picked on the same predictions) and the ThresholdOptimizer
  (fitted on them). Each model is refitted on the whole development set.
- Reported: ranking and probability metrics of Dummy, logistic regression,
  random forest and LightGBM with bootstrap intervals, and paired differences;
  LightGBM calibration (raw and Platt), overall and by sex; the fairness
  before / after (same rules as src/fairness.py); the SHAP split of the score
  gap between women and men.
- Refuses to run if reports/test_results.json exists (use --force to redo);
  --dry-run runs the same code on a held-out 20% of the development set, to
  check the code without opening the test set.

Run from the project root (uses the tuned parameters in src/config.BEST_PARAMS):
    python -m src.evaluate calibration
    python -m src.evaluate shap
    python -m src.evaluate test --dry-run --out <folder>   # no test set
    python -m src.evaluate test                             # once
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
from src.cv import best_f1_point, cross_validate_pipeline
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
    if np.ptp(p) == 0:  # a constant prediction (the Dummy) is one single bin
        bins = np.zeros(len(p), dtype=int)
    else:
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


CANDIDATE = "LightGBM"
# Each column after preprocessing belongs to one of the 6 model inputs
# (engineered columns go back to the input they are computed from, F12)
SOURCE_INPUT = {"age_squared": "age", "has_degree": "education-num",
                "part_time": "hours-per-week", "overtime": "hours-per-week",
                "is_married": "marital-status"}


def column_input(column):
    """Model input a preprocessed column comes from (num__age -> age,
    cat__occupation_Sales -> occupation, bin__is_married -> marital-status)."""
    kind, name = column.split("__", 1)
    if kind == "cat":
        return next(c for c in ("workclass", "marital-status", "occupation")
                    if name.startswith(c + "_"))
    return SOURCE_INPUT.get(name, name)


def oof_shap(X, y, params, cv):
    """Out-of-fold TreeSHAP values (log-odds), the preprocessed columns and the base value."""
    import shap

    shap_values = np.zeros((len(X), 0))
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        pipe = build_pipeline(build_model(CANDIDATE, params)).fit(X.iloc[train_idx], y.iloc[train_idx])
        prep = pipe[:-1]
        Xt = prep.transform(X.iloc[val_idx])
        explainer = shap.TreeExplainer(pipe[-1])
        values = np.asarray(explainer.shap_values(Xt))
        if fold == 0:
            columns = list(pipe.named_steps["preprocess"].get_feature_names_out())
            shap_values = np.zeros((len(X), len(columns)))
            prepared = np.zeros((len(X), len(columns)))
            base = np.zeros(len(X))
        shap_values[val_idx], prepared[val_idx] = values, Xt
        base[val_idx] = float(np.ravel(explainer.expected_value)[-1])
        print(f"  fold {fold + 1}: {len(val_idx)} people explained", flush=True)
    return shap_values, prepared, base, columns


def run_shap():
    import shap

    dev_df, _ = load_data()  # the test set is left untouched
    X, y = dev_df[FEATURES], dev_df[TARGET]
    sex = dev_df["sex"].to_numpy()
    params = json.loads(BEST_PARAMS.read_text())[CANDIDATE]
    print(f"Out-of-fold TreeSHAP: {CANDIDATE} (tuned)", flush=True)
    values, prepared, base, columns = oof_shap(X, y, params, get_cv())
    names = [c.split("__", 1)[1] for c in columns]
    inputs = [column_input(c) for c in columns]
    score = base + values.sum(axis=1)  # = LightGBM's raw score (log-odds), exactly

    # Per input: add its columns back together (exact, SHAP values are additive)
    by_input = pd.DataFrame(values, columns=names).T.groupby(inputs).sum().T
    women, men = sex == "Female", sex == "Male"
    input_table = pd.DataFrame({
        "mean_abs": by_input.abs().mean(),
        "mean_abs_women": by_input[women].abs().mean(),
        "mean_abs_men": by_input[men].abs().mean(),
        "mean_women": by_input[women].mean(),
        "mean_men": by_input[men].mean(),
    })
    input_table["gap_women_minus_men"] = input_table.mean_women - input_table.mean_men
    input_table = input_table.sort_values("mean_abs", ascending=False)
    column_table = pd.DataFrame({"column": names, "input": inputs,
                                 "mean_abs": np.abs(values).mean(axis=0),
                                 "mean": values.mean(axis=0)}
                                ).sort_values("mean_abs", ascending=False)
    gap = {"mean_score_women": float(score[women].mean()), "mean_score_men": float(score[men].mean()),
           "base_value_gap": float(base[women].mean() - base[men].mean())}
    gap["score_gap"] = gap["mean_score_women"] - gap["mean_score_men"]
    gap["sum_of_input_gaps"] = float(input_table.gap_women_minus_men.sum())

    print("\nMean |SHAP| per model input (log-odds), all people and by sex\n")
    print(input_table.round(4).to_string())
    print("\nTop 15 columns\n")
    print(column_table.head(15).round(4).to_string(index=False))
    print(f"\nMean score (log-odds): women {gap['mean_score_women']:.3f}, men "
          f"{gap['mean_score_men']:.3f}, gap {gap['score_gap']:.3f} = inputs "
          f"{gap['sum_of_input_gaps']:.3f} + base value {gap['base_value_gap']:.4f}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    order = list(input_table.index)

    # A. Global importance: per input (left), per column (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), width_ratios=[1, 1.4])
    ax1.barh(order[::-1], input_table.mean_abs[::-1], color=COLORS["raw"])
    ax1.set(title="Per model input (its columns added together)", xlabel="Mean |SHAP| (log-odds)")
    top = column_table.head(15)[::-1]
    ax2.barh(top.column, top.mean_abs, color=COLORS["raw"])
    ax2.set(title="Top 15 columns after preprocessing", xlabel="Mean |SHAP| (log-odds)")
    for ax in (ax1, ax2):
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(f"What drives {CANDIDATE} (tuned): global SHAP importance (out-of-fold, development set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_importance_dev.png", dpi=150)
    plt.close(fig)

    # B. Beeswarm on the columns (a random sample of people, for readability)
    sample = np.random.default_rng(RANDOM_STATE).choice(len(X), size=5000, replace=False)
    shap.summary_plot(values[sample], prepared[sample], feature_names=names,
                      max_display=15, show=False, plot_size=(9, 6.5))
    fig = plt.gcf()
    fig.suptitle("SHAP value of each column for 5,000 people (colour = value of the column)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_beeswarm_dev.png", dpi=150)
    plt.close(fig)

    # C. By sex: mean contribution of each input for women and men, and the gap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    yy = np.arange(len(order))
    ax1.barh(yy - 0.2, input_table.mean_women, height=0.4, color=COLORS["Female"], label="Female")
    ax1.barh(yy + 0.2, input_table.mean_men, height=0.4, color=COLORS["Male"], label="Male")
    ax1.set(title="Mean SHAP value by sex (log-odds; < 0 lowers the score)",
            yticks=yy, yticklabels=order)
    ax1.legend(fontsize=8)
    colors = ["#b2182b" if g < 0 else "#2166ac" for g in input_table.gap_women_minus_men]
    ax2.barh(yy, input_table.gap_women_minus_men, color=colors)
    ax2.set(title=f"Share of the score gap, women minus men (total {gap['score_gap']:.2f})")
    for ax in (ax1, ax2):
        ax.axvline(0, color="#555555", lw=0.8)
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()
    fig.suptitle("Which inputs separate the scores of women and men (out-of-fold, development set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_by_sex_dev.png", dpi=150)
    plt.close(fig)

    # D. Dependence: SHAP of the 3 numeric inputs vs their value, and of marital status by category
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, name in zip(axes[:3], ["age", "education-num", "hours-per-week"]):
        for group in ("Male", "Female"):  # women drawn last, on top
            mask = sex[sample] == group
            ax.scatter(X[name].to_numpy()[sample][mask], by_input[name].to_numpy()[sample][mask],
                       s=4, alpha=0.35, color=COLORS[group], label=group)
        ax.set(title=name, xlabel=f"{name} (original value)")
    marital = dev_df["marital-status"].to_numpy()
    cats = pd.Series(by_input["marital-status"].to_numpy()).groupby(marital).median().sort_values().index
    axes[3].boxplot([by_input["marital-status"].to_numpy()[marital == c] for c in cats],
                    showfliers=False)
    axes[3].set_xticks(range(1, len(cats) + 1), cats, rotation=35, ha="right", fontsize=8)
    axes[3].set(title="marital-status")
    axes[0].set_ylabel("SHAP value of the input (log-odds)")
    axes[0].legend(fontsize=8, markerscale=3)
    for ax in axes:
        ax.axhline(0, color="#555555", lw=0.8)
        ax.grid(alpha=0.25)
    fig.suptitle("How each input moves the score (5,000 people; colour = sex, which the model never sees)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_dependence_dev.png", dpi=150)
    plt.close(fig)

    result = {"model": CANDIDATE, "version": BEST_PARAMS.parent.name,
              "units": "log-odds (LightGBM raw score)", "n_people": len(X),
              "base_value_mean": float(base.mean()), "score_gap_by_sex": gap,
              "by_input": input_table.rename_axis("input").reset_index().to_dict("records"),
              "by_column": column_table.to_dict("records")}
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "shap_dev.json").write_text(json.dumps(result, indent=2))

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="SHAP (development, out-of-fold)"):
        mlflow.log_params({"model": CANDIDATE, "version": result["version"],
                           "explainer": "TreeSHAP, out-of-fold", "units": result["units"]})
        mlflow.log_metrics({f"mean_abs_shap_{i.replace('-', '_')}": float(v)
                            for i, v in input_table.mean_abs.items()})
        for name in ["importance", "beeswarm", "by_sex", "dependence"]:
            mlflow.log_artifact(str(FIGURES_DIR / f"shap_{name}_dev.png"), "figures")
        mlflow.log_artifact(str(REPORTS_DIR / "shap_dev.json"))
    print(f"\nFigures in {FIGURES_DIR} (shap_*_dev.png); values in {REPORTS_DIR / 'shap_dev.json'}")
    return result


TEST_MODELS = ["Dummy (prior)", "Logistic Regression", "Random Forest", "LightGBM"]
N_BOOTSTRAP_TEST = 1000


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def bootstrap_test(y, preds, n=N_BOOTSTRAP_TEST):
    """95% intervals of PR-AUC, ROC-AUC and Brier on the test set (resampling
    people), and of the paired PR-AUC / Brier differences between models."""
    rng = np.random.default_rng(RANDOM_STATE)
    draws = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        draws.append({(name, metric): fn(y[idx], p[idx])
                      for name, p in preds.items()
                      for metric, fn in (("pr_auc", average_precision_score),
                                         ("roc_auc", roc_auc_score),
                                         ("brier", brier_score_loss))})
    draws = pd.DataFrame(draws)
    interval = lambda s: [float(s.quantile(0.025)), float(s.quantile(0.975))]
    per_model = {name: {m: interval(draws[(name, m)]) for m in ("pr_auc", "roc_auc", "brier")}
                 for name in preds}
    paired = {}
    for other in [m for m in preds if m != CANDIDATE]:
        key = f"{CANDIDATE} - {other}"
        paired[key] = {}
        for m, fn in (("pr_auc", average_precision_score), ("brier", brier_score_loss)):
            d = draws[(CANDIDATE, m)] - draws[(other, m)]
            paired[key][m] = {"estimate": float(fn(y, preds[CANDIDATE]) - fn(y, preds[other])),
                              "ci95": interval(d), "share_of_resamples_above_0": float((d > 0).mean())}
    return per_model, paired


def run_test(dev_df, test_df, out_dir, label, log_mlflow=True):
    """Fit on the development set, score the test set once. See the docstring."""
    from fairlearn.postprocessing import ThresholdOptimizer
    import shap
    from sklearn.metrics import precision_recall_curve
    from src import fairness
    from src.models import get_models

    out_dir.mkdir(parents=True, exist_ok=True)
    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)
    X_dev, y_dev = dev_df[FEATURES], dev_df[TARGET].to_numpy()
    X_test, y_test = test_df[FEATURES], test_df[TARGET].to_numpy()
    sex_dev, sex_test = dev_df["sex"].to_numpy(), test_df["sex"].to_numpy()
    best_params = json.loads(BEST_PARAMS.read_text())
    cv = get_cv()

    # 1. Each model: development out-of-fold (reference) and test (fitted on all of development)
    oof, test_p, rows, fitted = {}, {}, [], {}
    for name in TEST_MODELS:
        model = get_models()[name] if name.startswith("Dummy") else build_model(name, best_params[name])
        pipe = build_pipeline(model)
        oof[name], _ = cross_validate_pipeline(pipe, X_dev, pd.Series(y_dev), cv)
        fitted[name] = pipe.fit(X_dev, y_dev)
        test_p[name] = fitted[name].predict_proba(X_test)[:, 1]
        for split, y, p in (("development (out-of-fold)", y_dev, oof[name]), ("test", y_test, test_p[name])):
            rows.append({"model": name, "split": split, **probability_metrics(y, p)})
        print(f"  {name}: test PR-AUC {average_precision_score(y_test, test_p[name]):.4f}", flush=True)
    models_table = pd.DataFrame(rows)
    ci, paired = bootstrap_test(y_test, {m: test_p[m] for m in TEST_MODELS[1:]})

    # 2. LightGBM calibration on the test set: raw and Platt (fitted on the development OOF)
    platt = LogisticRegression(C=1e6).fit(logit(oof[CANDIDATE]).reshape(-1, 1), y_dev)
    apply_platt = lambda p: platt.predict_proba(logit(p).reshape(-1, 1))[:, 1]
    p_dev, p_test = apply_platt(oof[CANDIDATE]), apply_platt(test_p[CANDIDATE])
    calib_rows = []
    for variant, p in (("raw", test_p[CANDIDATE]), ("platt", p_test)):
        calib_rows.append({"variant": variant, "group": "all", **probability_metrics(y_test, p)})
        for group in ("Female", "Male"):
            mask = sex_test == group
            calib_rows.append({"variant": variant, "group": group,
                               **probability_metrics(y_test[mask], p[mask])})
    calib_table = pd.DataFrame(calib_rows)

    # 3. Fairness before / after: rules fixed on the development OOF (Platt) probabilities
    thresholds = {"best_f1": float(best_f1_point(y_dev, p_dev)["threshold"]),
                  "best_balanced_accuracy": fairness.balanced_accuracy_threshold(y_dev, p_dev)}
    decisions = {
        "single threshold (best F1)": (p_test >= thresholds["best_f1"]).astype(int),
        "single threshold (best balanced accuracy)": (p_test >= thresholds["best_balanced_accuracy"]).astype(int),
    }
    for rule, constraint in (("equalized odds", "equalized_odds"),
                             ("equal opportunity", "true_positive_rate_parity"),
                             ("demographic parity", "demographic_parity")):
        optimizer = ThresholdOptimizer(estimator=fairness.ScoreEstimator().fit(), constraints=constraint,
                                       objective="balanced_accuracy_score", prefit=True,
                                       predict_method="predict_proba")
        optimizer.fit(p_dev.reshape(-1, 1), y_dev, sensitive_features=sex_dev)
        decisions[f"{rule} (ThresholdOptimizer)"] = optimizer.predict(
            p_test.reshape(-1, 1), sensitive_features=sex_test, random_state=RANDOM_STATE)
    fair_rows, by_sex = [], {}
    for rule, pred in decisions.items():
        gap, table = fairness.gaps(y_test, pred, sex_test)
        fair_rows.append({"rule": rule, **fairness.performance(y_test, pred), **gap})
        by_sex[rule] = table
    fair_table = pd.DataFrame(fair_rows)
    race_table = fairness.group_frame(y_test, decisions["single threshold (best F1)"],
                                      test_df["race"].to_numpy())
    race_table["small_group"] = race_table.n < fairness.SMALL_GROUP
    intervals = {rule: fairness.bootstrap_gaps(y_test, decisions[rule], sex_test)
                 for rule in ("single threshold (best F1)", "single threshold (best balanced accuracy)",
                              "equalized odds (ThresholdOptimizer)")}

    # 4. SHAP split of the score gap between women and men, on the test set
    pipe = fitted[CANDIDATE]
    explainer = shap.TreeExplainer(pipe[-1])
    values = np.asarray(explainer.shap_values(pipe[:-1].transform(X_test)))
    columns = list(pipe.named_steps["preprocess"].get_feature_names_out())
    by_input = pd.DataFrame(values, columns=columns).T.groupby([column_input(c) for c in columns]).sum().T
    women, men = sex_test == "Female", sex_test == "Male"
    shap_table = pd.DataFrame({"mean_abs": by_input.abs().mean(), "mean_women": by_input[women].mean(),
                               "mean_men": by_input[men].mean()})
    shap_table["gap_women_minus_men"] = shap_table.mean_women - shap_table.mean_men
    shap_table = shap_table.sort_values("mean_abs", ascending=False)
    score = float(np.ravel(explainer.expected_value)[-1]) + values.sum(axis=1)
    shap_gap = {"mean_score_women": float(score[women].mean()), "mean_score_men": float(score[men].mean())}
    shap_gap["score_gap"] = shap_gap["mean_score_women"] - shap_gap["mean_score_men"]

    # Figures
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, color in zip(TEST_MODELS[1:], ["#7570b3", "#1b9e77", "#1f77b4"]):
        precision, recall, _ = precision_recall_curve(y_test, test_p[name])
        ax.plot(recall, precision, color=color, lw=1.6,
                label=f"{name} (PR-AUC {average_precision_score(y_test, test_p[name]):.3f})")
    ax.axhline(y_test.mean(), color="#888888", ls="--", lw=1, label=f"Dummy ({y_test.mean():.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1),
           title=f"Precision-recall curves, {label}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(figures / "test_pr_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for variant, p in (("raw", test_p[CANDIDATE]), ("platt", p_test)):
        plot_curve(axes[0], y_test, p, variant, COLORS[variant])
    style_reliability(axes[0], f"{CANDIDATE}: raw vs Platt", True)
    for ax, (variant, p) in zip(axes[1:], (("raw", test_p[CANDIDATE]), ("Platt", p_test))):
        for group in ("Female", "Male"):
            mask = sex_test == group
            plot_curve(ax, y_test[mask], p[mask], group, COLORS[group])
        style_reliability(ax, f"{CANDIDATE} {variant}, by sex", False)
    fig.suptitle(f"Calibration of the final model, {label}")
    fig.tight_layout()
    fig.savefig(figures / "test_calibration.png", dpi=150)
    plt.close(fig)

    fairness.plot_groups(decisions, y_test, sex_test, figures / "test_fairness_by_sex.png",
                         title=f"Women vs men under each decision rule, {CANDIDATE} + Platt, {label} "
                               "(rules fixed on the development set)")

    records = lambda t: t.rename_axis("group").reset_index().to_dict("records")
    result = {
        "label": label, "version": BEST_PARAMS.parent.name, "candidate": CANDIDATE,
        "n_development": len(dev_df), "n_test": len(test_df),
        "base_rate_test": float(y_test.mean()),
        "base_rate_test_by_sex": {g: float(y_test[sex_test == g].mean()) for g in ("Female", "Male")},
        "models": models_table.to_dict("records"),
        "test_ci95": ci, "paired_differences_test": paired,
        "calibration_test": calib_table.to_dict("records"),
        "fairness_test": {"thresholds_from_development": thresholds,
                          "rules": fair_table.to_dict("records"),
                          "by_sex": {r: records(t) for r, t in by_sex.items()},
                          "by_race_best_f1": records(race_table),
                          "bootstrap_95": intervals},
        "shap_test": {"by_input": shap_table.rename_axis("input").reset_index().to_dict("records"),
                      "score_gap_by_sex": shap_gap},
    }
    path = out_dir / "test_results.json"
    path.write_text(json.dumps(result, indent=2, default=float))

    pd.set_option("display.width", 220)
    cols = ["model", "split", "pr_auc", "roc_auc", "brier", "ece"]
    print(f"\nModels: development out-of-fold vs {label}\n")
    print(models_table[cols].round(4).to_string(index=False))
    print(f"\nPaired differences on {label}:", json.dumps(paired, indent=1))
    print(f"\n{CANDIDATE} calibration on {label}\n")
    print(calib_table[["variant", "group", "brier", "ece", "mean_predicted", "observed_rate"]]
          .round(4).to_string(index=False))
    print(f"\nFairness on {label} (thresholds from development: "
          f"{ {k: round(v, 3) for k, v in thresholds.items()} })\n")
    print(fair_table[["rule", "accuracy", "balanced_accuracy", "f1", "dp_diff", "tpr_diff",
                      "fpr_diff", "equalized_odds_gap", "ppv_diff", "four_fifths_ratio"]]
          .round(3).to_string(index=False))
    print(f"\nSHAP on {label}: score gap {shap_gap['score_gap']:.3f}\n")
    print(shap_table.round(3).to_string())

    if log_mlflow:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name="final evaluation (test set, once)"):
            mlflow.log_params({"model": CANDIDATE, "version": result["version"],
                               "fitted_on": "whole development set", "n_test": len(test_df)})
            for r in rows:
                if r["split"] == "test":
                    key = r["model"].split(" (")[0].replace(" ", "_")
                    mlflow.log_metrics({f"test_{key}_pr_auc": r["pr_auc"],
                                        f"test_{key}_roc_auc": r["roc_auc"],
                                        f"test_{key}_brier": r["brier"]})
            mlflow.log_artifacts(str(figures), artifact_path="figures")
            mlflow.log_artifact(str(path))
    print(f"\nSaved {path} and {figures}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["calibration", "shap", "test"])
    parser.add_argument("--dry-run", action="store_true",
                        help="test stage: use a held-out 20%% of the development set instead of the test set")
    parser.add_argument("--out", help="test stage, dry run: output folder")
    parser.add_argument("--force", action="store_true", help="test stage: run again although results exist")
    args = parser.parse_args()
    if args.stage == "calibration":
        run_calibration()
    elif args.stage == "shap":
        run_shap()
    elif args.dry_run:
        from pathlib import Path
        from sklearn.model_selection import train_test_split
        dev_df, _ = load_data()  # the test set is left untouched
        fit_df, check_df = train_test_split(dev_df, test_size=0.2, stratify=dev_df[TARGET],
                                            random_state=RANDOM_STATE + 1)
        run_test(fit_df, check_df, Path(args.out), "dry run (development hold-out)", log_mlflow=False)
    else:
        if (REPORTS_DIR / "test_results.json").exists() and not args.force:
            raise SystemExit("reports/test_results.json exists: the test set has already been used. "
                             "Use --force only if the team agrees to re-run it.")
        dev_df, test_df = load_data()
        run_test(dev_df, test_df, REPORTS_DIR, "test set")
