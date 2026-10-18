"""Model training, Optuna tuning with cross-validation, MLflow tracking (requirements 4, 6, 7).

Uses the cleaned data and split from src/data.py:
    from src.data import load_data, get_cv, FEATURES, SENSITIVE, TARGET

Stage 1 (`baseline`): comparison of the model progression (Dummy -> Logistic
Regression -> Random Forest -> LightGBM) with default parameters, on the 5 CV
folds of the development set.
Stage 2 (`tune`): one Optuna study per required model (Logistic Regression,
Random Forest, LightGBM), same budget and same folds; every study and every
trial is logged in MLflow (parameters, metrics, artifacts).
The test set is NOT used in either stage.

Run from the project root:
    python -m src.train baseline
    python -m src.train tune --trials 25
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the logged runs
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # figures are only saved, never shown
import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)

from src.data import RANDOM_STATE, TARGET, FEATURES, get_cv, load_data
from src.pipeline import build_pipeline

# Positive class = high_income (income > $50K, ~24%): the rare class (S6), so
# PR-AUC is informative (a random ranking scores ~0.24, not ~0.76).


def get_models():
    """Model progression with default parameters (tuning comes in stage 2)."""
    return {
        "Dummy (prior)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Logistic Regression (balanced)": LogisticRegression(
            max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1,
            random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
        "LightGBM (balanced)": LGBMClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, verbose=-1),
    }


def best_f1_point(y_true, scores):
    """Operating point that maximises F1 (a neutral choice: no single costly error).

    Returns a dict with threshold, precision, recall, F1 and false positive rate.
    The threshold is picked on the same out-of-fold scores it is measured on, so
    the numbers are slightly optimistic; they are only for comparing models.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best = int(np.argmax(f1))
    flagged = scores >= thresholds[best]
    negatives = (y_true == 0)
    return {"threshold": thresholds[best], "precision": precision[best],
            "recall": recall[best], "f1": f1[best],
            "fpr": (flagged & negatives).sum() / negatives.sum()}


def cross_validate_pipeline(pipe, X, y, cv):
    """Out-of-fold probabilities + per-fold ranking metrics (fit inside each fold)."""
    oof = np.zeros(len(y))
    folds = []
    for train_idx, val_idx in cv.split(X, y):
        fitted = clone(pipe).fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = fitted.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = proba
        y_val = y.iloc[val_idx]
        folds.append({"roc_auc": roc_auc_score(y_val, proba),
                      "pr_auc": average_precision_score(y_val, proba),
                      "brier": brier_score_loss(y_val, proba)})
    return oof, pd.DataFrame(folds)


def run_baseline():
    dev_df, _ = load_data()  # the test set is left untouched
    X, y = dev_df[FEATURES], dev_df[TARGET]
    cv = get_cv()

    rows = []
    for engineered in (False, True):
        for name, model in get_models().items():
            if engineered and name.startswith("Dummy"):
                continue  # features cannot change a constant prediction
            pipe = build_pipeline(model, engineered=engineered)
            oof, folds = cross_validate_pipeline(pipe, X, y, cv)
            point = best_f1_point(y.to_numpy(), oof)
            rows.append({
                "model": name,
                "features": "engineered" if engineered else "base",
                "ROC-AUC": f"{folds.roc_auc.mean():.3f} ± {folds.roc_auc.std():.3f}",
                "PR-AUC": f"{folds.pr_auc.mean():.3f} ± {folds.pr_auc.std():.3f}",
                "Brier": f"{folds.brier.mean():.3f}",
                "best F1": round(point["f1"], 3),
                "precision": round(point["precision"], 3),
                "recall": round(point["recall"], 3),
                "FPR": round(point["fpr"], 3),
            })
            print(f"done: {name} ({'engineered' if engineered else 'base'})", flush=True)

    result = pd.DataFrame(rows)
    print(f"\nBaseline, 5-fold CV on the development set "
          f"(positive class = high_income, base rate {y.mean():.3f})\n")
    print(result.to_string(index=False))
    return result


# ----------------------------------------------------------------------------
# Stage 2: Optuna tuning with cross-validation, tracked in MLflow
# ----------------------------------------------------------------------------

N_TRIALS = 25  # same budget for every model, so the comparison is fair
EXPERIMENT = "adult-income-bias-audit"
# MLflow's plain-folder store is in maintenance mode, so runs go to a SQLite file
# (git-ignored); artifacts such as figures go to ./mlruns (also git-ignored).
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
TUNED_MODELS = ["Logistic Regression", "Random Forest", "LightGBM"]
REPORTS_DIR = Path("reports")


def suggest_params(trial, name):
    """Search space of each model. The imbalance remedy (class weight) is a
    hyperparameter too, so it is validated like any other (S6)."""
    if name == "Logistic Regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if name == "Random Forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30, log=True),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced_subsample"]),
        }
    if name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 96),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            # ~3.2 = N_negative / N_positive is the S6 starting heuristic
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 4.0),
        }
    raise ValueError(f"unknown model: {name}")


def build_model(name, params):
    """Estimator for a model name and a dict of hyperparameters."""
    if name == "Logistic Regression":
        return LogisticRegression(solver="liblinear", max_iter=2000, **params)
    if name == "Random Forest":
        return RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE, **params)
    if name == "LightGBM":
        return LGBMClassifier(subsample_freq=1, random_state=RANDOM_STATE,
                              verbose=-1, **params)
    raise ValueError(f"unknown model: {name}")


def tune_model(name, X, y, cv, n_trials):
    """One Optuna study. Each trial = one configuration + one 5-fold CV score of
    the complete Pipeline (S5). The score is the mean CV PR-AUC."""

    def objective(trial):
        params = suggest_params(trial, name)
        pipe = build_pipeline(build_model(name, params))
        _, folds = cross_validate_pipeline(pipe, X, y, cv)
        scores = {"cv_pr_auc": folds.pr_auc.mean(), "cv_pr_auc_std": folds.pr_auc.std(),
                  "cv_roc_auc": folds.roc_auc.mean(), "cv_roc_auc_std": folds.roc_auc.std(),
                  "cv_brier": folds.brier.mean()}
        with mlflow.start_run(run_name=f"{name} trial {trial.number}", nested=True):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics(scores)
            mlflow.log_dict(folds.to_dict("list"), "fold_metrics.json")
        for key, value in scores.items():
            trial.set_user_attr(key, float(value))
        return scores["cv_pr_auc"]

    def progress(study, trial):
        print(f"  {name} trial {trial.number:>2}: PR-AUC {trial.value:.4f} "
              f"(best so far {study.best_value:.4f})", flush=True)

    study = optuna.create_study(direction="maximize", study_name=name,
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"model": name, "n_trials": n_trials,
                           "objective": "mean 5-fold CV PR-AUC",
                           "cv": "StratifiedKFold(5, shuffle=True, random_state=42)"})
        study.optimize(objective, n_trials=n_trials, callbacks=[progress])
        best = study.best_trial
        mlflow.log_params({f"best_{k}": str(v) for k, v in best.params.items()})
        mlflow.log_metrics({f"best_{k}": v for k, v in best.user_attrs.items()})
        mlflow.log_dict(best.params, "best_params.json")
        try:  # optimisation history: PR-AUC of every trial
            from optuna.visualization.matplotlib import plot_optimization_history
            ax = plot_optimization_history(study)
            mlflow.log_figure(ax.figure, "optimization_history.png")
        except Exception as error:  # a missing plot must never stop the tuning
            print("  (optimisation-history figure skipped:", error, ")")
    return study


def run_tuning(n_trials=N_TRIALS):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    dev_df, _ = load_data()  # the test set is left untouched
    X, y = dev_df[FEATURES], dev_df[TARGET]
    cv = get_cv()

    rows, best_params = [], {}
    for name in TUNED_MODELS:
        print(f"\nTuning {name} ({n_trials} trials)", flush=True)
        study = tune_model(name, X, y, cv, n_trials)
        attrs = study.best_trial.user_attrs
        best_params[name] = study.best_trial.params
        rows.append({
            "model": name,
            "CV PR-AUC": f"{attrs['cv_pr_auc']:.3f} ± {attrs['cv_pr_auc_std']:.3f}",
            "CV ROC-AUC": f"{attrs['cv_roc_auc']:.3f} ± {attrs['cv_roc_auc_std']:.3f}",
            "CV Brier": f"{attrs['cv_brier']:.3f}",
        })

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2))
    result = pd.DataFrame(rows)
    print(f"\nTuned models, mean ± std over the 5 CV folds ({n_trials} trials each)\n")
    print(result.to_string(index=False))
    print("\nBest parameters saved in reports/best_params.json; runs in MLflow "
          f"experiment '{EXPERIMENT}'.")
    return result, best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", nargs="?", default="baseline",
                        choices=["baseline", "tune"])
    parser.add_argument("--trials", type=int, default=N_TRIALS,
                        help="Optuna trials per model (tune stage)")
    args = parser.parse_args()
    if args.stage == "baseline":
        run_baseline()
    else:
        run_tuning(args.trials)
