"""Stage 2: Optuna tuning with cross-validation, tracked in MLflow (requirements 6, 7).

One Optuna study per model, same budget and same 5 folds; the objective is the
mean CV PR-AUC of the complete Pipeline. Every study and every trial is logged
in MLflow. The test set is NOT used.

The search spaces are in suggest_params() below: this is the place to edit
after the diagnosis of each model (validation curves in tuning/<model>.py).

Results go to tuning/results/<version>/best_params.json. Each model only
updates its own entry, so the Random Forest and Logistic Regression can be
tuned separately into the same version.

Run from the project root:
    python -m tuning.tune --version v2 --model rf --trials 25
    python -m tuning.tune --version v2              # all three models
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the logged runs
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")  # figures are only saved, never shown
import mlflow
import optuna
import pandas as pd

from src.config import EXPERIMENT, TRACKING_URI, TUNING_RESULTS_DIR
from src.cv import cross_validate_pipeline
from src.data import FEATURES, RANDOM_STATE, TARGET, get_cv, load_data
from src.models import TUNED_MODELS, build_model
from src.pipeline import build_pipeline

N_TRIALS = 25  # same budget for every model, so the comparison is fair
SHORT_NAMES = {"lr": "Logistic Regression", "rf": "Random Forest", "lgbm": "LightGBM"}


def suggest_params(trial, name):
    """Search space of each model. The imbalance remedy (class weight) is a
    hyperparameter too, so it is validated like any other (S6).
    v1 spaces (Soumaya, git tag `baseline-v1`); edit them for v2."""
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


def tune_model(name, X, y, cv, n_trials, version):
    """One Optuna study. Each trial = one configuration + one 5-fold CV score of
    the complete Pipeline (S5). The score is the mean CV PR-AUC."""

    def objective(trial):
        params = suggest_params(trial, name)
        pipe = build_pipeline(build_model(name, params))
        _, folds = cross_validate_pipeline(pipe, X, y, cv)
        scores = {"cv_pr_auc": folds.pr_auc.mean(), "cv_pr_auc_std": folds.pr_auc.std(),
                  "cv_roc_auc": folds.roc_auc.mean(), "cv_roc_auc_std": folds.roc_auc.std(),
                  "cv_brier": folds.brier.mean()}
        with mlflow.start_run(run_name=f"{name} {version} trial {trial.number}", nested=True):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics(scores)
            mlflow.log_dict(folds.to_dict("list"), "fold_metrics.json")
        for key, value in scores.items():
            trial.set_user_attr(key, float(value))
        return scores["cv_pr_auc"]

    def progress(study, trial):
        print(f"  {name} trial {trial.number:>2}: PR-AUC {trial.value:.4f} "
              f"(best so far {study.best_value:.4f})", flush=True)

    study = optuna.create_study(direction="maximize", study_name=f"{name} {version}",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    with mlflow.start_run(run_name=f"{name} {version}"):
        mlflow.log_params({"model": name, "version": version, "n_trials": n_trials,
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


def run_tuning(version, models=TUNED_MODELS, n_trials=N_TRIALS):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    dev_df, _ = load_data()  # the test set is left untouched
    X, y = dev_df[FEATURES], dev_df[TARGET]
    cv = get_cv()

    out_dir = TUNING_RESULTS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "best_params.json"
    best_params = json.loads(path.read_text()) if path.exists() else {}

    rows = []
    for name in models:
        print(f"\nTuning {name} ({n_trials} trials, version {version})", flush=True)
        study = tune_model(name, X, y, cv, n_trials, version)
        attrs = study.best_trial.user_attrs
        best_params[name] = study.best_trial.params  # only this model's entry changes
        path.write_text(json.dumps(best_params, indent=2))
        rows.append({
            "model": name,
            "CV PR-AUC": f"{attrs['cv_pr_auc']:.3f} ± {attrs['cv_pr_auc_std']:.3f}",
            "CV ROC-AUC": f"{attrs['cv_roc_auc']:.3f} ± {attrs['cv_roc_auc_std']:.3f}",
            "CV Brier": f"{attrs['cv_brier']:.3f}",
        })

    result = pd.DataFrame(rows)
    print(f"\nTuned models, mean ± std over the 5 CV folds ({n_trials} trials each)\n")
    print(result.to_string(index=False))
    print(f"\nBest parameters saved in {path.relative_to(TUNING_RESULTS_DIR.parents[1])}; "
          f"runs in MLflow experiment '{EXPERIMENT}'.")
    return result, best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True,
                        help="results folder, e.g. v2 (v1 = Soumaya's first run, keep it)")
    parser.add_argument("--model", choices=list(SHORT_NAMES), action="append",
                        help="lr, rf or lgbm (repeat it for several); default: all three")
    parser.add_argument("--trials", type=int, default=N_TRIALS,
                        help="Optuna trials per model")
    args = parser.parse_args()
    models = [SHORT_NAMES[m] for m in args.model] if args.model else TUNED_MODELS
    run_tuning(args.version, models, args.trials)
