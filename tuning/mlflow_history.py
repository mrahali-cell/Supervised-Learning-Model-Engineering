"""Optuna history from MLflow: every tuning trial of the three models, v1 and v2.

Reads the runs logged by tuning/tune.py (one parent run per study, one nested
run per trial) from the local MLflow store and saves, for the report:
- trials.csv  : one row per trial (model, version, trial, CV scores, parameters);
- summary.csv : one row per study (trials, best PR-AUC, how many trials are
                within 0.002 of the best);
- history.png : validation PR-AUC of every trial and the best so far, per model.
If a study was run several times, only its latest run is kept.

Nothing is trained here. Results: tuning/results/mlflow/.

Run from the project root:
    python3 -m tuning.mlflow_history
"""

import re

import matplotlib
matplotlib.use("Agg")  # figures are only saved, never shown
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.config import EXPERIMENT, TRACKING_URI, TUNING_RESULTS_DIR
from src.models import TUNED_MODELS

OUT = TUNING_RESULTS_DIR / "mlflow"
TRIAL_NAME = re.compile(r"^(?P<model>.+?)(?: (?P<version>v\d+))? trial (?P<trial>\d+)$")
FLOOR = 0.69  # y-axis floor of the figure; lower trials are drawn on it


def load_trials():
    """One row per trial of the latest study of each (model, version)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], max_results=50_000)
    rows = []
    for _, run in runs.iterrows():
        match = TRIAL_NAME.match(str(run["tags.mlflow.runName"]))
        if not match or match["model"] not in TUNED_MODELS:
            continue
        params = {k[len("params."):]: v for k, v in run.items()
                  if k.startswith("params.") and pd.notna(v)}
        rows.append({"model": match["model"],
                     "version": match["version"] or "v1",  # v1 runs had no version in their name
                     "trial": int(match["trial"]),
                     "study run": run["tags.mlflow.parentRunId"],
                     "start": run["start_time"],
                     "cv PR-AUC": run["metrics.cv_pr_auc"],
                     "cv PR-AUC std": run["metrics.cv_pr_auc_std"],
                     "cv ROC-AUC": run["metrics.cv_roc_auc"],
                     "cv Brier": run["metrics.cv_brier"],
                     "params": "; ".join(f"{k}={v}" for k, v in sorted(params.items()))})
    trials = pd.DataFrame(rows)
    latest = trials.groupby(["model", "version"])["start"].transform("max")
    latest_study = trials.loc[trials["start"] == latest, ["model", "version", "study run"]]
    trials = trials.merge(latest_study.drop_duplicates(), on=["model", "version", "study run"])
    return trials.sort_values(["model", "version", "trial"]).reset_index(drop=True)


def summarise(trials):
    rows = []
    for (model, version), g in trials.groupby(["model", "version"], sort=False):
        best = g.loc[g["cv PR-AUC"].idxmax()]
        rows.append({"model": model, "version": version, "trials": len(g),
                     "best cv PR-AUC": best["cv PR-AUC"], "best cv PR-AUC std": best["cv PR-AUC std"],
                     "best trial": int(best["trial"]), "best cv Brier": best["cv Brier"],
                     "trials within 0.002 of best": int((g["cv PR-AUC"] >= best["cv PR-AUC"] - 0.002).sum()),
                     "lowest cv PR-AUC": g["cv PR-AUC"].min()})
    return pd.DataFrame(rows)


def plot(trials):
    fig, axes = plt.subplots(1, len(TUNED_MODELS), figsize=(12, 3.6), sharey=True)
    for ax, model in zip(axes, TUNED_MODELS):
        for version, color in (("v1", "tab:grey"), ("v2", "tab:blue")):
            g = trials[(trials["model"] == model) & (trials["version"] == version)]
            if g.empty:
                continue
            score = g["cv PR-AUC"]
            low = score < FLOOR
            ax.scatter(g["trial"][~low], score[~low], s=14, color=color, alpha=0.6)
            ax.scatter(g["trial"][low], [FLOOR] * low.sum(), s=18, color=color, marker="v")
            ax.step(g["trial"], score.cummax().clip(lower=FLOOR), where="post", color=color,
                    label=f"{version}: best {score.max():.4f}")
        ax.set_title(model)
        ax.set_xlabel("Optuna trial")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("mean 5-fold validation PR-AUC")
    axes[0].set_ylim(FLOOR - 0.002, 0.725)
    fig.suptitle(f"Every tuning trial logged in MLflow (dots) and the best so far (line); "
                 f"triangles = trials below {FLOOR}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "history.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    trials = load_trials()
    summary = summarise(trials)
    trials.drop(columns=["study run", "start"]).round(4).to_csv(OUT / "trials.csv", index=False)
    summary.round(4).to_csv(OUT / "summary.csv", index=False)
    plot(trials)
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved trials.csv, summary.csv and history.png in {OUT}")
