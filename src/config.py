"""Shared paths and settings, imported by src/ and tuning/ (nothing runs here)."""

import os
from pathlib import Path

# Project root, so paths work from the root, from notebooks/ or from tuning/
ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = ROOT / "reports"                  # calibration and fairness outputs
TUNING_RESULTS_DIR = ROOT / "tuning" / "results"  # every tuning result, one folder per version

# Hyperparameters used by src/evaluate.py and src/fairness.py.
# v1 = Soumaya's first Optuna run (git tag `baseline-v1`).
# Switch to "v2" once tuning v2 is finished and agreed by the team.
BEST_PARAMS_VERSION = "v1"
BEST_PARAMS = TUNING_RESULTS_DIR / BEST_PARAMS_VERSION / "best_params.json"

# MLflow: runs go to a SQLite file (git-ignored); artifacts to ./mlruns (git-ignored)
EXPERIMENT = "adult-income-bias-audit"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
