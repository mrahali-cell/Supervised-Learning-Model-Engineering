"""The models of the project (requirement 4). Library only: nothing is trained here.

- get_models(): the model progression with default parameters (the baseline);
- build_model(name, params): one model with given hyperparameters (used by
  tuning/, src/evaluate.py and src/fairness.py).
"""

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data import RANDOM_STATE

# The three models that are tuned (requirement 6)
TUNED_MODELS = ["Logistic Regression", "Random Forest", "LightGBM"]


def get_models():
    """Model progression with default parameters (Dummy -> LR -> RF -> LightGBM)."""
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
