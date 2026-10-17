"""Preprocessing + model Pipeline (requirements 2, 3, 5).

Decisions from notebooks/01_eda.ipynb:
- numeric columns: fill any missing value with the training median, then scale;
- categorical columns: fill missing values with "Unknown", then one-hot encode.

Every step is fitted on the training data only, inside the Pipeline
(Session 5, slides 9 and 11), so it is re-learned correctly in each CV fold.

Run `python -m src.pipeline` from the project root for a quick check.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# The 6 model inputs (src.data.FEATURES), split by semantic type (EDA Step 3)
NUMERIC = ["age", "education-num", "hours-per-week"]
CATEGORICAL = ["workclass", "marital-status", "occupation"]


def build_preprocessor():
    """ColumnTransformer: a different preprocessing chain per column type."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        # handle_unknown="ignore": a category never seen in training does not crash
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def build_pipeline(model=None):
    """Full Pipeline: preprocessing -> model.

    Pass any scikit-learn classifier as `model`; defaults to logistic
    regression (the baseline).
    TODO (team): add the feature engineering step here, before "preprocess".
    """
    if model is None:
        model = LogisticRegression(max_iter=1000)
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])


if __name__ == "__main__":
    from sklearn.model_selection import cross_validate
    from src.data import load_data, get_cv, FEATURES, TARGET

    dev_df, _ = load_data()  # the test set is NOT used here
    X, y = dev_df[FEATURES], dev_df[TARGET]

    scores = cross_validate(build_pipeline(), X, y, cv=get_cv(),
                            scoring=["recall", "roc_auc"])
    print("5-fold CV on the development set (logistic regression baseline)")
    print(f"Recall (low income): {scores['test_recall'].mean():.3f}")
    print(f"ROC AUC:             {scores['test_roc_auc'].mean():.3f}")

