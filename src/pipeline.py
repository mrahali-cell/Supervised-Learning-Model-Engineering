"""Preprocessing for the model Pipeline (requirements 2, 5).

Decisions from notebooks/01_eda.ipynb:
- numeric columns: fill any missing value with the training median, then scale;
- categorical columns: fill missing values with "Unknown", then one-hot encode.

The preprocessor is fitted on the training data only and is meant to be
placed inside the model Pipeline, so it is re-fitted in each CV fold
(Session 5, slides 9 and 11). No model is chosen or trained here.

Run `python3 -m src.pipeline` from the project root for a quick check.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def build_pipeline(model):
    """Full Pipeline for the team: preprocessing -> the model they choose.

    Example (in train.py):
        pipe = build_pipeline(SomeClassifier())
    TODO (team): add the feature engineering step here, before "preprocess".
    """
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])


if __name__ == "__main__":
    # Check the preprocessing only: fit on the training data, look at the result.
    from src.data import load_data, FEATURES

    dev_df, _ = load_data()  # the test set is NOT used here
    X_train = dev_df[FEATURES]

    preprocessor = build_preprocessor()
    X_prepared = preprocessor.fit_transform(X_train)
    names = preprocessor.get_feature_names_out()

    print("Before preprocessing:", X_train.shape, "| missing values:",
          int(X_train.isna().sum().sum()))
    print("After preprocessing: ", X_prepared.shape, "| missing values:",
          int((X_prepared != X_prepared).sum()))
    print("\nColumns created:")
    for name in names:
        print("  ", name)
