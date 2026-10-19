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
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# The 6 model inputs (src.data.FEATURES), split by semantic type (EDA Step 3)
NUMERIC = ["age", "education-num", "hours-per-week"]
CATEGORICAL = ["workclass", "marital-status", "occupation"]

# Engineered features (requirement 2), created by add_features() below
ENGINEERED_NUMERIC = ["age_squared"]
ENGINEERED_BINARY = ["is_married", "has_degree", "part_time", "overtime"]


def add_features(X):
    """Create the engineered features from the 6 model inputs.

    Fixed rules only (no value is learned from the data), so it is leakage-safe.
    Each feature is justified by a pattern in the development set (% earning <= $50K):
    - is_married : married-civ-spouse 55.5% vs 93.6% for everyone else;
    - has_degree : education-num >= 13 (bachelor+) 52.0% vs 83.9%;
    - part_time / overtime : < 35 h about 93%, 40 h 79%, > 40 h 56-65%.
      The EDA medians were both 40, but the rate changes by hours regime;
    - age_squared : U-shape, 98.9% (<= 24) -> 60.6% (45-54) -> 79.6% (65+),
      which a linear model cannot capture from `age` alone.
    """
    X = X.copy()
    X["is_married"] = (X["marital-status"] == "Married-civ-spouse").astype(int)
    X["has_degree"] = (X["education-num"] >= 13).astype(int)
    X["part_time"] = (X["hours-per-week"] < 35).astype(int)
    X["overtime"] = (X["hours-per-week"] > 40).astype(int)
    X["age_squared"] = X["age"] ** 2
    return X


def build_preprocessor(engineered=False):
    """ColumnTransformer: a different preprocessing chain per column type.

    engineered=True expects the extra columns made by add_features().
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        # handle_unknown="ignore": a category never seen in training does not crash
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = [
        ("num", numeric, NUMERIC + (ENGINEERED_NUMERIC if engineered else [])),
        ("cat", categorical, CATEGORICAL),
    ]
    if engineered:
        transformers.append(("bin", "passthrough", ENGINEERED_BINARY))
    return ColumnTransformer(transformers)


def build_pipeline(model, engineered=True):
    """Full Pipeline for the team: features -> preprocessing -> the model.

    Example (in train.py):
        pipe = build_pipeline(SomeClassifier())
    engineered=False gives the preprocessing only (to measure what the
    engineered features bring).
    """
    steps = []
    if engineered:
        steps.append(("features", FunctionTransformer(add_features)))
    steps += [("preprocess", build_preprocessor(engineered)), ("model", model)]
    return Pipeline(steps)


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
