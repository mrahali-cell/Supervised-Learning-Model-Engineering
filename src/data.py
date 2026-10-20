"""Load, clean and split the Adult dataset.

All decisions are justified in notebooks/01_eda.ipynb.
Run `python -m src.data` from the project root to check the output.
"""

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold

RANDOM_STATE = 42

# Model inputs vs columns kept aside for the fairness audit only
FEATURES = ["age", "workclass", "education-num", "marital-status",
            "occupation", "hours-per-week"]
SENSITIVE = ["race", "sex", "native-country"]
TARGET = "high_income"  # 1 = income > $50K (the rare class, ~24%), 0 = income <= $50K


def load_raw():
    """Download the raw Adult dataset (OpenML, version 2)."""
    return fetch_openml(name="adult", version=2, as_frame=True).frame


def clean_adult(raw):
    """Apply the fixed cleaning rules decided in the EDA (no learning from data)."""
    df = raw.copy()

    # 1. Remove exact duplicate rows (EDA Step 2)
    df = df.drop_duplicates()

    # Category columns -> plain text, so new labels can be written into them
    for col in df.select_dtypes("category").columns:
        df[col] = df[col].astype("object")

    # 2. People who never worked: occupation is "not applicable", not unknown (Step 2)
    never = (df["workclass"] == "Never-worked") & df["occupation"].isna()
    df.loc[never, "occupation"] = "Never-worked"

    # 3. Merge rare categories (Steps 4b-4c)
    df["workclass"] = df["workclass"].replace({"Without-pay": "No-pay",
                                               "Never-worked": "No-pay"})
    df["marital-status"] = df["marital-status"].replace(
        {"Married-AF-spouse": "Married-civ-spouse"})
    df["occupation"] = df["occupation"].replace({"Armed-Forces": "Protective-serv"})

    # 4. Native country -> United-States / Mexico / Other (missing stays missing)
    keep = df["native-country"].isin(["United-States", "Mexico"]) | df["native-country"].isna()
    df["native-country"] = df["native-country"].where(keep, "Other")

    # 5. Drop redundant (Step 3), sampling-weight and leaking (Step 6) columns
    df = df.drop(columns=["education", "relationship", "fnlwgt",
                          "capital-gain", "capital-loss"])

    # 6. Target as 0/1, and drop class so it can never be used as an input
    df[TARGET] = (df["class"] == ">50K").astype(int)
    df = df.drop(columns="class")

    return df


def split(df):
    """80% development / 20% test, stratified on the target.

    Stratifies on the reverse of the target (the former low_income column) so the
    rows in each part are exactly those of the split made before the target flip.
    """
    return train_test_split(df, test_size=0.2, stratify=1 - df[TARGET],
                            random_state=RANDOM_STATE)


def get_cv():
    """5-fold stratified cross-validation for the development set."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def load_data():
    """Full chain: download -> clean -> split. Returns (dev_df, test_df)."""
    return split(clean_adult(load_raw()))


if __name__ == "__main__":
    dev_df, test_df = load_data()
    print("Development:", dev_df.shape, "| Test:", test_df.shape)
    print("% high income, dev:", round(dev_df[TARGET].mean() * 100, 2),
          "| test:", round(test_df[TARGET].mean() * 100, 2))
