"""Stage 1: the model progression with default parameters (requirement 4).

Dummy -> Logistic Regression -> Random Forest -> LightGBM, with and without the
engineered features, on the 5 CV folds of the development set. The test set is
NOT used. This is the reference every tuning result is compared to.

Run from the project root:
    python -m tuning.baseline
"""

import pandas as pd

from src.cv import best_f1_point, cross_validate_pipeline
from src.data import FEATURES, TARGET, get_cv, load_data
from src.models import get_models
from src.pipeline import build_pipeline


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


if __name__ == "__main__":
    run_baseline()
