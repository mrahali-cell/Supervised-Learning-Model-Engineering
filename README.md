# Supervised-Learning-Model-Engineering

MSA-DATB09-04 — Supervised Learning & Model Engineering (Albert School, Fall 2026).

Binary classification on the UCI Adult / Census Income dataset (1994 U.S. Census):
predict whether a person's income is **≤ $50K** (`low_income = 1`) from a short
profile (age, education, job, hours worked, marital status).

---

## Architecture

```
notebooks/
    01_eda.ipynb      EDA: exploration, findings and every decision (the "why")
src/
    data.py           load → clean → split (the result of the EDA)
    pipeline.py       preprocessing + model Pipeline          (reqs 2, 3, 5)
    train.py          training, Optuna tuning, MLflow tracking (reqs 4, 6, 7)
    evaluate.py       test-set metrics, calibration, SHAP      (reqs 8, 9)
    fairness.py       fairness audit and mitigation            (req 10)
requirements.txt      Python libraries needed
.gitignore            files Git ignores (*.csv, caches, mlruns…)
```

### How the pieces fit together

```
01_eda.ipynb  ──decides──▶  src/data.py
                             load_raw()     download Adult (OpenML)
                             clean_adult()  fixed rules only (no learning from data)
                             split()        80% development / 20% test, stratified
                                  │
                                  ▼
                 dev_df (80%)            test_df (20%)
                     │                        │
                     ▼                        │  untouched until the end
        src/pipeline.py + src/train.py        │
        impute → encode → scale → model       │
        fitted on dev_df only,                │
        5-fold CV for tuning (get_cv())       │
                     │                        │
                     ▼                        ▼
        src/evaluate.py + src/fairness.py  ◀──┘  evaluated ONCE on test_df
```

**The rule:** anything that *learns from the data* (imputation, encoding,
scaling, learned features) happens **after the split, inside the Pipeline**,
fitted on the training data only (Session 1 slide 21; Session 5 slides 9, 11, 13).
Only fixed rules are applied before the split, in `clean_adult()`.

---

## Setup (once)

```bash
pip install -r requirements.txt
```

## How to run the EDA

- **To read it:** open `notebooks/01_eda.ipynb` directly on GitHub. It is saved
  with all its outputs.
- **To run it:** open it in VS Code, select the Python kernel (top right), then
  **Run All**.

## How to check the data step

From the project root:

```bash
python -m src.data
```

Expected output:

```
Development: (39032, 10) | Test: (9758, 10)
% low income, dev: 76.06 | test: 76.06
```

## How to use the data in your script

Do **not** re-run the EDA or copy the cleaning code. Import it:

```python
from src.data import load_data, get_cv, FEATURES, SENSITIVE, TARGET

dev_df, test_df = load_data()
X_train, y_train = dev_df[FEATURES], dev_df[TARGET]
cv = get_cv()   # 5-fold StratifiedKFold for model selection
```

---

## Key decisions (details in the notebook)

| Item | Decision |
|---|---|
| Target | `low_income`: 1 = income ≤ $50K, 0 = > $50K (≈ 76% / 24%) |
| Model inputs (`FEATURES`) | `age`, `workclass`, `education-num`, `marital-status`, `occupation`, `hours-per-week` |
| Kept aside for the fairness audit only (`SENSITIVE`) | `race`, `sex`, `native-country`: never model inputs |
| Dropped | `education` (duplicate), `relationship` (sex proxy), `fnlwgt` (sampling weight), `capital-gain` / `capital-loss` (leakage, not available at prediction time) |
| Split | 80% development / 20% test, stratified, `random_state=42` |
| Cross-validation | 5-fold `StratifiedKFold` on the development set |
| Missing values | Filled with `"Unknown"` **inside the Pipeline** |
| Key metric | Recall on low earners (`low_income = 1`) |

## Git workflow

- Each member works on their own branch (`manel/dev`, `hadiqa/dev`, `soumaya/dev`).
- Changes reach `main` through a pull request reviewed by a teammate.
- Never commit data files (`*.csv`): the data is downloaded by `load_raw()`.
