## For teammates: quick start

You don't need to run `data.py` or `pipeline.py` first. You **import** them at
the top of your own script (e.g. a script in `tuning/`), and the imports do the work.

### Step 1: get the clean data and the split

    from src.data import load_data, get_cv, FEATURES, TARGET

| Imported | What it is | What it gives you |
|---|---|---|
| `load_data()` | function | Downloads the Adult dataset, cleans it (EDA decisions), splits it 80/20. Returns `dev_df, test_df` |
| `get_cv()` | function | The 5-fold stratified cross-validation, to use for tuning and model comparison |
| `FEATURES` | list of 6 column names | The model inputs: age, workclass, education-num, marital-status, occupation, hours-per-week |
| `TARGET` | column name | `"high_income"`: 1 = income > $50K, 0 = income ≤ $50K |

    dev_df, test_df = load_data()
    X_train, y_train = dev_df[FEATURES], dev_df[TARGET]

- `dev_df`: 80% of the data (39,032 rows), for training and cross-validation.
- `test_df`: 20% (9,758 rows). **Do not touch it** until the final evaluation.
- `X_train`: only the 6 input columns. `y_train`: only the answer (0/1).
- Every call returns exactly the same split (`random_state=42`), for everyone.

### Step 2: build the Pipeline with your model

    from src.pipeline import build_pipeline

    pipe = build_pipeline(YourModel())
    pipe.fit(X_train, y_train)

| Imported | What it is | What it gives you |
|---|---|---|
| `build_pipeline(model)` | function | A Pipeline: preprocessing → your model, as one object |

The preprocessing inside the Pipeline:
- numeric columns (age, education-num, hours-per-week): missing values → median, then scaling;
- categorical columns (workclass, marital-status, occupation): missing values → "Unknown", then one-hot encoding.

It is fitted on the training data only, and re-fitted automatically in each
cross-validation fold, so there is no leakage. **Never preprocess the data
yourself outside the Pipeline.**

`YourModel()` is any scikit-learn classifier, for example
`LogisticRegression(max_iter=1000)` or `RandomForestClassifier()`.

### Notes
- Feature engineering: inside `build_pipeline()` (`engineered=True` by default):
  `is_married`, `has_degree`, `part_time`, `overtime`, `age_squared`.
- `SENSITIVE` (`race`, `sex`, `native-country`) can also be imported from
  `src.data`: these columns are **not** model inputs, they are only used for
  the fairness audit.
- To see what the data looks like: `notebooks/01_eda.ipynb`
  (Steps 1–8 = raw data and decisions, Step 9 = data after preprocessing).

# Supervised-Learning-Model-Engineering

MSA-DATB09-04 — Supervised Learning & Model Engineering (Albert School, Fall 2026).

Binary classification on the UCI Adult / Census Income dataset (1994 U.S. Census):
predict whether a person's income is **> $50K** (`high_income = 1`) from a short
profile (age, education, job, hours worked, marital status).

---

## Architecture

```
src/                  LIBRARY: imported by everything else, never trains a model by itself
    config.py           paths and settings (results folders, which tuned version is used, MLflow)
    data.py             load → clean → split (the result of the EDA)
    pipeline.py         feature engineering + preprocessing + model Pipeline  (reqs 2, 3, 5)
    models.py           the models: defaults (baseline) and build_model(name, params)  (req 4)
    cv.py               cross-validation helpers: out-of-fold scores, train vs validation
    evaluate.py         calibration (later: test-set metrics, SHAP)               (reqs 8, 9)
    fairness.py         fairness audit and mitigation                             (req 10)
tuning/               ALL model training and hyperparameter tuning: scripts you run
    baseline.py         stage 1: the model progression with default parameters   (req 4)
    random_forest.py    RF diagnosis: baseline + validation curves (Manel)
    tune.py             stage 2: Optuna + MLflow, one search space per model     (reqs 6, 7)
    results/
        v1/best_params.json       first Optuna run (Soumaya), = git tag `baseline-v1`
        v2/best_params.json       tuning after the diagnosis (created by tune.py)
        random_forest/            RF diagnosis results (baseline.csv, curves.csv, curves/*.png)
notebooks/            EXPLAIN: read the results files and comment them, no training
    01_eda.ipynb                  EDA: exploration, findings and every decision (the "why")
    02_logistic_regression.ipynb  one notebook per model: baseline, diagnosis, curves, tuning
    03_random_forest.ipynb
    04_lightgbm.ipynb
reports/              calibration and fairness outputs (json + figures)
docs/findings_log.md  the log of every finding and decision (for the report and the oral)
requirements.txt      Python libraries needed
.gitignore            files Git ignores (data *.csv, caches, mlruns…)
```

**Who does what:** `src/` = the tools, `tuning/` = the experiments (scripts),
`tuning/results/` = their numbers, `notebooks/` = the explanations,
`docs/findings_log.md` = the written log.

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
        src/pipeline.py + tuning/ scripts     │
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
% high income, dev: 23.94 | test: 23.94
```

## How to run the model stages

From the project root, in this order:

```bash
python -m tuning.baseline                              # stage 1: default parameters, 5-fold CV
python -m tuning.random_forest baseline                # RF diagnosis: where do we start?
python -m tuning.random_forest curves                  # RF validation curves (one per hyperparameter)
python -m tuning.tune --version v2 --model rf          # stage 2: Optuna + MLflow (lr, rf or lgbm)
python -m src.evaluate calibration                     # reliability diagrams, Brier, ECE, Platt / isotonic
python -m src.fairness audit                           # fairness by sex, one mitigation (development set)
mlflow ui --backend-store-uri sqlite:///mlflow.db      # browse the logged runs
```

`src/evaluate.py` and `src/fairness.py` use the tuned parameters of the
version set in `src/config.py` (`BEST_PARAMS_VERSION`, currently `v1`). Switch it
to `v2` once tuning v2 is done.

Findings so far are written in `docs/findings_log.md`.

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
| Target | `high_income`: 1 = income > $50K, 0 = ≤ $50K (≈ 24% / 76%) |
| Model inputs (`FEATURES`) | `age`, `workclass`, `education-num`, `marital-status`, `occupation`, `hours-per-week` |
| Kept aside for the fairness audit only (`SENSITIVE`) | `race`, `sex`, `native-country`: never model inputs |
| Dropped | `education` (duplicate), `relationship` (sex proxy), `fnlwgt` (sampling weight), `capital-gain` / `capital-loss` (leakage, not available at prediction time) |
| Split | 80% development / 20% test, stratified, `random_state=42` |
| Cross-validation | 5-fold `StratifiedKFold` on the development set |
| Missing values | Filled with `"Unknown"` **inside the Pipeline** |
| Key metrics | ROC-AUC and PR-AUC (ranking); per-group selection rate, recall, FPR and precision (fairness audit, `sex`); Brier score and ECE (calibration) |

## Git workflow

- Each member works on their own branch (`manel/dev`, `hadiqa/dev`, `soumaya/dev`).
- Changes reach `main` through a pull request reviewed by a teammate.
- Never commit data files (`*.csv`): the data is downloaded by `load_raw()`.
