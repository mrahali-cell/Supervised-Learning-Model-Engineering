# Supervised Learning & Model Engineering: auditing an income model for sex bias

MSA-DATB09-04, Albert School (Fall 2026). Team: Manel, Soumaya, Hadiqa.

**Use case.** A company builds models that predict whether a person earns more
than $50K a year; its clients use these predictions for decisions such as
audits. If a model treats some groups worse than others, those decisions become
unfair, and the clients face sanctions, legal action and financial loss. Our
work is the **model-audit step**: measure how biased such a model is by sex,
reduce part of that bias, and show what it costs.

- **Data:** UCI Adult / Census Income (OpenML version 2, 1994 US Census, 48,842 people).
- **Target:** `high_income` = 1 if income > $50K (about 24% of people), 0 otherwise.
- **Model inputs (6):** `age`, `workclass`, `education-num`, `marital-status`,
  `occupation`, `hours-per-week`.
- **Never inputs, audit only:** `sex`, `race`, `native-country`.
- **Hypothesis:** the model rebuilds the gender gap through proxies (marital
  status, occupation, hours worked).

Every finding and decision, with its numbers, is in
[`docs/findings_log.md`](docs/findings_log.md).

---

## The project in 11 steps

| # | Step | Course | Where | Status |
|---|---|---|---|---|
| 1 | Use case and framing | Session 1 | this README, `docs/findings_log.md` | done |
| 2 | EDA and cleaning (leakage, duplicates, rare categories) | Session 1 | `notebooks/01_eda.ipynb`, `src/data.py` | done |
| 3 | Stratified split: 80% development / 20% test, test locked | Session 1 | `src/data.py` | done |
| 4 | Pipeline: preprocessing + feature engineering | Session 5 | `src/pipeline.py` | done |
| 5 | Baseline models: Dummy, logistic regression, random forest, LightGBM | Sessions 2-4 | `tuning/baseline.py` | done |
| 6 | Diagnosis: train vs validation, validation curves | Sessions 3-5 | `tuning/random_forest.py`, `notebooks/0*_*.ipynb` | random forest, LR & LightGBM done |
| 7 | Hyperparameter tuning (Optuna, tracked in MLflow) | Session 5 | `tuning/tune.py` | v1 done; v2 done for all three models (used from now on) |
| 8 | Calibration | Session 7 | `src/evaluate.py` | done on v1; to re-run on v2 |
| 9 | Explainability (SHAP) | Session 8 | `src/evaluate.py` | to do |
| 10 | Fairness audit by sex + mitigation | Session 9 | `src/fairness.py` | done on the development set (v1); to re-run on v2 |
| 11 | Test set, once: final metrics, fairness before / after, model card | | | to do |

5-fold stratified cross-validation on the development set measures every
comparison in steps 5 to 8. The test set is opened only once, in step 11.

---

## Architecture

```
src/                  LIBRARY: imported by everything else, never trains a model by itself
    config.py           paths and settings (results folders, tuned version used, MLflow)
    data.py             load -> clean (fixed rules from the EDA) -> split; FEATURES, SENSITIVE, TARGET
    pipeline.py         feature engineering + preprocessing + model, as one Pipeline
    models.py           the models: defaults (baseline) and build_model(name, params)
    cv.py               cross-validation helpers: out-of-fold scores, train vs validation scores
    evaluate.py         calibration (later: test-set metrics, SHAP)
    fairness.py         fairness audit and mitigation
tuning/               EXPERIMENTS: every training and tuning script you run
    baseline.py         stage 1: all models with default parameters
    random_forest.py    random forest diagnosis: baseline table + validation curves (Manel)
    lightgbm.py         LightGBM diagnosis: baseline table + validation curves (Hadiqa)
    tune.py             stage 2: Optuna search + MLflow tracking, one search space per model
    results/            NUMBERS: everything the scripts produce
        v1/best_params.json         first Optuna run, all three models (Soumaya)
        v2/best_params.json         tuning after the diagnosis, all three models (used now)
        random_forest/              baseline.csv, curves.csv, curves/*.png
        lightgbm/                   same files for LightGBM (Hadiqa)
notebooks/            EXPLANATIONS: read the results files and comment them, no training
    01_eda.ipynb                  EDA: every data decision and why
    02_logistic_regression.ipynb  one notebook per model: baseline, diagnosis, curves, tuning, findings
    03_random_forest.ipynb
    04_lightgbm.ipynb
reports/              calibration and fairness outputs (json + figures)
docs/findings_log.md  the log of every finding and decision (for the report and the oral)
requirements.txt      Python libraries
```

**The rule:** `src/` = the tools, `tuning/` = the experiments, `tuning/results/` =
their numbers, `notebooks/` = the explanations, `docs/findings_log.md` = the log.

**No leakage:** anything that learns from the data (imputation, scaling, one-hot
encoding) happens after the split, inside the Pipeline, re-fitted in each
cross-validation fold. Only fixed rules are applied before the split, in
`clean_adult()` (Session 1; Session 5 on Pipelines).

---

## Setup (once)

From the project folder, in the VS Code terminal:

```bash
python3 -m venv .venv                 # create the environment (only the first time)
source .venv/bin/activate             # activate it: the line now starts with (.venv)
pip install -r requirements.txt       # install the libraries
```

**Mac only: LightGBM needs OpenMP (`libomp`).** Without it, any script stops with
`Library not loaded: @rpath/libomp.dylib`. Install it with Homebrew:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo >> ~/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
brew install libomp
```

The Homebrew installer asks for your Mac password (the one that unlocks your
session; nothing shows while you type).

**In VS Code:** `Cmd + Shift + P` -> `Python: Select Interpreter` -> choose the
one with `.venv`. In each notebook, the kernel (top right) should also show `.venv`.

---

## How to run

**Always from the project root, with the environment active, with `python3 -m`:**

```bash
source .venv/bin/activate
python3 -m tuning.random_forest baseline
```

Do **not** use the "Run Python File" button of VS Code for these scripts. It runs
the file from inside its own folder, so Python cannot find `src`
(`ModuleNotFoundError: No module named 'src'`); it may also use the Mac's Python
instead of `.venv`, and it passes no argument (`baseline`, `curves`...).

| Command | What it does | Output |
|---|---|---|
| `python3 -m src.data` | Checks the data step | `Development: (39032, 10) \| Test: (9758, 10)`, 23.94% high income in both |
| `python3 -m src.pipeline` | Checks the preprocessing | 6 inputs -> 32 columns, no missing value |
| `python3 -m tuning.baseline` | All models with default parameters, 5-fold CV | table in the terminal |
| `python3 -m tuning.random_forest baseline` | Dummy, one tree, three forests (+ v2 when it exists): train vs validation | `tuning/results/random_forest/baseline.csv` |
| `python3 -m tuning.random_forest curves` | One validation curve per random forest setting (about 10 min) | `curves.csv`, `curves/*.png` |
| `python3 -m tuning.random_forest curves --param max_depth` | One curve only | same files, that curve replaced |
| `python3 -m tuning.tune --version v2 --model rf` | Optuna, 25 trials (`--trials N` to change); `--model` = `lr`, `rf` or `lgbm`, repeatable; no `--model` = all three | `tuning/results/v2/best_params.json` + MLflow runs |
| `python3 -m src.evaluate calibration` | Calibration on out-of-fold predictions: Brier, ECE, Platt, isotonic, by sex | `reports/calibration_dev.json`, `reports/figures/` |
| `python3 -m src.fairness audit` | Fairness audit by sex + mitigation (development set) | `reports/fairness_dev.json`, `reports/figures/` |
| `mlflow ui --backend-store-uri sqlite:///mlflow.db` | Browse every Optuna trial | opens in the browser |

**Notebooks:** open them in VS Code and click **Run All**. They only read the
files in `tuning/results/` and `reports/`, so they run in seconds.

**Which tuned version is used:** `src/evaluate.py` and `src/fairness.py` read
`tuning/results/<version>/best_params.json`, with the version set in
`src/config.py` (`BEST_PARAMS_VERSION`, currently `v2`). That file must contain
all three models.

**Tuning a new model or version:** copy `tuning/random_forest.py` for the
diagnosis of your model, edit its search space in `suggest_params()` in
`tuning/tune.py`, then run `tune.py` with a new `--version`. Each model only
updates its own entry, so earlier versions are never overwritten.

---

## Using the code in your own script

Do not copy the cleaning code or preprocess the data yourself: import it.

```python
from src.data import load_data, get_cv, FEATURES, SENSITIVE, TARGET
from src.pipeline import build_pipeline
from src.models import build_model

dev_df, test_df = load_data()            # 39,032 / 9,758 rows; do not touch test_df until the end
X, y = dev_df[FEATURES], dev_df[TARGET]
cv = get_cv()                            # 5-fold StratifiedKFold, random_state=42

pipe = build_pipeline(build_model("Random Forest", {"min_samples_leaf": 5}))
pipe.fit(X, y)
```

`build_pipeline(model)` = feature engineering (`is_married`, `has_degree`,
`part_time`, `overtime`, `age_squared`) -> preprocessing (numbers: median, then
scaling; categories: "Unknown", then one-hot) -> your model.

---

## Key decisions

| Item | Decision |
|---|---|
| Target | `high_income` = 1 if income > $50K (rare class, 24%) |
| Dropped | `capital-gain`, `capital-loss` (leakage), `education` (duplicate), `relationship` (reveals sex), `fnlwgt` (sampling weight) |
| Kept aside for the audit | `sex`, `race`, `native-country`: never model inputs |
| Split | 80% development / 20% test, stratified on the target, `random_state=42` |
| Cross-validation | 5-fold `StratifiedKFold` on the development set |
| Metrics | PR-AUC (ranking, floor 0.239) and ROC-AUC; Brier and ECE (probabilities); by sex: selection rate, TPR, FPR, precision |
| Fairness definition | Equalized odds (primary), four-fifths rule (legal screen) |
| Final logistic regression | Tuned v2 (C = 6.80, L2): same as v1, 0.708 PR-AUC, Brier 0.111 (F40) |
| Final random forest | Tuned v2: same ranking as v1 (0.715 PR-AUC), Brier 0.110 instead of 0.136 (F36) |
| Final LightGBM | Tuned v2: same ranking as v1 (0.719 PR-AUC), smaller gap, Brier 0.1096 instead of 0.1112 (F39) |
| Candidate final model | LightGBM + Platt scaling; logistic regression as the transparent benchmark |

---

## Common problems

| Message | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the project root with `python3 -m ...`, not the Run button |
| `Library not loaded: @rpath/libomp.dylib` | `brew install libomp` (Setup) |
| `zsh: command not found: brew` | Run the three `brew shellenv` lines of the Setup |
| `command not found: python` | Use `python3` |
| `pip` fails on a path with spaces | Activate the environment first: `source .venv/bin/activate` |
| `fatal: Need to specify how to reconcile divergent branches` | Once: `git config pull.rebase false`, then `git pull --no-edit origin <branch>` and push |
| A notebook does not show the latest results | Close its tab without saving, reopen it, Run All |

---

## Git workflow

- Each member works on their own branch (`manel/dev`, `soumaya/dev`, `hadiqa/dev`).
- Changes reach `main` through a pull request.
- Never commit the data (`*.csv` is ignored, except the small result tables in `tuning/results/`).
