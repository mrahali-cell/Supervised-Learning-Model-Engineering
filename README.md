# Supervised-Learning-Model-Engineering

MSA-DATB09-04 — Supervised Learning & Model Engineering (Albert School, Fall 2026).
Binary classification on the UCI Adult / Census Income dataset (1994 U.S. Census):
predict whether a person's income is ≤ $50K.

## Project structure

```
notebooks/
    01_eda.ipynb      EDA, problem framing, leakage audit (supplementary)
src/
    data.py           load, clean, split
    pipeline.py       preprocessing + model Pipeline
    train.py          training, Optuna tuning, MLflow tracking
    evaluate.py       metrics, calibration, SHAP
    fairness.py       fairness audit
requirements.txt
```

## Setup

```
pip install -r requirements.txt
python -m src.data      # checks the data loads, cleans and splits correctly
```
