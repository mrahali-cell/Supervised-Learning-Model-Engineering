"""Preprocessing + model Pipeline (requirements 2, 3, 5).
Impute missing values with "Unknown", one-hot encode the nominal columns,
scale the numeric columns, then attach the model.

Uses the cleaned data and split from src/data.py:
    from src.data import load_data, get_cv, FEATURES, SENSITIVE, TARGET
"""
