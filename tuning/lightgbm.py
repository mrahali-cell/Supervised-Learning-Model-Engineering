"""LightGBM: diagnosis before tuning (Hadiqa).

To write: same structure as tuning/random_forest.py (Manel), applied to LightGBM,
the team's candidate final model.

Stage `baseline`: Dummy, LightGBM defaults, tuned v1 (and v2 once it exists)
                  -> train vs validation PR-AUC, gap, ROC-AUC, Brier.
Stage `curves`  : one validation curve per course hyperparameter
                  (learning_rate x n_estimators, num_leaves, max_depth,
                  min_child_samples, subsample, colsample_bytree, reg_alpha,
                  reg_lambda, scale_pos_weight).

Development set only, 5-fold CV; the test set is NOT used.
Results go to: tuning/results/lightgbm/ (baseline.csv, curves.csv, curves/*.png),
read by notebooks/04_lightgbm.ipynb.

Then: edit the LightGBM search space in suggest_params() in tuning/tune.py and run
    python3 -m tuning.tune --version v2 --model lgbm

Run from the project root (never with the VS Code "Run" button):
    python3 -m tuning.lightgbm baseline
    python3 -m tuning.lightgbm curves
"""
