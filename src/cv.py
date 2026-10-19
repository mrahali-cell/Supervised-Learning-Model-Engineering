"""Cross-validation helpers (requirement 5). Library only: nothing runs here.

- cross_validate_pipeline: out-of-fold probabilities + per-fold metrics
  (used by tuning/tune.py, src/evaluate.py, src/fairness.py);
- train_val_scores: train vs validation scores, to diagnose over/underfitting
  (used by the per-model diagnosis scripts in tuning/);
- best_f1_point: the operating point that maximises F1.

The Pipeline is re-fitted inside each fold, so there is no leakage (S5).
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import cross_validate


def cross_validate_pipeline(pipe, X, y, cv):
    """Out-of-fold probabilities + per-fold ranking metrics (fit inside each fold)."""
    oof = np.zeros(len(y))
    folds = []
    for train_idx, val_idx in cv.split(X, y):
        fitted = clone(pipe).fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = fitted.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = proba
        y_val = y.iloc[val_idx]
        folds.append({"roc_auc": roc_auc_score(y_val, proba),
                      "pr_auc": average_precision_score(y_val, proba),
                      "brier": brier_score_loss(y_val, proba)})
    return oof, pd.DataFrame(folds)


SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "brier": "neg_brier_score"}


def train_val_scores(pipe, X, y, cv):
    """Mean train and validation scores over the CV folds (S5 diagnosis).

    A big train - validation gap = overfitting (variance);
    both scores low and close = underfitting (bias).
    """
    start = time.time()
    s = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, return_train_score=True)
    return {
        "train PR-AUC": s["train_pr_auc"].mean(),
        "val PR-AUC": s["test_pr_auc"].mean(),
        "val PR-AUC std": s["test_pr_auc"].std(),
        "gap PR-AUC": s["train_pr_auc"].mean() - s["test_pr_auc"].mean(),
        "train ROC-AUC": s["train_roc_auc"].mean(),
        "val ROC-AUC": s["test_roc_auc"].mean(),
        "train Brier": -s["train_brier"].mean(),
        "val Brier": -s["test_brier"].mean(),
        "seconds": time.time() - start,
    }


def best_f1_point(y_true, scores):
    """Operating point that maximises F1 (a neutral choice: no single costly error).

    Returns a dict with threshold, precision, recall, F1 and false positive rate.
    The threshold is picked on the same out-of-fold scores it is measured on, so
    the numbers are slightly optimistic; they are only for comparing models.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best = int(np.argmax(f1))
    flagged = scores >= thresholds[best]
    negatives = (y_true == 0)
    return {"threshold": thresholds[best], "precision": precision[best],
            "recall": recall[best], "f1": f1[best],
            "fpr": (flagged & negatives).sum() / negatives.sum()}
