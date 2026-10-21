"""Logistic regression: diagnosis before tuning.

How a logistic regression learns: it gives each column a weight, adds them up
and turns the sum into a probability with the sigmoid. Training minimises ONE
global loss on the training folds (S2):
    C x (log-loss summed over the training rows) + penalty on the weights,
so C = 1 / lambda: a small C means a strong penalty.
- L2 (ridge): sum of squared weights, shrinks every weight a little;
- L1 (lasso): sum of absolute weights, can set weights to exactly 0;
- Elastic Net: l1_ratio x L1 + (1 - l1_ratio) x L2 (l1_ratio 0 = L2, 1 = L1).
Strong penalty = simpler model (more bias); weak penalty = closer to the
unpenalised fit (more variance). The model is linear in its columns, so a curve
or an interaction exists only if it is created as a column (src/pipeline.py,
and the `interactions` stage below).

Stage `baseline`    : Dummy, default LR (6 inputs, then + engineered features),
                      tuned v1 -> train vs validation PR-AUC, gap, ROC-AUC, Brier.
Stage `curves`      : validation curves of C from 0.0001 to 100, for L2 and L1
                      separately, and of class_weight (S5).
Stage `lasso`       : Elastic Net for several l1_ratio values, and which columns
                      L1 sets to exactly zero, from which C.
Stage `interactions`: three product columns, added one at a time to tuned v1,
                      kept only if they pass the rule fixed before the run.
Stage `search`      : optional, grid vs random vs Optuna TPE with the same budget
                      on the v1 search space (does NOT write best_params.json).

Development set only, 5-fold CV; the test set is NOT used.
Results: tuning/results/logistic_regression/ (read by notebooks/02_logistic_regression.ipynb).

Run from the project root:
    python3 -m tuning.logistic_regression baseline
    python3 -m tuning.logistic_regression curves                 # all curves
    python3 -m tuning.logistic_regression curves --param C_L1    # one curve
    python3 -m tuning.logistic_regression lasso
    python3 -m tuning.logistic_regression interactions
    python3 -m tuning.logistic_regression search
"""

import argparse
import json
import time
import warnings

import matplotlib
matplotlib.use("Agg")  # figures are only saved, never shown
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import TUNING_RESULTS_DIR
from src.cv import SCORING
from src.data import FEATURES, RANDOM_STATE, TARGET, get_cv, load_data
from src.models import build_model, get_models
from src.pipeline import add_features, build_pipeline, build_preprocessor

OUT = TUNING_RESULTS_DIR / "logistic_regression"

# Validation curves: ONE setting changes, the others stay at the sklearn defaults
# (C = 1, L2, no class weights), with the solver used by the tuning (src/models.py).
# liblinear shuffles the data: without a seed, L1 can zero a different one of two
# duplicate columns (e.g. is_married and marital-status_Married-civ-spouse) per run.
BASE = dict(C=1.0, l1_ratio=0.0, solver="liblinear", max_iter=2000,
            random_state=RANDOM_STATE)
C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
CURVES = {  # name: (setting that changes, its values, settings fixed for this curve)
    "C_L2": ("C", C_GRID, {"l1_ratio": 0.0}),  # ridge
    "C_L1": ("C", C_GRID, {"l1_ratio": 1.0}),  # lasso
    "class_weight": ("class_weight", [None, "balanced"], {}),  # watch the Brier score
}

# Elastic Net needs the saga solver (liblinear does pure L1 or pure L2 only)
EN_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]
EN_C = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100]
# Lasso path: pure L1 with liblinear (as in the curves), 4 values of C per decade
LASSO_C = np.logspace(-4, 2, 25)

# Interactions, tested one at a time on top of tuned v1.
# Rule fixed BEFORE the run (2026-10-20): an interaction "helps" only if it adds
# at least 0.001 mean validation PR-AUC AND is better in at least 4 of the 5
# folds (same folds, so a paired comparison). Otherwise it is not kept.
MIN_GAIN = 0.001
MIN_FOLDS = 4
INTERACTIONS = {
    "education x hours": ("education-num", "hours-per-week"),
    "age x education": ("age", "education-num"),
    "married x hours": ("is_married", "hours-per-week"),
}

# Search methods: the same budget for each, on the v1 space of tuning/tune.py
SEARCH_BUDGET = 20
SEARCH_GRID = {"C": [float(c) for c in np.logspace(-3, 2, 5)],
               "penalty": ["l1", "l2"],
               "class_weight": [None, "balanced"]}  # 5 x 2 x 2 = 20 = the budget


def load_best(version):
    """Tuned LR parameters of a version, or None if that version has no LR yet."""
    path = TUNING_RESULTS_DIR / version / "best_params.json"
    return json.loads(path.read_text()).get("Logistic Regression") if path.exists() else None


def cv_scores(pipe, X, y, cv, **kwargs):
    """src.cv.train_val_scores, plus the fold std of every validation metric and
    the number of folds whose solver did not converge. Also returns the raw
    cross_validate output (per-fold scores, and fitted Pipelines if asked)."""
    start = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        s = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, return_train_score=True,
                           **kwargs)
    others = {str(w.message): w for w in caught if not issubclass(w.category, ConvergenceWarning)}
    for w in others.values():  # show every other warning, once
        warnings.showwarning(w.message, w.category, w.filename, w.lineno)
    r = {
        "train PR-AUC": s["train_pr_auc"].mean(),
        "val PR-AUC": s["test_pr_auc"].mean(),
        "val PR-AUC std": s["test_pr_auc"].std(),
        "gap PR-AUC": s["train_pr_auc"].mean() - s["test_pr_auc"].mean(),
        "train ROC-AUC": s["train_roc_auc"].mean(),
        "val ROC-AUC": s["test_roc_auc"].mean(),
        "val ROC-AUC std": s["test_roc_auc"].std(),
        "train Brier": -s["train_brier"].mean(),
        "val Brier": -s["test_brier"].mean(),
        "val Brier std": s["test_brier"].std(),
        "not converged": sum(issubclass(w.category, ConvergenceWarning) for w in caught),
        "seconds": time.time() - start,
    }
    return r, s


def coefs(fitted):
    """Weights of a fitted Pipeline, indexed by column name after preprocessing."""
    names = fitted.named_steps["preprocess"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]  # drop the num__ / cat__ / bin__ prefix
    return pd.Series(fitted.named_steps["model"].coef_[0], index=names)


def show(label, r):
    warning = f" | NOT CONVERGED in {r['not converged']} folds" if r["not converged"] else ""
    print(f"{label:<48} train {r['train PR-AUC']:.3f} | val {r['val PR-AUC']:.3f} "
          f"± {r['val PR-AUC std']:.3f} | gap {r['gap PR-AUC']:.3f} | "
          f"ROC val {r['val ROC-AUC']:.3f} | Brier {r['val Brier']:.3f} | "
          f"{r['seconds']:.0f}s{warning}", flush=True)


def run_baseline(X, y, cv):
    v1 = load_best("v1")
    experiments = {  # name: (model, engineered features?)
        "Dummy (predicts the base rate)": (DummyClassifier(strategy="prior"), False),
        "LR defaults, 6 inputs only (C=1, L2)": (get_models()["Logistic Regression"], False),
        "LR defaults + engineered features (C=1, L2)": (
            get_models()["Logistic Regression"], True),
        f"LR tuned v1 (C={v1['C']:.2f}, {v1['penalty'].upper()})": (
            build_model("Logistic Regression", v1), True),
    }
    v2 = load_best("v2")
    if v2:  # add tuning v2 to the comparison once it exists
        experiments["LR tuned v2"] = (build_model("Logistic Regression", v2), True)
    rows = []
    for name, (model, engineered) in experiments.items():
        r, _ = cv_scores(build_pipeline(model, engineered=engineered), X, y, cv)
        show(name, r)
        rows.append({"experiment": name, **r})
    result = pd.DataFrame(rows).round(4)
    result.to_csv(OUT / "baseline.csv", index=False)
    print(f"\nSaved {OUT / 'baseline.csv'}")
    return result


def run_curve(name, X, y, cv):
    param, values, fixed = CURVES[name]
    rows = []
    print(f"\nValidation curve: {name}", flush=True)
    for value in values:
        model = LogisticRegression(**{**BASE, **fixed, param: value})
        r, _ = cv_scores(build_pipeline(model), X, y, cv)
        show(f"  {name}: {param} = {value}", r)
        lam = {"lambda = 1/C": 1 / value} if param == "C" else {}
        rows.append({"curve": name, "param": param, "value": str(value), **lam, **r})
    result = pd.DataFrame(rows).round(4)

    # Figure: train vs validation PR-AUC (left), validation Brier (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    x = values if param == "C" else range(len(values))
    ax1.plot(x, result["train PR-AUC"], "o-", label="train")
    ax1.errorbar(x, result["val PR-AUC"], yerr=result["val PR-AUC std"],
                 fmt="o-", capsize=3, label="validation")
    ax1.set_title(f"PR-AUC vs {param} ({name})")
    ax1.legend()
    ax2.plot(x, result["val Brier"], "o-", color="tab:red")
    ax2.set_title("Validation Brier (lower = better probabilities)")
    for ax in (ax1, ax2):
        if param == "C":
            ax.set_xscale("log")
            ax.axvline(1, color="grey", ls=":", lw=1)  # sklearn default C = 1
            ax.set_xlabel("C (log scale; small C = strong penalty, lambda = 1/C)")
        else:
            ax.set_xticks(list(x), result["value"])
            ax.set_xlabel(param)
    fig.tight_layout()
    (OUT / "curves").mkdir(exist_ok=True)
    fig.savefig(OUT / "curves" / f"{name}.png", dpi=120)
    plt.close(fig)
    return result


def run_curves(names, X, y, cv):
    path = OUT / "curves.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    for name in names:
        result = run_curve(name, X, y, cv)
        if not old.empty:
            old = old[old["curve"] != name]  # replace only this curve
        old = pd.concat([old, result], ignore_index=True)
        old.to_csv(path, index=False)
    print(f"\nSaved {path} and the figures in {OUT / 'curves'}")


def run_elastic_net(X, y, cv):
    rows = []
    print("\nElastic Net (saga solver): every l1_ratio x C", flush=True)
    for ratio in EN_RATIOS:
        for c in EN_C:
            model = LogisticRegression(C=c, l1_ratio=ratio, solver="saga", max_iter=5000,
                                       random_state=RANDOM_STATE)
            r, s = cv_scores(build_pipeline(model), X, y, cv, return_estimator=True)
            show(f"  l1_ratio {ratio}, C = {c}", r)
            weights = [coefs(p) for p in s["estimator"]]
            rows.append({"l1_ratio": ratio, "C": c, "columns": len(weights[0]),
                         "zero weights (mean of folds)": np.mean([(w == 0).sum() for w in weights]),
                         **r})
    result = pd.DataFrame(rows).round(4)
    result.to_csv(OUT / "elastic_net.csv", index=False)

    # Figure: validation PR-AUC (left) and number of zero weights (right) vs C
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    for ratio, g in result.groupby("l1_ratio"):
        ax1.errorbar(g["C"], g["val PR-AUC"], yerr=g["val PR-AUC std"], fmt="o-",
                     capsize=3, label=f"l1_ratio {ratio}")
        ax2.plot(g["C"], g["zero weights (mean of folds)"], "o-", label=f"l1_ratio {ratio}")
    ax1.set_title("Validation PR-AUC (0 = L2, 1 = L1)")
    ax2.set_title(f"Weights set to exactly 0 (of {result['columns'].iloc[0]})")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("C (log scale; small C = strong penalty)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "lasso" / "elastic_net.png", dpi=120)
    plt.close(fig)
    return result


def run_lasso_path(X, y, cv):
    rows, path = [], []
    print("\nLasso path (pure L1, liblinear): which weights are exactly 0", flush=True)
    for c in LASSO_C:
        model = LogisticRegression(**{**BASE, "C": c, "l1_ratio": 1.0})
        r, s = cv_scores(build_pipeline(model), X, y, cv, return_estimator=True)
        show(f"  C = {c:.4g}", r)
        w = pd.concat([coefs(p) for p in s["estimator"]], axis=1)  # columns x folds
        path.append(pd.DataFrame({"C": c, "column": w.index, "mean weight": w.mean(axis=1),
                                  "folds at zero": (w == 0).sum(axis=1)}))
        rows.append({"C": c, "non-zero weights (mean of folds)": (w != 0).sum().mean(), **r})
    scores = pd.DataFrame(rows).round(4)
    path = pd.concat(path, ignore_index=True).round({"mean weight": 4})
    scores.to_csv(OUT / "lasso_scores.csv", index=False)
    path.to_csv(OUT / "lasso_path.csv", index=False)

    # Per column: zero in all folds for every C up to this value = "out of the model"
    n_folds = cv.get_n_splits()
    summary = []
    for column, g in path.groupby("column", sort=False):
        g = g.sort_values("C")
        out = (g["folds at zero"] == n_folds).cumprod().astype(bool)
        at_1 = g[np.isclose(g["C"], 1.0)].iloc[0]
        summary.append({"column": column,
                        "zero in all folds for every C up to": g["C"][out].max(),
                        "folds at zero, C = 1": int(at_1["folds at zero"]),
                        "folds at zero, C = 100": int(g["folds at zero"].iloc[-1]),
                        "mean weight, C = 1": at_1["mean weight"]})
    summary = pd.DataFrame(summary).sort_values("zero in all folds for every C up to")
    summary.to_csv(OUT / "lasso_zeros.csv", index=False)

    # Figure: the path of every weight (left); columns kept and PR-AUC (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    wide = path.pivot(index="C", columns="column", values="mean weight")
    top = list(wide.loc[np.isclose(wide.index, 1.0)].iloc[0].abs().nlargest(8).index)
    for column in wide.columns:  # the 8 biggest weights at C = 1 in colour, the rest grey
        if column in top:
            ax1.plot(wide.index, wide[column], lw=1.8, color=plt.cm.tab10(top.index(column)),
                     label=column)
        else:
            ax1.plot(wide.index, wide[column], lw=0.6, color="lightgrey")
    ax1.set_title("Lasso path: mean weight over the folds")
    ax1.legend(fontsize=7)
    ax2.plot(scores["C"], scores["non-zero weights (mean of folds)"], "o-", color="tab:grey")
    ax2.set_ylabel("non-zero weights")
    ax3 = ax2.twinx()
    ax3.plot(scores["C"], scores["val PR-AUC"], "o-", color="tab:orange")
    ax3.set_ylabel("validation PR-AUC", color="tab:orange")
    ax2.set_title("Columns kept (grey) and validation PR-AUC (orange)")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("C (log scale; small C = strong penalty)")
    fig.tight_layout()
    fig.savefig(OUT / "lasso" / "lasso_path.png", dpi=120)
    plt.close(fig)
    return summary


def run_lasso(X, y, cv):
    (OUT / "lasso").mkdir(exist_ok=True)
    run_lasso_path(X, y, cv)
    run_elastic_net(X, y, cv)
    print(f"\nSaved lasso_scores.csv, lasso_path.csv, lasso_zeros.csv, elastic_net.csv "
          f"and the figures in {OUT / 'lasso'}")


def add_interactions(X, pairs):
    """Product columns, e.g. education-num x hours-per-week (fixed rule, leakage-safe)."""
    X = X.copy()
    for name, (a, b) in pairs.items():
        X[name] = X[a] * X[b]
    return X


def build_interaction_pipeline(model, pairs):
    """build_pipeline() plus product columns, imputed and scaled like the other
    numbers, inside the Pipeline (so fitted on the training folds only)."""
    preprocess = build_preprocessor(engineered=True)
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    preprocess.transformers.append(("inter", numeric, list(pairs)))
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("interactions", FunctionTransformer(add_interactions, kw_args={"pairs": pairs})),
        ("preprocess", preprocess),
        ("model", model),
    ])


def run_interactions(X, y, cv):
    v1 = load_best("v1")
    ref, ref_s = cv_scores(build_pipeline(build_model("Logistic Regression", v1)), X, y, cv)
    show("Reference: LR tuned v1, no interaction", ref)
    rows = [{"experiment": "reference: LR tuned v1", **ref}]
    folds = pd.DataFrame({"fold": range(1, cv.get_n_splits() + 1),
                          "reference": ref_s["test_pr_auc"]})

    def compare(name, pairs):
        pipe = build_interaction_pipeline(build_model("Logistic Regression", v1), pairs)
        r, s = cv_scores(pipe, X, y, cv)
        show(name, r)
        delta = s["test_pr_auc"] - ref_s["test_pr_auc"]  # same folds: paired
        folds[name] = s["test_pr_auc"]
        weights = coefs(pipe.fit(X, y))  # whole development set, for the sign only
        row = {"experiment": name, **r,
               "gain PR-AUC": delta.mean(),
               "folds better": int((delta > 0).sum()),
               "gain ROC-AUC": r["val ROC-AUC"] - ref["val ROC-AUC"],
               "change Brier": r["val Brier"] - ref["val Brier"],
               "weight (scaled)": "; ".join(f"{weights[p]:+.3f}" for p in pairs),
               "passes rule": delta.mean() >= MIN_GAIN and (delta > 0).sum() >= MIN_FOLDS}
        rows.append(row)
        return row["passes rule"]

    kept = {name: pair for name, pair in INTERACTIONS.items()
            if compare(name, {name: pair})}
    if len(kept) > 1:  # the ones that passed, together
        compare("kept together: " + " + ".join(kept), kept)
    result = pd.DataFrame(rows).round(4)
    result.to_csv(OUT / "interactions.csv", index=False)
    folds.round(4).to_csv(OUT / "interactions_folds.csv", index=False)

    # Figure: per-fold gain of each interaction vs the rule
    fig, ax = plt.subplots(figsize=(8, 3.5))
    tested = [c for c in folds.columns if c not in ("fold", "reference")]
    width = 0.8 / len(tested)
    for i, name in enumerate(tested):
        ax.bar(folds["fold"] + (i - (len(tested) - 1) / 2) * width,
               folds[name] - folds["reference"], width, label=name)
    ax.axhline(MIN_GAIN, color="black", ls="--", lw=1, label=f"rule: +{MIN_GAIN}")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("fold")
    ax.set_ylabel("validation PR-AUC gain")
    ax.set_title(f"Gain over tuned v1, per fold (helps = mean >= {MIN_GAIN} "
                 f"and better in >= {MIN_FOLDS} folds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "interactions.png", dpi=120)
    plt.close(fig)
    print(f"\nKept: {list(kept) or 'none'}. Saved {OUT / 'interactions.csv'}")
    return result


def run_search(X, y, cv):
    """Same objective, same budget, three ways of choosing the trials (S5).
    Optuna is only the engine here: GridSampler = grid search, RandomSampler =
    random search, TPESampler = Bayesian (10 random trials, then guided)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    samplers = {
        "grid": optuna.samplers.GridSampler(SEARCH_GRID, seed=RANDOM_STATE),
        "random": optuna.samplers.RandomSampler(seed=RANDOM_STATE),
        "Optuna TPE": optuna.samplers.TPESampler(seed=RANDOM_STATE),
    }

    def objective(trial):  # the v1 logistic regression space of tuning/tune.py
        params = {"C": trial.suggest_float("C", 1e-3, 1e2, log=True),
                  "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                  "class_weight": trial.suggest_categorical("class_weight",
                                                            [None, "balanced"])}
        r, _ = cv_scores(build_pipeline(build_model("Logistic Regression", params)), X, y, cv)
        for key in ("val PR-AUC std", "val ROC-AUC", "val Brier", "gap PR-AUC"):
            trial.set_user_attr(key, float(r[key]))
        return r["val PR-AUC"]

    trials, rows = [], []
    for method, sampler in samplers.items():
        print(f"\nSearch: {method} ({SEARCH_BUDGET} trials)", flush=True)
        start = time.time()
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=SEARCH_BUDGET)
        best = study.best_trial
        history = study.trials_dataframe()
        history["best so far"] = history["value"].cummax()
        history.insert(0, "method", method)
        trials.append(history)
        rows.append({"method": method, "trials": len(study.trials),
                     "best val PR-AUC": best.value,
                     "best val PR-AUC std": best.user_attrs["val PR-AUC std"],
                     "best val ROC-AUC": best.user_attrs["val ROC-AUC"],
                     "best val Brier": best.user_attrs["val Brier"],
                     "best trial": best.number,
                     "best C": best.params["C"], "best penalty": best.params["penalty"],
                     "best class_weight": str(best.params["class_weight"]),
                     "seconds": time.time() - start})
        print(f"  best {best.value:.4f} at trial {best.number}: {best.params}", flush=True)
    result = pd.DataFrame(rows).round(4)
    result.to_csv(OUT / "search.csv", index=False)
    trials = pd.concat(trials, ignore_index=True)
    trials.to_csv(OUT / "search_trials.csv", index=False)

    # Figure: best validation PR-AUC so far, trial after trial
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for method, g in trials.groupby("method", sort=False):
        ax.plot(g["number"], g["best so far"], "o-", ms=3, label=method)
    ax.set_xlabel("trial")
    ax.set_ylabel("best validation PR-AUC so far")
    ax.set_title(f"Same budget ({SEARCH_BUDGET} trials), v1 search space")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "search.png", dpi=120)
    plt.close(fig)
    print(f"\nSaved {OUT / 'search.csv'} (best_params.json is NOT changed)")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic regression diagnosis")
    parser.add_argument("stage", choices=["baseline", "curves", "lasso", "interactions",
                                          "search"])
    parser.add_argument("--param", choices=list(CURVES), action="append",
                        help="curves stage: one curve (repeatable); default all")
    args = parser.parse_args()

    dev_df, _ = load_data()  # the test set is left untouched
    X, y = dev_df[FEATURES], dev_df[TARGET]
    cv = get_cv()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "baseline":
        run_baseline(X, y, cv)
    elif args.stage == "curves":
        run_curves(args.param or list(CURVES), X, y, cv)
    elif args.stage == "lasso":
        run_lasso(X, y, cv)
    elif args.stage == "interactions":
        run_interactions(X, y, cv)
    else:
        run_search(X, y, cv)
