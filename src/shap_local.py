"""Local SHAP explanations: why the final model scores individual people as it does (requirement 8).

Complements the global SHAP stage (`python -m src.evaluate shap`) with the same
out-of-fold TreeSHAP on the candidate model (LightGBM, tuned): each person is
explained by the fold model that never saw them. The test set is not used.

Who is explained is fixed by a rule, not chosen by hand: at the best-F1
threshold of the out-of-fold probabilities, the person with the median
probability in each of the four outcomes (true positive, false positive, false
negative, true negative). Each person's score (log-odds) is split exactly into
the base value plus one contribution per model input (its columns added
together), as in the global stage.

Run from the project root:
    python -m src.shap_local
"""

import json

import matplotlib
matplotlib.use("Agg")  # figures are only saved, never shown
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.config import BEST_PARAMS, EXPERIMENT, REPORTS_DIR, TRACKING_URI
from src.cv import best_f1_point
from src.data import FEATURES, TARGET, get_cv, load_data
from src.evaluate import CANDIDATE, FIGURES_DIR, column_input, oof_shap

# outcome: (actual class, flagged by the model)
OUTCOMES = {"true positive": (1, 1), "false positive": (0, 1),
            "false negative": (1, 0), "true negative": (0, 0)}


def pick_people(y, p, threshold):
    """Position of the person with the median probability in each outcome."""
    flagged = (p >= threshold).astype(int)
    picks = {}
    for outcome, (actual, flag) in OUTCOMES.items():
        idx = np.flatnonzero((y == actual) & (flagged == flag))
        ordered = idx[np.argsort(p[idx], kind="stable")]
        picks[outcome] = int(ordered[len(ordered) // 2])
    return picks


def plain(value):
    """JSON-friendly value of one input (numbers, text, or None if missing)."""
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def waterfall(ax, person, threshold):
    """Base value, then each input's contribution (largest first), then the score."""
    contributions = person["contributions"]
    position = person["base_value"]
    cutoff = np.log(threshold / (1 - threshold))
    edges, labels = [position, cutoff], []
    for k, (name, value) in enumerate(contributions.items()):
        ax.barh(k, value, left=position, color="#2166ac" if value > 0 else "#b2182b")
        position += value
        edges.append(position)
        shown = person["inputs"][name]
        labels.append(f"{name} = {'missing' if shown is None else shown}")
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(min(edges) - 0.2, max(edges) + 0.2)
    ax.axvline(person["base_value"], color="#555555", lw=0.8, ls=":")
    ax.axvline(cutoff, color="black", lw=0.8, ls="--")
    ax.plot([person["score"]], [len(labels) - 0.5], marker="v", color="black")
    ax.set_xlabel("score (log-odds): dotted = base value, dashed = threshold", fontsize=8)
    actual = "> $50K" if person["actual"] else "<= $50K"
    ax.set_title(f"{person['outcome']}: {person['sex'].lower()}, earns {actual}\n"
                 f"predicted probability {person['probability']:.2f}", fontsize=9)
    ax.grid(axis="x", alpha=0.25)


def run_local():
    dev_df, _ = load_data()  # the test set is left untouched
    X, y_series = dev_df[FEATURES], dev_df[TARGET]
    y = y_series.to_numpy()
    params = json.loads(BEST_PARAMS.read_text())[CANDIDATE]
    print(f"Out-of-fold TreeSHAP: {CANDIDATE} (tuned)", flush=True)
    values, _, base, columns = oof_shap(X, y_series, params, get_cv())
    score = base + values.sum(axis=1)  # LightGBM's raw score (log-odds), exactly
    p = 1 / (1 + np.exp(-score))
    threshold = float(best_f1_point(y, p)["threshold"])
    by_input = pd.DataFrame(values, columns=columns).T.groupby(
        [column_input(c) for c in columns]).sum().T

    people = []
    for outcome, i in pick_people(y, p, threshold).items():
        row = dev_df.iloc[i]
        contributions = by_input.iloc[i].sort_values(key=np.abs, ascending=False)
        people.append({"outcome": outcome, "row_id": int(dev_df.index[i]), "sex": row["sex"],
                       "actual": int(y[i]), "probability": float(p[i]), "score": float(score[i]),
                       "base_value": float(base[i]),
                       "inputs": {f: plain(row[f]) for f in FEATURES},
                       "contributions": {k: float(v) for k, v in contributions.items()}})
        print(f"\n{outcome}: row {dev_df.index[i]}, {row['sex']}, probability {p[i]:.3f}")
        print("  " + ", ".join(f"{k} {v:+.2f}" for k, v in contributions.items()))

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for ax, person in zip(axes.ravel(), people):
        waterfall(ax, person, threshold)
    fig.suptitle(f"Why {CANDIDATE} (tuned) scores four people as it does: the median person of each "
                 f"outcome at the best-F1 threshold {threshold:.3f} (out-of-fold, development set)\n"
                 "Blue raises the score, red lowers it; the triangle is the final score", fontsize=10)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "shap_local_dev.png", dpi=150)
    plt.close(fig)

    result = {"model": CANDIDATE, "version": BEST_PARAMS.parent.name,
              "units": "log-odds (LightGBM raw score)", "threshold_best_f1": threshold,
              "rule": "median out-of-fold probability within each outcome at the best-F1 threshold",
              "people": people}
    (REPORTS_DIR / "shap_local_dev.json").write_text(json.dumps(result, indent=2))

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="SHAP local (development, out-of-fold)"):
        mlflow.log_params({"model": CANDIDATE, "version": result["version"],
                           "explainer": "TreeSHAP, out-of-fold", "rule": result["rule"]})
        mlflow.log_metric("threshold_best_f1", threshold)
        mlflow.log_artifact(str(FIGURES_DIR / "shap_local_dev.png"), "figures")
        mlflow.log_artifact(str(REPORTS_DIR / "shap_local_dev.json"))
    print(f"\nSaved {FIGURES_DIR / 'shap_local_dev.png'} and {REPORTS_DIR / 'shap_local_dev.json'}")
    return result


if __name__ == "__main__":
    run_local()
