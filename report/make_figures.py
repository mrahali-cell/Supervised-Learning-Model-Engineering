"""Figures made for the report from saved results only (nothing is trained here).

- figures/validation_curves.pdf: one validation curve per model family, read from
  tuning/results/<model>/curves.csv (written by `python -m tuning.<model> curves`).

Run from the project root:
    python -m report.make_figures
"""

import matplotlib
matplotlib.use("Agg")  # figures are only saved, never shown
import matplotlib.pyplot as plt
import pandas as pd

from src.config import ROOT, TUNING_RESULTS_DIR

OUT = ROOT / "report" / "figures"


def validation_curves():
    lr = pd.read_csv(TUNING_RESULTS_DIR / "logistic_regression" / "curves.csv")
    rf = pd.read_csv(TUNING_RESULTS_DIR / "random_forest" / "curves.csv")
    lgbm = pd.read_csv(TUNING_RESULTS_DIR / "lightgbm" / "curves.csv")
    plt.rcParams.update({"font.size": 8})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(11, 3.1))

    # (a) Logistic regression: C = 1/lambda, L2 and L1
    for name, style, label in (("C_L2", "-", "L2"), ("C_L1", ":", "L1")):
        g = lr[lr["curve"] == name].assign(C=lambda d: d["value"].astype(float))
        a.errorbar(g["C"], g["val PR-AUC"], yerr=g["val PR-AUC std"], fmt="o" + style, ms=3,
                   capsize=2, color="tab:blue", label=f"{label} validation")
        if name == "C_L2":
            a.plot(g["C"], g["train PR-AUC"], "--", color="tab:grey", label="L2 train")
    a.set(xscale="log", xlabel="C (log scale; small C = strong penalty)",
          ylabel="PR-AUC (5-fold CV)", title="(a) Logistic regression")

    # (b) Random forest: minimum leaf size, the other settings at sklearn defaults
    g = rf[rf["param"] == "min_samples_leaf"].assign(leaf=lambda d: d["value"].astype(int))
    b.plot(g["leaf"], g["train PR-AUC"], "--", color="tab:grey", label="train")
    b.errorbar(g["leaf"], g["val PR-AUC"], yerr=g["val PR-AUC std"], fmt="o-", ms=3,
               capsize=2, color="tab:green", label="validation")
    b.set(xscale="log", xlabel="min_samples_leaf (people per leaf, log scale)",
          title="(b) Random forest (200 trees)")

    # (c) LightGBM: learning rate x number of trees, validation only
    g = lgbm[lgbm["param"] == "learning_rate"]
    for rate, color in zip(sorted(g["learning_rate"].unique()),
                           ["tab:purple", "tab:blue", "tab:orange", "tab:red"]):
        r = g[g["learning_rate"] == rate].sort_values("n_estimators")
        c.plot(r["n_estimators"], r["val PR-AUC"], "o-", ms=3, color=color,
               label=f"rate {rate:g}")
    c.set(xscale="log", xlabel="number of trees (log scale)", title="(c) LightGBM: validation")

    for ax in (a, b, c):
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "validation_curves.pdf")
    plt.close(fig)


if __name__ == "__main__":
    validation_curves()
    print(f"Saved {OUT / 'validation_curves.pdf'}")
