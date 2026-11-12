"""Shared helpers: covariates, metrics, bootstrap, model fitting and plotting."""
import os

import matplotlib

matplotlib.use("Agg")  # figures are saved to disk, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

H = 60  # evaluation horizon in months (5 years)
FIG_DIR, RES_DIR = "figures", "results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

CLINICAL = [
    "age_at_diagnosis",
    "tumor_size",
    "neoplasm_histologic_grade",
    "lymph_nodes_examined_positive",
    "er_status",
    "her2_status",
    "chemotherapy",
    "hormone_therapy",
    "radio_therapy",
]
BASELINE = ["age_at_diagnosis", "tumor_size", "neoplasm_histologic_grade"]
CONTINUOUS = ["age_at_diagnosis", "tumor_size", "lymph_nodes_examined_positive"]


# ---------------------------------------------------------------- data preparation
def prepare(d, extra=()):
    """Complete-case table `m` and one-hot encoded clinical matrix `X` (with time and event)."""
    m = d[CLINICAL + list(extra) + ["time", "event", "cohort"]].dropna().copy()
    cat = [c for c in CLINICAL if not pd.api.types.is_numeric_dtype(m[c])]  # text columns (any pandas version)
    X = pd.get_dummies(m[CLINICAL + ["time", "event"]], columns=cat, drop_first=True, dtype=float)
    return m, X


def covariate_columns(X):
    return [c for c in X.columns if c not in ("time", "event")]


# ---------------------------------------------------------------- metrics
def trunc(t, e, h=H):
    """Administrative censoring at horizon h."""
    return np.minimum(t, h), ((e == 1) & (t <= h)).astype(int)


def c_index_h(t, e, s, h=H):
    """C-index truncated at h. `s` is a survival probability (higher = better prognosis)."""
    t_, e_ = trunc(t, e, h)
    return concordance_index(t_, s, e_)


def km_risk(t, e, h=H):
    """Kaplan-Meier risk at horizon h with 95% CI: (risk, low, high)."""
    km = KaplanMeierFitter().fit(t, e)
    ci = km.confidence_interval_survival_function_
    ci = ci[ci.index <= h].iloc[-1]
    return 1 - float(km.predict(h)), 1 - ci.iloc[1], 1 - ci.iloc[0]


def boot_ci_c_index(t, e, s, B=1000, seed=0, h=H):
    """Patient-level bootstrap 95% CI of the truncated C-index."""
    rng = np.random.default_rng(seed)
    n = len(t)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        try:
            vals.append(c_index_h(t[i], e[i], s[i], h))
        except ZeroDivisionError:
            pass
    return np.percentile(vals, [2.5, 97.5])


def survival_array(t, e):
    from sksurv.util import Surv

    return Surv.from_arrays(event=np.asarray(e).astype(bool), time=np.asarray(t, dtype=float))


def brier_at(y, s, h=H):
    """IPCW Brier score at horizon h; censoring distribution estimated from y."""
    from sksurv.metrics import brier_score

    return float(brier_score(y, y, np.asarray(s)[:, None], [h])[1][0])


# ---------------------------------------------------------------- modelling
def fit_predict_cox(X, tr_idx, te_idx, feats, strata=None, h=H):
    """Fit a Cox model on rows `tr_idx`; return S(h) for rows `te_idx` (aligned by patient label)."""
    cols = list(feats) + ["time", "event"]
    cph = CoxPHFitter().fit(X.iloc[tr_idx][cols], duration_col="time", event_col="event", strata=strata)
    s = cph.predict_survival_function(X.iloc[te_idx][list(feats)], times=[h]).iloc[0]
    return s.reindex(X.index[te_idx]).values  # reindex: stratified models return columns grouped by stratum


# ---------------------------------------------------------------- output
def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def savecsv(df, name, index=False):
    path = os.path.join(RES_DIR, name)
    df.to_csv(path, index=index)
    print("saved", path)


def legend_below(ax, ncol=1, fontsize=9):
    """Place the legend under the axes so that it never hides data points."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, frameon=False, fontsize=fontsize)


def ypos(n):
    return np.arange(n)[::-1].astype(float)


def forest(ax, labels, est, lo, hi, color="tab:blue", offset=0.0, label=None, marker="o"):
    """Horizontal point-and-interval plot, first label on top."""
    est, lo, hi = np.asarray(est, float), np.asarray(lo, float), np.asarray(hi, float)
    y = ypos(len(labels)) + offset
    xerr = [np.clip(est - lo, 0, None), np.clip(hi - est, 0, None)]
    ax.errorbar(est, y, xerr=xerr, fmt=marker, color=color, capsize=3, label=label)
    ax.set_yticks(ypos(len(labels)))
    ax.set_yticklabels(labels)


def plot_selected_genes(sel, coefs, name="fig12_expr_selected_genes.png"):
    """Genes selected per outer fold (left) and most frequently selected genes (right).
    The right panel shows model G1, and also G2 when it selected any gene."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    xs = np.arange(len(sel))
    ax.bar(xs - 0.2, sel["genes_G1"], width=0.4, label="G1 (expression only)", color="tab:gray")
    ax.bar(xs + 0.2, sel["genes_G2"], width=0.4, label="G2 (clinical + expression)", color="tab:red")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Cohort {c}" for c in sel["held_out_cohort"]])
    ax.set_ylabel("Number of genes with non-zero coefficient")
    ax.set_title("Genes selected when each cohort is held out")
    ax.set_ylim(0, max(1, sel[["genes_G1", "genes_G2"]].values.max()) * 1.3)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=8)
    ax = axes[1]
    n_folds = len(sel)
    models = [m for m in ("G1", "G2") if (coefs["model"] == m).any()]
    note = "" if "G2" in models else "\n(model G2 selected no genes in any fold)"
    if models:
        m = models[0] if len(models) == 1 else "G1"
        sub = coefs[coefs["model"] == m]
        freq = sub.groupby("gene").agg(folds=("coefficient", "size"), mean_abs=("coefficient", lambda v: np.abs(v).mean()))
        top = freq.sort_values(["folds", "mean_abs"], ascending=False).head(15).iloc[::-1]
        ax.barh(top.index, top["folds"], color="tab:gray" if m == "G1" else "tab:red")
        ax.set_xlim(0, n_folds + 0.2)
        ax.set_xticks(range(n_folds + 1))
        ax.set_xlabel(f"Number of outer folds (of {n_folds}) in which the gene was selected (model {m})")
        ax.set_title(f"Most frequently selected genes, model {m}{note}")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no genes selected by G1 or G2", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    savefig(fig, name)