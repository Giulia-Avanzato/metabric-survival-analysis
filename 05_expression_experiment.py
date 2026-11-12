"""Step 5 - Extension experiment: does gene expression add predictive value over the clinical model,
and does it hold up when a whole cohort is held out?

PRE-SPECIFIED HYPOTHESES (copy them into the README BEFORE running the full experiment)
  H1 (primary)   Under leave-one-cohort-out (LOCO) validation, adding mRNA expression to the clinical model
                 increases the pooled 60-month C-index relative to model C. The effect is considered relevant
                 only if the 95% paired-bootstrap CI of the difference excludes 0 AND the difference is >= 0.02.
  H2 (secondary) The benefit of expression is heterogeneous across cohorts, and calibration-in-the-large per
                 cohort does not deteriorate when expression is added.
  A null result is an accepted outcome and is reported as such.

MODELS (all evaluated on a cohort that was excluded from training)
  C   clinical Cox model stratified by ER (same as step 03)
  G0  clinical Cox model, ER as a covariate (isolates the effect of stratification from the effect of genes)
  G1  elastic-net Cox model on expression only
  G2  elastic-net Cox model on clinical variables + expression (clinical variables almost unpenalised)

TUNING: the elastic-net penalty (alpha) is chosen by 5-fold CV *inside* the training cohorts (nested), genes are
standardised on the training data only, l1_ratio is fixed at 0.5.

Usage
  python 05_expression_experiment.py --smoke   quick test (50 genes, 5 alphas); results are meaningless
  python 05_expression_experiment.py           full experiment

Outputs
  figures/fig09_expr_delta_c.png          Differences in C-index (paired bootstrap), pooled and per cohort
  figures/fig10_expr_c_by_cohort.png      C-index of every model, pooled and per cohort
  figures/fig11_expr_calibration.png      Calibration-in-the-large per cohort, model C vs G2
  figures/fig12_expr_selected_genes.png   Number of genes selected and most frequently selected genes
  results/05_expr_loco.csv, 05_expr_calibration.csv, 05_expr_selected.csv, 05_expr_gene_coefficients.csv
"""
import sys
import warnings

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis

from data import load
from utils import (
    H, brier_at, c_index_h, covariate_columns, fit_predict_cox, forest, km_risk, legend_below, prepare,
    plot_selected_genes, savecsv, savefig, survival_array, trunc,
)

import matplotlib.pyplot as plt  # noqa: E402

SMOKE = "--smoke" in sys.argv
B, L1 = 1000, 0.5  # bootstrap resamples, elastic-net mixing parameter
N_ALPHAS = 5 if SMOKE else 15
PF_CLIN = 1e-3  # penalty factor of the clinical variables in G2 (almost unpenalised)


def main():
    warnings.filterwarnings("ignore")
    d = load()
    all_cols = list(d.columns)
    i0 = all_cols.index("death_from_cancer") + 1
    i1 = next(i for i, c in enumerate(all_cols) if c.endswith("_mut"))
    expr = all_cols[i0:i1]
    if SMOKE:
        expr = expr[:50]
    print("Expression genes:", len(expr), "| missing values:", int(d[expr].isna().sum().sum()))

    m, X = prepare(d, extra=expr)
    n = len(m)
    print("Patients (complete cases):", n)
    clin = covariate_columns(X)
    Xg = m[expr].astype(float)
    Xfull = pd.concat([X[clin], Xg], axis=1)
    t_all, e_all, coh = m["time"].values, m["event"].values.astype(int), m["cohort"].values
    y_all = survival_array(t_all, e_all)

    try:
        CoxnetSurvivalAnalysis(penalty_factor=np.ones(3))
        pf = np.r_[np.full(len(clin), PF_CLIN), np.ones(len(expr))]
    except TypeError:
        pf = None
        print("WARNING: this scikit-survival version has no penalty_factor; G2 also penalises clinical variables")

    def coxnet(Xdf, tr_idx, te_idx, pen=None):
        Xtr, Xte, ytr = Xdf.iloc[tr_idx].values, Xdf.iloc[te_idx].values, y_all[tr_idx]

        def make(fit_baseline, alphas=None):
            kw = dict(l1_ratio=L1, fit_baseline_model=fit_baseline, max_iter=10000)
            if pen is not None:
                kw["penalty_factor"] = pen
            if alphas is None:
                kw.update(n_alphas=N_ALPHAS, alpha_min_ratio=0.01)
            else:
                kw["alphas"] = alphas
            return make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(**kw))

        path = make(False).fit(Xtr, ytr)
        alphas = path.named_steps["coxnetsurvivalanalysis"].alphas_
        gcv = GridSearchCV(
            make(True),
            param_grid={"coxnetsurvivalanalysis__alphas": [[a] for a in alphas]},
            cv=KFold(5, shuffle=True, random_state=0),
            error_score=0.5,
            n_jobs=-1,
        )
        gcv.fit(Xtr, ytr)
        best = gcv.best_estimator_
        scaler, cox = best.named_steps["standardscaler"], best.named_steps["coxnetsurvivalanalysis"]
        fns = cox.predict_survival_function(scaler.transform(Xte))
        s60 = np.array([fn(H) for fn in fns])
        return s60, cox.coef_.ravel().copy(), float(cox.alphas_[0])

    # ------------------------------------------------------------ leave-one-cohort-out
    names = ["C", "G0", "G1", "G2"]
    pred = {k: np.full(n, np.nan) for k in names + ["null"]}
    sel_rows, coef_rows = [], []
    for c in sorted(np.unique(coh)):
        te, tr = np.where(coh == c)[0], np.where(coh != c)[0]
        pred["C"][te] = fit_predict_cox(X, tr, te, clin, strata=["er_status_Positive"])
        pred["G0"][te] = fit_predict_cox(X, tr, te, clin)
        pred["G1"][te], coef1, a1 = coxnet(Xg, tr, te)
        pred["G2"][te], coef2, a2 = coxnet(Xfull, tr, te, pf)
        pred["null"][te] = float(KaplanMeierFitter().fit(t_all[tr], e_all[tr]).predict(H))
        g2 = coef2[len(clin):]
        sel_rows.append(dict(held_out_cohort=int(c), n_test=len(te), alpha_G1=a1, genes_G1=int((coef1 != 0).sum()),
                             alpha_G2=a2, genes_G2=int((g2 != 0).sum())))
        for model, coefs in (("G1", coef1), ("G2", g2)):
            for gene, v in zip(expr, coefs):
                if v != 0:
                    coef_rows.append(dict(model=model, held_out_cohort=int(c), gene=gene, coefficient=v))
        print(f"held-out cohort {int(c)} done | genes selected G1/G2: {sel_rows[-1]['genes_G1']}/"
              f"{sel_rows[-1]['genes_G2']}", flush=True)
    assert not any(np.isnan(v).any() for v in pred.values())

    sel = pd.DataFrame(sel_rows)
    coefs = pd.DataFrame(coef_rows, columns=["model", "held_out_cohort", "gene", "coefficient"])
    print("\n== Chosen penalty and number of selected genes per outer fold ==")
    print(sel.round(4).to_string(index=False))
    savecsv(sel.round(5), "05_expr_selected.csv")
    savecsv(coefs, "05_expr_gene_coefficients.csv")

    # ------------------------------------------------------------ paired bootstrap comparisons
    diffs = {"G2-C": ("G2", "C"), "G2-G0": ("G2", "G0"), "G1-C": ("G1", "C")}

    def compare(mask, label):
        idx = np.where(mask)[0]
        t, e = t_all[idx], e_all[idx]
        S = {k: pred[k][idx] for k in names}
        pt = {k: c_index_h(t, e, S[k]) for k in names}
        pt.update({name: pt[a] - pt[b] for name, (a, b) in diffs.items()})
        rng = np.random.default_rng(2)
        bs = {k: [] for k in pt}
        for _ in range(B):
            i = rng.integers(0, len(idx), len(idx))
            try:
                cc = {k: c_index_h(t[i], e[i], S[k][i]) for k in names}
            except ZeroDivisionError:
                continue
            cc.update({name: cc[a] - cc[b] for name, (a, b) in diffs.items()})
            for k in pt:
                bs[k].append(cc[k])
        row = dict(group=label, n=len(idx), events_within_60=int(trunc(t, e)[1].sum()))
        for k in pt:
            lo, hi = np.percentile(bs[k], [2.5, 97.5])
            row.update({k: pt[k], f"{k}_low": lo, f"{k}_high": hi})
        return row

    rows = [compare(np.ones(n, bool), "Pooled")]
    for c in sorted(np.unique(coh)):
        rows.append(compare(coh == c, f"Cohort {int(c)}"))
    res = pd.DataFrame(rows)
    savecsv(res.round(4), "05_expr_loco.csv")
    show = ["group", "n", "events_within_60", "C", "G0", "G1", "G2", "G2-C", "G2-C_low", "G2-C_high",
            "G2-G0", "G2-G0_low", "G2-G0_high"]
    print("\n== C-index at 60 months, leave-one-cohort-out; differences with paired-bootstrap 95% CI ==")
    print(res[show].round(3).to_string(index=False))
    per_cohort = res.iloc[1:]
    print("\nSD of the C-index across the five cohorts:",
          {k: round(float(per_cohort[k].std()), 3) for k in names})

    # ------------------------------------------------------------ Brier and calibration-in-the-large
    b_null = brier_at(y_all, pred["null"])
    print("\n== Brier score at 60 months (pooled) and skill vs the null model ==")
    for k in names:
        b = brier_at(y_all, pred[k])
        print(k, round(b, 4), "skill:", round(1 - b / b_null, 3))
    cal_rows = []
    for c in sorted(np.unique(coh)):
        mk = coh == c
        obs, lo, hi = km_risk(t_all[mk], e_all[mk])
        cal_rows.append(dict(cohort=int(c), observed=obs, obs_low=lo, obs_high=hi,
                             predicted_C=(1 - pred["C"][mk]).mean(), predicted_G2=(1 - pred["G2"][mk]).mean()))
    cal = pd.DataFrame(cal_rows)
    print("\n== Calibration-in-the-large per held-out cohort (5-year risk) ==")
    print(cal.round(3).to_string(index=False))
    savecsv(cal.round(4), "05_expr_calibration.csv")

    # ------------------------------------------------------------ figures
    labels = list(res["group"])

    # fig09: differences in C-index
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for key, color, off, lab in (("G2-C", "tab:red", -0.2, "G2 - C  (clinical + expression vs clinical, ER-stratified)"),
                                 ("G2-G0", "tab:orange", 0.0, "G2 - G0 (same, unstratified clinical)"),
                                 ("G1-C", "tab:gray", 0.2, "G1 - C  (expression only vs clinical)")):
        forest(ax, labels, res[key], res[f"{key}_low"], res[f"{key}_high"], color=color, offset=off, label=lab)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(0.02, color="red", ls="--", lw=1)
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y, hi_y + 0.7)  # head-room for the threshold label
    ax.text(0.0205, 0.99, "pre-specified threshold 0.02", color="red", fontsize=8, va="top",
            transform=ax.get_xaxis_transform())
    ax.set_xlabel("Difference in C-index at 60 months (paired bootstrap 95% CI)")
    ax.set_title("Does gene expression add predictive value? (leave-one-cohort-out)")
    legend_below(ax, fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig09_expr_delta_c.png")

    # fig10: C-index of every model
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = {"C": "tab:blue", "G0": "tab:cyan", "G1": "tab:gray", "G2": "tab:red"}
    for key, off in zip(names, (-0.24, -0.08, 0.08, 0.24)):
        forest(ax, labels, res[key], res[f"{key}_low"], res[f"{key}_high"], color=colors[key], offset=off,
               label=key)
    ax.set_xlabel("C-index truncated at 60 months (bootstrap 95% CI)")
    ax.set_title("Discrimination on the held-out cohort")
    legend_below(ax, ncol=4)
    fig.tight_layout()
    savefig(fig, "fig10_expr_c_by_cohort.png")

    # fig11: calibration-in-the-large
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    y = np.arange(len(cal))[::-1].astype(float)
    ax.errorbar(cal["observed"], y,
                xerr=[np.clip(cal["observed"] - cal["obs_low"], 0, None),
                      np.clip(cal["obs_high"] - cal["observed"], 0, None)],
                fmt="o", color="tab:blue", capsize=3, label="observed (Kaplan-Meier, 95% CI)")
    ax.scatter(cal["predicted_C"], y + 0.12, marker="x", color="tab:green", s=45, label="predicted, model C")
    ax.scatter(cal["predicted_G2"], y - 0.12, marker="+", color="tab:red", s=60, label="predicted, model G2")
    ax.set_yticks(y)
    ax.set_yticklabels([f"Cohort {c}" for c in cal["cohort"]])
    ax.set_xlabel("5-year risk of disease-specific death")
    ax.set_title("Calibration-in-the-large on the held-out cohort")
    legend_below(ax, fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig11_expr_calibration.png")

    # fig12: selected genes
    plot_selected_genes(sel, coefs)


if __name__ == "__main__":
    main()