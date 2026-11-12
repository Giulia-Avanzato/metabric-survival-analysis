"""Step 4 - Does the clinical model generalise to a cohort it has never seen?

For each METABRIC cohort, model C (ER-stratified clinical Cox model) is trained on the other four cohorts
(leave-one-cohort-out, LOCO) and evaluated on the held-out cohort. The result is compared with the
out-of-fold predictions of standard cross-validation from step 03 (cohorts mixed in the folds).

If LOCO performance is not worse than cross-validation performance, there is no evidence that losing a
cohort from the training data hurts, i.e. no evidence of harmful cohort shift for this model.

Requires: results/oof_repeat0.csv (created by 03_cv_evaluation.py)

Outputs
  figures/fig08_cohort_shift.png         C-index (CV vs LOCO) and calibration-in-the-large per cohort
  results/04_cohort_shift.csv            Full table per cohort
"""
import os

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from data import load
from utils import (
    RES_DIR, boot_ci_c_index, c_index_h, covariate_columns, fit_predict_cox, forest, km_risk, legend_below,
    prepare, savecsv, savefig, trunc,
)

import matplotlib.pyplot as plt  # noqa: E402

d = load()
m, X = prepare(d)
n = len(X)
t_all, e_all, coh = m["time"].values, m["event"].values.astype(int), m["cohort"].values
other_cause = (d.loc[m.index, "death_from_cancer"] == "Died of Other Causes").values

oof = pd.read_csv(os.path.join(RES_DIR, "oof_repeat0.csv"))
assert (oof["patient_id"].values == m.index.values).all(), "oof_repeat0.csv does not match the data; rerun step 03"
s_cv = oof["s60_C"].values

feats = covariate_columns(X)
s_loco = np.full(n, np.nan)
for c in sorted(np.unique(coh)):
    te_idx, tr_idx = np.where(coh == c)[0], np.where(coh != c)[0]
    s_loco[te_idx] = fit_predict_cox(X, tr_idx, te_idx, feats, strata=["er_status_Positive"])
assert not np.isnan(s_loco).any()

rows = []
for c in sorted(np.unique(coh)):
    mk = coh == c
    t, e = t_all[mk], e_all[mk]
    sl, sc = s_loco[mk], s_cv[mk]
    t120, e120 = trunc(t, e, 120)
    lo_l, hi_l = boot_ci_c_index(t, e, sl)
    lo_c, hi_c = boot_ci_c_index(t, e, sc)
    obs, olo, ohi = km_risk(t, e)
    rows.append(
        dict(
            cohort=int(c), n=int(mk.sum()), events_within_60=int(trunc(t, e)[1].sum()),
            c_full_followup_loco=concordance_index(t, sl, e),
            c60_loco=c_index_h(t, e, sl), c60_loco_low=lo_l, c60_loco_high=hi_l,
            c60_cv=c_index_h(t, e, sc), c60_cv_low=lo_c, c60_cv_high=hi_c,
            c120_loco=concordance_index(t120, sl, e120),
            predicted_risk_loco=1 - sl.mean(), observed_risk=obs, obs_low=olo, obs_high=ohi,
            other_cause_death_pct=other_cause[mk].mean() * 100,
            iqr_predicted_risk=float(np.subtract(*np.percentile(1 - sl, [75, 25]))),
        )
    )
res = pd.DataFrame(rows)
res["c60_loco_minus_cv"] = res["c60_loco"] - res["c60_cv"]
print("\n== Per-cohort results for model C ==")
print(res.round(3).to_string(index=False))
savecsv(res.round(4), "04_cohort_shift.csv")

# ---------------------------------------------------------------- figure
labels = [f"Cohort {r.cohort} (n={r.n})" for r in res.itertuples()]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
ax = axes[0]
forest(ax, labels, res["c60_cv"], res["c60_cv_low"], res["c60_cv_high"], color="tab:blue", offset=0.12,
       label="Cross-validation (cohorts mixed)")
forest(ax, labels, res["c60_loco"], res["c60_loco_low"], res["c60_loco_high"], color="tab:red", offset=-0.12,
       label="Leave-one-cohort-out", marker="s")
ax.set_xlabel("C-index truncated at 60 months (bootstrap 95% CI)")
ax.set_title("Discrimination: seen vs unseen cohort")
legend_below(ax)
ax = axes[1]
y = np.arange(len(res))[::-1].astype(float)
ax.errorbar(res["observed_risk"], y,
            xerr=[np.clip(res["observed_risk"] - res["obs_low"], 0, None),
                  np.clip(res["obs_high"] - res["observed_risk"], 0, None)],
            fmt="o", color="tab:blue", capsize=3, label="observed (Kaplan-Meier, 95% CI)")
ax.scatter(res["predicted_risk_loco"], y, marker="x", color="tab:red", s=45, label="mean predicted (LOCO)")
ax.set_xlabel("5-year risk of disease-specific death")
ax.set_title("Calibration-in-the-large per held-out cohort")
legend_below(ax)
fig.tight_layout()
savefig(fig, "fig08_cohort_shift.png")
