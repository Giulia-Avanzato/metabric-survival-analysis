"""Step 3 - Repeated cross-validation, baseline comparison, subgroup analysis and calibration.

Models (all Cox proportional hazards models, fitted with lifelines)
  A  age + tumour size + grade                      (simple baseline)
  B  A + number of positive lymph nodes
  C  full clinical model, stratified by ER status
  null  constant Kaplan-Meier survival of the training fold (reference for the Brier score)

Evaluation
  * 5-fold cross-validation repeated 5 times (stratified by event). No hyper-parameters are tuned,
    so plain (non-nested) cross-validation is appropriate.
  * Primary metric: C-index truncated at 60 months, computed on out-of-fold predictions of S(60).
  * Secondary: IPCW Brier score at 60 months and its skill relative to the null model.
  * 95% CIs: patient-level bootstrap on the out-of-fold predictions of repeat 0 (paired for differences).

Outputs
  figures/fig05_model_comparison.png     C-index per model + paired differences
  figures/fig06_subgroups.png            Subgroup discrimination and predicted vs observed risk
  figures/fig07_calibration_deciles.png  Calibration of models A and C
  results/03_cv_per_repeat.csv, 03_cv_summary.csv, 03_cv_bootstrap.csv, 03_subgroups.csv,
  results/03_calibration_A.csv, 03_calibration_C.csv, oof_repeat0.csv (used by step 04)
"""
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sklearn.model_selection import StratifiedKFold

from data import load
from utils import (
    BASELINE, H, boot_ci_c_index, brier_at, c_index_h, covariate_columns, fit_predict_cox, forest,
    km_risk, legend_below, prepare, savecsv, savefig, survival_array, trunc, ypos,
)

import matplotlib.pyplot as plt  # noqa: E402

K, R, B = 5, 5, 1000  # folds, repeats, bootstrap resamples

d = load()
m, X = prepare(d)
n = len(X)
t_all, e_all = m["time"].values, m["event"].values.astype(int)
print("Patients:", n, "| events (any time):", int(e_all.sum()))

feats = {
    "A": BASELINE,
    "B": BASELINE + ["lymph_nodes_examined_positive"],
    "C": covariate_columns(X),
}
strata = {"C": ["er_status_Positive"]}
desc = {"A": "A: age + size + grade", "B": "B: A + lymph nodes", "C": "C: full clinical, ER-stratified"}

# ---------------------------------------------------------------- 1. repeated cross-validation
oof = {k: np.full((R, n), np.nan) for k in list(feats) + ["null"]}
for r in range(R):
    for tr_idx, te_idx in StratifiedKFold(K, shuffle=True, random_state=r).split(X, e_all):
        for k, f in feats.items():
            oof[k][r, te_idx] = fit_predict_cox(X, tr_idx, te_idx, f, strata.get(k))
        km = KaplanMeierFitter().fit(t_all[tr_idx], e_all[tr_idx])
        oof["null"][r, te_idx] = float(km.predict(H))
    print(f"repeat {r + 1}/{R} done", flush=True)
assert not any(np.isnan(v).any() for v in oof.values()), "some patients have no out-of-fold prediction"

y_all = survival_array(t_all, e_all)
rows = []
for r in range(R):
    b_null = brier_at(y_all, oof["null"][r])
    for k in feats:
        s = oof[k][r]
        b = brier_at(y_all, s)
        rows.append(
            dict(repeat=r, model=k, c_index_60=c_index_h(t_all, e_all, s), brier_60=b, brier_null=b_null,
                 brier_skill=1 - b / b_null)
        )
cv = pd.DataFrame(rows)
summary = cv.groupby("model").agg(
    c_mean=("c_index_60", "mean"), c_min=("c_index_60", "min"), c_max=("c_index_60", "max"),
    brier_mean=("brier_60", "mean"), brier_skill_mean=("brier_skill", "mean"),
)
print("\n== Repeated 5x5 CV: C-index truncated at 60 months and Brier score at 60 months ==")
print(desc)
print(summary.round(3))
savecsv(cv, "03_cv_per_repeat.csv")
savecsv(summary.reset_index().round(4), "03_cv_summary.csv")

# ---------------------------------------------------------------- 2. paired bootstrap (repeat 0)
S0 = {k: oof[k][0] for k in feats}
diffs = {"B-A": ("B", "A"), "C-A": ("C", "A"), "C-B": ("C", "B")}


def all_c(i):
    c = {k: c_index_h(t_all[i], e_all[i], S0[k][i]) for k in feats}
    c.update({name: c[a] - c[b] for name, (a, b) in diffs.items()})
    return c


point = all_c(np.arange(n))
rng = np.random.default_rng(1)
boots = {k: [] for k in point}
for _ in range(B):
    c = all_c(rng.integers(0, n, n))
    for k in boots:
        boots[k].append(c[k])
boot = pd.DataFrame(
    {k: dict(estimate=point[k], ci_low=np.percentile(boots[k], 2.5), ci_high=np.percentile(boots[k], 97.5))
     for k in point}
).T
print("\n== C-index at 60 months with 95% bootstrap CI (repeat 0); paired differences ==")
print(boot.round(3))
savecsv(boot.round(4).reset_index().rename(columns={"index": "quantity"}), "03_cv_bootstrap.csv")

# ---------------------------------------------------------------- 3. subgroup analysis (model C)
s = S0["C"]
meta = m[["age_at_diagnosis", "er_status", "her2_status", "cohort"]].reset_index(drop=True)
age = meta["age_at_diagnosis"]
groups = {
    "ER positive": meta["er_status"] == "Positive",
    "ER negative": meta["er_status"] == "Negative",
    "HER2 positive": meta["her2_status"] == "Positive",
    "HER2 negative": meta["her2_status"] == "Negative",
    "Age < 50": age < 50,
    "Age 50-69": (age >= 50) & (age < 70),
    "Age >= 70": age >= 70,
}
for c in sorted(meta["cohort"].unique()):
    groups[f"Cohort {int(c)}"] = meta["cohort"] == c

rows = []
for name, mask in groups.items():
    mk = mask.values
    t, e, ss = t_all[mk], e_all[mk], s[mk]
    row = dict(subgroup=name, n=int(mk.sum()), events_within_60=int(trunc(t, e)[1].sum()))
    try:
        lo, hi = boot_ci_c_index(t, e, ss)
        row.update(c_index_60=c_index_h(t, e, ss), c_low=lo, c_high=hi)
    except ZeroDivisionError:
        row.update(c_index_60=np.nan, c_low=np.nan, c_high=np.nan)
    obs, olo, ohi = km_risk(t, e)
    row.update(predicted_risk=1 - ss.mean(), observed_risk=obs, obs_low=olo, obs_high=ohi)
    rows.append(row)
sub = pd.DataFrame(rows)
print("\n== Subgroups, model C (out-of-fold predictions) ==")
print(sub.round(3).to_string(index=False))
savecsv(sub.round(4), "03_subgroups.csv")
print(
    "\nNote: calibration within ER groups is close to perfect by construction, because the model has a "
    "separate baseline hazard for each ER stratum."
)


# ---------------------------------------------------------------- 4. calibration by risk decile
def calibration(s, q=10):
    g = pd.qcut(1 - s, q, labels=False, duplicates="drop")
    out = []
    for k in np.unique(g):
        mk = g == k
        obs, lo, hi = km_risk(t_all[mk], e_all[mk])
        out.append(dict(group=int(k) + 1, n=int(mk.sum()), predicted=(1 - s[mk]).mean(),
                        observed=obs, low=lo, high=hi))
    return pd.DataFrame(out)


cal = {k: calibration(S0[k]) for k in ("A", "C")}
for k, df in cal.items():
    savecsv(df.round(4), f"03_calibration_{k}.csv")

# ---------------------------------------------------------------- figures
# fig05: model comparison
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
keys = ["A", "B", "C"]
forest(ax, [desc[k] for k in keys], [point[k] for k in keys],
       [boot.loc[k, "ci_low"] for k in keys], [boot.loc[k, "ci_high"] for k in keys])
for j, k in enumerate(keys):
    vals = cv.loc[cv["model"] == k, "c_index_60"].values
    ax.scatter(vals, np.full(len(vals), ypos(3)[j] + 0.18), s=14, color="grey", alpha=0.8,
               label="5 CV repeats" if j == 0 else None)
ax.set_xlabel("C-index truncated at 60 months")
ax.set_title("Discrimination (bootstrap 95% CI, repeat 0)")
legend_below(ax)
ax = axes[1]
dk = ["B-A", "C-A", "C-B"]
forest(ax, ["B - A", "C - A", "C - B"], [point[k] for k in dk],
       [boot.loc[k, "ci_low"] for k in dk], [boot.loc[k, "ci_high"] for k in dk], color="tab:red")
ax.axvline(0, color="grey", ls="--")
ax.set_xlabel("Difference in C-index (paired bootstrap, 95% CI)")
ax.set_title("What each block of variables adds")
fig.tight_layout()
savefig(fig, "fig05_model_comparison.png")

# fig06: subgroups
labels = [f"{r.subgroup} (n={r.n})" for r in sub.itertuples()]
fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), sharey=True)
ax = axes[0]
forest(ax, labels, sub["c_index_60"], sub["c_low"], sub["c_high"])
ax.axvline(point["C"], color="grey", ls="--", label="pooled model C")
ax.set_xlabel("C-index truncated at 60 months (bootstrap 95% CI)")
ax.set_title("Discrimination by subgroup")
legend_below(ax)
ax = axes[1]
y = ypos(len(sub))
ax.errorbar(sub["observed_risk"], y,
            xerr=[np.clip(sub["observed_risk"] - sub["obs_low"], 0, None),
                  np.clip(sub["obs_high"] - sub["observed_risk"], 0, None)],
            fmt="o", color="tab:blue", capsize=3, label="observed (Kaplan-Meier, 95% CI)")
ax.scatter(sub["predicted_risk"], y, marker="x", color="tab:red", s=45, label="mean predicted")
ax.set_xlabel("5-year risk of disease-specific death")
ax.set_title("Predicted vs observed risk by subgroup")
legend_below(ax)
fig.tight_layout()
savefig(fig, "fig06_subgroups.png")

# fig07: calibration
fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), sharex=True, sharey=True)
lim = max(max(df["high"].max(), df["predicted"].max()) for df in cal.values()) * 1.08
for ax, (k, df) in zip(axes, cal.items()):
    ax.errorbar(df["predicted"], df["observed"],
                yerr=[np.clip(df["observed"] - df["low"], 0, None), np.clip(df["high"] - df["observed"], 0, None)],
                fmt="o", capsize=3)
    ax.plot([0, lim], [0, lim], "--", color="grey")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title(f"Model {desc[k]}")
    ax.set_xlabel("Predicted 5-year risk (decile mean)")
axes[0].set_ylabel("Observed 5-year risk (Kaplan-Meier, 95% CI)")
fig.suptitle("Calibration by risk decile (out-of-fold predictions)")
fig.tight_layout()
savefig(fig, "fig07_calibration_deciles.png")

# ---------------------------------------------------------------- out-of-fold predictions for step 04
oof_df = pd.DataFrame({"patient_id": m.index.values, "time": t_all, "event": e_all, "cohort": m["cohort"].values})
for k in feats:
    oof_df[f"s60_{k}"] = S0[k]
savecsv(oof_df, "oof_repeat0.csv")
