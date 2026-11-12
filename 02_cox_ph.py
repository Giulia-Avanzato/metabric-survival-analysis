"""Step 2 - Cox models on the full data set: description and proportional-hazards diagnostics.

This step is descriptive (no held-out evaluation; that is step 3). It shows that ER status violates the
proportional-hazards assumption and that stratifying by ER removes the violation.

Outputs
  figures/fig03_cox_forest.png       Hazard ratios of the ER-stratified Cox model
  figures/fig04_ph_tests.png         Proportional-hazards test p-values, before and after stratifying
  results/02_cox_summary.csv         Coefficients of the stratified model
  results/02_ph_tests.csv            Proportional-hazards test results (both models)
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from data import load
from utils import CONTINUOUS, forest, prepare, savecsv, savefig

import matplotlib.pyplot as plt  # noqa: E402

d = load()
m, X = prepare(d)
print("Complete-case patients:", len(X), "of", len(d), "| disease-specific events:", int(X["event"].sum()))

# Standardise continuous covariates so hazard ratios are per 1 SD (grade stays per step).
Z = X.copy()
for c in CONTINUOUS:
    Z[c] = (Z[c] - Z[c].mean()) / Z[c].std()

cph_u = CoxPHFitter().fit(Z, duration_col="time", event_col="event")
cph_s = CoxPHFitter().fit(Z, duration_col="time", event_col="event", strata=["er_status_Positive"])


def ph_table(cph, name):
    res = proportional_hazard_test(cph, Z, time_transform="rank")
    s = res.summary.reset_index()
    out = pd.DataFrame(
        {"covariate": s.iloc[:, 0].astype(str), "test_statistic": s["test_statistic"], "p": s["p"], "model": name}
    )
    return out


ph = pd.concat(
    [ph_table(cph_u, "ER as covariate"), ph_table(cph_s, "stratified by ER")], ignore_index=True
)
print("\nProportional-hazards tests (Schoenfeld residuals, rank time transform):")
print(ph.round(4).to_string(index=False))
savecsv(ph, "02_ph_tests.csv")

print("\nStratified Cox model (HR per 1 SD for continuous variables):")
cph_s.print_summary(decimals=2)
savecsv(cph_s.summary.reset_index().round(4), "02_cox_summary.csv")

# ---------------------------------------------------------------- forest plot of hazard ratios
names = {
    "age_at_diagnosis": "Age at diagnosis (per SD)",
    "tumor_size": "Tumour size (per SD)",
    "neoplasm_histologic_grade": "Histologic grade (per step)",
    "lymph_nodes_examined_positive": "Positive lymph nodes (per SD)",
    "her2_status_Positive": "HER2 positive",
    "chemotherapy": "Chemotherapy*",
    "hormone_therapy": "Hormone therapy*",
    "radio_therapy": "Radiotherapy*",
}
s = cph_s.summary.loc[list(names)]
fig, ax = plt.subplots(figsize=(7.5, 4.8))
forest(
    ax,
    [names[k] for k in names],
    s["exp(coef)"],
    s["exp(coef) lower 95%"],
    s["exp(coef) upper 95%"],
)
ax.axvline(1, color="grey", ls="--")
ax.set_xscale("log")
ax.set_xlabel("Hazard ratio (95% CI), disease-specific death")
ax.set_title("Cox model stratified by ER status")
fig.text(0.01, -0.02, "* treatment effects are confounded by indication and must not be read causally.", fontsize=8)
fig.tight_layout()
savefig(fig, "fig03_cox_forest.png")

# ---------------------------------------------------------------- PH test figure
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
for ax, (name, sub) in zip(axes, ph.groupby("model", sort=False)):
    sub = sub.sort_values("p")
    ax.barh(sub["covariate"], -np.log10(sub["p"].clip(lower=1e-300)), color="tab:blue")
    ax.axvline(-np.log10(0.05), color="red", ls="--", label="p = 0.05")
    ax.set_title(name)
    ax.set_xlabel("-log10(p) of the proportional-hazards test")
    ax.invert_yaxis()
axes[0].legend()
fig.tight_layout()
savefig(fig, "fig04_ph_tests.png")
