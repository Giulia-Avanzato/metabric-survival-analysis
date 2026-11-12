"""Step 1 - Data checks, endpoint definition, Kaplan-Meier curves and cohort case-mix.

Outputs
  figures/fig01_km_by_er.png        Kaplan-Meier curves by ER status, with numbers at risk
  figures/fig02_km_by_cohort.png    Kaplan-Meier curves by METABRIC cohort
  results/01_missingness.csv        Fraction of missing values in the clinical columns
  results/01_cohort_casemix.csv     Case-mix, treatment rates and follow-up per cohort
"""
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test

from data import load
from utils import savecsv, savefig

import matplotlib.pyplot as plt  # noqa: E402  (after utils so the Agg backend is set)

d = load()
print("Patients with a valid endpoint:", len(d))

# ---------------------------------------------------------------- coding check
print("\nCoding check: overall_survival vs death_from_cancer")
print("(in this file overall_survival is 1 = living, 0 = deceased)")
print(pd.crosstab(d["overall_survival"], d["death_from_cancer"], dropna=False))

print("\nEndpoint: disease-specific survival")
print("events (died of disease):", int(d["event"].sum()))
print("deaths from other causes (censored):", int((d["death_from_cancer"] == "Died of Other Causes").sum()))
print("follow-up in months:")
print(d["time"].describe().round(1))

# ---------------------------------------------------------------- missing data
clin_cols = list(d.columns[: list(d.columns).index("death_from_cancer") + 1])
miss = d[clin_cols].isna().mean().sort_values(ascending=False).head(15).rename("fraction_missing")
print("\nMost incomplete clinical columns:")
print(miss.round(3))
savecsv(miss.to_frame(), "01_missingness.csv", index=True)

# ---------------------------------------------------------------- KM by ER status
fig, ax = plt.subplots(figsize=(7.5, 6.2))
fitters = []
for status, color in (("Positive", "tab:blue"), ("Negative", "tab:red")):
    sub = d[d["er_status"] == status]
    kmf = KaplanMeierFitter().fit(sub["time"], sub["event"], label=f"ER {status} (n={len(sub)})")
    kmf.plot_survival_function(ax=ax, color=color, ci_show=True)
    fitters.append(kmf)
pos, neg = d[d["er_status"] == "Positive"], d[d["er_status"] == "Negative"]
p = logrank_test(pos["time"], neg["time"], pos["event"], neg["event"]).p_value
print("\nLog-rank test ER+ vs ER-: p =", p)
ax.set_xlabel("Months since diagnosis")
ax.set_ylabel("Disease-specific survival probability")
ax.set_title(f"Disease-specific survival by ER status (log-rank p = {p:.1e})")
ax.set_ylim(0, 1.02)
add_at_risk_counts(*fitters, ax=ax)
fig.tight_layout()
savefig(fig, "fig01_km_by_er.png")

# ---------------------------------------------------------------- KM by cohort
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for c, sub in d.groupby("cohort"):
    KaplanMeierFitter().fit(sub["time"], sub["event"], label=f"Cohort {int(c)} (n={len(sub)})").plot_survival_function(
        ax=ax, ci_show=False
    )
ax.set_xlabel("Months since diagnosis")
ax.set_ylabel("Disease-specific survival probability")
ax.set_title("Disease-specific survival by METABRIC cohort")
ax.set_ylim(0, 1.02)
fig.tight_layout()
savefig(fig, "fig02_km_by_cohort.png")

# ---------------------------------------------------------------- cohort case-mix
g = d.groupby("cohort")
casemix = pd.DataFrame(
    {
        "n": g.size(),
        "disease_deaths": g["event"].sum(),
        "other_cause_deaths_pct": g["death_from_cancer"].apply(lambda x: (x == "Died of Other Causes").mean() * 100),
        "median_age": g["age_at_diagnosis"].median(),
        "median_tumour_size": g["tumor_size"].median(),
        "mean_grade": g["neoplasm_histologic_grade"].mean(),
        "mean_positive_nodes": g["lymph_nodes_examined_positive"].mean(),
        "ER_pos_pct": g["er_status"].apply(lambda x: (x == "Positive").mean() * 100),
        "HER2_pos_pct": g["her2_status"].apply(lambda x: (x == "Positive").mean() * 100),
        "chemotherapy_pct": g["chemotherapy"].mean() * 100,
        "hormone_therapy_pct": g["hormone_therapy"].mean() * 100,
        "radiotherapy_pct": g["radio_therapy"].mean() * 100,
    }
)
followup = {}
for c, sub in d.groupby("cohort"):
    # reverse Kaplan-Meier: censoring is treated as the event of interest
    followup[c] = KaplanMeierFitter().fit(sub["time"], 1 - sub["event"]).median_survival_time_
casemix["median_followup_months"] = pd.Series(followup)
print("\nCohort case-mix:")
print(casemix.round(1).to_string())
savecsv(casemix.round(2), "01_cohort_casemix.csv", index=True)
