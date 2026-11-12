# metabric-survival-analysis
Disease-specific survival in METABRIC: a validated clinical model and a pre-specified test of gene expression

# Disease-specific survival in METABRIC: a validated clinical model and a pre-specified test of gene expression

A small, fully reproducible survival-analysis project on the METABRIC breast-cancer cohort (Kaggle file
`METABRIC_RNA_Mutation.csv`). The goals were methodological: use the right endpoint, check model assumptions,
evaluate with honest validation (repeated cross-validation and leave-one-cohort-out), report uncertainty, and
test one extension with hypotheses written **before** the experiment was run.

**Main findings**

1. A clinical Cox model stratified by ER status reaches a 5-year (60-month) truncated C-index of **0.738** in repeated
   5x5 cross-validation, against 0.671 for age + size + grade. Brier skill over a no-covariate model is 0.139.
2. Adding lymph nodes is the clear gain (+0.043, 95% CI 0.028 to 0.060). The last step to the full clinical model is
   small and fragile (+0.022, 95% CI 0.001 to 0.043).
3. Discrimination on a cohort never seen in training is almost the same as in cross-validation
   (per-cohort difference between -0.017 and +0.009). Cohort 3 is consistently the hardest; the cause was not identified.
4. **Gene expression did not improve the clinical model.** Pre-specified hypothesis H1 is not supported:
   change in 60-month C-index +0.004 (95% CI -0.008 to 0.016), below the 0.02 relevance threshold. The upper confidence limit is
   below the threshold, so the data rule out a gain of that size. This is a null result and is reported as such.

---

## Data and endpoint

| Item | Value |
|---|---|
| Patients with a valid endpoint | 1903 |
| Complete cases used for modelling | 1814 (601 disease-specific deaths) |
| Cohorts (as coded in the dataset) | 5 |
| Expression features | 489 genes (z-scores) |

* **Endpoint: disease-specific survival.** Event = `death_from_cancer == "Died of Disease"` (622 events overall).
  Deaths from other causes (480) are **censored**. Competing risks are not modelled.
* **Coding trap:** in this file `overall_survival` is 1 for *living* and 0 for deceased, the opposite of the usual
  convention. Step 01 prints the cross-tabulation that proves it.
* Follow-up: median 115.6 months (range 0 to 355).
* Evaluation horizon: 60 months for the C-index, Brier score and calibration.

## Methods

* **Kaplan-Meier and log-rank test** (ER+ vs ER-: p = 4e-10).
* **Cox proportional hazards.** The Schoenfeld-residual test rejects proportional hazards for ER status when it enters
  as a covariate (statistic 34.8, p < 0.001). The model is therefore **stratified by ER**; after stratification no
  covariate violates the assumption (all p > 0.14).
* **C-index truncated at 60 months**, IPCW Brier score, Brier skill against a Kaplan-Meier null model, and
  decile calibration.
* **Validation:** 5x5 repeated stratified cross-validation (no tuning, so no nesting is needed), paired
  patient-level bootstrap (B = 1000) for confidence intervals and differences, and leave-one-cohort-out (LOCO).
* **Expression experiment:** elastic-net Cox model (scikit-survival, `l1_ratio` = 0.5), penalty chosen by 5-fold
  CV nested inside the training cohorts, genes standardised on training data only, clinical variables almost
  unpenalised in model G2.

## Results

### 1. Cohort case-mix (step 01)

| cohort | n | disease deaths | other-cause deaths (%) | median age | ER+ (%) | HER2+ (%) | chemo (%) | median follow-up (months) |
|---|---|---|---|---|---|---|---|---|
| 1 | 481 | 134 | 15.6 | 60.9 | 78.6 | 10.4 | 34.7 | 114.3 |
| 2 | 286 | 96 | 18.2 | 56.8 | 75.2 | 14.3 | 17.8 | 235.4 |
| 3 | 733 | 283 | 31.1 | 64.3 | 76.9 | 12.3 | 12.6 | 186.4 |
| 4 | 236 | 63 | 25.0 | 61.5 | 73.3 | 12.7 | 22.9 | 122.8 |
| 5 | 167 | 46 | 39.5 | 66.3 | 76.6 | 15.0 | 19.2 | 170.3 |

### 2. Stratified Cox model (step 02, 1814 patients, 601 events)

Hazard ratios: continuous variables (age, tumour size, positive nodes) per 1 SD, grade per one grade level,
binary variables yes vs no.

| covariate | HR | 95% CI |
|---|---|---|
| Age at diagnosis | 1.21 | 1.10 to 1.33 |
| Tumour size | 1.17 | 1.11 to 1.24 |
| Histologic grade | 1.39 | 1.19 to 1.62 |
| Positive lymph nodes | 1.27 | 1.20 to 1.34 |
| HER2 positive | 1.50 | 1.20 to 1.88 |
| Chemotherapy | 1.57 | 1.24 to 2.00 |
| Hormone therapy | 0.99 | 0.82 to 1.19 |
| Radiotherapy | 0.81 | 0.68 to 0.96 |

**Treatment hazard ratios are not causal.** Treatment is given according to disease severity (confounding by
indication), which is why chemotherapy has HR > 1. The in-sample concordance (0.68, full follow-up) is apparent
performance and is not used anywhere as a result.

### 3. Repeated 5x5 cross-validation (step 03)

| model | C-index at 60 months (mean over 5 repeats) | Brier at 60 months | Brier skill |
|---|---|---|---|
| A: age + size + grade | 0.671 | 0.139 | 0.059 |
| B: A + positive nodes | 0.718 | 0.134 | 0.093 |
| C: full clinical, ER-stratified | 0.738 | 0.128 | 0.139 |

Paired bootstrap on the out-of-fold predictions of one repeat (95% CI):

| contrast | difference in C-index |
|---|---|
| B - A | +0.043 (0.028 to 0.060) |
| C - A | +0.066 (0.042 to 0.091) |
| C - B | +0.022 (0.001 to 0.043) |

The spread across repeats (about 0.003) only reflects the random split; the bootstrap interval (about +/- 0.025) is
the relevant uncertainty.

Subgroups, model C (out-of-fold predictions):

| subgroup | n | events (60 m) | C-index (95% CI) | predicted risk | observed risk (95% CI) |
|---|---|---|---|---|---|
| ER positive | 1383 | 167 | 0.697 (0.658 to 0.735) | 0.126 | 0.126 (0.110 to 0.146) |
| ER negative | 431 | 147 | 0.662 (0.619 to 0.702) | 0.356 | 0.356 (0.311 to 0.404) |
| HER2 positive | 227 | 73 | 0.679 (0.619 to 0.736) | 0.352 | 0.337 (0.279 to 0.405) |
| HER2 negative | 1587 | 241 | 0.729 (0.697 to 0.760) | 0.156 | 0.158 (0.141 to 0.178) |
| Age < 50 | 398 | 88 | 0.744 (0.697 to 0.791) | 0.197 | 0.229 (0.190 to 0.274) |
| Age 50-69 | 932 | 147 | 0.747 (0.710 to 0.787) | 0.178 | 0.163 (0.140 to 0.188) |
| Age >= 70 | 484 | 79 | 0.697 (0.639 to 0.758) | 0.173 | 0.176 (0.144 to 0.215) |

These are exploratory: 12 subgroups (including the 5 cohorts, see the next table) with no multiplicity correction and
mostly overlapping intervals. Calibration inside the ER groups is perfect **by construction**, because the model has
a separate baseline hazard per ER stratum; it is not evidence of good calibration.

### 4. Does the model transfer to an unseen cohort? (step 04)

| held-out cohort | n | events (60 m) | C-index, LOCO | C-index, CV | LOCO - CV | predicted risk (LOCO) | observed risk (95% CI) |
|---|---|---|---|---|---|---|---|
| 1 | 466 | 77 | 0.734 | 0.745 | -0.012 | 0.213 | 0.177 (0.144 to 0.217) |
| 2 | 283 | 48 | 0.771 | 0.788 | -0.017 | 0.143 | 0.171 (0.132 to 0.220) |
| 3 | 682 | 122 | 0.683 | 0.682 | +0.001 | 0.170 | 0.186 (0.158 to 0.218) |
| 4 | 227 | 42 | 0.808 | 0.799 | +0.009 | 0.197 | 0.190 (0.144 to 0.249) |
| 5 | 156 | 25 | 0.782 | 0.776 | +0.005 | 0.200 | 0.169 (0.117 to 0.240) |

No evidence that holding out a cohort harms discrimination, and the mean predicted 5-year risk lies inside the
observed interval in every cohort. Cohort 3 is consistently the lowest (C-index at 60 months 0.683, at 120 months 0.638);
it is not explained by follow-up length, the share of other-cause deaths or the spread of predicted risk. The cause is
unknown. The C-index over the whole follow-up is lower than at 60 months (for example cohort 2: 0.685 vs 0.771): the model
separates early deaths better than late ones. Always state the horizon.

### 5. Pre-specified extension: gene expression (step 05)

**Pre-specified hypotheses** (written in the header of `05_expression_experiment.py` before the full run):

* **H1 (primary).** Under leave-one-cohort-out validation, adding mRNA expression to the clinical model increases the
  pooled 60-month C-index relative to model C. The effect is considered relevant only if the 95% paired-bootstrap CI of the
  difference excludes 0 **and** the difference is >= 0.02.
* **H2 (secondary).** The benefit of expression is heterogeneous across cohorts, and calibration-in-the-large per cohort
  does not deteriorate when expression is added.
* A null result is an accepted outcome.

Models, all evaluated on a cohort excluded from training: **C** clinical Cox stratified by ER; **G0** clinical Cox with ER as a
covariate; **G1** elastic net on expression only; **G2** elastic net on clinical variables + expression.

C-index at 60 months, leave-one-cohort-out (differences with 95% paired-bootstrap CI):

| group | n | C | G0 | G1 | G2 | G2 - C | G2 - G0 |
|---|---|---|---|---|---|---|---|
| Pooled | 1814 | 0.729 | 0.718 | 0.716 | 0.732 | +0.004 (-0.008 to 0.016) | +0.015 (0.005 to 0.025) |
| Cohort 1 | 466 | 0.734 | 0.709 | 0.708 | 0.708 | -0.026 (-0.055 to 0.001) | -0.001 (-0.021 to 0.017) |
| Cohort 2 | 283 | 0.771 | 0.772 | 0.802 | 0.801 | +0.029 (0.003 to 0.055) | +0.029 (0.017 to 0.041) |
| Cohort 3 | 682 | 0.683 | 0.677 | 0.703 | 0.696 | +0.013 (-0.006 to 0.034) | +0.019 (-0.002 to 0.040) |
| Cohort 4 | 227 | 0.808 | 0.812 | 0.710 | 0.805 | -0.003 (-0.035 to 0.030) | -0.007 (-0.027 to 0.013) |
| Cohort 5 | 156 | 0.782 | 0.775 | 0.703 | 0.790 | +0.008 (-0.035 to 0.052) | +0.014 (-0.020 to 0.051) |

Brier score at 60 months (pooled) and skill: C 0.1284 (0.133), G0 0.1319 (0.109), G1 0.1368 (0.076), G2 0.1330 (0.102).

**H1 is not supported.** The pooled change from adding expression to the clinical model is +0.004 (95% CI -0.008 to 0.016),
below the 0.02 threshold and compatible with zero. The Brier score does not improve either.

**Reading the rest, with care.**

* G2 selected **zero genes in all five outer folds**: nested cross-validation always preferred the sparsest model.
  G2 is therefore effectively a lightly penalised clinical model, and its differences from C (per cohort, between -0.026 and
  +0.029) and from G0 (+0.015 pooled) reflect how the clinical model is fitted, **not expression**.
  The G2 - G0 contrast was not pre-specified and must not be read as a gene effect.
* H2: the calibration part holds (G2 predictions are almost identical to C and inside the observed intervals in every
  cohort), but trivially, since G2 contains no genes. The heterogeneity part is **not informative about expression**.
* G1 (expression only) reaches 0.716 pooled, about the same as the clinical model G0 (0.718), so expression carries
  prognostic signal that is largely redundant with the clinical variables. It is unstable across cohorts (for
  example cohort 4: 0.710 vs 0.812 for G0).
* Genes selected by G1 in all five folds: AURKA, STAT5A, MAPT, STAT5B, NR2F1, DIRAS3. This is consistent with a
  signal about proliferation and ER status, which grade and ER status already capture, but this was **not tested**.
  Elastic-net selection among correlated genes is unstable; these are not claimed as biomarkers.

## Limitations

* Complete-case analysis (1814 of 1903 patients); missingness was not modelled.
* Competing risks are not modelled: other-cause deaths are censored, so risk estimates are cause-specific hazards-based,
  not cumulative incidence.
* Gene z-scores were computed on the full dataset before the cross-validation (mild leakage that does not
  affect the sign of the null result).
* Power is limited for the expression question (314 events within 60 months; as few as 25 in one cohort).
* One value of `l1_ratio`, 489 mRNA z-scores only, no other omics, and the five cohorts are used as given: the
  dataset does not tell us how they differ technically.
* Treatment effects are confounded by indication and must not be read causally.
* The subgroup analyses are exploratory and uncorrected for multiple comparisons.
* No external validation outside METABRIC.

## Process disclosure

* H1 and H2 were written in the script header before the full expression experiment was run. A copy of that header, with a
  date line, is saved in `hypotheses_prespecified.txt`.
* A run of step 05 on 50 genes (`--smoke`) was executed first only to check that the code runs; its results were not used to
  change any design choice or threshold.
* After the full run, only the plotting code of `fig12` was changed (in `utils.py`; figure regenerated by
  `06_replot_fig12.py`). The models, hyper-parameters and hypotheses were not touched.

## Reproduce

```powershell
python -m venv venv_metabric
.\venv_metabric\Scripts\Activate.ps1
$env:PYTHONUTF8="1"
python -m pip install -r requirements.txt

python .\01_exploration.py       # endpoint, KM, case-mix
python .\02_cox_ph.py            # Cox, PH tests
python .\03_cv_evaluation.py     # repeated CV, bootstrap, subgroups, calibration  (slowest of 01-04)
python .\04_cohort_shift.py      # leave-one-cohort-out vs CV
python .\05_expression_experiment.py --smoke   # quick check (50 genes)
python .\05_expression_experiment.py           # full experiment (can take tens of minutes)
python .\06_replot_fig12.py      # optional: regenerate fig12 from saved results
```

Place `METABRIC_RNA_Mutation.csv` in the working directory, or let `data.py` download it with `kagglehub`. The
dataset is not redistributed here. Run on Windows (PowerShell) with lifelines and scikit-survival; exact package
versions are not pinned in `requirements.txt` (add the output of `pip freeze` before publishing).

## Files

| script | output |
|---|---|
| `01_exploration.py` | fig01 KM by ER, fig02 KM by cohort, `01_cohort_casemix.csv`, `01_missingness.csv` |
| `02_cox_ph.py` | fig03 forest plot, fig04 PH tests, `02_ph_tests.csv`, `02_cox_summary.csv` |
| `03_cv_evaluation.py` | fig05 model comparison, fig06 subgroups, fig07 calibration by decile, `03_*.csv` |
| `04_cohort_shift.py` | fig08 seen vs unseen cohort, `04_cohort_shift.csv` |
| `05_expression_experiment.py` | fig09 to fig12, `05_expr_*.csv` |
| `data.py`, `utils.py` | data loading and shared helpers |
