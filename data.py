"""Data loading for the METABRIC survival project."""
import os

import pandas as pd

KAGGLE_DATASET = "raghadalharbi/breast-cancer-gene-expression-profiles-metabric"
CSV_NAME = "METABRIC_RNA_Mutation.csv"


def _csv_path():
    """Use a local copy of the CSV if present, otherwise download it with kagglehub."""
    local = os.path.join(os.getcwd(), CSV_NAME)
    if os.path.exists(local):
        return local
    import kagglehub

    return os.path.join(kagglehub.dataset_download(KAGGLE_DATASET), CSV_NAME)


def load():
    """Return the patients that have a valid disease-specific survival endpoint.

    time  : overall_survival_months
    event : 1 = died of disease; deaths from other causes and living patients are censored.

    NOTE: in this Kaggle file `overall_survival` is coded 1 = living, 0 = deceased,
    so it is NOT used as the event indicator (see step 01 for the cross-tabulation check).
    """
    df = pd.read_csv(_csv_path(), low_memory=False)
    d = df.dropna(subset=["death_from_cancer", "overall_survival_months"]).copy()
    d["event"] = (d["death_from_cancer"] == "Died of Disease").astype(int)
    d["time"] = d["overall_survival_months"]
    return d
