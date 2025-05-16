import pandas as pd
from .utils import load_csv

def integrate_single_cell_scores(path: str, mapping: dict) -> pd.DataFrame:
    """
    Read a single‐cell scores CSV with columns ['Cell','SampleID','Score'],
    map each SampleID → patient_id via `mapping`, then compute the mean Score
    per patient.
    """
    df = load_csv(path)
    df["patient_id"] = df["SampleID"].map(mapping)
    out = (
        df
        .groupby("patient_id")["Score"]
        .mean()
        .reset_index()
        .rename(columns={"Score": "avg_score"})
    )
    return out
