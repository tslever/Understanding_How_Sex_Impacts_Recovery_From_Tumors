import pandas as pd
from .utils import load_csv

def merge_molecular_qc(paths: list[str]) -> pd.DataFrame:
    """
    Read each QC CSV in `paths`, then merge them one-by-one on the sample ID column
    (assumed to be the first column, e.g. 'SampleID').
    """
    dfs = [load_csv(p) for p in paths]
    merged = dfs[0]
    key = merged.columns[0]  # e.g. 'SampleID'
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on=key)
    return merged
