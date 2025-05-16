import pandas as pd
from .utils import load_csv

def load_clinical_table(path: str) -> pd.DataFrame:
    """
    Load one or more clinical CSVs, normalize column names to lowercase,
    and return a single patient‐level DataFrame.
    """
    df = load_csv(path)
    # lowercase all column names
    df.columns = [c.lower() for c in df.columns]
    return df
