import argparse
import pandas as pd
import re

def load_csv(path: str) -> pd.DataFrame:
    """Read a CSV file, handling UTF-8 BOM if present."""
    return pd.read_csv(path, encoding='utf-8-sig')

def clean_id_string(id_str: str) -> str:
    """Strip suffixes after hyphens or dots, e.g. 'S1-RNA' -> 'S1'."""
    return re.split(r'[-\.]', id_str)[0]

def create_map_from_qc(
    path: str,
    sample_col: str = None,
    patient_col: str = None,
    clean_ids: bool = False
) -> dict[str, str]:
    """
    Build a mapping from sample IDs to patient IDs based on a QC CSV.
    If sample_col or patient_col aren't provided, auto-detect columns containing 'sample' or 'patient'.
    If clean_ids is True, also strip suffixes (after '-' or '.') and map the cleaned ID.
    """
    df = load_csv(path)
    cols = list(df.columns)
    # Auto-detect columns if needed
    if sample_col is None:
        sample_col = next((c for c in cols if 'sample' in c.lower()), cols[0])
    if patient_col is None:
        patient_col = next((c for c in cols if 'patient' in c.lower()), cols[1] if len(cols) > 1 else cols[0])

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        sample = row[sample_col]
        patient = row[patient_col]
        mapping[sample] = patient
        if clean_ids:
            cleaned = clean_id_string(sample)
            mapping[cleaned] = patient
    return mapping


def main():
    """Command-line interface for testing the QC-to-patient map."""
    parser = argparse.ArgumentParser(description='Generate sample→patient map from QC CSV')
    parser.add_argument('qc_file', help='Path to QC metrics CSV file')
    parser.add_argument('--sample-col', help='Override sample ID column name')
    parser.add_argument('--patient-col', help='Override patient ID column name')
    parser.add_argument('--clean-ids', action='store_true', help='Also include cleaned sample IDs')
    args = parser.parse_args()
    mapping = create_map_from_qc(
        args.qc_file,
        sample_col=args.sample_col,
        patient_col=args.patient_col,
        clean_ids=args.clean_ids
    )
    for sample, patient in mapping.items():
        print(f"{sample}\t{patient}")

if __name__ == '__main__':
    main()
