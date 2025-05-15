import os
import sys
import subprocess
import pandas as pd
import pytest

from codes.src.data_processing import (
    create_map_from_qc,
    load_clinical_table,
    merge_molecular_qc,
    integrate_single_cell_scores
)
from codes.src.data_processing.utils import clean_id_string, load_csv


def test_create_map_from_qc_basic(tmp_path):
    df = pd.DataFrame({
        "SampleID": ["S1", "S2", "S3"],
        "PatientID": ["P1", "P2", "P1"]
    })
    file = tmp_path / "qc.csv"
    df.to_csv(file, index=False)
    mapping = create_map_from_qc(str(file), sample_col="SampleID", patient_col="PatientID")
    assert mapping == {"S1": "P1", "S2": "P2", "S3": "P1"}


def test_create_map_from_qc_clean_ids(tmp_path):
    df = pd.DataFrame({
        "SampleID": ["S1-RNA", "S2.1", "S3-extra"],
        "PatientID": ["P1", "P2", "P3"]
    })
    file = tmp_path / "qc.csv"
    df.to_csv(file, index=False)
    mapping = create_map_from_qc(str(file), clean_ids=True)
    assert mapping.get("S1") == "P1"
    assert mapping.get("S2") == "P2"
    assert mapping.get("S3") == "P3"


def test_load_clinical_table(tmp_path):
    df = pd.DataFrame({
        "Patient_ID": ["P1", "P2"],
        "Stage": ["I", "II"],
        "Treatment": ["A", "B"]
    })
    file = tmp_path / "clinical.csv"
    df.to_csv(file, index=False)
    clinical = load_clinical_table(str(file))
    # standardized column names
    assert "patient_id" in clinical.columns
    assert "stage" in clinical.columns
    assert "treatment" in clinical.columns
    assert len(clinical) == 2


def test_merge_molecular_qc(tmp_path):
    df1 = pd.DataFrame({"SampleID": ["S1", "S2"], "TMB": [10, 20]})
    df2 = pd.DataFrame({"SampleID": ["S1", "S2"], "MSI": [0.1, 0.2]})
    f1 = tmp_path / "tmb.csv"
    df1.to_csv(f1, index=False)
    f2 = tmp_path / "msi.csv"
    df2.to_csv(f2, index=False)
    merged = merge_molecular_qc([str(f1), str(f2)])
    assert all(col in merged.columns for col in ["SampleID", "TMB", "MSI"])
    assert merged.shape == (2, 3)


def test_integrate_single_cell_scores(tmp_path):
    df = pd.DataFrame({
        "Cell": ["C1", "C2"],
        "SampleID": ["S1", "S1"],
        "Score": [0.5, 0.8]
    })
    file = tmp_path / "scores.csv"
    df.to_csv(file, index=False)
    mapping = {"S1": "P1"}
    integrated = integrate_single_cell_scores(str(file), mapping)
    # expect aggregated per patient P1
    assert "patient_id" in integrated.columns
    assert integrated.loc[0, "patient_id"] == "P1"
    assert integrated.loc[0, "avg_score"] == pytest.approx(0.65)


def test_cli_harness(tmp_path, capsys, monkeypatch):
    df = pd.DataFrame({"SampleID": ["S1"], "PatientID": ["P1"]})
    file = tmp_path / "qc.csv"
    df.to_csv(file, index=False)
    monkeypatch.setattr(sys, 'argv', [
        'utils.py', str(file)
    ])
    from src.data_processing.utils import main
    main()
    captured = capsys.readouterr()
    assert "S1" in captured.out
    assert "P1" in captured.out


def test_clean_id_string():
    assert clean_id_string("X-1-RNA") == "X"
    assert clean_id_string("Y.2") == "Y"


def test_load_csv_encoding(tmp_path):
    data = "\ufeffcol1,col2\n1,2\n"
    file = tmp_path / "bom.csv"
    file.write_text(data, encoding="utf-8")
    df = load_csv(str(file))
    assert list(df.columns) == ["col1", "col2"]
    assert df.loc[0, "col1"] == 1
