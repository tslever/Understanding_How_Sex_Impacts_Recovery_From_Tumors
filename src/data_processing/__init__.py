"""
Data Processing module for clinical, molecular, and single-cell data integration.

This package provides utilities for:
- ID mapping between different sample types
- Clinical data processing and harmonization
- Molecular data integration (TMB, MSI, RNA-Seq QC)
- Single-cell scores integration with clinical data
"""

# Version information
__version__ = "1.0.0"

# Core utilities
from .utils import create_map_from_qc, clean_id_string, load_csv

# Clinical data processing
from .clinical import load_clinical_table

# Molecular QC integration
from .molecular import merge_molecular_qc

# Single-cell score integration
from .single_cell import integrate_single_cell_scores
