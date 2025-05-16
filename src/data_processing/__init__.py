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

# Make key functions available at package level
try:
    from .utils import create_map_from_qc, clean_id_string, load_csv
    from .clinical import load_clinical_table
    from .molecular import merge_molecular_qc
    from .single_cell import integrate_single_cell_scores
except ImportError:
    pass  # Allow partial imports if some modules aren't available 
