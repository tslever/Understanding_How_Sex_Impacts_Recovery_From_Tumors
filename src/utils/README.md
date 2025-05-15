# Utils Module

This module contains shared utility functions used across multiple analysis modules.

## Files

- `shared_functions.py`: Common utility functions for data loading, processing, and analysis

## Functions

The shared functions module provides the following functionality:

### Data Loading

- `load_clinical_data`: Load clinical data from processed file
- `load_rnaseq_data`: Load RNA-seq data from processed file
- `load_gene_signatures`: Load gene signatures from file

### Data Processing

- `filter_by_diagnosis`: Filter data by diagnosis
- `calculate_survival_months`: Calculate survival months from age columns
- `normalize_gene_expression`: Normalize gene expression data

### Data Analysis

- `save_results`: Save results to CSV file

## Usage

The utils module is imported by other modules:

```python
from utils.shared_functions import load_clinical_data, load_rnaseq_data, filter_by_diagnosis
```

## Dependencies

This module has the following dependencies:

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- statsmodels 