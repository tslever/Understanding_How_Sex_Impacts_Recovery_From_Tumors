# Microenvironment Analysis Pipeline

This pipeline analyzes tumor microenvironment using RNA-seq data and xCell deconvolution.

## Files
- `microenv.py`: Main pipeline script
- `analyze_ids.py`: Helper script for ID mapping
- `merged_immune_clinical.csv`: Final processed data with immune scores and clinical info

## Pipeline Steps

1. **Data Loading**
   - Loads RNA-seq expression data from genes.results files
   - Filters lowly expressed genes (TPM > 1 in at least 20% of samples)
   - Loads clinical data

2. **xCell Analysis**
   - Converts ENSEMBL IDs to gene symbols
   - Runs xCell deconvolution algorithm
   - Generates scores for 67 immune cell types
   - Saves raw scores to `cell_scores.csv`

3. **Data Integration**
   - Maps sample IDs to patient IDs using QC metrics
   - Merges immune scores with clinical data
   - Averages scores for patients with multiple samples
   - Saves final merged data to `merged_immune_clinical.csv`

## Output Files

1. **cell_scores.csv**
   - Raw xCell scores
   - Rows: 333 samples
   - Columns: 67 immune cell types

2. **merged_immune_clinical.csv**
   - Final processed dataset
   - Rows: 321 unique patients
   - Columns (72 total):
     - Clinical: PATIENT_ID, AGE, YearOfClinicalRecordCreation, SEX, Race
     - Immune: 67 cell type scores

## Usage
```bash
python microenv.py
```

## Next Steps
- Analyze sex differences in immune composition
- Create visualizations (heatmaps, boxplots)
- Perform statistical testing
- Consider age and race as covariates

## Dependencies
- Python 3.6+
- pandas
- numpy
- R with xCell package
- rpy2 for R-Python interface 