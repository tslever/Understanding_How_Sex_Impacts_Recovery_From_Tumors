# CD8+ T Cell Analysis

## Overview

This module analyzes CD8+ T cell populations using single-cell RNA-seq derived signatures. It includes both cluster-level analysis (6 clusters) and broader functional group analysis (CD8_B and CD8_G groups).

## Components

### Scripts

- `cd8_analysis.py`: Core CD8 cluster analysis
- `cd8_groups_analysis.py`: CD8 functional group analysis
- `clinical_analysis.py`: Analysis of clinical variables in relation to CD8 signatures

### Key Features

- **CD8 Cluster Analysis**: Analyzes 6 distinct CD8+ T cell clusters derived from single-cell RNA-seq data
- **CD8 Group Analysis**: Analyzes broader functional groups (CD8_B and CD8_G)
- **Signature Scoring**: Calculates signature scores for each cluster/group
- **Sex-Stratified Analysis**: Compares CD8 signatures between males and females
- **Survival Analysis**: Analyzes survival outcomes by CD8 signature
- **Z-Score Analysis**: Uses z-score normalization for comparing high vs. low expression
- **Absolute Cutoff Analysis**: Uses absolute expression cutoffs for more consistent comparisons

## Data Flow

1. **Input**:
   - RNAseq data (TPM values)
   - Clinical data with patient information
   - CD8 signature gene lists

2. **Processing**:
   - Calculate signature scores for each cluster/group
   - Analyze expression patterns by sex and other clinical variables
   - Perform survival analysis with various stratifications

3. **Output**:
   - CD8 signature scores
   - Expression distribution analyses
   - Survival analysis results
   - Visualizations (distribution plots, survival curves, forest plots)

## CD8+ T Cell Groups

- **CD8_B** (Non-responder enriched, Clusters 1-3):
  - Associated with T cell exhaustion and dysfunction
  - Contains 641 genes including markers like PDCD1 (PD-1), HAVCR2 (TIM-3), LAG3, CTLA4

- **CD8_G** (Responder enriched, Clusters 4-6):
  - Associated with T cell memory and stemness
  - Contains 35 genes including markers like IL7R, TCF7, CCR7

## Usage

### Basic Usage

```bash
# Run CD8 cluster analysis
python src/cd8_analysis/cd8_analysis.py

# Run CD8 group analysis
python src/cd8_analysis/cd8_groups_analysis.py
```

### Advanced Usage

```bash
# Run clinical analysis with CD8 signatures
python src/cd8_analysis/clinical_analysis.py
```

## Dependencies

- `utils/shared_functions.py`: For shared utility functions

## Results

Results are saved to:
- `output/results/cd8_analysis/`: CSV files with analysis results
- `output/plots/cd8_analysis/`: Visualizations

## Key Findings

- CD8_B and CD8_G signatures show sex-specific patterns
- CD8_B signature has opposite prognostic effects in males vs. females
- CD8_G/CD8_B ratio may serve as a potential biomarker

## References

- Sade-Feldman et al. (2018). Defining T Cell States Associated with Response to Checkpoint Immunotherapy in Melanoma. Cell, 175(4), 998-1013.e20. PMID: 30388456 