# ICB Analysis

## Overview

This module analyzes Immune Checkpoint Blockade (ICB) medications and their relationship with CD8+ T cell signatures. It focuses on identifying ICB treatments, analyzing CD8+ T cell signatures by ICB status, and examining sex-specific differences in ICB response.

## Components

### Scripts

- `icb_main.py`: Main entry point for ICB analysis
- `icb_analysis.py`: Core ICB analysis functions
- `icb_data_loader.py`: Functions for loading ICB-related data
- `icb_treatment_identifier.py`: Functions for identifying ICB treatments
- `advanced_group_analysis.py`: Advanced group comparison methods
- `group_analysis.py`: Group analysis functions
- `group_data_loader.py`: Functions for loading group-related data

### Key Features

- **ICB Treatment Identification**: Identifies patients who received ICB treatments (PD-1, PD-L1, CTLA-4 inhibitors)
- **ICB Type Analysis**: Analyzes CD8 signatures by specific ICB type (PD-1 only, CTLA-4 only, combination)
- **Treatment Duration Analysis**: Examines the effect of ICB treatment duration on CD8 signatures
- **Sex-Stratified Analysis**: Compares ICB response between males and females
- **Survival Analysis**: Analyzes survival outcomes by ICB status and CD8 signature

## Data Flow

1. **Input**:
   - CD8+ T cell signature scores from CD8 group analysis
   - Clinical data with patient information
   - Medication data with treatment information

2. **Processing**:
   - Identify ICB treatments and classify by type
   - Merge ICB status with clinical data and CD8 scores
   - Analyze CD8 signatures by ICB status and sex
   - Perform survival analysis

3. **Output**:
   - CD8 signature scores by ICB status
   - Survival analysis results
   - Visualizations (distribution plots, survival curves, forest plots)

## Usage

### Basic Usage

```bash
python src/icb_analysis/icb_main.py
```

### Advanced Usage

For specific analyses:

```bash
# Analyze by ICB type
python src/icb_analysis/icb_main.py --by-type

# Analyze treatment duration
python src/icb_analysis/icb_main.py --duration

# Perform propensity score matching
python src/icb_analysis/icb_main.py --propensity-matching
```

## Dependencies

- `cd8_groups_analysis.py`: For CD8+ T cell signature analysis
- `utils/shared_functions.py`: For shared utility functions

## Results

Results are saved to:
- `output/results/icb_analysis/`: CSV files with analysis results
- `output/plots/icb_analysis/`: Visualizations

## Key Findings

- CD8_B signature shows opposite trends in ICB-naive males vs. females
- High CD8_B expression is strongly protective in ICB-naive females (HR=0.08, p=0.04)
- ICB treatment may neutralize the prognostic impact of CD8 signatures 