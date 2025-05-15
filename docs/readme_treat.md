# Treatment Response Analysis Pipeline

This pipeline analyzes the relationship between immune cell composition and treatment responses in cancer patients, with a focus on immunotherapy outcomes.

## Overview

The analysis pipeline examines:
- Treatment response patterns across immune cell types
- Drug-specific immune signatures
- Survival outcomes based on immune profiles
- Combination therapy effects
- Temporal changes in immune composition

## Data Sources

1. **Clinical Data**
   - `Medications_V4.csv`: Treatment information
   - `Outcomes_V4.csv`: Response and progression data
   - `VitalStatus_V4.csv`: Survival information
   - `PatientHistory_V4.csv`: Clinical history and performance status

2. **Immune Data**
   - Cell type scores from xCell deconvolution
   - 67 immune cell types analyzed
   - Sample-level immune profiles

## Response Categories

Treatment responses are classified as:
- CR: Complete Response/Remission
- PR: Partial Response
- SD: Stable Disease
- PD: Progressive Disease
- OR: Objective Response
- NA: Not Applicable/Unknown

## Analysis Components

### 1. Treatment Response Analysis
```python
analyze_response_patterns()
```
- Examines associations between immune cell types and treatment responses
- Uses Kruskal-Wallis tests for multi-group comparisons
- Applies Bonferroni correction for multiple testing
- Generates boxplots for significant associations

### 2. Drug-Specific Analysis
```python
analyze_drug_specific_patterns()
```
- Analyzes immune patterns for specific immunotherapy drugs:
  - Pembrolizumab
  - Nivolumab
  - Atezolizumab
  - Durvalumab
  - Ipilimumab
  - Avelumab
  - Cemiplimab
- Compares treated vs untreated immune profiles

### 3. Survival Analysis
```python
analyze_survival()
```
- Stratifies survival by immune cell levels
- Generates Kaplan-Meier curves
- Performs log-rank tests
- Adjusts for multiple testing

### 4. Combination Therapy Analysis
```python
analyze_combination_therapy()
```
- Compares mono vs combination therapy effects
- Examines immune profile differences
- Assesses treatment sequence impacts

## Output Files

1. **Analysis Results**
   - `response_patterns.csv`: Immune-response associations
   - `survival_patterns.csv`: Survival analysis results
   - `drug_patterns.csv`: Drug-specific immune patterns

2. **Visualizations**
   - Response pattern boxplots
   - Survival curves
   - Drug-specific immune profiles
   - Temporal change plots

## Usage

```python
from treatment_analysis import TreatmentResponseAnalysis

# Initialize analysis
analysis = TreatmentResponseAnalysis()

# Run response analysis
response_results = analysis.analyze_response_patterns()

# Analyze survival
survival_results = analysis.analyze_survival()

# Examine drug patterns
drug_results = analysis.analyze_drug_specific_patterns()
```

## Key Findings

1. **Response Patterns**
   - Found 14 significant immune cell associations with treatment response
   - CD4+ T-cells show strong correlation with response

2. **Patient Demographics**
   - 321 total patients analyzed
   - 122 death events recorded
   - Median survival time: 0.485 years

3. **Response Distribution**
   - Complete Response: 97 patients (30.2%)
   - Stable Disease: 37 patients (11.5%)
   - Progressive Disease: 27 patients (8.4%)
   - Partial Response: 4 patients (1.2%)

## Dependencies

- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- lifelines (for survival analysis)

## Notes

- Age values over 89 are aggregated as "Age 90 or Older"
- Missing response data is coded as "Unknown/Not Applicable"
- Survival times are calculated from diagnosis date
- Multiple samples per patient are averaged for immune scores

## Future Directions

1. Implement temporal analysis of immune changes
2. Add multivariate response prediction models
3. Incorporate molecular features (TMB, MSI)
4. Analyze treatment sequence effects 

## Comparing Male and Female Survival Patterns

To directly compare survival patterns between males and females, we implemented three complementary approaches:

### 1. Sex-Specific Median Approach

This method uses sex-specific medians as thresholds:
- Males: Split into high/low groups based on median expression among males
- Females: Split into high/low groups based on median expression among females
- Advantage: Equal-sized groups within each sex
- Limitation: Thresholds differ between sexes, complicating direct comparison

### 2. Common Reference Point Approach

This method uses the same threshold for both sexes:
- Overall median: Calculated from the entire cohort (both sexes)
- Same threshold applied to both male and female patients
- Advantage: Directly comparable between sexes
- Limitation: May result in unbalanced group sizes if expression differs by sex

### 3. Z-Score Normalization Approach

This method uses standardized expression values:
- Z-scores calculated based on the entire cohort's mean and standard deviation
- Patients with z-scores > 0 (above mean) classified as "high"
- Advantage: Statistically robust and accounts for distribution characteristics
- Limitation: Assumes normal distribution of expression values

These complementary approaches provide a comprehensive view of how CD8+ T cell signatures affect survival in a sex-specific manner, allowing us to identify:
1. Signatures with consistent effects across sexes
2. Signatures with sex-specific prognostic value
3. Potential biological differences in immune function between sexes 

## Cohort Selection

For consistency and biological relevance, this analysis excludes patients with the following primary diagnoses:
- Prostate gland 
- Vulva, NOS

This filtering ensures a more homogeneous cohort for evaluating treatment effects and sex-specific patterns in immune responses. Sex-specific cancers can introduce biases particularly when comparing male and female patients, as their biology and treatment approaches differ substantially from other cancer types. 