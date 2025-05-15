# CD8+ T Cell Signature Analysis

This pipeline analyzes CD8+ T cell populations using single-cell derived signatures and integrates with clinical outcomes.

## Features

- Calculates signature scores for 6 CD8+ T cell populations:
  - CD8_C1: Proliferative/Effector
  - CD8_C2: Exhaustion
  - CD8_C3: IFN-Response
  - CD8_C4: Memory
  - CD8_C5: Activation
  - CD8_C6: Naive/Memory

- Performs comprehensive analyses:
  - Survival analysis (overall and sex-stratified)
  - Sex differences in signature expression
  - Stage associations
  - Signature correlations
  - Clinical-molecular integration

## Recent Updates

- Added sex-stratified survival analysis to identify sex-specific prognostic signatures
- Improved stage mapping with standardized categories (I, II, III, IV, Unknown)
- Added robust handling of duplicate patient IDs
- Enhanced correlation analysis with warnings for highly correlated signatures
- Added detailed cohort characteristics summary
- Improved TNM stage reporting (clinical and pathological)
- Fixed index alignment issues in data merging

## Usage

```python
# Initialize analysis
cd8_analysis = CD8ClusterAnalysis(base_path)

# Load and process data
expr_data = cd8_analysis.load_rnaseq_data(base_path)
scores = cd8_analysis.score_signatures(expr_data)
clinical_data = pd.read_csv("processed_clinical_molecular.csv")

# Run analyses
cd8_analysis.plot_signature_heatmap(scores, clinical_data)
survival_results = cd8_analysis.analyze_survival(scores, clinical_data)
male_results, female_results = cd8_analysis.analyze_survival_by_sex(scores, clinical_data)
sex_results = cd8_analysis.analyze_sex_differences(scores, clinical_data)

# Generate summary
cd8_analysis.summarize_results(survival_results, sex_results, clinical_data)
```

## Output Files

- Signature scores: `signature_scores.csv`
- Survival analysis: `survival_analysis.csv`
- Sex-stratified survival: `male_survival_analysis.csv`, `female_survival_analysis.csv`
- Correlation results: `signature_correlations.csv`

### Plots
- Signature heatmap: `signature_heatmap.png`
- Survival curves: `{signature}_survival.png`
- Sex-stratified survival curves: `{signature}_male_survival.png`, `{signature}_female_survival.png`
- Forest plot: `survival_forest_plot.png`
- Sex-stratified forest plot: `sex_stratified_survival_forest.png`
- Correlation heatmap: `signature_correlations.png`

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- lifelines
- scipy
- statsmodels
- rpy2 (for gene ID conversion)

## Key Methods

### score_signatures()
- Calculates mean expression of signature genes
- Handles ENSEMBL ID conversion
- Maps sample IDs to patient IDs

### analyze_survival()
- Performs Cox regression analysis
- Generates KM plots
- Calculates hazard ratios and confidence intervals

### analyze_survival_by_sex()
- Performs sex-stratified survival analysis
- Generates separate male and female KM plots
- Creates combined forest plot showing sex-specific associations
- For each sex, divides patients into high/low expression groups based on sex-specific median
- Reference group (denominator in HR) is the low expression group within each sex

### analyze_sex_differences()
- Compares signature scores between males and females
- Performs t-tests with FDR correction
- Reports effect sizes and sample sizes

### plot_signature_heatmap()
- Creates clustered heatmap of signatures
- Includes clinical annotations (sex, stage)
- Handles duplicate samples
- Uses standardized stage categories

### summarize_results()
- Provides comprehensive analysis summary
- Reports cohort characteristics
- Highlights significant findings
- Details TNM staging distribution

## Clinical Variables

### Stage Information
We use the PathGroupStage variable from the clinical data, which records the anatomic extent of disease based on T, N, and M elements of the tumor following the completion of surgical therapy per AJCC TNM Cancer Staging:
- T: Tumor size and extent
- N: Lymph node involvement
- M: Presence of metastasis (spread to distant sites)

For analysis purposes, we standardized the staging into five main categories:
- Stage 0: Carcinoma in situ
- Stage I: Early-stage cancer, localized
- Stage II: Locally advanced, no distant spread
- Stage III: Regional spread, often to lymph nodes
- Stage IV: Distant metastasis present
- Unknown: Any unknowns/not reported were combined into one group

This standardization allows for more robust statistical analysis across different cancer types.

## Interpreting Sex-Stratified Survival Results

In the sex-stratified survival analysis, hazard ratios are calculated separately for males and females:

1. **Male-specific analysis**:
   - Male patients are divided into "high" and "low" expression groups based on the median expression value among males only
   - The reference group (denominator in HR) is males with low expression
   - HR > 1: Males with high expression have worse survival than males with low expression
   - HR < 1: Males with high expression have better survival than males with low expression

2. **Female-specific analysis**:
   - Female patients are divided into "high" and "low" expression groups based on the median expression value among females only
   - The reference group (denominator in HR) is females with low expression
   - HR > 1: Females with high expression have worse survival than females with low expression
   - HR < 1: Females with high expression have better survival than females with low expression

This approach allows detection of sex-specific prognostic effects that might be masked in a combined analysis. For example, a signature might be protective in females (HR < 1) but detrimental in males (HR > 1).

## Key Findings

### Survival Analysis
- CD8_C5 (Activation) showed the strongest trend toward improved survival (HR=0.75, p=0.116)
- CD8_C3 (IFN-Response) also showed a protective trend (HR=0.79, p=0.186)
- Sex-stratified analysis revealed different patterns between males and females

### Sex Differences
- Sex distribution: 214 males (61.5%), 134 females (38.5%)
- No significant differences in CD8+ T cell signature expression between sexes
- CD8_C2 (Exhaustion) showed the largest difference (higher in males by 1.47 units, p=0.459)

### Signature Correlations
- High correlations between several signatures:
  - CD8_C1 and CD8_C2: r = 0.957
  - CD8_C2 and CD8_C4: r = 0.878
  - CD8_C1 and CD8_C4: r = 0.851
- These correlations suggest potential redundancy in the signatures

### Stage Distribution
- Stage III disease is most common (45.6% of cohort)
- Stage IV: 15.5%
- Stage II: 12.4%
- Stage I: 3.6%
- Unknown: 22.9%

## Notes

- Stage mapping standardizes clinical stages into I-IV categories
- Duplicate patient IDs are handled by keeping first occurrence
- High correlations (|r| > 0.8) between signatures are flagged
- Sex distribution shows male predominance (61.5% vs 38.5%)
- Stage III disease is most common (45.6% of cohort)

## Contact

For questions or issues, please contact [Your Contact Information]

## Clinical Variables Analysis

The `clinical_analysis.py` module extends our CD8+ T cell analysis to explore relationships between signature scores and important clinical variables. This analysis provides insights into how immune cell populations may differ across clinical subgroups.

### Key Clinical Variables Analyzed

1. **Medications**
   - Counts and percentages of patients for each medication by sex
   - Visualization of top medications used in male vs. female patients
   - For melanoma-specific treatments, filtered by MedPrimaryDiagnosisSite

2. **Systemic Surgery Sequence**
   - Analysis of the sequence in which systemic therapy and surgical procedures were performed
   - Comparison of treatment sequencing patterns between males and females

3. **Age at Diagnosis**
   - Distribution of patients by decade of age (< 20 years, 20-29 years, 30-39, etc.)
   - Sex-specific age patterns at diagnosis
   - Correlation between age and CD8+ T cell signature scores

4. **Primary Diagnosis Site**
   - Anatomical distribution of primary tumors by sex
   - Visualization of the most common diagnosis sites
   - Association between primary site and immune signatures

5. **Pathological Group Stage**
   - Standardized staging (0, I, II, III, IV, Unknown) distribution by sex
   - Analysis of signature scores across different disease stages
   - ANOVA tests to identify stage-associated signatures

6. **Primary vs. Recurrent Disease**
   - Comparison of patients seen for primary vs. recurrent disease
   - Sex-specific patterns in recurrence
   - Differences in CD8+ T cell signatures between primary and recurrent cases

### Statistical Approaches

For each clinical variable, we perform:
- Descriptive statistics (counts, percentages) stratified by sex
- Visualizations (bar plots, heatmaps) to highlight patterns
- Statistical tests to identify significant associations:
  - Pearson correlation for continuous variables (e.g., age)
  - ANOVA for categorical variables with multiple groups (e.g., stage)
  - t-tests for binary comparisons (e.g., sex, primary vs. recurrent)
- Multiple testing correction using Benjamini-Hochberg FDR method

### Output Files

- **CSV Files**:
  - `medication_counts_by_sex.csv` / `medication_percentages_by_sex.csv`
  - `surgery_sequence_counts_by_sex.csv` / `surgery_sequence_percentages_by_sex.csv`
  - `age_group_counts_by_sex.csv` / `age_group_percentages_by_sex.csv`
  - `diagnosis_site_counts_by_sex.csv` / `diagnosis_site_percentages_by_sex.csv`
  - `stage_counts_by_sex.csv` / `stage_percentages_by_sex.csv`
  - `primary_recurrent_counts_by_sex.csv` / `primary_recurrent_percentages_by_sex.csv`
  - `clinical_correlations.csv` (summary of all statistical tests)

- **Plots**:
  - `top_medications_by_sex.png`
  - `surgery_sequence_by_sex.png`
  - `age_distribution_by_sex.png`
  - `top_diagnosis_sites_by_sex.png`
  - `stage_distribution_by_sex.png`
  - `primary_recurrent_by_sex.png`
  - `clinical_associations_heatmap.png` (heatmap of significant associations)

### Interpretation

This comprehensive analysis helps identify:
1. Clinical factors that may influence CD8+ T cell populations
2. Potential confounding variables in survival analyses
3. Patient subgroups that might benefit from targeted immunotherapies
4. Sex-specific patterns in disease presentation and immune response

The results complement our survival analyses by providing clinical context for the observed prognostic effects of CD8+ T cell signatures.

## Z-Score Based Comparative Analysis

To enable direct comparison between males and females, we implemented a Z-score normalization approach that provides a common reference point:

### Standardization Approach

For each CD8+ T cell signature:
1. Z-scores are calculated based on the entire cohort's mean and standard deviation
2. Z-score = (raw score - mean) / standard deviation
3. Patients are classified as "high" (Z > 0) or "low" (Z ≤ 0) expression
4. This creates a biologically meaningful cutoff where "high" = above the population mean

### Distribution Analysis

Distribution plots show how signature expression differs between sexes:
- Histograms of Z-scores for males vs females
- Statistical comparison (t-test) of mean expression levels
- Visual representation of sex-specific expression patterns
- Identification of signatures with significant sex differences

### Survival Analysis

The Z-score approach offers several advantages for survival analysis:
- Common reference point (Z = 0) for both males and females
- Directly comparable hazard ratios between sexes
- Statistically robust classification of "high" vs "low" expression
- Accounting for overall expression distribution

This method helps identify:
1. Signatures with similar prognostic value across sexes (consistent HRs)
2. Signatures with sex-specific effects (divergent HRs)
3. Potential biological mechanisms underlying sex differences in immune function 

## Cohort Selection

To maintain a biologically homogeneous cohort and reduce confounding factors, we excluded patients with the following primary diagnoses:
- Prostate gland
- Vulva, NOS

These cancers have distinct biological properties and sex-specific patterns that could bias the analysis of immune cell populations and their associations with clinical outcomes. The filtering was applied consistently across all analyses including survival analysis, sex differences, and clinical correlations. 