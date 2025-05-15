# CD8+ T Cell Group Analysis

This analysis focuses on two functionally distinct CD8+ T cell groups derived from single-cell RNA-seq data:

## CD8+ T Cell Groups

- **CD8_B**: "Bad" prognostic group (enriched in non-responders)
  - Combines clusters 1-3 from the original analysis
  - Associated with exhaustion, proliferation, and IFN-response
  - Contains 641 genes including markers like PDCD1 (PD-1), HAVCR2 (TIM-3), LAG3, CTLA4
  - Characterized by T cell dysfunction and exhaustion phenotypes

- **CD8_G**: "Good" prognostic group (enriched in responders)
  - Combines clusters 4-6 from the original analysis
  - Associated with memory, activation, and naive phenotypes
  - Contains 35 genes including markers like IL7R, TCF7, CCR7
  - Characterized by T cell stemness and memory-like properties

## CD8_G/CD8_B Ratio

We calculate the ratio of CD8_G to CD8_B expression as a potential biomarker:
- **CD8_GtoB_ratio**: Direct ratio of CD8_G / CD8_B scores
- **CD8_GtoB_log**: Log2-transformed ratio for better statistical properties

This ratio represents the balance between "good" and "bad" CD8+ T cell states in the tumor microenvironment. Higher values indicate a more favorable immune environment potentially associated with better response to immunotherapy.

## Analysis Methods

### Expression Distribution Analysis
- Raw expression distributions of CD8_B and CD8_G by sex
- Statistical comparison (t-test) of mean expression levels
- Visual representation of sex-specific expression patterns

### Z-Score Based Survival Analysis
- Z-scores calculated based on entire cohort's mean and standard deviation
- Patients classified as "high" (Z > 0) or "low" (Z ≤ 0) expression
- Sex-stratified survival analysis using Cox proportional hazards models
- Kaplan-Meier plots for each group and sex

## Cohort Selection

To maintain a biologically homogeneous cohort and reduce confounding factors, we excluded patients with the following primary diagnoses:
- Prostate gland
- Vulva, NOS

These cancers have distinct biological properties and sex-specific patterns that could bias the analysis of immune cell populations and their associations with clinical outcomes.

## Relationship to 6-Cluster Analysis

This analysis simplifies the original 6-cluster approach by combining functionally related clusters:
- CD8_B combines clusters 1-3 (exhaustion/dysfunction-related)
- CD8_G combines clusters 4-6 (memory/activation-related)

This grouping is based on:
1. Functional similarities observed in the original publication (PMID: 30388456)
2. High correlations detected between clusters within each group
3. Biological relevance to response/resistance mechanisms

## Output Files

- **Group scores**: `group_scores.csv`
- **Sex differences**: `group_sex_differences.csv`
- **Survival analysis**: `male_group_survival.csv`, `female_group_survival.csv`

### Plots
- Distribution plots: `CD8_B_distribution.png`, `CD8_G_distribution.png`, `CD8_GtoB_ratio_distribution.png`
- Survival curves: `CD8_B_male_zscore.png`, `CD8_B_female_zscore.png`, etc.
- Forest plot: `group_survival_forest.png`

## Interpretation

The CD8_B and CD8_G groups represent distinct functional states of CD8+ T cells in the tumor microenvironment:

- **CD8_B (Non-responder enriched)**: High expression indicates an exhausted/dysfunctional T cell phenotype, potentially associated with resistance to immunotherapy and poorer outcomes.

- **CD8_G (Responder enriched)**: High expression indicates a memory/stemness T cell phenotype, potentially associated with response to immunotherapy and better outcomes.

- **CD8_G/CD8_B ratio**: A higher ratio suggests a more favorable balance of T cell states and may be a more robust biomarker than either signature alone.

Sex-specific differences in these signatures may help explain differential responses to immunotherapy between males and females observed in clinical trials.