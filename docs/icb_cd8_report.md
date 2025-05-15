# Immune Checkpoint Blockade (ICB) and CD8+ T Cell Signature Analysis

## Overview

This report summarizes the analysis of immune checkpoint blockade (ICB) treatment patterns and their relationship with CD8+ T cell signatures in our cohort. We examined sex-specific differences in ICB treatment patterns and how CD8+ T cell functional groups correlate with treatment status and outcomes.

## ICB Treatment Distribution

### Patient ICB Status

A total of 340 patients had known ICB treatment status:
- **ICB-naive**: 101 patients (29.7%)
- **ICB-experienced**: 239 patients (70.3%)

The sex distribution was similar between treatment groups:
- **Males**: 64 ICB-naive (30.5%), 146 ICB-experienced (69.5%)
- **Females**: 37 ICB-naive (28.5%), 93 ICB-experienced (71.5%)

This balanced distribution suggests no sex bias in ICB treatment allocation in our cohort.

![Patient ICB Status](output/cd8_analysis/icb_medication_analysis/plots/patient_icb_status.png)
*Figure 1: Distribution of ICB treatment status by sex*

### ICB Treatment Types

The following ICB treatment types were observed:

| ICB Class | Female | Male | Total |
|-----------|--------|------|-------|
| PD-1 | 52 | 88 | 140 |
| PD-1+CTLA-4 | 43 | 59 | 102 |
| CTLA-4 | 6 | 13 | 19 |
| PD-1+PD-L1 | 1 | 0 | 1 |
| None | 43 | 73 | 116 |
| **All** | **145** | **233** | **378** |

PD-1 inhibitors were the most common treatment, followed by combination therapy with PD-1+CTLA-4 inhibitors. The distribution of treatment types was proportionally similar between males and females.

## CD8+ T Cell Signatures

We analyzed two functionally distinct CD8+ T cell signatures:

1. **CD8_B** (Non-responder enriched, Clusters 1-3): Associated with T cell exhaustion and dysfunction
2. **CD8_G** (Responder enriched, Clusters 4-6): Associated with T cell memory and stemness

### CD8 Signature Expression by ICB Status

#### CD8_B Expression

| Group | ICB Status | Mean | SD | Count |
|-------|------------|------|----|----|
| Male | ICB-naive | 61.22 | 17.11 | 64 |
| Male | ICB-experienced | 56.29 | 19.94 | 146 |
| Female | ICB-naive | 54.93 | 21.50 | 37 |
| Female | ICB-experienced | 56.93 | 17.77 | 93 |

Statistical comparisons:
- Overall difference: -2.38 (p=0.294)
- Male difference: -4.93 (p=0.087)
- Female difference: +1.99 (p=0.588)

![CD8_B by ICB Status](output/cd8_analysis/icb_medication_analysis/plots/CD8_B_by_icb.png)
*Figure 2: CD8_B expression by ICB status and sex*

#### CD8_G Expression

| Group | ICB Status | Mean | SD | Count |
|-------|------------|------|----|----|
| Male | ICB-naive | 36.13 | 14.38 | 64 |
| Male | ICB-experienced | 34.60 | 16.18 | 146 |
| Female | ICB-naive | 37.48 | 17.50 | 37 |
| Female | ICB-experienced | 35.25 | 17.81 | 93 |

Statistical comparisons:
- Overall difference: -1.77 (p=0.364)
- Male difference: -1.53 (p=0.515)
- Female difference: -2.23 (p=0.518)

![CD8_G by ICB Status](output/cd8_analysis/icb_medication_analysis/plots/CD8_G_by_icb.png)
*Figure 3: CD8_G expression by ICB status and sex*

### Key Observations on Expression Patterns

1. **Sex-specific CD8_B patterns**: 
   - Males show a trend toward lower CD8_B expression in ICB-experienced patients (approaching significance at p=0.087)
   - Females show an opposite pattern with slightly higher CD8_B in ICB-experienced patients

2. **CD8_G patterns**:
   - Both sexes show slightly lower CD8_G expression in ICB-experienced patients
   - None of these differences reached statistical significance

## Survival Analysis by ICB Status

### Overall Survival by ICB Status

ICB treatment is associated with overall survival outcomes, as shown in Figure 4:

![Overall Survival by ICB Status](output/cd8_analysis/icb_medication_analysis/plots/overall_survival_by_icb.png)
*Figure 4: Kaplan-Meier curves showing overall survival by ICB treatment status*

### ICB-Naive Patients

#### Males (n=63)

| Signature | Hazard Ratio | 95% CI | p-value |
|-----------|--------------|--------|---------|
| CD8_B | 2.31 | 0.93-5.70 | 0.071 |
| CD8_G | 0.91 | 0.39-2.11 | 0.818 |

![CD8_B ICB-Naive Male Survival](output/cd8_analysis/icb_medication_analysis/plots/CD8_B_ICB_naive_male_survival.png)
*Figure 5: Survival curves for ICB-naive males stratified by CD8_B expression*

![CD8_G ICB-Naive Male Survival](output/cd8_analysis/icb_medication_analysis/plots/CD8_G_ICB_naive_male_survival.png)
*Figure 6: Survival curves for ICB-naive males stratified by CD8_G expression*

#### Females (n=37)

| Signature | Hazard Ratio | 95% CI | p-value |
|-----------|--------------|--------|---------|
| CD8_B | 0.08 | 0.01-0.90 | 0.040 |
| CD8_G | 2.41 | 0.46-12.72 | 0.300 |

![CD8_B ICB-Naive Female Survival](output/cd8_analysis/icb_medication_analysis/plots/CD8_B_ICB_naive_female_survival.png)
*Figure 7: Survival curves for ICB-naive females stratified by CD8_B expression*

![CD8_G ICB-Naive Female Survival](output/cd8_analysis/icb_medication_analysis/plots/CD8_G_ICB_naive_female_survival.png)
*Figure 8: Survival curves for ICB-naive females stratified by CD8_G expression*

### ICB-Experienced Patients

#### Males (n=142)

| Signature | Hazard Ratio | 95% CI | p-value |
|-----------|--------------|--------|---------|
| CD8_B | 1.00 | 0.59-1.71 | 0.993 |
| CD8_G | 0.69 | 0.40-1.19 | 0.184 |

![CD8_B ICB-Experienced Male Survival](output/cd8_analysis/icb_medication_analysis/plots/CD8_B_ICB_experienced_male_survival.png)
*Figure 9: Survival curves for ICB-experienced males stratified by CD8_B expression*

![CD8_G ICB-Experienced Male Survival](output/cd8_analysis/icb_medication_analysis/plots/CD8_G_ICB_experienced_male_survival.png)
*Figure 10: Survival curves for ICB-experienced males stratified by CD8_G expression*

#### Females (n=93)

| Signature | Hazard Ratio | 95% CI | p-value |
|-----------|--------------|--------|---------|
| CD8_B | 1.04 | 0.51-2.12 | 0.911 |
| CD8_G | 0.84 | 0.42-1.70 | 0.630 |

![CD8_B ICB-Experienced Female Survival](output/cd8_analysis/icb_medication_analysis/plots/CD8_B_ICB_experienced_female_survival.png)
*Figure 11: Survival curves for ICB-experienced females stratified by CD8_B expression*

![CD8_G ICB-Experienced Female Survival](output/cd8_analysis/icb_medication_analysis/plots/CD8_G_ICB_experienced_female_survival.png)
*Figure 12: Survival curves for ICB-experienced females stratified by CD8_G expression*

### Summary Forest Plots

The forest plots below summarize the hazard ratios for CD8_B and CD8_G signatures in ICB-naive and ICB-experienced patients:

![ICB-Naive Survival Forest](output/cd8_analysis/icb_medication_analysis/plots/ICB_naive_survival_forest.png)
*Figure 13: Forest plot of hazard ratios for CD8 signatures in ICB-naive patients*

![ICB-Experienced Survival Forest](output/cd8_analysis/icb_medication_analysis/plots/ICB_experienced_survival_forest.png)
*Figure 14: Forest plot of hazard ratios for CD8 signatures in ICB-experienced patients*

### Key Observations on Survival

1. **Striking sex difference in ICB-naive patients**:
   - In ICB-naive females, high CD8_B expression is strongly protective (HR=0.08, p=0.04)
   - In ICB-naive males, high CD8_B expression shows an opposite trend toward worse outcomes (HR=2.31, p=0.07)

2. **No significant effects in ICB-experienced patients**:
   - Neither signature shows significant association with survival in ICB-experienced patients of either sex
   - This suggests that ICB treatment may neutralize the prognostic impact of these signatures

## Treatment Selection Bias and Methodological Considerations

### Global Median Approach for Survival Analysis

In our survival analyses, we used the global median expression of CD8 signatures as cutoff points for all patients, regardless of ICB treatment status. This approach has several important implications:

1. **Consistent Reference Point**: Using a single threshold allows direct comparison across all subgroups, making interpretation more straightforward.

2. **Statistical Stability**: The global median leverages the full dataset, providing a more stable cutoff point than subgroup-specific medians, especially for smaller subgroups like ICB-naive females (n=37).

3. **Biological Relevance**: The absolute expression level of CD8+ T cell signatures likely has biological significance that exists independently of treatment status.

4. **Limitations**: This approach assumes the biological significance of a given expression level is the same across treatment groups, which may not always be true.

### Treatment Selection Patterns

Our analysis reveals potential treatment selection patterns that may influence the interpretation of results:

1. **Sex-Specific CD8_B Patterns**:
   - Males: ICB-experienced males have lower CD8_B expression (56.29) compared to ICB-naive males (61.22), approaching significance (p=0.087)
   - Females: ICB-experienced females have slightly higher CD8_B expression (56.93) than ICB-naive females (54.93)

2. **Assumption of Non-Sex-Specific Selection**:
   - For this analysis, we assume that treatment selection criteria were not explicitly sex-specific
   - However, the opposite patterns observed in males and females suggest that either:
     a) Treatment selection may have been influenced by different factors in males versus females
     b) The biological response to treatment differs by sex
     c) There are underlying differences in disease biology between sexes that affect both treatment selection and outcomes

3. **Potential Confounding Factors**:
   - Disease stage, tumor burden, and comorbidities likely influence both treatment selection and outcomes
   - These factors may distribute differently between males and females

### Pre- vs. Post-Treatment Tumor Samples

An important distinction in our dataset is between tumor samples collected before ICB treatment versus those collected after treatment initiation:

1. **Sample Timing**:
   - **Pre-treatment samples**: Reflect the baseline immune microenvironment before ICB intervention
   - **Post-treatment samples**: Reflect the immune microenvironment that has been modified by ICB

2. **Current Dataset Limitations**:
   - Our current analysis primarily distinguishes between ICB-naive patients (never received ICB) and ICB-experienced patients (received ICB at some point)
   - We have limited information on whether tumor samples from ICB-experienced patients were collected before or after treatment initiation
   - The tumor status is marked as "Unknown" for most ICB-experienced patients in our dataset

3. **Implications for Interpretation**:
   - If most samples from ICB-experienced patients were collected pre-treatment, differences in CD8 signatures may reflect selection bias rather than treatment effects
   - If samples were collected post-treatment, differences may reflect both selection bias and treatment-induced changes
   - The lack of significant prognostic impact in ICB-experienced patients could be due to either successful treatment equalization or pre-existing differences in tumor biology

4. **Future Directions**:
   - Obtaining clear information on sample timing relative to treatment would significantly enhance our ability to distinguish selection effects from treatment effects
   - Longitudinal samples (pre- and post-treatment from the same patients) would be ideal for assessing treatment-induced changes

## Discussion and Implications

### Sex-Specific Immune Responses

The opposite direction of CD8_B effects in ICB-naive males and females suggests fundamental sex differences in how the immune system responds to tumors. High levels of exhausted/dysfunctional T cells (CD8_B) appear protective in treatment-naive females but potentially harmful in treatment-naive males.

### Treatment Considerations

1. **Sample size limitations**:
   - The female ICB-naive group (n=37) is relatively small, so the strong protective effect of CD8_B should be interpreted with caution
   - Larger validation studies are needed to confirm these findings

2. **Biomarker potential**:
   - CD8_B signature may have sex-specific biomarker potential in ICB-naive patients
   - The lack of prognostic value in ICB-experienced patients suggests that treatment may override the baseline immune configuration

3. **Treatment selection**:
   - These findings suggest that treatment decisions based on immune signatures may need to be sex-specific
   - Males and females with similar immune profiles may benefit from different treatment approaches

## Limitations and Future Directions

1. **Heterogeneity of ICB treatments**:
   - Our analysis combined different ICB types (PD-1, CTLA-4, combinations)
   - Future analyses should examine signature-specific effects for different treatment types

2. **Time on treatment**:
   - Duration of ICB treatment was not considered in this analysis
   - Future work should examine how treatment duration affects these signatures

3. **Propensity score matching**:
   - Implementing propensity score matching would better balance ICB-naive and ICB-experienced groups
   - This would help control for confounding factors that influence treatment selection and provide more reliable estimates of treatment effects

4. **Absolute vs. Relative Expression Cutoffs**:
   - Future analyses should compare results using different cutoff approaches (global median, sex-specific medians, quartiles)
   - This would test the robustness of findings to methodological choices

5. **Validation in larger cohorts**:
   - The striking sex-specific findings in ICB-naive patients should be validated in larger, independent cohorts
   - Particular attention should be paid to the protective effect of CD8_B in females 