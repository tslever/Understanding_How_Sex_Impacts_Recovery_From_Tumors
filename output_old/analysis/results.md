# Immune Cell Analysis Results

## Overview
Analysis of immune cell composition differences between sexes in tumor samples.
- Total samples analyzed: 321 patients
- Multiple testing correction: Bonferroni (threshold: 7.46e-04)
- Cell types analyzed: 67 total

## Key Findings

### Innate Immune Cells
- Total tests: 12 cell types
- No significant differences between sexes
- Closest to significance:
  1. Activated Dendritic Cells (p = 0.061)
  2. Basophils (p = 0.270)

![aDC Distribution](aDC%HPCA%1.txt_SEX_dist.png)

### Adaptive Immune Cells
- Total tests: 9 cell types
- 1 nominally significant cell type
- Most significant findings:
  1. B cells (p = 0.021) *
  2. B cells variant 2 (p = 0.158)

![B Cells Distribution](B-cells%FANTOM%1.txt_SEX_dist.png)

### Stromal Cells
- Total tests: 9 cell types
- 1 nominally significant cell type
- Most significant findings:
  1. Adipocytes (p = 0.046) *
  2. Adipocytes variant 2 (p = 0.132)

![Adipocytes Distribution](Adipocytes%HPCA%1.txt_SEX_dist.png)

## Cell Type Correlations

### Overall Correlations
![Overall Correlation Matrix](correlation_matrix.png)

### Category-Specific Correlations
- Innate Immune Cells
![Innate Correlations](correlation_matrix_Innate.png)

- Adaptive Immune Cells
![Adaptive Correlations](correlation_matrix_Adaptive.png)

- Stromal Cells
![Stromal Correlations](correlation_matrix_Stromal.png)

## Statistical Summary

### Significance Levels
- *: p < 0.05 (nominal significance)
- **: p < 0.01
- ***: p < 7.46e-04 (Bonferroni-corrected significance)

### Results by Category
1. **Innate Immune Cells**
   - Nominally significant: 0
   - Bonferroni significant: 0

   | Cell Type | p-value | Effect Size | Significance |
   |-----------|---------|-------------|--------------|
   | Activated Dendritic Cells | 0.061 | 0.004 | ns |
   | Activated Dendritic Cells (v2) | 0.261 | 0.016 | ns |
   | Basophils | 0.270 | 0.036 | ns |
   | Activated Dendritic Cells (v3) | 0.451 | 0.003 | ns |
   | Basophils (v2) | 0.475 | 0.045 | ns |

2. **Adaptive Immune Cells**
   - Nominally significant: 1
   - Bonferroni significant: 0

   | Cell Type | p-value | Effect Size | Significance |
   |-----------|---------|-------------|--------------|
   | B cells | 0.021 | 0.001 | * |
   | B cells (v2) | 0.158 | 0.007 | ns |
   | B cells (v3) | 0.180 | 0.258 | ns |
   | B cells (v4) | 0.341 | 0.238 | ns |
   | B cells (v5) | 0.403 | 0.004 | ns |

3. **Stromal Cells**
   - Nominally significant: 1
   - Bonferroni significant: 0

   | Cell Type | p-value | Effect Size | Significance |
   |-----------|---------|-------------|--------------|
   | Adipocytes | 0.046 | 0.052 | * |
   | Adipocytes (v2) | 0.132 | 0.011 | ns |
   | Adipocytes (v3) | 0.313 | 0.010 | ns |
   | Adipocytes (v4) | 0.344 | 0.030 | ns |
   | Adipocytes (v5) | 0.427 | 0.014 | ns |

Note: 
- Effect size represents mean difference between groups
- Significance codes: 
  * p < 0.05
  ** p < 0.01
  *** p < 7.46e-04 (Bonferroni)
  ns: not significant
- (v2), (v3), etc. represent different signature versions from xCell

## Methods
- Mann-Whitney U test for group comparisons
- Bonferroni correction for multiple testing
- Pearson correlation for cell type relationships
- xCell scores used for cell type quantification

## Conclusions
1. Limited sexual dimorphism in tumor immune composition
   - Only 2/67 cell types show nominal differences
   - Effect sizes are very small (< 0.06)
   - No robust differences after multiple testing correction

2. B cells show nominal differences between sexes
   - p = 0.021, but effect size = 0.001
   - Multiple B cell signatures show no difference
   - Likely a false positive given multiple testing

3. Adipocyte content shows nominal sex-specific patterns
   - p = 0.046, effect size = 0.052
   - Most consistent finding across signatures
   - May reflect general sex differences in adiposity

4. No findings survive strict multiple testing correction
   - Bonferroni threshold: p < 7.46e-04
   - Closest finding (aDCs): p = 0.061
   - Suggests tumor immune composition is largely similar between sexes

## Implications
1. Tumor immune microenvironment appears largely sex-independent
2. Any sex-specific treatment strategies should not assume major immune differences
3. Other factors (age, stage, treatment) may be more important than sex
4. Future studies might focus on specific contexts where sex differences emerge 

## Discussion: Sex Differences in Immunotherapy Response

Despite finding minimal sex-based differences in tumor immune composition, clinical studies consistently show sex-based differences in immunotherapy outcomes. This apparent paradox might be explained by:

1. **Dynamic vs. Static Measurements**
   - Our analysis captures a static snapshot of immune composition
   - Sex differences may emerge in dynamic immune responses to therapy
   - Functional differences may exist despite similar cell proportions

2. **Molecular Mechanisms**
   - Sex hormones (estrogen, testosterone) influence immune cell function
   - X-chromosome genes affect immune responses
   - These factors might not affect baseline composition but impact response

3. **Complex Interactions**
   - Sex differences might emerge through:
     - Drug metabolism differences
     - Hormone-immune system interactions
     - Sex-specific molecular pathways
     - Microbiome differences

4. **Limitations of Current Analysis**
   - Bulk tissue measurements may miss spatial relationships
   - Cell proportions don't reflect activation states
   - Limited to pre-treatment samples
   - May miss rare but important cell populations

## Future Directions

To better understand sex-based differences in immunotherapy response:

1. **Temporal Analysis**
   - Study immune changes during treatment
   - Compare male vs. female response dynamics

2. **Functional Studies**
   - Examine immune cell activation states
   - Measure cytokine/chemokine profiles
   - Assess immune cell functionality

3. **Integration with Other Data**
   - Hormone levels
   - Genomic data
   - Microbiome profiles
   - Treatment outcomes

4. **Advanced Technologies**
   - Single-cell RNA sequencing
   - Spatial transcriptomics
   - Multiplex imaging
   - Functional assays 