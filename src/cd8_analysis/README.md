# CD8 Analysis Module

This module contains scripts for analyzing CD8+ T cell signatures in RNA-seq data.

## Files

- `cd8_analysis.py`: Core CD8 analysis functions
- `cd8_groups_analysis.py`: Analysis of CD8 groups and their ratios
- `clinical_analysis.py`: Analysis of clinical data in relation to CD8 signatures

## Usage

The CD8 analysis module is typically used by other modules, but can also be run directly:

```bash
python cd8_groups_analysis.py [--base-path PATH]
```

## Analysis Workflow

1. Load RNA-seq data
2. Calculate CD8 signature scores
3. Group samples based on CD8 signatures
4. Calculate CD8 group ratios
5. Analyze clinical correlations

## CD8 Signatures

The module analyzes two main CD8+ T cell signatures:

- **CD8_B**: Non-responder enriched (Clusters 1-3)
- **CD8_G**: Responder enriched (Clusters 4-6)

From these, two derived metrics are calculated:

- **CD8_GtoB_ratio**: Responder/Non-responder ratio
- **CD8_GtoB_log**: Log2(Responder/Non-responder ratio)

## Output

Results are saved to the following directory:

- `output/cd8_analysis/cd8_groups_analysis/`: Analysis results and plots

## Dependencies

This module depends on the following modules:

- `utils`: For shared functions

## References

The CD8+ T cell signatures are based on the following publications:

1. Jiang P, Gu S, Pan D, et al. Signatures of T cell dysfunction and exclusion predict cancer immunotherapy response. Nat Med. 2018;24(10):1550-1558.
2. Danaher P, Warren S, Lu R, et al. Pan-cancer adaptive immune resistance as defined by the Tumor Inflammation Signature (TIS): results from The Cancer Genome Atlas (TCGA). J Immunother Cancer. 2018;6(1):63. 