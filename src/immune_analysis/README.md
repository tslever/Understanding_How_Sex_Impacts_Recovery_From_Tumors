# Immune Analysis Module

This module contains scripts for analyzing immune signatures and tumor microenvironment in RNA-seq data.

## Files

- `immune_analysis.py`: Core immune analysis functions
- `microenv.py`: Analysis of tumor microenvironment
- `treatment_analysis.py`: Analysis of treatment effects on immune signatures

## Usage

The immune analysis module can be run directly:

```bash
python immune_analysis.py [--base-path PATH]
```

## Analysis Workflow

1. Load RNA-seq data
2. Calculate immune signature scores
3. Analyze tumor microenvironment composition
4. Correlate immune signatures with clinical outcomes
5. Analyze treatment effects on immune signatures

## Immune Signatures

The module analyzes several immune signatures, including:

- **Cytolytic activity (CYT)**: Based on GZMA and PRF1 expression
- **Interferon gamma (IFNg)**: IFNg response genes
- **T cell inflamed**: T cell inflammation signature
- **Immune checkpoint**: Expression of immune checkpoint genes
- **Myeloid inflammation**: Myeloid cell inflammation signature

## Tumor Microenvironment Analysis

The microenvironment analysis includes:

- Estimation of immune cell infiltration
- Tumor purity assessment
- Stromal content estimation
- Analysis of immune cell interactions

## Output

Results are saved to the following directory:

- `output/immune_analysis/`: Analysis results and plots

## Dependencies

This module depends on the following modules:

- `utils`: For shared functions
- `cd8_analysis`: For CD8 signature scores (optional)

## References

The immune signatures are based on the following publications:

1. Rooney MS, Shukla SA, Wu CJ, Getz G, Hacohen N. Molecular and genetic properties of tumors associated with local immune cytolytic activity. Cell. 2015;160(1-2):48-61.
2. Ayers M, Lunceford J, Nebozhyn M, et al. IFN-γ-related mRNA profile predicts clinical response to PD-1 blockade. J Clin Invest. 2017;127(8):2930-2940.
3. Newman AM, Liu CL, Green MR, et al. Robust enumeration of cell subsets from tissue expression profiles. Nat Methods. 2015;12(5):453-457. 