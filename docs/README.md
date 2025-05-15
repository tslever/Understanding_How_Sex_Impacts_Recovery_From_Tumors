# ICI Sex Project

This repository contains code for analyzing sex-based differences in ICI data.

## Environment Setup

### Initial Setup

1. Clone or navigate to the project directory:

```bash
cd /project/orien/data/aws/24PRJ217UVA_IORIG/codes
```

2. Make the setup script executable and run it:

```bash
chmod +x setup.sh
./setup.sh
```

This script will:
- Install Miniconda3 in the project directory if not already installed
- Create a Python environment named `ici_sex`
- Install required Python packages:
  - numpy
  - matplotlib
  - scikit-learn
  - pandas
  - scipy

### Activating the Environment

To use the environment, you need to:

1. First source conda initialization:
```bash
source /project/orien/data/aws/24PRJ217UVA_IORIG/codes/miniconda3/etc/profile.d/conda.sh 
```

2. Then activate the environment:
```bash
conda activate ici_sex
```

You can verify the environment is active:
```bash
conda info --envs  # Shows all environments with active one marked
python --version   # Should show Python 3.9
```

Optional: Add to your .bashrc for automatic sourcing:
```bash
# Add these lines to ~/.bashrc
export PATH="/project/orien/data/aws/24PRJ217UVA_IORIG/codes/miniconda3/bin:$PATH"
source /project/orien/data/aws/24PRJ217UVA_IORIG/codes/miniconda3/etc/profile.d/conda.sh
```

## Project Structure

### Code Directory
```
/project/orien/data/aws/24PRJ217UVA_IORIG/codes/
├── setup.sh          # Environment setup script
├── README.md         # This documentation
└── miniconda3/       # Conda installation (created by setup.sh)
```

### Data Directory Structure
```
/project/orien/data/aws/24PRJ217UVA_IORIG/
├── Clinical_Data/
│   ├── 20240726_ORIEN_SITE_CytogeneticAbnormalities_V4
│   ├── 20240726_ORIEN_SITE_Diagnosis_V4
│   ├── 20240726_ORIEN_SITE_FamilyHistory_V4
│   ├── 20240726_ORIEN_SITE_Imaging_V4
│   ├── 20240726_ORIEN_SITE_Labs_V4
│   ├── 20240726_ORIEN_SITE_Medications_V4
│   ├── 20240726_ORIEN_SITE_MetastaticDisease_V4
│   ├── 20240726_ORIEN_SITE_Outcomes_V4
│   ├── 20240726_ORIEN_SITE_PatientHistory_V4
│   ├── 20240726_ORIEN_SITE_PatientMaster_V4
│   └── ... [additional clinical files]
├── Manifest_and_QC_Files/
│   ├── 20240726_ORIEN_SITE_RNASeq_QCMetrics.csv     # RNA-SeQC metrics
│   ├── 20240726_ORIEN_SITE_TMB_MSI_table.csv        # MSI and TMB scores
│   ├── 20240726_ORIEN_SITE_WES_QCMetrics.csv        # WES metrics
│   └── 20240726_ORIEN_SITE_WES_TN_Concordance.csv   # Concordance metrics
├── RNAseq/
│   ├── alignment_crams/                  # Compressed alignment files
│   ├── fusion_results/                   # STAR-Fusion and Arriba results
│   └── gene_and_transcript_expression_results/
└── WES/
    ├── alignment_crams/                  # Compressed alignment files
    ├── annotated_germline_vcfs/         # Funcotator annotated germline VCFs
    ├── annotated_somatic_vcfs/          # Paired tumor/germline VCFs
    ├── g.VCFs/                          # Single tumor and germline G.VCFs
    ├── msisensor2/                      # MSI analysis results
    ├── scarHRD/                         # Homologous recombination deficiency
    └── somatic_CNV/                     # Copy number variation results
```

## Data Description

### Data Processing Overview
All data has been processed using the GRCh38.p13 human genome build and GenCode gene build version 32.

### WES (Whole Exome Sequencing) Data Processing
1. **Read Alignment and Quality Control**
   - Processed using Sentieon App with GRCh38/hg38 reference
   - BWA mem aligner (sentieon_release_201911)
   - Includes sorting, duplicate marking, base quality recalibration
   - Outputs stored in .cram format with .crai index

2. **Variant Detection**
   - Single sample: Haplotyper algorithm (equivalent to GATK 4.0)
   - Somatic mutations: TNhaplotyper2 with --trim_soft_clip option
   - Outputs in .vcf.gz and .g.vcf.gz formats

3. **Variant Annotation**
   - Using Funcotator (GATK v4.1.6.0)
   - Germline samples: *.ftg.vcf.gz
   - Somatic/tumor samples: *.ft.vcf.gz

4. **Additional Analyses**
   - **Panel of Normals (PoN)**: Filters mutations present in >0.5% of ORIEN Avatar population
   - **TMB**: Calculated from filtered indel, missense, stop gain/loss mutations
   - **MSI**: Calculated using msisensor2 (v0.1)
   - **CNV and HRD**: 
     - Using Sequenza (v3.0.0) and scarHRD (v0.1.0)
     - Outputs include gene-level copy number and HRD scores

### RNA-Seq Data Processing
1. **Read Alignment**
   - STAR aligner (v.2.7.3a)
   - Using GRCh38/hg38 reference and Gencode v32 annotation
   - Outputs include:
     - Junction files for fusion detection
     - Genome and transcriptome alignments
     - QC VCF files

2. **Gene Fusion Analysis**
   - Tools: STAR-Fusion (v1.8.0) and Arriba (v1.1.0)
   - Results merged and filtered
   - Includes graphical outputs in PDF format

3. **Expression Quantification**
   - Using RSEM (v1.3.1)
   - Provides:
     - Estimated mapped reads
     - FPKM values
     - TPM values
   - Available at both transcript and gene level

### Data Organization
Contains patient-level information including diagnoses, medications, outcomes, and various clinical assessments.

### Manifest and QC Files
- RNA-Seq quality metrics
- TMB (Tumor Mutation Burden) and MSI (Microsatellite Instability) scores
- WES quality metrics and tumor-normal concordance data

### RNA-Seq Data
- Alignment files (CRAM format)
- Gene fusion predictions from STAR-Fusion and Arriba
- Gene and transcript-level expression results

### WES (Whole Exome Sequencing) Data
- Alignment files (CRAM format)
- Germline and somatic variant calls (VCF format)
- MSI sensor results
- Copy number variation analysis
- Homologous recombination deficiency scores

## Usage

[This section will be updated as the project develops with instructions for running analyses and descriptions of key scripts.]

