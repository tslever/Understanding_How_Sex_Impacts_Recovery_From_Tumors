import os
import pandas as pd
import logging
import glob
from collections import defaultdict
import pysam

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_PATH = '/project/orien/data/aws/24PRJ217UVA_IORIG'
WES_DIR = os.path.join(BASE_PATH, 'WES', 'annotated_somatic_vcfs')  # Directory with somatic VCF files
QC_FILE = os.path.join(BASE_PATH, 'Manifest_and_QC_Files', '24PRJ217UVA_20250130_WES_QCMetrics.csv')  # QC metrics file
CLINICAL_CSV = os.path.join(BASE_PATH, 'codes/output/melanoma_patients_with_sequencing.csv')
OUTPUT_DIR = os.path.join(BASE_PATH, 'codes/output/melanoma_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_clinical_data(clinical_csv):
    """
    Load clinical data from CSV and parse list columns.
    
    Args:
        clinical_csv (str): Path to the clinical CSV file.
    
    Returns:
        pd.DataFrame: Clinical data with parsed list columns.
    """
    try:
        df = pd.read_csv(clinical_csv)
        list_cols = ['MelanomaHistologyCodes', 'ClinStages', 'PathStages', 'ICB_Treatments',
                     'MelanomaSequencingSamples', 'SequencingAges', 'SequencingBeforeICB']
        for col in list_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: eval(x) if pd.notnull(x) else [])
        logger.info(f"Loaded clinical data with {len(df)} patients.")
        return df
    except Exception as e:
        logger.error(f"Failed to load clinical data: {e}")
        return None

def load_qc_mapping(qc_file):
    """
    Load QC metrics file to create a mapping from sample IDs to VCF file identifiers.
    
    Args:
        qc_file (str): Path to the QC metrics CSV file.
    
    Returns:
        dict: Mapping of sample IDs to VCF file identifiers.
    """
    try:
        qc_df = pd.read_csv(qc_file)
        # Use actual column names from the QC file
        sample_id_col = 'SLID'  # Contains sample IDs like 'FT-SA165650D'
        vcf_id_col = 'FullID'   # Contains VCF identifiers like 'TA32569_st_t_NA32570_st_g'
        
        # Log available columns for verification
        logger.info(f"QC file columns: {list(qc_df.columns)}")
        
        # Check if the required columns exist
        if sample_id_col not in qc_df.columns or vcf_id_col not in qc_df.columns:
            logger.error(f"QC file missing required columns: {sample_id_col}, {vcf_id_col}. Available columns: {list(qc_df.columns)}")
            return {}
        
        # Create mapping (convert to strings for consistency)
        mapping = dict(zip(qc_df[sample_id_col].astype(str), qc_df[vcf_id_col].astype(str)))
        logger.info(f"Loaded QC mapping for {len(mapping)} samples. Sample mapping example: {list(mapping.items())[:5]}")
        return mapping
    except Exception as e:
        logger.error(f"Failed to load QC mapping: {e}")
        return {}

def find_wes_files(wes_dir, patient_samples, qc_mapping):
    """
    Find WES files for each patient using the QC mapping.
    
    Args:
        wes_dir (str): Directory containing WES data files.
        patient_samples (dict): Mapping of patient IDs to their WES sample IDs.
        qc_mapping (dict): Mapping from sample IDs to VCF file identifiers.
    
    Returns:
        dict: Mapping of patient IDs to lists of WES file paths.
    """
    wes_files = defaultdict(list)
    for patient, samples in patient_samples.items():
        for sample in samples:
            if 'WES:' in sample:
                sample_id = sample.split(':')[1].strip()  # Remove 'WES:' prefix
                vcf_id = qc_mapping.get(sample_id)
                if vcf_id:
                    pattern = os.path.join(wes_dir, f'*{vcf_id}*.vcf.gz')
                    files = glob.glob(pattern)
                    if files:
                        wes_files[patient].extend(files)
                        # logger.info(f"Found {len(files)} WES files for patient {patient} with VCF ID {vcf_id}: {files}")
                    else:
                        logger.warning(f"No WES files found for VCF ID {vcf_id} of patient {patient} with pattern {pattern}")
                else:
                    logger.warning(f"No VCF ID mapping for sample {sample_id} of patient {patient}")
    return wes_files

def extract_somatic_mutations(vcf_file):
    """
    Extract the number of somatic mutations from a VCF file using pysam.
    
    Args:
        vcf_file (str): Path to the VCF file.
    
    Returns:
        int: Number of somatic mutations.
    """
    try:
        vcf = pysam.VariantFile(vcf_file)
        num_variants = sum(1 for _ in vcf.fetch())
        vcf.close()
        logger.debug(f"Extracted {num_variants} mutations from {vcf_file}")
        return num_variants
    except Exception as e:
        logger.error(f"Error parsing VCF file {vcf_file}: {e}")
        return 0

def process_wes_data(wes_files):
    """
    Process WES files to extract features for each patient.
    
    Args:
        wes_files (dict): Mapping of patient IDs to lists of WES file paths.
    
    Returns:
        pd.DataFrame: DataFrame with patient IDs and extracted features.
    """
    features = []
    for patient, files in wes_files.items():
        total_mutations = sum(extract_somatic_mutations(f) for f in files)
        features.append({'PATIENT_ID': patient, 'TotalSomaticMutations': total_mutations})
    df = pd.DataFrame(features)
    logger.info(f"Extracted WES features for {len(df)} patients.")
    return df

def integrate_wes_with_clinical(clinical_df, wes_df):
    """
    Merge WES features with clinical data.
    
    Args:
        clinical_df (pd.DataFrame): Clinical data.
        wes_df (pd.DataFrame): WES-derived features.
    
    Returns:
        pd.DataFrame: Integrated DataFrame.
    """
    integrated_df = clinical_df.merge(wes_df, on='PATIENT_ID', how='left')
    integrated_df['TotalSomaticMutations'] = integrated_df['TotalSomaticMutations'].fillna(0)
    logger.info(f"Integrated WES data with clinical data for {len(integrated_df)} patients.")
    return integrated_df

def main():
    """Main function to run the WES integration pipeline."""
    try:
        # Load clinical data
        clinical_df = load_clinical_data(CLINICAL_CSV)
        if clinical_df is None or clinical_df.empty:
            logger.warning("No clinical data loaded. Exiting.")
            return
        
        # Load QC mapping
        qc_mapping = load_qc_mapping(QC_FILE)
        if not qc_mapping:
            logger.warning("No QC mapping loaded. Exiting.")
            return
        
        # Extract patient samples with WES data
        patient_samples = {
            row['PATIENT_ID']: [s for s in row['MelanomaSequencingSamples'] if 'WES:' in s]
            for _, row in clinical_df.iterrows()
            if 'MelanomaSequencingSamples' in clinical_df.columns and row['MelanomaSequencingSamples']
        }
        
        # Find WES files using QC mapping
        wes_files = find_wes_files(WES_DIR, patient_samples, qc_mapping)
        if not wes_files:
            logger.warning("No WES files found. Exiting.")
            return
        
        # Process WES data
        wes_df = process_wes_data(wes_files)
        if wes_df is None or wes_df.empty:
            logger.warning("No WES features extracted. Exiting.")
            return
        
        # Integrate with clinical data
        integrated_df = integrate_wes_with_clinical(clinical_df, wes_df)
        
        # Save integrated data
        output_file = os.path.join(OUTPUT_DIR, 'integrated_clinical_wes.csv')
        integrated_df.to_csv(output_file, index=False)
        logger.info(f"Saved integrated data to {output_file}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()