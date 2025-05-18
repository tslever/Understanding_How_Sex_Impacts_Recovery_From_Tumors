"""
Melanoma Clinical Data Processing Pipeline
==========================================
Processes melanoma patient data from ORIEN datasets, focusing on:
- Loading and integrating clinical data
- Filtering melanoma cases
- Processing ICB treatment and stage information
- Generating demographic plots
"""

import os
import logging
import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

logger = logging.getLogger(__name__)
plt.style.use('seaborn-v0_8-whitegrid')
try:
    from src.data_processing.utils import create_map_from_qc
except ImportError:
    try:
        from utils import create_map_from_qc
    except ImportError:
        print("Error: Cannot import create_map_from_qc function")
        sys.exit(1)
    

ICB_DRUGS = {
    'PD-1': ['PEMBROLIZUMAB', 'NIVOLUMAB', 'CEMIPLIMAB', 'SINTILIMAB'],
    'PD-L1': ['ATEZOLIZUMAB', 'AVELUMAB', 'DURVALUMAB'],
    'CTLA-4': ['IPILIMUMAB', 'TREMELIMUMAB'],
    'LAG3': ['RELATLIMAB']
}
ICB_DRUGS_FLAT = [drug.lower() for sublist in ICB_DRUGS.values() for drug in sublist]

MELANOMA_CODES = ['8720', '8721', '8722', '8723', '8728', '8730', '8740', '8741', '8742', '8743', '8744', '8745', '8746', '8761', '8770', '8771', '8772', '8773', '8774', '8780']

def is_melanoma_histology_code(code):
    """
    Check if a histology code corresponds to melanoma.
    
    Args:
        code (str): Histology code (e.g., '8720/3').
    
    Returns:
        bool: True if the code indicates melanoma, False otherwise.
    """
    if not isinstance(code, str) or len(code) < 5 or not code[:5].isdigit():
        return False
    # Extract the code parts
    code_prefix = code[:5]  # e.g., "8720/3" -> "87203"
    base_code = code_prefix[:4]  # e.g., "8720"
    behavior_code = code_prefix[4]  # e.g., "3"
    # Define melanoma codes (adjust as needed)
    MELANOMA_CODES = {'8720', '8721', '8730', '8740', '8742', '8761', '8771', '8772'}
    # Check if it's a melanoma code with malignant behavior
    return base_code in MELANOMA_CODES and behavior_code == '3'

def load_clinical_data(base_path):
    """
    Load clinical data from the specified base path and subdirectory.

    Args:
        base_path (str): Base path to the project directory.

    Returns:
        dict: Dictionary of DataFrames loaded from the clinical data files.
    """
    # Define the subdirectory where the files are located
    sub_dir = os.path.join(base_path, 'Clinical_Data', '24PRJ217UVA_NormalizedFiles')
    
    if not os.path.exists(sub_dir):
        logger.error(f"Subdirectory not found: {sub_dir}")
        return {}
    
    # List all CSV files in the subdirectory
    files = [f for f in os.listdir(sub_dir) if f.endswith('.csv')]
    
    # Define keywords for each data type to match files
    keyword_mapping = {
        'patients': ['PatientMaster'],
        'diagnoses': ['Diagnosis'],
        'treatments': ['Medications'],  # Assuming 'treatments' corresponds to 'Medications'
        'outcomes': ['Outcomes'],       # Assuming 'outcomes' corresponds to 'Outcomes'
        'sequencing': ['TumorSequencing'],
        'clinical_mol_linkage': ['ClinicalMolLinkage']  # Add this
    }
    
    dfs = {}
    for data_type, keywords in keyword_mapping.items():
        # Find files that contain any of the keywords (case-insensitive)
        matching_files = [f for f in files if any(kw.lower() in f.lower() for kw in keywords)]
        
        if not matching_files:
            logger.warning(f"No file found for {data_type}")
            continue
        
        if len(matching_files) > 1:
            logger.warning(f"Multiple files found for {data_type}: {matching_files}. Using the first one.")
        
        # Use the first matching file
        file_path = os.path.join(sub_dir, matching_files[0])
        
        try:
            df = pd.read_csv(file_path)
            # Rename 'AvatarKey' to 'PATIENT_ID' if present (for compatibility)
            if 'AvatarKey' in df.columns:
                df = df.rename(columns={'AvatarKey': 'PATIENT_ID'})
            dfs[data_type] = df
            logger.info(f"Loaded {data_type} from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    return dfs
def convert_age(age):
    """Convert age to numeric value."""
    if pd.isna(age):
        return np.nan
    return 90.0 if isinstance(age, str) and '90 or older' in age else float(age) if isinstance(age, (int, float, str)) and str(age).replace('.','',1).isdigit() else np.nan

def classify_icb(med):
    """Identify ICB drugs in medication names."""
    if not isinstance(med, str):
        return None
    med_lower = med.lower()
    return next((drug for drug in ICB_DRUGS_FLAT if drug in med_lower), None)

def get_cancer_type(code):
    """Classify cancer type from histology code."""
    if pd.isna(code) or not str(code):
        return 'Unknown'
    return 'Melanoma' if any(str(code).split('/')[0].startswith(m) for m in MELANOMA_CODES) else 'Other'

def clean_stage(stage):
    """Standardize stage values."""
    if pd.isna(stage) or not str(stage) or 'unknown' in str(stage).lower() or 'not reported' in str(stage).lower() or 'not applicable' in str(stage).lower(): # Added more checks
        return 'Unknown'
    # Handle simple Roman numerals possibly preceded by Stage
    stage_str = str(stage).upper().replace('STAGE', '').strip()
    stage_str = re.sub(r'^[CPTYNM]+', '', stage_str).strip() # Remove common prefixes

    # More robust Roman numeral handling and direct mapping
    roman_map = {
        '0': 'Stage 0', 'I': 'Stage I', 'IA': 'Stage IA', 'IB': 'Stage IB', 
        'II': 'Stage II', 'IIA': 'Stage IIA', 'IIB': 'Stage IIB', 'IIC': 'Stage IIC', 
        'III': 'Stage III', 'IIIA': 'Stage IIIA', 'IIIB': 'Stage IIIB', 'IIIC': 'Stage IIIC', 'IIID': 'Stage IIID', 
        'IV': 'Stage IV', 'IVA': 'Stage IVA', 'IVB': 'Stage IVB', 'IVC': 'Stage IVC'
    }
    # Direct match after stripping prefixes and 'STAGE'
    if stage_str in roman_map:
        return roman_map[stage_str]

    # Handle cases like '3', '4' etc.
    if stage_str.isdigit():
        num_map = {'0':'Stage 0', '1':'Stage I', '2':'Stage II', '3':'Stage III', '4':'Stage IV'}
        if stage_str in num_map:
             return num_map[stage_str]

    # If it wasn't mapped, return Unknown or original cleaned string if desired
    logger.debug(f"Could not map stage: '{stage}' -> '{stage_str}'")
    return 'Unknown' # Default to Unknown if no pattern matches

def get_stage_at_icb(patient_id, icb_start_age, melanoma_diag_df):
    """Get the most relevant melanoma stage at or just before ICB start.

    Prioritizes Pathological stage closest to ICB start without exceeding it.
    Falls back to Clinical stage if Pathological is unavailable.
    If no stage before ICB, uses the earliest available melanoma stage.
    """
    if pd.isna(icb_start_age):
        return 'Unknown'

    patient_diag = melanoma_diag_df[melanoma_diag_df['PATIENT_ID'] == patient_id].copy()
    if patient_diag.empty:
        return 'Unknown'

    # Convert diagnosis age and clean stages
    patient_diag['DiagAge'] = patient_diag['AgeAtDiagnosis'].apply(convert_age)
    patient_diag['CleanPathStage'] = patient_diag['PathGroupStage'].apply(clean_stage)
    patient_diag['CleanClinStage'] = patient_diag['ClinGroupStage'].apply(clean_stage)

    # Filter diagnoses occurring at or before ICB start
    relevant_diag = patient_diag[patient_diag['DiagAge'] <= icb_start_age].sort_values('DiagAge', ascending=False)

    if not relevant_diag.empty:
        # Get the most recent diagnosis at or before ICB
        latest_relevant = relevant_diag.iloc[0]
        # Prioritize Pathological stage, then Clinical
        if latest_relevant['CleanPathStage'] != 'Unknown':
            return latest_relevant['CleanPathStage']
        elif latest_relevant['CleanClinStage'] != 'Unknown':
            return latest_relevant['CleanClinStage']
        else:
             # If the closest one has Unknown stage, check slightly earlier ones
             for _, row in relevant_diag.iloc[1:].iterrows():
                  if row['CleanPathStage'] != 'Unknown': return row['CleanPathStage']
                  if row['CleanClinStage'] != 'Unknown': return row['CleanClinStage']
             # If still nothing found before ICB, fall through to use earliest overall stage

    # Fallback: If no relevant pre-ICB stage found, use the earliest known melanoma stage
    earliest_diag = patient_diag.sort_values('DiagAge').iloc[0]
    if earliest_diag['CleanPathStage'] != 'Unknown':
        return earliest_diag['CleanPathStage']
    elif earliest_diag['CleanClinStage'] != 'Unknown':
        return earliest_diag['CleanClinStage']
    
    return 'Unknown' # Final fallback

def process_clinical_data(dfs, output_dir, qc_file_path=None):
    """Process clinical data and create patient-level dataset."""
    if not all(key in dfs for key in ['patients', 'diagnoses', 'treatments']):
        logger.error("Missing required dataframes")
        return None

    # Start with patient master data
    patients = dfs['patients'].copy()
    
    # Process treatments (medications) to identify ICB usage
    treatments = dfs['treatments'].copy()
    treatments['is_icb'] = treatments['Medication'].apply(classify_icb).notna()
    
    # Group by patient to determine ICB status and first ICB date
    icb_summary = treatments[treatments['is_icb']].groupby('PATIENT_ID').agg({
        'AgeAtMedStart': 'min',  # Get earliest ICB age
        'is_icb': 'any'  # True if any ICB treatment
    }).rename(columns={
        'AgeAtMedStart': 'age_at_first_icb',
        'is_icb': 'HAS_ICB'
    })
    
    # Merge ICB information with patient data
    patients = patients.merge(icb_summary, on='PATIENT_ID', how='left')
    patients['HAS_ICB'] = patients['HAS_ICB'].fillna(False)
    
    # Process diagnoses
    diagnoses = dfs['diagnoses'].copy()
    melanoma_diagnoses = diagnoses[diagnoses['HistologyCode'].apply(get_cancer_type) == 'Melanoma']
    
    # Get earliest melanoma diagnosis age for each patient
    melanoma_age = melanoma_diagnoses.groupby('PATIENT_ID')['AgeAtDiagnosis'].min()
    patients = patients.merge(melanoma_age.rename('age_at_melanoma_dx'), 
                            on='PATIENT_ID', how='left')
    
    # Get stage at ICB start
    patients['stage_at_icb'] = patients.apply(
        lambda x: get_stage_at_icb(x['PATIENT_ID'], 
                                 x.get('age_at_first_icb'), 
                                 melanoma_diagnoses), axis=1)
    
    # Process sequencing data if available
    if 'sequencing' in dfs:
        patients['SequencingBeforeICB'] = patients.apply(get_sequencing_before_icb, axis=1)
    
    # Additional processing as needed...
    
    return patients

def create_stage_simple(df, diag_df=None):
    """Deduce stage at ICB treatment start."""
    df['STAGE_SIMPLE'] = df.get('STAGE', 'Unknown')
    print("Columns available:", df.columns.tolist())
    
    if 'HAS_ICB' in df and 'TreatmentStartAge' in df and diag_df is not None:
        diag_df['DiagnosisAge'] = diag_df['DiagnosisAge'].apply(convert_age)
        diag_df['STAGE'] = diag_df['STAGE'].apply(clean_stage)
        for idx, row in df[df['HAS_ICB'] == 1].iterrows():
            icb_age = convert_age(row['TreatmentStartAge'])
            if pd.isna(icb_age):
                continue
            patient_diag = diag_df[diag_df['PATIENT_ID'] == row['PATIENT_ID']]
            pre_icb = patient_diag[patient_diag['DiagnosisAge'] <= icb_age]
            stage = pre_icb.sort_values('DiagnosisAge', ascending=False)['STAGE'].iloc[0] if not pre_icb.empty else patient_diag.sort_values('DiagnosisAge')['STAGE'].iloc[0] if not patient_diag.empty else 'Unknown'
            df.loc[idx, 'STAGE_SIMPLE'] = stage
    return df

def plot_demographics(df, plots_dir):
    """Generate demographic plots and save to plots_dir."""
    if df.empty:
        logger.warning("No data to plot demographics.")
        return
    
    # Plot age distribution
    if 'DiagnosisAge' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df['DiagnosisAge'].dropna(), bins=15, kde=True)
        plt.title('Age at Diagnosis')
        plt.xlabel('Age')
        plt.ylabel('Count')
        age_plot_file = os.path.join(plots_dir, 'age_distribution.png')
        plt.savefig(age_plot_file)
        plt.close()
        logger.info(f"Saved age distribution plot to {age_plot_file}")
    
    # Plot sex distribution
    if 'Sex' in df.columns:
        plt.figure(figsize=(6, 4))
        df['Sex'].value_counts().plot(kind='bar')
        plt.title('Sex Distribution')
        plt.xlabel('Sex')
        plt.ylabel('Count')
        sex_plot_file = os.path.join(plots_dir, 'sex_distribution.png')
        plt.savefig(sex_plot_file)
        plt.close()
        logger.info(f"Saved sex distribution plot to {sex_plot_file}")

def generate_reports(df, reports_dir):
    """Generate reports and save to reports_dir."""
    if df.empty:
        logger.warning("No data to generate reports.")
        return
    
    # Example report: summary statistics
    summary = df.describe(include='all')
    summary_file = os.path.join(reports_dir, 'summary_statistics.csv')
    summary.to_csv(summary_file)
    logger.info(f"Saved summary statistics to {summary_file}")
    
    # Example report: ICB treatment distribution
    if 'HAS_ICB' in df.columns:
        icb_dist = df['HAS_ICB'].value_counts()
        icb_file = os.path.join(reports_dir, 'icb_distribution.csv')
        icb_dist.to_csv(icb_file)
        logger.info(f"Saved ICB distribution to {icb_file}")


def main(base_path, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)  # Optional
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)  # Optional
    # Set up logging after ensuring the directory exists
    logging.basicConfig(filename=os.path.join(output_dir, 'processing_log.txt'), level=logging.INFO)
    logger = logging.getLogger()

    # Rest of your code here...
    logger.info("Logging setup complete. Starting processing...")
    qc_file = os.path.join(base_path, "Manifest_and_QC_Files/24PRJ217UVA_20241112_RNASeq_QCMetrics.csv")



    # Load clinical data
    files = {
        'patients': '24PRJ217UVA_20241112_PatientMaster_V4.csv',
        'diagnoses': '24PRJ217UVA_20241112_Diagnosis_V4.csv',
        'treatments': '24PRJ217UVA_20241112_Medications_V4.csv',
        'outcomes': '24PRJ217UVA_20241112_Outcomes_V4.csv',
        'sequencing': '24PRJ217UVA_20241112_TumorSequencing_V4.csv',
        'vital_status': '24PRJ217UVA_20241112_VitalStatus_V4.csv',
        'clinical_mol_linkage': '24PRJ217UVA_20241112_ClinicalMolLinkage_V4.csv'
    }
    dfs = {}
    clinical_data_dir = os.path.join(base_path, 'Clinical_Data', '24PRJ217UVA_NormalizedFiles')
    for key, file in files.items():
        path = os.path.join(clinical_data_dir, file)
        if os.path.exists(path):
            dfs[key] = pd.read_csv(path).rename(columns={'AvatarKey': 'PATIENT_ID'})
            logger.info(f"Loaded {key} from {path}")
        else:
            logger.warning(f"File not found: {path}")

    # Find QC metrics file
    qc_dir = os.path.join(base_path, "Manifest_and_QC_Files")
    qc_file = os.path.join(qc_dir, "24PRJ217UVA_20241112_RNASeq_QCMetrics.csv")

    # If the exact file doesn't exist, search for alternatives
    if not os.path.exists(qc_file):
        if os.path.exists(qc_dir):
            logger.info(f"Searching for RNASeq QC files in {qc_dir}")
            all_files = os.listdir(qc_dir)

            # First, try files with "RNA" and "QC"
            qc_files = [f for f in all_files if "RNA" in f and "QC" in f and f.endswith(".csv")]
            if qc_files:
                qc_file = os.path.join(qc_dir, qc_files[0])
                logger.info(f"Found alternative QC file: {qc_file}")
            else:
                # Broader search for RNA-related QC files
                logger.info("No RNASeq QC files found with standard pattern, trying broader search")
                qc_files = [f for f in all_files if any(term in f.lower() for term in ["rna", "seq"]) and "qc" in f.lower() and f.endswith(".csv")]
                if qc_files:
                    qc_file = os.path.join(qc_dir, qc_files[0])
                    logger.info(f"Found potential QC file: {qc_file}")
                else:
                    logger.error(f"No suitable RNASeq QC files found in {qc_dir}")
                    logger.info(f"Available files in directory: {all_files[:10]}")
                    return
        else:
            logger.error(f"QC directory not found: {qc_dir}")
            return

    if not os.path.exists(qc_file):
        logger.error(f"QC file not found at {qc_file}")
        return

    # Create ID mapping using utility function
    id_map = create_map_from_qc(qc_file, sample_col=None, patient_col=None)

    # Save ID mapping for future use
    if id_map:
        map_file = os.path.join(output_dir, "sample_to_patient_map.csv")
        logger.info(f"Saving ID mapping to: {map_file}")
        map_df = pd.DataFrame(list(id_map.items()), columns=['SampleID', 'PatientID'])
        map_df.to_csv(map_file, index=False)
    else:
        logger.error("Failed to create ID mapping")
        return

    # Process clinical data with ID mapping
    patients_df = process_clinical_data(dfs, output_dir, qc_file_path=qc_file)

    # Generate basic EDA if data is available
    if not patients_df.empty:
        # Sex distribution plot
        plt.figure(figsize=(8, 6))
        sns.countplot(data=patients_df, x='Sex')
        plt.title('Sex Distribution')
        plt.savefig(os.path.join(output_dir, 'plots', 'sex_distribution.png'))
        plt.close()
        logger.info(f"Saved sex distribution plot to {output_dir}/plots/sex_distribution.png")

        # Summary statistics
        summary_stats = patients_df.describe(include='all')
        summary_stats.to_csv(os.path.join(output_dir, 'reports', 'summary_statistics.csv'))
        logger.info(f"Saved summary statistics to {output_dir}/reports/summary_statistics.csv")

        # ICB distribution
        icb_dist = patients_df['HAS_ICB'].value_counts().reset_index()
        icb_dist.columns = ['HAS_ICB', 'count']
        icb_dist.to_csv(os.path.join(output_dir, 'reports', 'icb_distribution.csv'), index=False)
        logger.info(f"Saved ICB distribution to {output_dir}/reports/icb_distribution.csv")
    else:
        logger.warning("No processed patient data available for EDA.")

def parse_args():
    parser = argparse.ArgumentParser(description="Process melanoma clinical data")
    parser.add_argument("--base-path", default="/project/orien/data/aws/24PRJ217UVA_IORIG")
    parser.add_argument("--output-dir", default="output/melanoma_analysis")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.base_path, os.path.join(args.base_path, args.output_dir))
