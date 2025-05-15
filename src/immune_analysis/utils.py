import os
import pandas as pd
import logging
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration dictionary for file paths and column names
CONFIG = {
    'base_path': '/project/orien/data/aws/24PRJ217UVA_IORIG',
    'norm_files_dir': 'Clinical_Data/24PRJ217UVA_NormalizedFiles',
    'qc_file': 'Manifest_and_QC_Files/24PRJ217UVA_20250130_RNASeq_QCMetrics.csv',
    'diag_file': '24PRJ217UVA_20241112_Diagnosis_V4.csv',
    'patient_file': '24PRJ217UVA_20241112_PatientMaster_V4.csv',
    'biopsy_file': '24PRJ217UVA_20241112_SurgeryBiopsy_V4.csv',
    'map_file': 'codes/output/sample_to_patient_map.csv',
    'output_dir': 'codes/output',
    'columns': {
        'diagnosis': {'histology': 'HistologyCode', 'patient_id': 'AvatarKey'},
        'patient': {'patient_id': 'AvatarKey'},
        'qc': {'slid': 'SLID', 'patient_id': 'ORIENAvatarKey'},
        'biopsy': {'patient_id': 'AvatarKey', 'slid': 'SLID'}
    }
}

def load_csv_with_logging(file_path, required_columns=None):
    """
    Load a CSV file and log the process. Optionally check for required columns.
    
    Args:
        file_path (str): Path to the CSV file.
        required_columns (list, optional): List of column names that must be present.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    
    Raises:
        ValueError: If required columns are missing.
        Exception: If file loading fails.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {file_path} with shape {df.shape}")
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise

def save_results(df, filename, output_dir=None):
    """
    Save a DataFrame to a CSV file in the output directory.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        filename (str): Name of the output file.
        output_dir (str, optional): Directory to save the file. Defaults to CONFIG['output_dir'].
    """
    if output_dir is None:
        output_dir = os.path.join(CONFIG['base_path'], CONFIG['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, filename)
    df.to_csv(output_file, index=False)
    logger.info(f"Saved results to {output_file}")

def plot_boxplot(data, x_col, y_col, title, output_file, adjust_p=False):
    """
    Create a boxplot with optional p-value adjustment.
    
    Args:
        data (pd.DataFrame): Data to plot.
        x_col (str): Column name for the x-axis.
        y_col (str): Column name for the y-axis.
        title (str): Plot title.
        output_file (str): Path to save the plot.
        adjust_p (bool): Whether to adjust p-values (placeholder for actual implementation).
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=x_col, y=y_col)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if adjust_p:
        # Placeholder for p-value adjustment; replace with actual p-values if available
        pvals = [0.05]  # Example; integrate with actual statistical results
        adjusted_p = multipletests(pvals, method='bonferroni')[1]
        plt.text(0.5, 0.95, f'Adjusted p = {adjusted_p[0]:.2e}', transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    logger.info(f"Saved boxplot to {output_file}")