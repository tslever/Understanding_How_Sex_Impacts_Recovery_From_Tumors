import pandas as pd
import numpy as np
import os
import argparse
import logging
import sys

# Handle relative imports differently based on how the script is run
try:
    from src.data_processing.utils import create_map_from_qc
except ImportError:
    try:
        from utils import create_map_from_qc
    except ImportError:
        print("Error: Cannot import create_map_from_qc function")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('analyze_ids')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze and map patient IDs between datasets")
    parser.add_argument("--base-path", default="/project/orien/data/aws/24PRJ217UVA_IORIG", 
                        help="Base path to project directory")
    parser.add_argument("--output-dir", default="codes/output", 
                        help="Directory for output files")
    
    # Parse either passed args or default to script args if empty
    if len(sys.argv) > 1 and "analyze_ids" not in sys.argv[0]:
        return parser.parse_args()
    else:
        return parser.parse_args(sys.argv[1:])

# Load the data
def load_data(base_path):
    """Load cell scores and clinical data"""
    try:
        # Load cell scores (output from microenv.py)
        cell_scores_path = os.path.join(base_path, "codes/output/cell_scores.csv")
        logger.info(f"Loading cell scores from: {cell_scores_path}")
        cell_scores = pd.read_csv(cell_scores_path, index_col=0)
        logger.info(f"Loaded cell scores with shape: {cell_scores.shape}")
        
        # Load clinical data
        clinical_path = os.path.join(base_path, "codes/processed_data/processed_clinical_molecular.csv")
        logger.info(f"Loading clinical data from: {clinical_path}")
        clinical_data = pd.read_csv(clinical_path)
        logger.info(f"Loaded clinical data with shape: {clinical_data.shape}")
        
        return cell_scores, clinical_data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None

def test_merge_with_mapping(cell_scores, clinical_data, id_map, output_dir=None):
    """Test merge using the ID mapping and optionally save the result"""
    logger.info("Merging data using ID mapping")
    
    # Map expression data IDs to ORIEN IDs
    cell_scores_mapped = cell_scores.copy()
    cell_scores_mapped.index = [id_map.get(x, x) for x in cell_scores_mapped.index]
    
    # Merge with clinical data
    merged = clinical_data.merge(
        cell_scores_mapped,
        left_on='PATIENT_ID',
        right_index=True,
        how='inner'
    )
    
    logger.info(f"Merged data has {len(merged)} rows and {merged.shape[1]} columns")
    
    # Save merged data if output directory is provided
    if output_dir and len(merged) > 0:
        output_path = os.path.join(output_dir, "merged_clinical_cell_scores.csv")
        logger.info(f"Saving merged data to: {output_path}")
        merged.to_csv(output_path, index=False)
        
    return merged

def analyze_mapping_results(cell_scores, clinical_data, id_map, merged_data):
    """Analyze the mapping and merge results"""
    logger.info("\nMapping Analysis:")
    logger.info(f"Original samples: {len(cell_scores)}")
    logger.info(f"Unique patients after mapping: {len(set(id_map.values()))}")
    
    # Count samples per patient
    value_counts = pd.Series(id_map.values()).value_counts()
    multi_samples = value_counts[value_counts > 1]
    logger.info(f"Patients with multiple samples: {len(multi_samples)}")
    
    # Check for unmatched samples
    unmatched_samples = [x for x in cell_scores.index if x not in id_map]
    logger.info(f"Unmatched samples: {len(unmatched_samples)}")
    if unmatched_samples and len(unmatched_samples) < 20:
        logger.info(f"Unmatched samples: {unmatched_samples}")
    elif unmatched_samples:
        logger.info(f"First 10 unmatched samples: {unmatched_samples[:10]}")
    
    # Check clinical data coverage
    mapped_patients = set(id_map.values())
    clinical_patients = set(clinical_data['PATIENT_ID'])
    logger.info(f"Total patients in clinical data: {len(clinical_patients)}")
    logger.info(f"Patients with RNA samples: {len(mapped_patients)}")
    logger.info(f"Patients in both: {len(mapped_patients & clinical_patients)}")
    
    # Analyze merged data
    logger.info(f"Total rows in merged data: {len(merged_data)}")
    logger.info(f"Unique patients in merged data: {merged_data['PATIENT_ID'].nunique()}")
    
    # Sample distribution
    logger.info("Sample distribution in merged data:")
    sample_counts = merged_data['PATIENT_ID'].value_counts()
    logger.info(f"Patients by number of samples: {sample_counts.value_counts().sort_index().to_dict()}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(args.base_path, args.output_dir), exist_ok=True)
    output_dir = os.path.join(args.base_path, args.output_dir)
    
    # Load data
    cell_scores, clinical_data = load_data(args.base_path)
    if cell_scores is None or clinical_data is None:
        return
    
    # Find QC metrics file - try with updated file pattern
    qc_file = os.path.join(args.base_path, "Manifest_and_QC_Files/24PRJ217UVA_20241112_RNASeq_QCMetrics.csv")
    
    # If the exact file doesn't exist, try to find an alternative
    if not os.path.exists(qc_file):
        qc_dir = os.path.join(args.base_path, "Manifest_and_QC_Files")
        if os.path.exists(qc_dir):
            # List all files in directory
            logger.info(f"Searching for RNASeq QC files in {qc_dir}")
            all_files = os.listdir(qc_dir)
            
            # First try files with "RNA" and "QC" in the name
            qc_files = [f for f in all_files if "RNA" in f and "QC" in f]
            if qc_files:
                qc_file = os.path.join(qc_dir, qc_files[0])
                logger.info(f"Found alternative QC file: {qc_file}")
            else:
                # If no match, try broader search
                logger.info("No RNASeq QC files found with standard pattern, trying broader search")
                qc_files = [f for f in all_files if any(term in f for term in ["rna", "RNA", "seq", "Seq"]) and f.endswith(".csv")]
                if qc_files:
                    qc_file = os.path.join(qc_dir, qc_files[0])
                    logger.info(f"Found potential QC file: {qc_file}")
                else:
                    logger.error(f"No suitable RNASeq QC files found in {qc_dir}")
                    # List available files for debugging
                    logger.info(f"Available files in directory: {all_files[:10]}")
    
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
    
    # Test merge with mapping and save result
    merged_data = test_merge_with_mapping(cell_scores, clinical_data, id_map, output_dir)
    
    # Analyze mapping results
    analyze_mapping_results(cell_scores, clinical_data, id_map, merged_data)

# Check if this is being run as a script vs imported as a module
if __name__ == "__main__" or "analyze_ids" in sys.argv[0]:
    main() 