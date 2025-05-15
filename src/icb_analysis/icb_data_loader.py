"""
ICB Data Loader
Loads and processes data for ICB analysis
"""

import pandas as pd
import numpy as np
import os
import sys
import traceback
from datetime import datetime

# Add parent directory to path to allow imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cd8_analysis.cd8_groups_analysis import CD8GroupAnalysis
from utils.shared_functions import load_clinical_data, load_rnaseq_data, filter_by_diagnosis

class ICBDataLoader:
    """Loads and processes data for ICB analysis"""
    
    def __init__(self, base_path):
        """Initialize the ICB data loader"""
        self.base_path = base_path
        self.output_dir = os.path.join(base_path, "codes/output/cd8_analysis")
        self.groups_dir = os.path.join(self.output_dir, 'cd8_groups_analysis')
        self.icb_dir = os.path.join(self.output_dir, 'icb_medication_analysis')
        self.icb_plots_dir = os.path.join(self.icb_dir, 'plots')
        
        # Create output directories
        os.makedirs(self.icb_dir, exist_ok=True)
        os.makedirs(self.icb_plots_dir, exist_ok=True)
        
        # Define signatures
        self.signatures = {
            'CD8_B': 'Non-responder enriched (Clusters 1-3)',
            'CD8_G': 'Responder enriched (Clusters 4-6)',
            'CD8_GtoB_ratio': 'Responder/Non-responder ratio',
            'CD8_GtoB_log': 'Log2(Responder/Non-responder ratio)'
        }
        
        # Define cluster descriptions
        self.cluster_desc = {
            'CD8_B': 'Non-responder enriched (Clusters 1-3)',
            'CD8_G': 'Responder enriched (Clusters 4-6)',
            'CD8_GtoB_ratio': 'Responder/Non-responder ratio',
            'CD8_GtoB_log': 'Log2(Responder/Non-responder ratio)'
        }
    
    def load_clinical_data(self):
        """Load clinical data"""
        try:
            print("\nLoading clinical data...")
            clinical_data = load_clinical_data(self.base_path)
            
            if clinical_data is None:
                print("ERROR: Failed to load clinical data")
                return None
            
            print(f"Loaded clinical data for {len(clinical_data)} patients")
            return clinical_data
            
        except Exception as e:
            print(f"Error loading clinical data: {e}")
            print(traceback.format_exc())
            return None
    
    def load_cd8_group_scores(self):
        """Load CD8 group scores from file"""
        try:
            print("\nLoading CD8 group scores...")
            
            # Define scores file path
            scores_file = os.path.join(self.groups_dir, 'cd8_group_scores.csv')
            
            # Check if file exists
            if not os.path.exists(scores_file):
                print(f"Scores file not found: {scores_file}")
                return None
            
            # Load scores
            scores = pd.read_csv(scores_file, index_col='PATIENT_ID')
            print(f"Loaded CD8 group scores for {len(scores)} patients")
            
            return scores
            
        except Exception as e:
            print(f"Error loading CD8 group scores: {e}")
            print(traceback.format_exc())
            return None
    
    def load_rnaseq_data(self):
        """Load RNA-seq data"""
        try:
            print("\nLoading RNA-seq data...")
            rnaseq_data = load_rnaseq_data(self.base_path)
            
            if rnaseq_data is None:
                print("ERROR: Failed to load RNA-seq data")
                return None
            
            print(f"Loaded RNA-seq data with {rnaseq_data.shape[0]} genes and {rnaseq_data.shape[1]} samples")
            return rnaseq_data
            
        except Exception as e:
            print(f"Error loading RNA-seq data: {e}")
            print(traceback.format_exc())
            return None

    def load_medication_data(self):
        """Load medication data from CSV"""
        try:
            medications_file = os.path.join(
                self.base_path,
                "Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_Medications_V4.csv"
            )
            if not os.path.exists(medications_file):
                print(f"ERROR: Medications file not found at {medications_file}")
                return None
            
            print(f"\nLoading medications data from {medications_file}")
            medications = pd.read_csv(medications_file)
            print(f"Loaded medication data for {len(medications)} records")
            
            # Check for required columns
            required_columns = ['AvatarKey', 'Medication']
            missing_columns = [col for col in required_columns if col not in medications.columns]
            if missing_columns:
                print(f"WARNING: Missing columns: {missing_columns}")
                
                # Try to find alternative columns
                if 'AvatarKey' not in medications.columns:
                    patient_id_cols = [col for col in medications.columns if 'patient' in col.lower() or 'avatar' in col.lower() or 'key' in col.lower()]
                    if patient_id_cols:
                        print(f"Using {patient_id_cols[0]} as patient ID column")
                        medications = medications.rename(columns={patient_id_cols[0]: 'AvatarKey'})
                    else:
                        print("ERROR: No patient ID column found")
                        return None
                
                if 'Medication' not in medications.columns:
                    med_cols = [col for col in medications.columns if 'med' in col.lower() and 'name' in col.lower()]
                    if med_cols:
                        print(f"Using {med_cols[0]} as medication name column")
                        medications = medications.rename(columns={med_cols[0]: 'Medication'})
                    else:
                        print("ERROR: No medication name column found")
                        return None
            
            # Standardize patient ID column name
            medications = medications.rename(columns={'AvatarKey': 'PATIENT_ID'})
            
            print(f"Processed medication data with columns: {medications.columns.tolist()}")
            return medications
        except Exception as e:
            print(f"Error loading medication data: {e}")
            print(traceback.format_exc())
            return None

    def load_icb_data(self):
        """Load pre-processed ICB data from file created by eda.py"""
        try:
            # Try different possible locations for the ICB data file
            potential_paths = [
                os.path.join(self.base_path, "processed_data/icb_data.csv"),
                os.path.join(os.path.dirname(self.base_path), "processed_data/icb_data.csv"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "processed_data/icb_data.csv")
            ]
            
            icb_file = None
            for path in potential_paths:
                if os.path.exists(path):
                    icb_file = path
                    break
            
            if icb_file is None:
                print("\nICB data file not found in standard locations.")
                print("Will fall back to processing medication data directly.")
                return self.load_medication_data()
            
            # Load the ICB data
            print(f"\nLoading pre-processed ICB data from {icb_file}")
            icb_data = pd.read_csv(icb_file)
            
            # Print summary information
            print(f"Loaded ICB data for {len(icb_data)} patients")
            print(f"ICB data columns: {icb_data.columns.tolist()}")
            
            return icb_data
            
        except Exception as e:
            print(f"Error loading ICB data: {e}")
            print(traceback.format_exc())
            # Fall back to medication data
            print("Falling back to medication data processing")
            return self.load_medication_data()