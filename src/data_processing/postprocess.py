import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def generate_plots(input_file, output_dir):
    # Load data
    df = pd.read_csv(input_file)
    
    # Ensure output directories exist
    os.makedirs(os.path.join(output_dir, 'icb_plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'non_icb_plots'), exist_ok=True)
    
    # Split into ICB and non-ICB patients
    icb_df = df[df['HAS_ICB'] == 1]
    non_icb_df = df[df['HAS_ICB'] == 0]
    
    # ICB Patients Plots
    # 1. Stage at ICB Start
    plt.figure(figsize=(10, 6))
    sns.countplot(data=icb_df, x='STAGE_AT_ICB', order=sorted(icb_df['STAGE_AT_ICB'].unique()))
    plt.title('Distribution of Stages at ICB Start')
    plt.xlabel('Stage at ICB Start')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'icb_plots', 'stage_at_icb_distribution.png'))
    plt.close()
    
    # 2. Sex Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=icb_df, x='Sex')
    plt.title('Sex Distribution of ICB Patients')
    plt.xlabel('Sex')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'icb_plots', 'sex_distribution_icb.png'))
    plt.close()
    
    # 3. Age at ICB Start
    plt.figure(figsize=(10, 6))
    sns.histplot(data=icb_df, x='ICB_START_AGE', bins=20, kde=True)
    plt.title('Age Distribution at ICB Start')
    plt.xlabel('Age at ICB Start')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'icb_plots', 'age_at_icb_distribution.png'))
    plt.close()
    
    # Non-ICB Patients Plots
    # 1. Sex Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=non_icb_df, x='Sex')
    plt.title('Sex Distribution of Non-ICB Patients')
    plt.xlabel('Sex')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'non_icb_plots', 'sex_distribution_non_icb.png'))
    plt.close()
    
    # 2. Age at Reference Point (Earliest Diagnosis)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=non_icb_df, x='REFERENCE_AGE', bins=20, kde=True)
    plt.title('Age Distribution at Diagnosis (Non-ICB Patients)')
    plt.xlabel('Age at Diagnosis')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'non_icb_plots', 'age_at_diagnosis_distribution.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from processed clinical data.")
    parser.add_argument('--input-file', required=True, help="Path to melanoma_patients_with_sequencing.csv")
    parser.add_argument('--output-dir', required=True, help="Directory to save plots")
    args = parser.parse_args()
    
    generate_plots(args.input_file, args.output_dir)
    print(f"Plots saved to {args.output_dir}/icb_plots and {args.output_dir}/non_icb_plots")