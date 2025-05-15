"""
Clinical Variables Analysis
Analyzes clinical variables in relation to CD8+ T cell signatures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import stats
import warnings
import traceback
from cd8_analysis import CD8ClusterAnalysis

class ClinicalAnalysis(CD8ClusterAnalysis):
    """Analyze clinical variables in relation to CD8+ T cell signatures"""
    
    def __init__(self, base_path):
        super().__init__(base_path)
        
        # Create additional output directories
        self.clinical_dir = os.path.join(self.output_dir, 'clinical_analysis')
        self.clinical_plots_dir = os.path.join(self.clinical_dir, 'plots')
        os.makedirs(self.clinical_dir, exist_ok=True)
        os.makedirs(self.clinical_plots_dir, exist_ok=True)
    
    def load_existing_scores(self):
        """Load previously calculated signature scores"""
        try:
            scores_file = os.path.join(self.output_dir, 'signature_scores.csv')
            if os.path.exists(scores_file):
                print(f"\nLoading existing signature scores from {scores_file}")
                scores = pd.read_csv(scores_file, index_col=0)
                print(f"Loaded scores for {len(scores)} patients and {len(scores.columns)} signatures")
                return scores
            else:
                print(f"\nNo existing scores found at {scores_file}")
                return None
        except Exception as e:
            print(f"Error loading existing scores: {e}")
            print(traceback.format_exc())
            return None
    
    def analyze_clinical_variables(self, scores, clinical_data):
        """Analyze clinical variables in relation to CD8+ T cell signatures"""
        try:
            print("\nAnalyzing clinical variables...")
            
            # Merge data
            merged = clinical_data.merge(
                scores,
                left_on='PATIENT_ID',
                right_index=True,
                how='inner'
            )
            
            # Filter by diagnosis
            merged = self.filter_by_diagnosis(merged)
            
            results = {}
            
            # 1. Medication Analysis
            if 'Medication' in merged.columns:
                print("\nAnalyzing medications...")
                
                # Filter for melanoma medications if needed
                if 'MedPrimaryDiagnosisSite' in merged.columns:
                    melanoma_meds = merged[merged['MedPrimaryDiagnosisSite'].str.contains('melanoma', case=False, na=False)]
                else:
                    melanoma_meds = merged
                    
                # Count medications by sex
                med_counts = pd.crosstab(
                    melanoma_meds['Medication'], 
                    melanoma_meds['SEX'],
                    margins=True
                )
                
                # Calculate percentages
                med_pcts = pd.crosstab(
                    melanoma_meds['Medication'], 
                    melanoma_meds['SEX'],
                    normalize='columns'
                ) * 100
                
                results['medications'] = {
                    'counts': med_counts,
                    'percentages': med_pcts
                }
                
                # Save results
                med_counts.to_csv(os.path.join(self.clinical_dir, 'medication_counts_by_sex.csv'))
                med_pcts.to_csv(os.path.join(self.clinical_dir, 'medication_percentages_by_sex.csv'))
                
                # Plot top medications
                top_meds = med_counts.sort_values('All', ascending=False).head(10).index
                top_meds = [x for x in top_meds if x != 'All']  # Exclude 'All' row
                if len(top_meds) > 0:
                    plt.figure(figsize=(12, 8))
                    med_pcts.loc[top_meds].plot(kind='bar')
                    plt.title('Top Medications by Sex')
                    plt.ylabel('Percentage of Patients (%)')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.clinical_plots_dir, 'top_medications_by_sex.png'), dpi=300)
                    plt.close()
            
            # 2. Systemic Surgery Sequence
            if 'SystemicSurgerySequence' in merged.columns:
                print("\nAnalyzing systemic surgery sequence...")
                
                # Count sequences by sex
                seq_counts = pd.crosstab(
                    merged['SystemicSurgerySequence'], 
                    merged['SEX'],
                    margins=True
                )
                
                # Calculate percentages
                seq_pcts = pd.crosstab(
                    merged['SystemicSurgerySequence'], 
                    merged['SEX'],
                    normalize='columns'
                ) * 100
                
                results['surgery_sequence'] = {
                    'counts': seq_counts,
                    'percentages': seq_pcts
                }
                
                # Save results
                seq_counts.to_csv(os.path.join(self.clinical_dir, 'surgery_sequence_counts_by_sex.csv'))
                seq_pcts.to_csv(os.path.join(self.clinical_dir, 'surgery_sequence_percentages_by_sex.csv'))
                
                # Plot sequences
                seq_items = [x for x in seq_counts.index if x != 'All']  # Exclude 'All' row
                if len(seq_items) > 0:
                    plt.figure(figsize=(10, 6))
                    seq_pcts.loc[seq_items].plot(kind='bar')
                    plt.title('Systemic Surgery Sequence by Sex')
                    plt.ylabel('Percentage of Patients (%)')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.clinical_plots_dir, 'surgery_sequence_by_sex.png'), dpi=300)
                    plt.close()
            
            # 3. Age at Diagnosis
            if 'AgeAtDiagnosis' in merged.columns:
                print("\nAnalyzing age at diagnosis...")
                
                # Create age groups by decade
                merged['AgeGroup'] = pd.cut(
                    merged['AgeAtDiagnosis'],
                    bins=[0, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    labels=['<20', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '≥90']
                )
                
                # Count age groups by sex
                age_counts = pd.crosstab(
                    merged['AgeGroup'], 
                    merged['SEX'],
                    margins=True
                )
                
                # Calculate percentages
                age_pcts = pd.crosstab(
                    merged['AgeGroup'], 
                    merged['SEX'],
                    normalize='columns'
                ) * 100
                
                results['age_groups'] = {
                    'counts': age_counts,
                    'percentages': age_pcts
                }
                
                # Save results
                age_counts.to_csv(os.path.join(self.clinical_dir, 'age_group_counts_by_sex.csv'))
                age_pcts.to_csv(os.path.join(self.clinical_dir, 'age_group_percentages_by_sex.csv'))
                
                # Plot age distribution
                age_items = [x for x in age_counts.index if x != 'All']  # Exclude 'All' row
                if len(age_items) > 0:
                    plt.figure(figsize=(10, 6))
                    age_pcts.loc[age_items].plot(kind='bar')
                    plt.title('Age Distribution by Sex')
                    plt.ylabel('Percentage of Patients (%)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.clinical_plots_dir, 'age_distribution_by_sex.png'), dpi=300)
                    plt.close()
            
            # 4. Primary Diagnosis Site
            if 'PrimaryDiagnosisSiteCode' in merged.columns and 'PrimaryDiagnosisSite' in merged.columns:
                print("\nAnalyzing primary diagnosis site...")
                
                # Count sites by sex
                site_counts = pd.crosstab(
                    merged['PrimaryDiagnosisSite'], 
                    merged['SEX'],
                    margins=True
                )
                
                # Calculate percentages
                site_pcts = pd.crosstab(
                    merged['PrimaryDiagnosisSite'], 
                    merged['SEX'],
                    normalize='columns'
                ) * 100
                
                results['diagnosis_sites'] = {
                    'counts': site_counts,
                    'percentages': site_pcts
                }
                
                # Save results
                site_counts.to_csv(os.path.join(self.clinical_dir, 'diagnosis_site_counts_by_sex.csv'))
                site_pcts.to_csv(os.path.join(self.clinical_dir, 'diagnosis_site_percentages_by_sex.csv'))
                
                # Plot top sites - FIX: Properly exclude 'All' from top_sites
                top_sites = site_counts.sort_values('All', ascending=False).head(10).index
                top_sites = [x for x in top_sites if x != 'All']  # Exclude 'All' row
                if len(top_sites) > 0:
                    plt.figure(figsize=(12, 8))
                    site_pcts.loc[top_sites].plot(kind='bar')
                    plt.title('Top Diagnosis Sites by Sex')
                    plt.ylabel('Percentage of Patients (%)')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.clinical_plots_dir, 'top_diagnosis_sites_by_sex.png'), dpi=300)
                    plt.close()
            
            # 5. Pathological Group Stage
            if 'PathGroupStage' in merged.columns:
                print("\nAnalyzing pathological group stage...")
                
                # Map stages to standardized categories
                stage_map = {
                    'IA': 'I', 'IB': 'I', 'I': 'I', '0': '0',
                    'IIA': 'II', 'IIB': 'II', 'IIC': 'II',
                    'IIIA': 'III', 'IIIB': 'III', 'IIIC': 'III', 'IIID': 'III', 'III': 'III',
                    'IV': 'IV', 'IVB': 'IV', 'IVC': 'IV'
                }
                
                # Apply mapping, with unknowns grouped together
                merged['StageStandardized'] = merged['PathGroupStage'].map(
                    lambda x: stage_map.get(x, 'Unknown')
                )
                
                # Count stages by sex
                stage_counts = pd.crosstab(
                    merged['StageStandardized'], 
                    merged['SEX'],
                    margins=True
                )
                
                # Calculate percentages
                stage_pcts = pd.crosstab(
                    merged['StageStandardized'], 
                    merged['SEX'],
                    normalize='columns'
                ) * 100
                
                results['stages'] = {
                    'counts': stage_counts,
                    'percentages': stage_pcts
                }
                
                # Save results
                stage_counts.to_csv(os.path.join(self.clinical_dir, 'stage_counts_by_sex.csv'))
                stage_pcts.to_csv(os.path.join(self.clinical_dir, 'stage_percentages_by_sex.csv'))
                
                # Plot stages
                plt.figure(figsize=(10, 6))
                # Ensure proper order of stages
                stage_order = ['0', 'I', 'II', 'III', 'IV', 'Unknown']
                stage_items = [s for s in stage_order if s in stage_pcts.index and s != 'All']
                if len(stage_items) > 0:
                    stage_pcts.loc[stage_items].plot(kind='bar')
                    plt.title('Pathological Stage by Sex')
                    plt.ylabel('Percentage of Patients (%)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.clinical_plots_dir, 'stage_distribution_by_sex.png'), dpi=300)
                    plt.close()
            
            # 6. Currently Seen For Primary or Recurrent
            if 'CurrentlySeenForPrimaryOrRecurr' in merged.columns:
                print("\nAnalyzing primary vs recurrent disease...")
                
                # Count primary/recurrent by sex
                recur_counts = pd.crosstab(
                    merged['CurrentlySeenForPrimaryOrRecurr'], 
                    merged['SEX'],
                    margins=True
                )
                
                # Calculate percentages
                recur_pcts = pd.crosstab(
                    merged['CurrentlySeenForPrimaryOrRecurr'], 
                    merged['SEX'],
                    normalize='columns'
                ) * 100
                
                results['primary_recurrent'] = {
                    'counts': recur_counts,
                    'percentages': recur_pcts
                }
                
                # Save results
                recur_counts.to_csv(os.path.join(self.clinical_dir, 'primary_recurrent_counts_by_sex.csv'))
                recur_pcts.to_csv(os.path.join(self.clinical_dir, 'primary_recurrent_percentages_by_sex.csv'))
                
                # Plot primary/recurrent
                recur_items = [x for x in recur_counts.index if x != 'All']  # Exclude 'All' row
                if len(recur_items) > 0:
                    plt.figure(figsize=(8, 6))
                    recur_pcts.loc[recur_items].plot(kind='bar')
                    plt.title('Primary vs Recurrent Disease by Sex')
                    plt.ylabel('Percentage of Patients (%)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.clinical_plots_dir, 'primary_recurrent_by_sex.png'), dpi=300)
                    plt.close()
            
            # 7. Correlation between clinical variables and CD8+ signatures
            print("\nAnalyzing correlations between clinical variables and CD8+ signatures...")
            
            # Create summary table
            summary_rows = []
            
            # For each signature
            for sig in self.signatures.keys():
                # Age correlation
                if 'AgeAtDiagnosis' in merged.columns:
                    # Fix: Handle NaN values by dropping them before correlation
                    age_data = merged['AgeAtDiagnosis'].dropna()
                    sig_data = merged[sig].dropna()
                    
                    # Get common indices
                    common_idx = age_data.index.intersection(sig_data.index)
                    
                    if len(common_idx) > 5:  # Ensure enough data points
                        try:
                            r, p = stats.pearsonr(age_data.loc[common_idx], sig_data.loc[common_idx])
                            summary_rows.append({
                                'signature': sig,
                                'clinical_variable': 'Age at Diagnosis',
                                'test': 'Pearson r',
                                'statistic': r,
                                'p_value': p
                            })
                        except Exception as e:
                            print(f"Warning: Could not calculate correlation for {sig} with Age: {e}")
                
                # Stage association
                if 'StageStandardized' in merged.columns:
                    # ANOVA
                    try:
                        # Group by stage and drop NaN values
                        stage_groups = []
                        for name, group in merged.groupby('StageStandardized'):
                            values = group[sig].dropna().values
                            if len(values) > 5:  # Minimum sample size
                                stage_groups.append(values)
                        
                        if len(stage_groups) > 1:
                            f_stat, p_val = stats.f_oneway(*stage_groups)
                            summary_rows.append({
                                'signature': sig,
                                'clinical_variable': 'Pathological Stage',
                                'test': 'ANOVA',
                                'statistic': f_stat,
                                'p_value': p_val
                            })
                    except Exception as e:
                        print(f"Warning: Could not perform ANOVA for {sig} with Stage: {e}")
                
                # Sex difference
                if 'SEX' in merged.columns:
                    try:
                        male = merged[merged['SEX'] == 'Male'][sig].dropna()
                        female = merged[merged['SEX'] == 'Female'][sig].dropna()
                        
                        if len(male) > 5 and len(female) > 5:  # Minimum sample size
                            t_stat, p_val = stats.ttest_ind(male, female)
                            summary_rows.append({
                                'signature': sig,
                                'clinical_variable': 'Sex',
                                'test': 't-test',
                                'statistic': t_stat,
                                'p_value': p_val
                            })
                    except Exception as e:
                        print(f"Warning: Could not perform t-test for {sig} with Sex: {e}")
                
                # Primary vs Recurrent
                if 'CurrentlySeenForPrimaryOrRecurr' in merged.columns:
                    try:
                        primary = merged[merged['CurrentlySeenForPrimaryOrRecurr'] == 'Primary'][sig].dropna()
                        recurrent = merged[merged['CurrentlySeenForPrimaryOrRecurr'] == 'Recurrent'][sig].dropna()
                        
                        if len(primary) > 5 and len(recurrent) > 5:  # Minimum sample size
                            t_stat, p_val = stats.ttest_ind(primary, recurrent)
                            summary_rows.append({
                                'signature': sig,
                                'clinical_variable': 'Primary vs Recurrent',
                                'test': 't-test',
                                'statistic': t_stat,
                                'p_value': p_val
                            })
                    except Exception as e:
                        print(f"Warning: Could not perform t-test for {sig} with Primary/Recurrent: {e}")
            
            # Create summary DataFrame
            summary_df = pd.DataFrame(summary_rows)
            if len(summary_df) > 0:
                # Add FDR correction
                summary_df['fdr'] = multipletests(summary_df['p_value'], method='fdr_bh')[1]
                
                # Save summary
                summary_df.to_csv(os.path.join(self.clinical_dir, 'clinical_correlations.csv'), index=False)
                
                # Create heatmap of significant associations
                sig_summary = summary_df[summary_df['fdr'] < 0.1].copy()
                if len(sig_summary) > 0:
                    try:
                        # Pivot for heatmap
                        pivot_df = sig_summary.pivot(
                            index='signature',
                            columns='clinical_variable',
                            values='p_value'
                        )
                        
                        # Plot heatmap
                        plt.figure(figsize=(12, 8))
                        sns.heatmap(
                            -np.log10(pivot_df),
                            cmap='YlOrRd',
                            annot=True,
                            fmt='.2f'
                        )
                        plt.title('Significant Associations (-log10 p-value)')
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.clinical_plots_dir, 'clinical_associations_heatmap.png'), dpi=300)
                        plt.close()
                    except Exception as e:
                        print(f"Warning: Could not create heatmap: {e}")
            
            return results
            
        except Exception as e:
            print(f"Error analyzing clinical variables: {e}")
            print(traceback.format_exc())
            return None

def main():
    """Main execution function"""
    # Set paths
    base_path = "/project/orien/data/aws/24PRJ217UVA_IORIG"
    
    # Initialize analysis
    clinical_analysis = ClinicalAnalysis(base_path)
    
    # Try to load existing scores first
    scores = clinical_analysis.load_existing_scores()
    
    # If no existing scores, calculate them
    if scores is None:
        # Load expression data
        expr_data = clinical_analysis.load_rnaseq_data(base_path)
        
        if expr_data is not None:
            # Calculate signature scores
            scores = clinical_analysis.score_signatures(expr_data)
    
    if scores is not None:
        # Load clinical data
        clinical_data = pd.read_csv(
            os.path.join(base_path, "codes/processed_data/processed_clinical_molecular.csv")
        )
        
        # Run sex-stratified survival analysis
        male_results, female_results = clinical_analysis.analyze_survival_by_sex(scores, clinical_data)
        
        # Run clinical variables analysis
        clinical_results = clinical_analysis.analyze_clinical_variables(scores, clinical_data)
        
        print("\nAnalysis complete!")
        print(f"Results saved to: {clinical_analysis.clinical_dir}")
        print(f"Plots saved to: {clinical_analysis.clinical_plots_dir}")

if __name__ == "__main__":
    main() 