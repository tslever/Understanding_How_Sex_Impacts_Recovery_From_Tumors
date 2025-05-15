import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter
import os
import traceback

class GroupAnalysis:
    def __init__(self, base_path, data_loader, filter_by_diagnosis_func):
        self.base_path = base_path
        self.data_loader = data_loader
        self.filter_by_diagnosis = filter_by_diagnosis_func
        self.groups_dir = os.path.join(base_path, 'output', 'cd8_groups_analysis')
        self.groups_plots_dir = os.path.join(self.groups_dir, 'plots')
        os.makedirs(self.groups_dir, exist_ok=True)
        os.makedirs(self.groups_plots_dir, exist_ok=True)

    def plot_group_distributions(self, scores, clinical_data):
        """Plot expression distributions by sex and perform t-tests."""
        try:
            print("Plotting CD8 group distributions by sex...")
            merged = clinical_data.merge(scores, left_on='PATIENT_ID', right_index=True, how='inner')
            merged = self.filter_by_diagnosis(merged)
            for sig in self.data_loader.signatures.keys():
                plt.figure(figsize=(10, 6))
                male_scores = merged[merged['SEX'] == 'Male'][sig]
                female_scores = merged[merged['SEX'] == 'Female'][sig]
                t_stat, p_val = stats.ttest_ind(male_scores, female_scores)
                sns.histplot(male_scores, kde=True, alpha=0.5, color='blue', label='Male')
                sns.histplot(female_scores, kde=True, alpha=0.5, color='red', label='Female')
                plt.axvline(male_scores.mean(), color='blue', linestyle='--', linewidth=1.5)
                plt.axvline(female_scores.mean(), color='red', linestyle='--', linewidth=1.5)
                plt.title(f'Distribution of {sig} ({self.data_loader.cluster_desc[sig]}) Expression by Sex')
                plt.xlabel('Expression (raw values)')
                plt.ylabel('Density')
                legend_text = (
                    f'Male (n={len(male_scores)}): μ={male_scores.mean():.2f}\n'
                    f'Female (n={len(female_scores)}): μ={female_scores.mean():.2f}\n'
                    f'p-value={p_val:.3f}'
                )
                plt.legend(loc='upper right')
                plt.text(0.05, 0.95, legend_text, transform=plt.gca().transAxes, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                plt.tight_layout()
                plt.savefig(os.path.join(self.groups_plots_dir, f'{sig}_distribution.png'), dpi=300)
                plt.close()
            summary_df = pd.DataFrame([
                {
                    'signature': sig,
                    'male_mean': merged[merged['SEX'] == 'Male'][sig].mean(),
                    'female_mean': merged[merged['SEX'] == 'Female'][sig].mean(),
                    'diff': merged[merged['SEX'] == 'Female'][sig].mean() - merged[merged['SEX'] == 'Male'][sig].mean(),
                    'male_n': len(merged[merged['SEX'] == 'Male'][sig]),
                    'female_n': len(merged[merged['SEX'] == 'Female'][sig]),
                    't_statistic': stats.ttest_ind(merged[merged['SEX'] == 'Male'][sig], 
                                                  merged[merged['SEX'] == 'Female'][sig])[0],
                    'p_value': stats.ttest_ind(merged[merged['SEX'] == 'Male'][sig], 
                                               merged[merged['SEX'] == 'Female'][sig])[1]
                } for sig in self.data_loader.signatures.keys()
            ])
            summary_df['fdr'] = multipletests(summary_df['p_value'], method='fdr_bh')[1]
            summary_df.to_csv(os.path.join(self.groups_dir, 'group_sex_differences.csv'), index=False)
            return summary_df
        except Exception as e:
            print(f"Error plotting group distributions: {e}")
            print(traceback.format_exc())
            return None

    def analyze_group_survival_with_zscore(self, scores, clinical_data):
        """Perform survival analysis with z-score normalization."""
        try:
            print("Analyzing survival with z-score normalization for CD8 groups...")
            merged = clinical_data.merge(scores, left_on='PATIENT_ID', right_index=True, how='inner')
            merged = self.filter_by_diagnosis(merged)
            merged['survival_time'] = merged['OS_TIME']
            merged['death_event'] = merged['OS_EVENT']
            merged = merged.dropna(subset=['survival_time', 'death_event'])
            if merged.index.duplicated().any():
                merged = merged.loc[~merged.index.duplicated(keep='first')]
            male_data = merged[merged['SEX'] == 'Male'].copy()
            female_data = merged[merged['SEX'] == 'Female'].copy()
            male_results, female_results = [], []
            for sig in self.data_loader.signatures.keys():
                mean, std = merged[sig].mean(), merged[sig].std()
                for data, results, sex in [(male_data, male_results, 'Male'), (female_data, female_results, 'Female')]:
                    data[f'{sig}_zscore'] = (data[sig] - mean) / std
                    data[f'{sig}_high'] = (data[f'{sig}_zscore'] > 0).astype(int)
                    if len(data) >= 20:
                        cox_data = data[['survival_time', 'death_event', f'{sig}_zscore', f'{sig}_high', 'AGE']].dropna()
                        high, low = cox_data[cox_data[f'{sig}_high'] == 1], cox_data[cox_data[f'{sig}_high'] == 0]
                        if len(cox_data) >= 10 and len(high) > 5 and len(low) > 5:
                            cph = CoxPHFitter()
                            cph.fit(cox_data, duration_col='survival_time', event_col='death_event')
                            results.append({
                                'signature': sig,
                                'hazard_ratio': np.exp(cph.params_[f'{sig}_high']),
                                'p_value': cph.summary.loc[f'{sig}_high', 'p'],
                                'ci_lower': np.exp(cph.confidence_intervals_.loc[f'{sig}_high', '0.025 lower-bound']),
                                'ci_upper': np.exp(cph.confidence_intervals_.loc[f'{sig}_high', '0.975 upper-bound']),
                                'n_patients': len(cox_data),
                                'n_high': len(high),
                                'n_low': len(low),
                                'mean_zscore': cox_data[f'{sig}_zscore'].mean()
                            })
            male_df = pd.DataFrame(male_results)
            female_df = pd.DataFrame(female_results)
            if len(male_df) > 0:
                male_df['fdr'] = multipletests(male_df['p_value'], method='fdr_bh')[1]
                male_df['sex'] = 'Male'
                male_df.to_csv(os.path.join(self.groups_dir, 'male_group_survival.csv'), index=False)
            if len(female_df) > 0:
                female_df['fdr'] = multipletests(female_df['p_value'], method='fdr_bh')[1]
                female_df['sex'] = 'Female'
                female_df.to_csv(os.path.join(self.groups_dir, 'female_group_survival.csv'), index=False)
            if len(male_df) > 0 and len(female_df) > 0:
                self.plot_group_forest(male_df, female_df)
            return male_df, female_df
        except Exception as e:
            print(f"Error in group survival analysis: {e}")
            print(traceback.format_exc())
            return None, None

    def plot_group_forest(self, male_results, female_results):
        """Create a forest plot for survival results."""
        # Simplified for brevity; full implementation as in original code
        pass

    def check_extreme_results(self, male_results, female_results):
        """Check for extreme hazard ratios or confidence intervals."""
        # Simplified for brevity; full implementation as in original code
        pass