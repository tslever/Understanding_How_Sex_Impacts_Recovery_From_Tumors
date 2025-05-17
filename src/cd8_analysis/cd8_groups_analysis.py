"""
CD8 Groups Analysis
Analyzes CD8+ T cell signatures and groups
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import traceback
from datetime import datetime
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Add parent directory to path to allow imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cd8_analysis.cd8_analysis import CD8Analysis
from utils.shared_functions import load_rnaseq_data, load_clinical_data, filter_by_diagnosis

class CD8GroupAnalysis(CD8Analysis):
    """Analyzes CD8+ T cell signatures and groups"""
    
    def __init__(self, base_path):
        """Initialize CD8 group analysis with base path"""
        self.base_path = base_path
        
        # Define output directories
        self.output_dir = os.path.join(self.base_path, "output/cd8_groups_analysis")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.results_dir = os.path.join(self.output_dir, "results")
        
        # Create output directories if they don't exist
        for directory in [self.output_dir, self.plots_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Define CD8 T cell groups and their marker genes
        self.cd8_groups = {
            'CD8_A': ['CD8A', 'CD8B', 'GZMA', 'GZMB', 'PRF1', 'IFNG'],
            'CD8_B': ['ENTPD1', 'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT'],
            'CD8_C': ['IL7R', 'TCF7', 'CCR7', 'SELL', 'LEF1', 'CD27'],
            'CD8_D': ['NR4A1', 'NR4A2', 'NR4A3', 'TOX', 'TOX2'],
            'CD8_E': ['GZMK', 'GZMM', 'CD74', 'CD27', 'XCL1', 'XCL2'],
            'CD8_F': ['FGFBP2', 'FCGR3A', 'CX3CR1', 'KLRG1', 'GNLY', 'NKG7'],
            'CD8_G': ['MKI67', 'TOP2A', 'CDK1', 'PCNA', 'TYMS']
        }
        
        # Define cluster descriptions
        self.cluster_desc = {
            'CD8_B': 'Non-responder enriched (Clusters 1-3)',
            'CD8_G': 'Responder enriched (Clusters 4-6)',
            'CD8_GtoB_ratio': 'Responder/Non-responder ratio',
            'CD8_GtoB_log': 'Log2(Responder/Non-responder ratio)'
        }
    
    def calculate_group_scores(self, rnaseq_data):
        """Calculate CD8 group scores"""
        try:
            print("\nCalculating CD8 group scores...")
            
            # Initialize scores DataFrame
            scores = pd.DataFrame(index=rnaseq_data.columns)
            
            # Calculate scores for each group
            for group, genes in self.cd8_groups.items():
                # Filter to genes in the group that are present in the data
                group_genes = [gene for gene in genes if gene in rnaseq_data.index]
                
                if len(group_genes) == 0:
                    print(f"Warning: No genes found for group {group}")
                    scores[group] = 0.0
                else:
                    print(f"Calculating {group} score using {len(group_genes)} genes")
                    # Calculate mean expression across genes
                    scores[group] = rnaseq_data.loc[group_genes].mean()
            
            # Calculate ratio and log ratio
            if 'CD8_B' in scores.columns and 'CD8_G' in scores.columns:
                # Add small constant to avoid division by zero
                scores['CD8_GtoB_ratio'] = scores['CD8_G'] / (scores['CD8_B'] + 0.01)
                scores['CD8_GtoB_log'] = np.log2(scores['CD8_GtoB_ratio'])
            
            # Save scores
            scores.to_csv(os.path.join(self.results_dir, 'cd8_group_scores.csv'))
            
            print(f"Calculated CD8 group scores for {len(scores)} samples")
            
            wanted_cols = list(self.cd8_groups.keys())
            
            return pd.DataFrame(scores).reindex(wanted_cols, fill_value = 0.0).pipe(zscore, axis = 1).T
            
        except Exception as e:
            print(f"Error calculating group scores: {e}")
            print(traceback.format_exc())
            return None
    
    def cluster_samples(self, scores, n_clusters=6):
        """Cluster samples based on CD8 group scores"""
        try:
            print(f"\nClustering samples into {n_clusters} clusters...")
            
            # Select features for clustering
            features = ['CD8_A', 'CD8_B', 'CD8_C', 'CD8_D', 'CD8_E', 'CD8_F', 'CD8_G']
            missing = [c for c in features if c not in scores.columns]
            if missing:
                logger.warning("Missing CD8 score columns %s - inserting zeros.", missing)
                scores = scores.assign(**{c: 0.0 for c in missing})
            X = scores[features].values
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to scores
            scores_with_clusters = scores.copy()
            scores_with_clusters['cluster'] = clusters
            
            # Calculate silhouette score
            silhouette = silhouette_score(X_scaled, clusters)
            print(f"Silhouette score: {silhouette:.3f}")
            
            # Save clustered data
            scores_with_clusters.to_csv(os.path.join(self.results_dir, 'cd8_clusters.csv'))
            
            # Plot clusters
            self.plot_clusters(scores_with_clusters, features)
            
            print(f"Clustered {len(scores)} samples into {n_clusters} clusters")
            
            return scores_with_clusters
            
        except Exception as e:
            print(f"Error clustering samples: {e}")
            print(traceback.format_exc())
            return None
    
    def plot_clusters(self, scores_with_clusters, features):
        """Plot clusters"""
        try:
            # Create scatter plot
            plt.figure(figsize=(10, 8))
            
            # Plot each cluster
            for cluster in sorted(scores_with_clusters['cluster'].unique()):
                cluster_data = scores_with_clusters[scores_with_clusters['cluster'] == cluster]
                plt.scatter(
                    cluster_data[features[0]],
                    cluster_data[features[1]],
                    label=f'Cluster {cluster+1}',
                    alpha=0.7,
                    s=50
                )
            
            # Add labels and title
            plt.xlabel(f"{features[0]} ({self.cluster_desc.get(features[0], 'Unknown')})")
            plt.ylabel(f"{features[1]} ({self.cluster_desc.get(features[1], 'Unknown')})")
            plt.title('CD8 Group Clusters')
            
            # Add legend
            plt.legend()
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'cd8_clusters.png'), dpi=300)
            plt.close()
            
            # Create PCA plot
            self.plot_pca(scores_with_clusters)
            
            print("Saved cluster plots")
            
        except Exception as e:
            print(f"Error plotting clusters: {e}")
            print(traceback.format_exc())
    
    def plot_pca(self, scores_with_clusters):
        """Plot PCA of CD8 group scores"""
        try:
            # Select features for PCA
            features = ['CD8_B', 'CD8_G', 'CD8_GtoB_ratio', 'CD8_GtoB_log']
            features = [f for f in features if f in scores_with_clusters.columns]
            X = scores_with_clusters[features].values
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Create scatter plot
            plt.figure(figsize=(10, 8))
            
            # Plot each cluster
            for cluster in sorted(scores_with_clusters['cluster'].unique()):
                cluster_mask = scores_with_clusters['cluster'] == cluster
                plt.scatter(
                    X_pca[cluster_mask, 0],
                    X_pca[cluster_mask, 1],
                    label=f'Cluster {cluster+1}',
                    alpha=0.7,
                    s=50
                )
            
            # Add labels and title
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
            plt.title('PCA of CD8 Group Scores')
            
            # Add legend
            plt.legend()
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'cd8_pca.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error plotting PCA: {e}")
            print(traceback.format_exc())
    
    def analyze_clusters_by_sex(self, scores_with_clusters, clinical_data):
        """Analyze clusters by sex"""
        try:
            print("\nAnalyzing clusters by sex...")
            
            # Merge with clinical data
            merged = clinical_data.merge(
                scores_with_clusters,
                left_on='PATIENT_ID',
                right_index=True,
                how='inner'
            )
            
            # Filter by diagnosis
            merged = filter_by_diagnosis(merged)
            
            # Create cluster distribution by sex
            cluster_by_sex = pd.crosstab(
                merged['cluster'],
                merged['SEX'],
                normalize='columns'
            ) * 100
            
            # Save distribution
            cluster_by_sex.to_csv(os.path.join(self.results_dir, 'cluster_by_sex.csv'))
            
            # Plot distribution
            plt.figure(figsize=(10, 6))
            cluster_by_sex.plot(kind='bar')
            plt.xlabel('Cluster')
            plt.ylabel('Percentage (%)')
            plt.title('Cluster Distribution by Sex')
            plt.xticks(rotation=0)
            plt.legend(title='Sex')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'cluster_by_sex.png'), dpi=300)
            plt.close()
            
            # Create summary statistics by sex
            summary = []
            
            for sex in ['Male', 'Female']:
                sex_data = merged[merged['SEX'] == sex]
                
                for feature in ['CD8_B', 'CD8_G', 'CD8_GtoB_ratio', 'CD8_GtoB_log']:
                    if feature in sex_data.columns:
                        summary.append({
                            'sex': sex,
                            'feature': feature,
                            'mean': sex_data[feature].mean(),
                            'median': sex_data[feature].median(),
                            'std': sex_data[feature].std(),
                            'count': len(sex_data)
                        })
            
            # Create DataFrame
            summary_df = pd.DataFrame(summary)
            
            # Save summary
            summary_df.to_csv(os.path.join(self.results_dir, 'cd8_by_sex.csv'), index=False)
            
            # Plot CD8 scores by sex
            self.plot_cd8_by_sex(summary_df)
            
            # Perform statistical tests
            self.test_cd8_by_sex(merged)
            
            print(f"Analyzed clusters by sex for {len(merged)} patients")
            
            return merged
            
        except Exception as e:
            print(f"Error analyzing clusters by sex: {e}")
            print(traceback.format_exc())
            return None
    
    def plot_cd8_by_sex(self, summary_df):
        """Plot CD8 scores by sex"""
        try:
            # Create bar plot
            plt.figure(figsize=(12, 6))
            
            # Plot each feature
            for i, feature in enumerate(['CD8_B', 'CD8_G', 'CD8_GtoB_ratio', 'CD8_GtoB_log']):
                feature_data = summary_df[summary_df['feature'] == feature]
                
                if len(feature_data) == 0:
                    continue
                
                # Create subplot
                plt.subplot(1, 4, i+1)
                
                # Create bar plot
                sns.barplot(x='sex', y='mean', data=feature_data)
                
                # Add error bars
                for j, row in feature_data.iterrows():
                    plt.errorbar(
                        x=j % 2,  # Position based on sex
                        y=row['mean'],
                        yerr=row['std'],
                        fmt='none',
                        color='black',
                        capsize=5
                    )
                
                # Add labels and title
                plt.xlabel('Sex')
                plt.ylabel('Mean Score')
                plt.title(f'{feature} ({self.cluster_desc.get(feature, "Unknown")})')
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'cd8_by_sex.png'), dpi=300)
            plt.close()
            
            print("Saved CD8 by sex plots")
            
        except Exception as e:
            print(f"Error plotting CD8 by sex: {e}")
            print(traceback.format_exc())
    
    def test_cd8_by_sex(self, merged):
        """Perform statistical tests for CD8 scores by sex"""
        try:
            # Create results list
            test_results = []
            
            # Test each feature
            for feature in ['CD8_B', 'CD8_G', 'CD8_GtoB_ratio', 'CD8_GtoB_log']:
                if feature not in merged.columns:
                    continue
                
                # Get data by sex
                male = merged[merged['SEX'] == 'Male'][feature]
                female = merged[merged['SEX'] == 'Female'][feature]
                
                # Skip if not enough samples
                if len(male) < 10 or len(female) < 10:
                    continue
                
                # Perform t-test
                t_stat, p_val = stats.ttest_ind(male, female, equal_var=False)
                
                # Add to results
                test_results.append({
                    'feature': feature,
                    'male_mean': male.mean(),
                    'female_mean': female.mean(),
                    'male_count': len(male),
                    'female_count': len(female),
                    't_stat': t_stat,
                    'p_value': p_val
                })
            
            # Create DataFrame
            test_df = pd.DataFrame(test_results)
            
            # Add significance indicator
            test_df['significant'] = test_df['p_value'] < 0.05
            
            # Save results
            test_df.to_csv(os.path.join(self.results_dir, 'cd8_by_sex_tests.csv'), index=False)
            
            print("Performed statistical tests for CD8 scores by sex")
            
            return test_df
            
        except Exception as e:
            print(f"Error testing CD8 by sex: {e}")
            print(traceback.format_exc())
            return None
    
    def analyze_clusters_by_diagnosis(self, scores_with_clusters, clinical_data):
        """Analyze clusters by diagnosis"""
        try:
            print("\nAnalyzing clusters by diagnosis...")
            
            # Merge with clinical data
            merged = clinical_data.merge(
                scores_with_clusters,
                left_on='PATIENT_ID',
                right_index=True,
                how='inner'
            )
            
            # Get top diagnoses
            diagnosis_counts = merged['DIAGNOSIS'].value_counts()
            top_diagnoses = diagnosis_counts[diagnosis_counts >= 20].index.tolist()
            
            if len(top_diagnoses) == 0:
                print("Warning: No diagnoses with at least 20 patients")
                return None
            
            # Filter to top diagnoses
            merged_top = merged[merged['DIAGNOSIS'].isin(top_diagnoses)]
            
            # Create cluster distribution by diagnosis
            cluster_by_diagnosis = pd.crosstab(
                merged_top['cluster'],
                merged_top['DIAGNOSIS'],
                normalize='columns'
            ) * 100
            
            # Save distribution
            cluster_by_diagnosis.to_csv(os.path.join(self.results_dir, 'cluster_by_diagnosis.csv'))
            
            # Plot distribution
            plt.figure(figsize=(12, 8))
            cluster_by_diagnosis.plot(kind='bar')
            plt.xlabel('Cluster')
            plt.ylabel('Percentage (%)')
            plt.title('Cluster Distribution by Diagnosis')
            plt.xticks(rotation=0)
            plt.legend(title='Diagnosis', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'cluster_by_diagnosis.png'), dpi=300)
            plt.close()
            
            # Create summary statistics by diagnosis
            summary = []
            
            for diagnosis in top_diagnoses:
                diag_data = merged[merged['DIAGNOSIS'] == diagnosis]
                
                for feature in ['CD8_B', 'CD8_G', 'CD8_GtoB_ratio', 'CD8_GtoB_log']:
                    if feature in diag_data.columns:
                        summary.append({
                            'diagnosis': diagnosis,
                            'feature': feature,
                            'mean': diag_data[feature].mean(),
                            'median': diag_data[feature].median(),
                            'std': diag_data[feature].std(),
                            'count': len(diag_data)
                        })
            
            # Create DataFrame
            summary_df = pd.DataFrame(summary)
            
            # Save summary
            summary_df.to_csv(os.path.join(self.results_dir, 'cd8_by_diagnosis.csv'), index=False)
            
            # Plot CD8 scores by diagnosis
            self.plot_cd8_by_diagnosis(summary_df)
            
            print(f"Analyzed clusters by diagnosis for {len(merged_top)} patients with top diagnoses")
            
            return merged_top
            
        except Exception as e:
            print(f"Error analyzing clusters by diagnosis: {e}")
            print(traceback.format_exc())
            return None
    
    def plot_cd8_by_diagnosis(self, summary_df):
        """Plot CD8 scores by diagnosis"""
        try:
            # Plot each feature
            for feature in ['CD8_B', 'CD8_G', 'CD8_GtoB_ratio', 'CD8_GtoB_log']:
                feature_data = summary_df[summary_df['feature'] == feature]
                
                if len(feature_data) == 0:
                    continue
                
                # Create bar plot
                plt.figure(figsize=(12, 6))
                sns.barplot(x='diagnosis', y='mean', data=feature_data)
                
                # Add error bars
                for i, row in feature_data.iterrows():
                    plt.errorbar(
                        x=i % len(feature_data['diagnosis'].unique()),  # Position based on diagnosis
                        y=row['mean'],
                        yerr=row['std'],
                        fmt='none',
                        color='black',
                        capsize=5
                    )
                
                # Add labels and title
                plt.xlabel('Diagnosis')
                plt.ylabel('Mean Score')
                plt.title(f'{feature} ({self.cluster_desc.get(feature, "Unknown")}) by Diagnosis')
                plt.xticks(rotation=45, ha='right')
                
                # Save plot
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, f'{feature}_by_diagnosis.png'), dpi=300)
                plt.close()
            
            print("Saved CD8 by diagnosis plots")
            
        except Exception as e:
            print(f"Error plotting CD8 by diagnosis: {e}")
            print(traceback.format_exc())
    
    def analyze_survival_by_cluster(self, scores_with_clusters, clinical_data):
        """Analyze survival by cluster"""
        try:
            print("\nAnalyzing survival by cluster...")
            
            # Merge with clinical data
            merged = clinical_data.merge(
                scores_with_clusters,
                left_on='PATIENT_ID',
                right_index=True,
                how='inner'
            )
            
            # Filter by diagnosis
            merged = filter_by_diagnosis(merged)
            
            # Check if survival data is available
            if 'OS_MONTHS' not in merged.columns or 'OS_STATUS' not in merged.columns:
                print("Warning: Survival data not available")
                return None
            
            # Create survival status indicator (1 for death, 0 for censored)
            merged['event'] = (merged['OS_STATUS'] == 'DECEASED').astype(int)
            
            # Group clusters into high and low CD8_GtoB_ratio
            if 'CD8_GtoB_ratio' in merged.columns:
                median_ratio = merged['CD8_GtoB_ratio'].median()
                merged['CD8_GtoB_group'] = (merged['CD8_GtoB_ratio'] > median_ratio).map({True: 'High', False: 'Low'})
                
                # Plot Kaplan-Meier curves by CD8_GtoB_group
                self.plot_survival_curves(merged, 'CD8_GtoB_group')
            
            # Plot Kaplan-Meier curves by cluster
            self.plot_survival_curves(merged, 'cluster')
            
            print(f"Analyzed survival by cluster for {len(merged)} patients")
            
            return merged
            
        except Exception as e:
            print(f"Error analyzing survival by cluster: {e}")
            print(traceback.format_exc())
            return None
    
    def plot_survival_curves(self, merged, group_col):
        """Plot Kaplan-Meier survival curves by group"""
        try:
            from lifelines import KaplanMeierFitter
            from lifelines.statistics import logrank_test
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Initialize Kaplan-Meier fitter
            kmf = KaplanMeierFitter()
            
            # Plot survival curve for each group
            for group in sorted(merged[group_col].unique()):
                group_data = merged[merged[group_col] == group]
                
                # Skip if not enough samples
                if len(group_data) < 10:
                    continue
                
                # Fit survival curve
                kmf.fit(
                    group_data['OS_MONTHS'],
                    group_data['event'],
                    label=f'{group} (n={len(group_data)})'
                )
                
                # Plot survival curve
                kmf.plot()
            
            # Add labels and title
            plt.xlabel('Months')
            plt.ylabel('Survival Probability')
            plt.title(f'Kaplan-Meier Survival Curves by {group_col}')
            
            # Add grid
            plt.grid(alpha=0.3)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'survival_by_{group_col}.png'), dpi=300)
            plt.close()
            
            # Perform log-rank test if there are exactly 2 groups
            if len(merged[group_col].unique()) == 2:
                groups = sorted(merged[group_col].unique())
                group1_data = merged[merged[group_col] == groups[0]]
                group2_data = merged[merged[group_col] == groups[1]]
                
                results = logrank_test(
                    group1_data['OS_MONTHS'],
                    group2_data['OS_MONTHS'],
                    group1_data['event'],
                    group2_data['event']
                )
                
                print(f"Log-rank test p-value: {results.p_value:.4f}")
                
                # Save results
                with open(os.path.join(self.results_dir, f'logrank_{group_col}.txt'), 'w') as f:
                    f.write(f"Log-rank test results for {group_col}:\n")
                    f.write(f"Group 1: {groups[0]} (n={len(group1_data)})\n")
                    f.write(f"Group 2: {groups[1]} (n={len(group2_data)})\n")
                    f.write(f"p-value: {results.p_value:.4f}\n")
            
            print(f"Saved survival curves by {group_col}")
            
        except Exception as e:
            print(f"Error plotting survival curves: {e}")
            print(traceback.format_exc())
    
    def run_full_analysis(self):
        """Run full CD8 group analysis"""
        try:
            print("\nRunning full CD8 group analysis...")
            
            # Load RNA-seq data
            rnaseq_data = load_rnaseq_data(self.base_path)
            if rnaseq_data is None:
                return None
            
            # Calculate CD8 group scores
            scores = self.calculate_group_scores(rnaseq_data)
            if scores is None:
                return None
            
            # Cluster samples
            scores_with_clusters = self.cluster_samples(scores)
            if scores_with_clusters is None:
                return None
            
            # Load clinical data
            clinical_data = load_clinical_data(self.base_path)
            if clinical_data is None:
                return None
            
            # Analyze clusters by sex
            self.analyze_clusters_by_sex(scores_with_clusters, clinical_data)
            
            # Analyze clusters by diagnosis
            self.analyze_clusters_by_diagnosis(scores_with_clusters, clinical_data)
            
            # Analyze survival by cluster
            self.analyze_survival_by_cluster(scores_with_clusters, clinical_data)
            
            print("\nCD8 group analysis complete!")
            print(f"Results saved to {self.results_dir}")
            print(f"Plots saved to {self.plots_dir}")
            
            return scores_with_clusters
            
        except Exception as e:
            print(f"Error running full analysis: {e}")
            print(traceback.format_exc())
            return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CD8 Group Analysis')
    parser.add_argument('--base-path', type=str, default='/project/orien/data/aws/24PRJ217UVA_IORIG',
                        help='Base path for data files')
    args = parser.parse_args()
    
    analysis = CD8GroupAnalysis(args.base_path)
    analysis.run_full_analysis() 