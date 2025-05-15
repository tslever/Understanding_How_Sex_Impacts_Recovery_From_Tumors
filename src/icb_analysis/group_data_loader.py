import pandas as pd
import numpy as np
import os
import traceback

class GroupDataLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.signatures = {}
        self.cluster_desc = {
            'CD8_B': 'Non-responder enriched (Clusters 1-3)',
            'CD8_G': 'Responder enriched (Clusters 4-6)'
        }
        self._load_group_signatures()

    def _load_group_signatures(self):
        """Load CD8+ T cell group signatures from a CSV file."""
        try:
            groups = pd.read_csv("two_groups.csv")
            for col in groups.columns:
                self.signatures[col] = groups[col].dropna().tolist()
            print(f"Loaded {len(self.signatures)} CD8+ T cell group signatures")
            for sig, genes in self.signatures.items():
                print(f"{sig}: {len(genes)} genes")
            print("Sample genes from CD8_B:", self.signatures['CD8_B'][:5])
            print("Sample genes from CD8_G:", self.signatures['CD8_G'][:5])
        except Exception as e:
            print(f"Error loading group signatures: {e}")
            print(traceback.format_exc())

    def calculate_group_ratio(self, scores):
        """Calculate the CD8_G to CD8_B ratio and its log transform."""
        try:
            scores_with_ratio = scores.copy()
            scores_with_ratio['CD8_GtoB_ratio'] = scores_with_ratio['CD8_G'] / (scores_with_ratio['CD8_B'] + 0.001)
            scores_with_ratio['CD8_GtoB_log'] = np.log2(scores_with_ratio['CD8_GtoB_ratio'])
            self.cluster_desc['CD8_GtoB_ratio'] = 'Responder/Non-responder ratio'
            self.cluster_desc['CD8_GtoB_log'] = 'Log2(Responder/Non-responder ratio)'
            self.signatures['CD8_GtoB_ratio'] = []
            self.signatures['CD8_GtoB_log'] = []
            print("Added CD8_G/CD8_B ratio to scores")
            return scores_with_ratio
        except Exception as e:
            print(f"Error calculating group ratio: {e}")
            print(traceback.format_exc())
            return scores