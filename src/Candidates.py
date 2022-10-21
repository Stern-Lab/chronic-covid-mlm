import os
import itertools
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
import statsmodels.api as sm
import statsmodels.stats.multitest as multi
import matplotlib.pyplot as plt

from matplotlib import pyplot

class KnownCandidates:
    def __init__(self, clade, mutation_to_consider=['num_aa_from_ref']):
        self.clade_data = clade
        self.mutation_to_consider = mutation_to_consider

        self.candidates_data = None

    def get_stats(self, pvalue_thresh=0.01):
        def fit_candidate_OLS(data):
            x = data['datenum']
            y = data[list(self.mutation_to_consider)].sum(axis=1)
            model = sm.OLS(y, sm.add_constant(x))
            results = model.fit()
            return results

        candidates = self.clade_data.candidates
        candidates['stats'] = candidates['reg_data'].apply(lambda data: fit_candidate_OLS(data))
        candidates['f_pvalue'] = candidates['stats'].apply(lambda x: x.f_pvalue)

        # correct for multiple tests
        candidates['corrected_f_pvalue'] = multi.fdrcorrection(candidates['f_pvalue'])[1]
        candidates['minus_log_f_pvalue'] = candidates['corrected_f_pvalue'].apply(lambda x: -np.log10(x))
        candidates['is_significant'] = candidates.apply(
            lambda x: x['corrected_f_pvalue'] < pvalue_thresh and x['stats'].params['datenum'] > 0, axis=1)

        self.candidates_data = candidates.reset_index(drop=True)

    def summarize(self):
        clade_data = self.clade_data
        data = [(clade_data.clade, clade_data.n, clade_data.extremes.shape[0],clade_data.candidates.shape[0])]
        columns = ['variant', 'num_sequences', 'num_extreme_sequence', 'num_candidates']
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join(clade_data.out, 'summary.csv'), index=False)

    def save_candidate_data(self):
        self.candidates_data.to_pickle(os.path.join(self.clade_data.out, 'candidates.pkl'))

    def plot_manhattan(self, pvalue_thresh=0.01):
        candidates = self.candidates_data
        if candidates['is_significant'].nunique() == 1:
            return
        sns.set_context('poster')
        f, ax = plt.subplots(figsize=(12, 5))
        sns.scatterplot(x=candidates.index, y='minus_log_f_pvalue', hue='is_significant', data=candidates,
                        ax=ax, alpha=0.75, palette=['#CBD6D5', 'tomato'])
        ax.axhline(-np.log10(pvalue_thresh), color='red', alpha=0.85, lw=1, linestyle='--')
        _ = ax.set_xticks([])
        plt.savefig(os.path.join(self.clade_data.out, 'manhattan.pdf'), transparent=True, bbox_inches="tight")
