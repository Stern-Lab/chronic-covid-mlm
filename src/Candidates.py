import os
import itertools
import math
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

from utils import get_longest_path_per_component, create_mutational_graph

from matplotlib import pyplot

class KnownCandidates:
    def __init__(self, clade, mutation_to_consider=['num_aa_from_ref'], chronic_thresh='20 days'):
        self.clade_data = clade
        self.mutation_to_consider = mutation_to_consider
        self.chronic_thresh = chronic_thresh

        self.candidates_data = None
        self.potential_chronic = None

    @staticmethod
    def get_candidate_time_interval(candidates):
        mapper = candidates.groupby(['candidate_id']).apply(lambda grp: grp['date'].max() - grp['date'].min())
        candidates['time_interval'] = candidates['candidate_id'].apply(lambda x: mapper[x])
        return candidates

    def get_stats(self, pvalue_thresh=0.01):
        def fit_candidate_OLS(data, y_labels=['num_aa_from_ref']):
            x = data['datenum']
            y = data[y_labels].sum(axis=1)
            model = sm.OLS(y, sm.add_constant(x))
            results = model.fit()

            n = data.shape[0]
            return results.f_pvalue, results.params['datenum'], n, data['date'].nunique()/n

        candidates = self.clade_data.candidates

        # generate candidate specific regressions
        reg_dict = candidates.groupby(['candidate_id']).apply(
            lambda grp: fit_candidate_OLS(grp, y_labels=['num_nuc_from_ref', 'num_deletions', 'num_insertions']))
        reg_dict_spike = candidates.groupby(['candidate_id']).apply(
            lambda grp: fit_candidate_OLS(grp, y_labels=['num_spike_mutations_from_ref']))

        # correct for multiple testing
        corrected_f_pvalues = multi.fdrcorrection([f[0] if not math.isnan(f[0]) else 1 for f in reg_dict.values])[1]
        corrected_f_pvalues_spike = \
        multi.fdrcorrection([f[0] if not math.isnan(f[0]) else 1 for f in reg_dict_spike.values])[1]

        candidates['f_pvalue'] = candidates['candidate_id'].apply(lambda x: corrected_f_pvalues[x])
        candidates['slope'] = candidates['candidate_id'].apply(lambda x: reg_dict[x][1])

        candidates['f_pvalue_spike'] = candidates['candidate_id'].apply(lambda x: corrected_f_pvalues_spike[x])
        candidates['slope_spike'] = candidates['candidate_id'].apply(lambda x: reg_dict_spike[x][1])

        candidates['minus_log_f_pvalue'] = candidates['f_pvalue'].apply(lambda x: -np.log10(x))
        candidates['minus_log_f_pvalue_spike'] = candidates['f_pvalue_spike'].apply(lambda x: -np.log10(x))

        candidates['is_significant'] = candidates.apply(
            lambda x: x['f_pvalue'] < pvalue_thresh and x['slope'] > 0, axis=1)
        candidates['is_significant_spike'] = candidates.apply(
            lambda x: x['f_pvalue_spike'] < pvalue_thresh or x['slope_spike'] > 0, axis=1)

        candidates['num_samples'] = candidates['candidate_id'].apply(lambda x: reg_dict[x][2])
        candidates['unique_dates_ratio'] = candidates['candidate_id'].apply(lambda x: reg_dict[x][3])

        candidates = self.get_candidate_time_interval(candidates)

        self.candidates_data = candidates


    def extract_potential_chronic(self):
        data = self.candidates_data
        data = data[(data['is_significant'] == True) & (data['slope'] > 0)]

        res = []
        for candidate_id in data['candidate_id'].unique():
            candidate = data[data['candidate_id'] == candidate_id]
            G = create_mutational_graph(candidate)
            paths = get_longest_path_per_component(G, time_interval=self.chronic_thresh)
            paths['candidate_id'] = candidate_id
            res.append(paths)

        df = pd.concat(res)
        self.potential_chronic = df


    def plot_manhattan(self, pvalue_thresh=0.01, by='is_significant'):
        candidates = self.candidates_data
        if candidates[by].nunique() == 1:
            return
        y_col = 'minus_log_f_pvalue'
        if 'spike' in by:
            y_col += '_spike'

        sns.set_context('poster')
        f, ax = plt.subplots(figsize=(12, 5))
        sns.scatterplot(x='candidate_id', y=y_col, hue=by, data=candidates,
                        ax=ax, alpha=0.75, palette=['#CBD6D5', 'tomato'])
        ax.axhline(-np.log10(pvalue_thresh), color='red', alpha=0.85, lw=1, linestyle='--')
        _ = ax.set_xticks([])
        plt.savefig(os.path.join(self.clade_data.out, f'manhattan_{by}.pdf'), transparent=True, bbox_inches="tight")

    def summarize(self):
        clade_data = self.clade_data
        data = [(clade_data.clade, clade_data.n, clade_data.extremes.shape[0],clade_data.candidates.shape[0])]
        columns = ['variant', 'num_sequences', 'num_extreme_sequence', 'num_candidates']
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join(clade_data.out, 'summary.csv'), index=False)

    def save_candidate_data(self):
        self.candidates_data.to_pickle(os.path.join(self.clade_data.out, 'candidates.pkl'))