import os
import pickle
import glob
import json
import sys

import pandas as pd
import numpy as np

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.stats.multitest as multi

genes = ['ORF1a','ORF3a','ORF9b','ORF1b','N','S','E','ORF8',
         'M','ORF7a','ORF7b','ORF6','rbd','ntd','nsp3','nsp6']
modes = ['overall', 'syn', 'non_syn']
dirnames = ['no_t0_no_lb_no_sp', 't0_lb_no_sp']

for dirname in dirnames:
    dfs = []
    for mode in modes:
        files = [f'/sternadi/home/volume3/chronic-corona-pred/data/case_control/rates/bootstrap/{dirname}/{gene}_{mode}.npy'
                 for gene in genes]
        res = [tuple(np.load(f)) for f in files]
        df = pd.DataFrame(res, columns=['gene','case','control','pvalue','direction'])
        df['mode'] = mode
        df['case'] = df['case'].apply(lambda x: np.round(float(x), 4))
        df['control'] = df['control'].apply(lambda x: np.round(float(x), 4))
        df['pvalue'] = df['pvalue'].astype(float)
        df['direction'] = df['direction'].apply(lambda x: "bigger" if x == 'right' else "smaller")
        df['corrected_pvalue'] = multi.fdrcorrection(df['pvalue'])[1]
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(f'/sternadi/home/volume3/chronic-corona-pred/data/case_control/rates/bootstrap/{dirname}/rates.csv',
              index=False)

    data = pd.melt(df, id_vars=['gene', 'direction', 'pvalue','corrected_pvalue', 'mode'], value_vars=['case', 'control'])
    data['rate'] = data.apply(lambda x: -1 * x['value'] if x['direction'] == "smaller" else x['value'], axis=1)








