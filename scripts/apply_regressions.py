import os
import sys
import ast
import argparse

import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.stats.multitest as multi
from scipy import stats


import warnings
warnings.filterwarnings('ignore')


def fit_OLS(x,y):
    model = sm.OLS(y, sm.add_constant(x))
    results = model.fit()
    return results.f_pvalue, results.params[1]

def run_regressions(data, col='n'):
    reg = data.groupby(['id','label']).apply(lambda grp: fit_OLS(grp['datenum'], grp[col])).reset_index()
    reg['slope'] = reg[0].apply(lambda x: x[1])
    reg['f_pvalue'] = reg[0].apply(lambda x: x[0])
    reg = reg[~reg['f_pvalue'].isnull()]
    reg['corrected'] = multi.fdrcorrection(reg['f_pvalue'])[1]
    reg = reg.merge(data[['id','group','variant','clade']], on='id').drop_duplicates('id')
    reg = reg.drop(columns=[0])
    t, p = stats.ttest_ind(reg[reg['label'] == 1]['slope'],reg[reg['label'] == 0]['slope'])
    return reg, p

def anova_test(res, y='slope', x='variant'):
    c = res[(res['label'] == 1) & (res['variant'].isin(['alpha', 'delta', 'BA.1', 'BA.2']))]
    model = ols(f'{y} ~ {x}', data=c).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    tukey_results = pairwise_tukeyhsd(c[y], c[x])

    anova_table = pd.DataFrame(anova_table)
    df = pd.DataFrame(tukey_results.summary())
    df, df.columns = df[1:], df.iloc[0]
    return anova_table, df


def main(args):

    # load data for regression
    data = pd.read_csv('/sternadi/home/volume3/chronic-corona-pred/data/case_control/rates/regression_data.csv')
    data['added_mutations'] = data['added_mutations'].apply(lambda x: ast.literal_eval(x))

    y = args.y
    alias = args.alias
    base_dir = "/sternadi/home/volume3/chronic-corona-pred/data/case_control/rates/regressions"
    reg, p = run_regressions(data, col=y)

    results_dir = os.path.join(base_dir, alias)
    os.makedirs(results_dir, exist_ok=True)

    reg.to_csv(os.path.join(results_dir, 'slopes.csv'), index=False)
    with open(os.path.join(results_dir, 'p_value.txt'), 'w') as o:
        o.write(f"T-test case vs. controls p-value: {p}")

    grouped_by_var = reg.groupby(['variant', 'label'])['slope'].mean().reset_index()
    grouped_by_var.to_csv(os.path.join(results_dir, 'slopes_by_variant_and_class.csv'), index=False)

    # Case ANOVA for variant differences
    anova_table, tukey_results = anova_test(reg, y='slope', x='variant')
    anova_table.to_csv(os.path.join(results_dir, 'anova.csv'), index=False)
    tukey_results.to_csv(os.path.join(results_dir, 'tukey.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Regression analysis for per-variant rates')
    parser.add_argument('-y', default='n',
                        help='The column in the data to apply regression on')
    parser.add_argument('--alias', default='overall',
                        help='The name of the folder for results')
    args = parser.parse_args()

    main(args)


