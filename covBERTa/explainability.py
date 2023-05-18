import numpy as np
import pandas as pd

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns

import re
from dataset.extract_mutations_by_epi import GENEMAP

def get_gene(x):
    if ":" in x:
        y = x.split(':')[0]
    else:
        y = [g for g in GENEMAP if GENEMAP[g][0] <= int(re.findall('\d+',x)[0]) <= GENEMAP[g][1]]
        if y == []:
            y='non-coding'
        else:
            y = y[0]
    return y

def load_data(fold, clade=''):
    explain = pd.read_csv(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/models/{fold}/{clade}explainer.csv')
    explain['exp_score'] = explain['exp_score'] / explain['n_samples']
    explain['exp_score'] = explain.apply(lambda x: -1 * x['exp_score'] if x['label'] == 0 else x['exp_score'], axis=1)

    remove_case = np.load(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/models/'
                          f'{fold}/explainability/{clade}exclude_case.npy')
    remove_control = np.load(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/models/'
                             f'{fold}/explainability/{clade}exclude_control.npy')

    exp = explain[((explain['label'] == 1) & (~explain['mt'].isin(remove_case))) |
                  ((explain['label'] == 0) & (~explain['mt'].isin(remove_control)))]

    exp = exp[exp['score'] > 0.05]
    exp.to_csv(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/models/'
               f'{fold}/explainability/{clade}explainer.csv', index=False)
    return exp

def plot_scatter(df, fold, clade=''):
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            if point['y'] < 0.1:
                continue
            ax.text(point['x'] + .002, point['y'] + 0.002, str(point['val']), fontsize=10)

    colors = ['#375E79', '#FB6542']
    for i, alias in enumerate(['control','case']):
        with sns.plotting_context('poster'):
            plt_data = df[(df['label'] == i) & (~df['mt'].str.contains('-'))]
            f, ax = plt.subplots(figsize=(6, 5))
            ax = sns.scatterplot(x='exp_score', y='score', data=plt_data, size='n_samples', alpha=0.6,
                                 color=colors[i], legend=False)
            label_point(plt_data['exp_score'], plt_data['score'], plt_data['mt'], ax)
            plt.ylim(0.04, 1.1)
            if i == 0:
                plt.xlim(-1, -0.6)
            else:
                plt.xlim(0.6,1)
            plt.yscale('log')
            plt.ylabel(r'Mutation Score', fontsize=18)
            plt.xlabel(r'Explaination relaibility ($R^2$)', fontsize=18)
            plt.tight_layout()
            plt.savefig(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/models/'
                        f'{fold}/explainability/{clade}{alias}_mutations.pdf')
            plt.close()

def plot_gene_bars(df, fold, clade=''):
    df['gene'] = df['mt'].apply(lambda x: get_gene(x))
    df['type'] = df['mt'].apply(lambda x: 'non-syn' if ':' in x else ('del' if '-' in x or x.isdigit() else 'syn'))
    grp = df.groupby(['label', 'gene', 'type']).agg({'n_samples': sum, 'score': sum}).reset_index()
    grp = grp[grp['gene'] != 'non-coding']
    order = ['ORF1a', 'ORF1b', 'S', 'ORF3a', 'E', 'M', 'ORF6', 'ORF7a', 'ORF7b', 'ORF8', 'N', 'ORF9b']
    with sns.plotting_context("talk"):
        sns.catplot(x='gene', y='score', data=grp[grp['type'] != 'del'], row='type',
                    hue='label', kind='bar', aspect=5, height=2, palette=['#375E79', '#FB6542'], order=order)
        plt.tight_layout()
        plt.savefig(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/models/'
                    f'{fold}/explainability/{clade}{fold}_gene_counts.pdf')
        plt.close()

def main():
    print("Here")
    folds = ['alpha', 'delta','omicron']
    for fold in folds:
        if fold == 'omicron':
            for clade in ['BA1_', 'BA2_']:
                print(clade)
                exp = load_data(fold, clade=clade)
                plot_scatter(exp, fold, clade=clade)
                plot_gene_bars(exp, fold, clade=clade)

if __name__ == "__main__":
    main()