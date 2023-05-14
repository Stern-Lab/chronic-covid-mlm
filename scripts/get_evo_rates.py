import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


def stratify_by_clade_and_size(df, dist, col):
    res = []
    for k, v in dist.items():
        cur = df[df['clade'] == k[0]]
        if cur.shape[0] == 0:
            continue
        cur = cur[(cur['size'] >= k[1] - 2) & (cur['size'] <= k[1] + 2)]
        if cur.shape[0] == 0:
            continue
        r = cur.sample(n=v, replace=True)[col].tolist()
        res.extend(r)
    return res


def bootstrap(df, dist, col, m=1000):
    res = []
    for i in tqdm(range(m)):
        rates = stratify_by_clade_and_size(df, dist, col)
        res.append(np.mean(rates))
    return res

def main(args):
    mode = args.mode
    m = args.m
    gene = args.gene
    input_dir = args.input

    cases = pd.read_csv(os.path.join(input_dir, f"cases_group_data_{mode}_rates.csv"))
    controls = pd.read_csv(os.path.join(input_dir, f"controls_group_data_{mode}_rates.csv"))

    clades = dict(cases.groupby(['clade', 'size']).size())
    res = bootstrap(controls, clades, col=gene, m=m)

    cases_mean = cases[gene].mean()
    background_mean = np.mean(res)

    if cases_mean > background_mean:
        pval = np.sum(np.asarray(res) >= cases_mean) / m
        val = (gene, cases_mean, background_mean, pval, 'right')
    else:
        pval = np.sum(np.asarray(res) <= cases_mean) / m
        val = (gene, cases_mean, background_mean, pval, 'left')

    out = args.out
    np.save(os.path.join(out, f"{gene}_{mode}"), val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rate bootstrap')
    parser.add_argument('--input', default='/sternadi/home/volume3/chronic-corona-pred/data/case_control/rates/no_sp/', type=str,
                        help='input directory for rate estimation')
    parser.add_argument('--gene', default='ORF1a',help='gene name')
    parser.add_argument('--mode', default='overall', type=str, help='overall, syn or non_syn modes')
    parser.add_argument('-m', default=1000, type=int, help='number of bootstrap')
    parser.add_argument('--out', default='./', type=str, help='directory to save results')
    args = parser.parse_args()
    main(args)