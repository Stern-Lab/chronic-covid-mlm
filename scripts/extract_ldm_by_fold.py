import json

import pandas as pd
import numpy as np

def get_ratio(df, fold, mt, mapper):
    ratio = df[(df['clade'].isin(mapper[fold]))&
                       (df['text'].str.contains(mt))].shape[0] / df.shape[0]
    return 100 * np.round(ratio, 4)

def get_clades(test, df, mt):
    txt = test[test['text'].str.contains(mt)]['text']
    return ';'.join(df[df['text'].isin(txt)]['clade'].unique())

with open('/sternadi/home/volume3/chronic-corona-pred/data/clades/clades_nad.json', 'r') as o:
    clades = json.load(o)

# make sets instead of lists:
for clade in clades:
    for c in clades[clade]:
        clades[clade][c] = set(clades[clade][c])

mapper = {'omicron':['21K', '21L', '22A', '22B', '22C'], 'alpha':['20I'], 'delta':['21J', '21I']}

data = pd.read_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/lovocv/sampled_case_controls.csv')

case = pd.read_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/data/cases_clades_with_text.csv')
control = pd.read_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/data/controls_clades_with_text.csv')

for fold in mapper:
    train = pd.read_csv(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/lovocv/'
                        f'fold_{fold}_train.csv')
    test = pd.read_csv(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/lovocv/'
                       f'fold_{fold}_test.csv')
    exp = pd.read_csv(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/models/{fold}/'
                      f'explainability/explainer.csv')

    l = []
    for n, mt in zip(exp['n_samples'], exp['mt']):
        cases_clades = get_clades(test, data, mt)
        if mt.isdigit():
            mt = int(mt)
        for clade in clades:
            for c in clades[clade]:
                if mt in clades[clade][c]:
                    l.append((mt, n, cases_clades, clade))
    df = pd.DataFrame(l, columns=['mt', 'n_samples', 'cases_clades', 'LDM_clade'])
    df['fold'] = fold
    df['mt'] = df['mt'].astype(str)

    df['control_ratio'] = df['mt'].apply(lambda x: get_ratio(control, fold, x, mapper))
    df['case_ratio'] = df['mt'].apply(lambda x: get_ratio(case, fold, x, mapper))
    df.to_csv(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/models/{fold}/'
                      f'explainability/LDM.csv', index=False)
