import json
import pickle
import functools

import pandas as pd
import numpy as np

from extract_mutations_by_epi import GENEMAP, comp, flatten

# helper function
def has_numbers(string):
    return any(char.isdigit() for char in string)

def remove_ld(text, clade, mapper):
    # if no sentence is available:
    if text == 'no-sentence':
        return 'no-sentence'
    mts = []
    for mt in text.split():
        # check if there is an intersection for del positions
        # deletion
        if '-' in mt or mt.isdigit():
            s, e = int(mt.split('-')[0]), int(mt.split('-')[-1])
            r = {x for x in range(s, e+1)}
            if len(r.intersection(mapper[clade]['del'])) == 0:
                mts.append(mt)
        # aa mutation
        elif ':' in mt and has_numbers(mt.split(':')[-1]):
            aa, pos = mt.split(':')
            gene_start = GENEMAP[aa][0]
            aa_pos = int(pos[1:-1])
            r = {gene_start + aa_pos * 3 - 1, gene_start + aa_pos * 3 - 2,
                     gene_start + aa_pos * 3 - 3}
            if (len(r.intersection(mapper[clade]['del'])) == 0) and (mt not in mapper[clade]['aa']):
                mts.append(mt)
        # nuc mutation
        elif ':' not in mt:
            r = {int(mt[1:-1])}
            if (len(r.intersection(mapper[clade]['del'])) == 0) and (mt not in mapper[clade]['nuc']):
                mts.append(mt)
        # all insertions are valid
        else:
            print(mt)
    return ' '.join(mts)


# load data
with open('/sternadi/home/volume3/chronic-corona-pred/data/clades/clades_nad.json', 'r') as o:
    clades = json.load(o)
with open('/sternadi/home/volume3/chronic-corona-pred/data/GISAID/mappers/meta_mapper.pkl', 'rb') as o:
    mapper = pickle.load(o)
with open('/sternadi/home/volume3/chronic-corona-pred/data/case_control/cases/epi_to_sentence.pkl', 'rb') as o:
    cases_text = pickle.load(o)
with open('/sternadi/home/volume3/chronic-corona-pred/data/case_control/controls/epi_to_sentence.pkl', 'rb') as o:
    controls_text = pickle.load(o)

case = pd.read_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/data/cases.csv')
control = pd.read_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/data/controls.csv')

for clade in clades:
    for c in clades[clade]:
        clades[clade][c] = set(clades[clade][c])

case['text'] = case['epi'].apply(lambda x: cases_text[x] if x in cases_text else 'no-sentence')
control['text'] = control['epi'].apply(lambda x: controls_text[x] if x in controls_text else 'no-sentence')

case[case['text'] == 'no-sentence'].to_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/data/cases_no_sentences.csv', index=False)
control[control['text'] == 'no-sentence'].to_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/data/controls_no_sentences.csv', index=False)

case = case[case['text'] != 'no-sentence']
control = control[control['text'] != 'no-sentence']

case['size'] = case.groupby("id")['size'].transform("count")
control['size'] = control.groupby("id")['size'].transform("count")

# filter out candidates with small size and time interval < 21 days
case = case[case['size'] > 2]
control = control[control['size'] > 2]

case['date'] = case['epi'].apply(lambda x: mapper[x]['date'])
dates_mapper = case.groupby('id')['date'].apply(list).to_dict()
case['dates'] = case['id'].apply(lambda x: dates_mapper[x])
case['time'] = case['dates'].apply(lambda x: (max(x) - min(x)).days)

control['date'] = control['epi'].apply(lambda x: mapper[x]['date'])
dates_mapper = control.groupby('id')['date'].apply(list).to_dict()
control['dates'] = control['id'].apply(lambda x: dates_mapper[x])
control['time'] = control['dates'].apply(lambda x: (max(x) - min(x)).days)

case = case[case['time'] >= 21]
control = control[control['time'] >= 21]

case['text'] = case.apply(lambda row: remove_ld(row['text'], row['clade'], clades), axis=1)
control['text'] = control.apply(lambda row: remove_ld(row['text'], row['clade'], clades), axis=1)

case.to_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/data/cases_with_text.csv', index=False)
control.to_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/data/controls_with_text.csv', index=False)

case_clades = case.groupby(['id', 'clade', 'size', 'time', 'label']).agg({'text':list}).reset_index()
case_clades['text'] = case_clades['text'].apply(lambda l:set(flatten([e.split() for e in l])))
case_clades['text'] = case_clades['text'].apply(lambda l:' '.join(sorted(l, key=functools.cmp_to_key(comp))))

control_clades = control.groupby(['id', 'clade','size', 'time', 'label']).agg({'text':list}).reset_index()
control_clades['text'] = control_clades['text'].apply(lambda l:set(flatten([e.split() for e in l])))
control_clades['text'] = control_clades['text'].apply(lambda l:' '.join(sorted(l, key=functools.cmp_to_key(comp))))

case_clades.to_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/data/cases_clades_with_text.csv', index=False)
control_clades.to_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/data/controls_clades_with_text.csv', index=False)