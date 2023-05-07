import json
import pickle
import functools

import pandas as pd
import numpy as np

from extract_mutations_by_epi import GENEMAP, comp, flatten

# helper function
def remove_ld(text, clade, mapper):
    mts = []
    for mt in text.split():
        # if we have a deletion
        if '-' in mt or mt.isdigit():
            s, e = int(mt.split('-')[0]), int(mt.split('-')[-1])
            r = {x for x in range(s, e+1)}
            if len(r.intersection(mapper[clade]['del'])) == 0:
                mts.append(mt)
        elif mt not in mapper[clade]['nuc'] and mt not in mapper[clade]['aa']:
            mts.append(mt)
    return ' '.join(mts)


# load data
with open('/sternadi/home/volume3/chronic-corona-pred/data/clades/lineage_def_clades_extended.json', 'r') as o:
    clades = json.load(o)
with open('/sternadi/home/volume3/chronic-corona-pred/data/case_control/cases/epi_to_sentence.pkl', 'rb') as o:
    cases_text = pickle.load(o)
with open('/sternadi/home/volume3/chronic-corona-pred/data/case_control/controls/epi_to_sentence.pkl', 'rb') as o:
    controls_text = pickle.load(o)

case = pd.read_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/data/cases.csv')
control = pd.read_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/data/controls.csv')

for clade in clades:
    for c in clades[clade]:
        clades[clade][c] = set(clades[clade][c])

case['text'] = case['epi'].apply(lambda x: cases_text[x])
control['text'] = control['epi'].apply(lambda x: controls_text[x])

case['text'] = case.apply(lambda row: remove_ld(row['text'], row['clade'], clades), axis=1)
control['text'] = control.apply(lambda row: remove_ld(row['text'], row['clade'], clades), axis=1)

case.to_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/data/cases_with_text.csv', index=False)
control.to_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/data/controls_with_text.csv', index=False)

case_clades = case.groupby(['id', 'clade', 'size', 'label']).agg({'text':list}).reset_index()
case_clades['text'] = case_clades['text'].apply(lambda l:set(flatten([e.split() for e in l])))
case_clades['text'] = case_clades['text'].apply(lambda l:' '.join(sorted(l, key=functools.cmp_to_key(comp))))

control_clades = control.groupby(['id', 'clade','size', 'label']).agg({'text':list}).reset_index()
control_clades['text'] = control_clades['text'].apply(lambda l:set(flatten([e.split() for e in l])))
control_clades['text'] = control_clades['text'].apply(lambda l:' '.join(sorted(l, key=functools.cmp_to_key(comp))))

case_clades.to_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/data/cases_clades_with_text.csv', index=False)
control_clades.to_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/data/controls_clades_with_text.csv', index=False)