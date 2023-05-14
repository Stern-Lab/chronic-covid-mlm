import pickle
import pandas as pd


def node_to_clade(node_data, mapper):
    epis = node_data['epi']
    clades = list(set([mapper[x]for x in epis  if x in mapper]))
    return clades[0]

with open('/sternadi/home/volume3/chronic-corona-pred/data/case_control/cases/epi_2_clade.pkl', 'rb') as o:
    case_epi_2_clade = pickle.load(o)
with open('/sternadi/home/volume3/chronic-corona-pred/data/case_control/controls/epi_2_clade.pkl', 'rb') as o:
    control_epi_2_clade = pickle.load(o)

with open('/sternadi/home/volume3/chronic-corona-pred/data/case_control/cases/known_branch_ids.txt', 'r') as o:
    case_ids = [x.strip() for x in o.readlines()]
with open('/sternadi/home/volume3/chronic-corona-pred/data/case_control/controls/known_branch_ids.txt', 'r') as o:
    control_ids = [x.strip() for x in o.readlines()]


# filter out unknown candidates + intersecting nodes in the tree
case = pd.read_csv("/sternadi/home/volume3/chronic-corona-pred/data/case_control/initial_cases.csv")
case = case[(case['isKnown'] == 'known-known') & (case['branch_id'].isin(case_ids))]

control = pd.read_csv("/sternadi/home/volume3/chronic-corona-pred/data/case_control/initial_controls.csv")
control = control[(~control['sex'].str.contains('unknown')) & (~control['age'].str.contains('unknown'))]
control = control[control['branch_id'].isin(control_ids)]

groups = {'control':{'label':0, 'data':control, 'mapper':control_epi_2_clade},
          'case':{'label':1, 'data':case, 'mapper':case_epi_2_clade}}
for grp in groups:
    data =groups[grp]['data']
    data['epi'] = data['accession_ids'].apply(lambda x: x.split(';'))
    data = data.explode('epi')
    data = data[['epi','branch_id', 'num_samples_in_major_group']].rename(columns={'branch_id':'id', 'num_samples_in_major_group':'size'})
    data = data.drop_duplicates()
    data['id'] = data['id'].apply(lambda x: x.split(':')[0])
    data['label'] = groups[grp]['label']

    # add clade information - this might take a while for controls
    res = dict()
    for node in data['id'].unique():
        cur = data[data['id'] == node]
        clade = node_to_clade(cur, groups[grp]['mapper'])
        res[node] = clade

    data['clade'] = data['id'].apply(lambda x: res[x])
    data.to_csv(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/data/{grp}.csv', index=False)

