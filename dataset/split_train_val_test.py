import itertools
import os
import pickle

import numpy as np
import pandas as pd

# RANDOM STATE 42

class Test:
    def __init__(self, candidates, path, type='unknown-unknown'):
        self.data = pd.read_csv(candidates)
        self.type = type
        self.p = path

    def parse(self):
        df = self.data
        df = df[df['isKnown'] == self.type][['branch_id', 'accession_ids']]
        df['accession_ids'] = df['accession_ids'].apply(lambda x: x.split(';'))
        df = df.explode('accession_ids')
        df.reset_index(inplace=True)

        return [tuple(x) for x in df[['accession_ids', 'index']].values]

    def save(self, obj):
        p = os.path.join(self.p, 'test.pkl')
        with open(p, 'wb') as fp:
            pickle.dump(obj, fp)



class Pairs:
    def __init__(self, label, age, branch, pool):
        self.label = label
        self.age = age
        self.branch = branch
        self.pool = pool

        self.samples = []


    def sample_by_age(self):
        age = self.age
        pool = self.pool

        # different age +- 5 years apart
        try:
            x = pool[(pool['age'] > age + 5) | (pool['age'] < age - 5)].sample(n=1, random_state=42)['epi'].values[0]
            self.samples.append(x)
        except:
            pass
        # similar age
        try:
            y = pool[(pool['age'] < age + 2) | (pool['age'] > age - 2)].sample(n=1, random_state=42)['epi'].values[0]
            self.samples.append(y)
        except:
            pass

    def sample_by_sex_and_clade(self):
        pool = self.pool
        grp = pool.groupby(['clade', 'sex']).sample(n=1, random_state=42)['epi'].values
        self.samples.extend(grp)

    def sample_by_branch(self):
        branch = self.branch
        pool = self.pool

        # closest & random branch
        x = pool.iloc[(pool['branch_id'] - branch).abs().argsort()[0], :]['epi']
        y = pool[pool['branch_id'] > branch].sample(n=1, random_state=42)['epi'].values[0]

        self.samples.extend([x,y])

    def sampler(self):
        self.sample_by_age()
        self.sample_by_branch()
        self.sample_by_sex_and_clade()


class Triplets:
    def __init__(self, case, control, path):
        case = pd.read_csv(case)
        control = pd.read_csv(control)

        self.case = case
        self.control = control
        self.p = path

        self.visited = set()


    def update_visited(self, epi):
        control = self.control
        labels = control[control['epi'].isin(epi)]['label'].values
        self.visited = self.visited.union(set(labels))


    def make_triplets(self):
        case = self.case
        control = self.control
        pool = control

        epi2label = dict(pool[['epi', 'label']].values)

        pos_anchors = []
        for label in case['label'].unique():
            cur = case[case['label'] == label]
            pairs = list(itertools.combinations(cur['epi'], 2))
            pool = pool[~pool['label'].isin(self.visited)].reset_index(drop=True)

            p = Pairs(label=label, age=cur['age'].values[0], branch=cur['branch_id'].values[0], pool=pool)
            p.sampler()


            # append new candidates to
            pos_anchors.extend([((p1, p2, n1), (label, epi2label[n1]))  for p1, p2 in pairs for n1 in p.samples])

            # update visited control clades
            self.update_visited(p.samples)

        return pos_anchors

    def extract_epis(self, triplets):
        case = self.case
        control = self.control

        # ((epis), (label_pos, label_neg))
        case_labels = [x[1][0] for x in triplets]
        control_labels = [x[1][1] for x in triplets]

        epis = case[case['label'].isin(case_labels)]['epi'].tolist() + \
               control[control['label'].isin(control_labels)]['epi'].tolist()
        return epis


# make test data
test = Test(candidates='/sternadi/home/volume3/chronic-corona-pred/data/monophyletic_candidates.csv',
            path='/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/few_shot/data/')
res = test.parse()
test.save(res)

# create triplets for few-shot learning
tp = Triplets(case='/sternadi/home/volume3/chronic-corona-pred/data/case_control/cases_metadata_for_triplets.csv',
         control='/sternadi/home/volume3/chronic-corona-pred/data/case_control/control_metadata_for_triplets.csv',
         path='./')

triplets = tp.make_triplets()
with open('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/few_shot/data/triplets.pkl', 'wb') as fp:
    pickle.dump(triplets, fp)

# extract all epi data to exclude from MLM
few_shot_epis = tp.extract_epis(triplets)
exclude_mlm =  few_shot_epis + [x[0] for x in res]

with open('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/few_shot/data/exclude_epis.pkl', 'wb') as fp:
    pickle.dump(exclude_mlm, fp)














