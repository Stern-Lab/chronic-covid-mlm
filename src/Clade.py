import glob
import os
import time
import json
import re

import pandas as pd
import numpy as np
import statsmodels.api as sm

from datetime import datetime as dt

class Clade:
    def __init__(self, clade_path, ref_path, clade, min_seqs=2, out='./'):
        self.clade = clade
        self.min_seqs = min_seqs
        self.out = out
        data = pd.concat([pd.read_pickle(f) for f in glob.glob(f'{clade_path}/*.pkl')])
        self.data = data
        self.n = data.shape[0]

        with open(ref_path) as json_handle:
            refs = json.load(json_handle)

        self.ref = refs[clade]

        self.extremes =  None
        self.candidates = None

    @staticmethod
    def toYearFraction(date):
        def sinceEpoch(date):  # returns seconds since epoch
            return time.mktime(date.timetuple())

        s = sinceEpoch

        year = date.year
        startOfThisYear = dt(year=year, month=1, day=1)
        startOfNextYear = dt(year=year + 1, month=1, day=1)

        yearElapsed = s(date) - s(startOfThisYear)
        yearDuration = s(startOfNextYear) - s(startOfThisYear)
        fraction = yearElapsed / yearDuration

        return date.year + fraction

    @staticmethod
    def get_gender(row):
        gender = row['gender'].lower()
        age = row['age'].lower()

        if 'female' in gender or 'feme' in gender or 'muj' in gender or 'f' == gender:
            return 'Female'
        elif 'female' in age or 'feme' in age or 'muj' in age or 'f' == age:
            return 'Female'
        elif 'male' in gender or 'homb' in gender or 'm' == gender:
            return 'Male'
        elif 'male' in age or 'homb' in age or 'm' == age:
            return 'Male'
        else:
            return "Unknown"

    @staticmethod
    def get_age(row):
        gender = row['gender'].lower()
        age = row['age'].lower()

        ints_age = re.findall(r'\d+', age)
        ints_gender = re.findall(r'\d+', gender)

        if len(ints_age) == 0 and len(ints_gender) == 0:
            return 'Unknown'
        elif len(ints_age) == 0 and len(ints_gender) > 0:
            return gender
        else:
            return age

    def calc_mutations_from_ref(self):
        data = self.data
        ref = self.ref

        data['nuc_from_ref'] = data['nuc'].apply(lambda x: set(x) - set(ref['nuc']))
        data['aa_from_ref'] = data['aa'].apply(lambda x: set(x) - set(ref['aa']))

        data['num_nuc_from_ref'] = data['nuc_from_ref'].map(len)
        data['num_aa_from_ref'] = data['aa_from_ref'].map(len)
        data['num_syn_from_ref'] = data['num_nuc_from_ref'] - data['num_aa_from_ref']
        data['num_spike_mutations_from_ref'] = data['aa_from_ref'].apply(lambda x: len([mt for mt in x if 'S:' in mt]))

        data['num_deletions'] = data['deletions'].apply(lambda x: len(x.split(',')) if ',' in x else 0)
        data['num_insertions'] = data['insertions'].apply(lambda x: len(x.split(',')) if ',' in x else 0)

        data['datenum'] = data['date'].apply(lambda x: self.toYearFraction(x))

        self.data = data

    def fit_OLS(self, by=['num_aa_from_ref']):
        data = self.data
        x = data['datenum']
        y = data[by].sum(axis=1)

        model = sm.OLS(y, sm.add_constant(x))
        results = model.fit()

        data['resid'] = results.resid
        self.data = data

    def filter_extremes(self):
        data = self.data
        assert 'resid' in data.columns

        # take the sequences with extreme number of mutations
        mean = np.mean(data['resid'], axis=0)
        sd = np.std(data['resid'], axis=0)
        extremes = data[data['resid'] > mean + 2 * sd].fillna('').reset_index(drop=True)
        extremes['gender'] = extremes.apply(lambda x: self.get_gender(x), axis=1)
        extremes['age'] = extremes.apply(lambda x: self.get_age(x), axis=1)

        self.extremes = extremes

    def save_extremes(self):
        self.extremes.to_pickle(os.path.join(self.out, 'extremes.pkl'))

    def set_known_candidates(self):
        extremes = self.extremes
        data = self.data

        # group by known attributes and filter sequences with less than min_seqs with extreme mutations
        grouped = extremes.groupby(['location', 'age', 'gender']).agg({'seqid': list}).reset_index()
        grouped['size'] = grouped['seqid'].map(len)
        grouped = grouped[(grouped['age'] != 'Unknown') & (grouped['gender'] != 'Unknown')]\
            .sort_values(by='size', ascending=False).reset_index(drop=True)
        grouped = grouped[grouped['size'] >= self.min_seqs]
        grouped['candidate_id'] = grouped.index

        # get the original data to add sequences with the same known attrs, but with smaller number of mutations
        data = data[data['location'].isin(grouped['location'].unique())].fillna('') # reduce number of
                                                                                    # samples before manipulations
        data['age'] = data.apply(lambda x: self.get_age(x), axis=1)
        data['gender'] = data.apply(lambda x: self.get_gender(x), axis=1)

        # filter data by ID
        data['filter_id'] = data.apply(lambda row: f'{row["location"]};{row["age"]};{row["gender"]}', axis=1)
        grouped['filter_id'] = grouped.apply(lambda row: f'{row["location"]};{row["age"]};{row["gender"]}', axis=1)

        # get full candidates
        mapper = dict(grouped[['filter_id', 'candidate_id']].values)
        candidates = data[data['filter_id'].isin(grouped['filter_id'])]
        candidates['candidate_id'] = candidates['filter_id'].apply(lambda x: mapper[x])

        self.candidates = candidates











