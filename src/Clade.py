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
    def __init__(self, clade_path, ref_path, clade, min_seqs=6, out='./'):
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

        data['num_deletions'] = data['deletions'].apply(lambda x: len(x.split(',')) if ',' in x else 0)
        data['num_insertions'] = data['insertions'].apply(lambda x: len(x.split(',')) if ',' in x else 0)

        data['datenum'] = data['date'].apply(lambda x: self.toYearFraction(x))

        self.data = data

    def fit_OLS(self, by=('num_aa_from_ref')):
        data = self.data
        x = data['datenum']
        y = data[list(by)].sum(axis=1)

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
        grouped = extremes.groupby(['location', 'age', 'gender']).agg({'seqid': list}).reset_index()
        grouped['size'] = grouped['seqid'].map(len)

        grouped = grouped[(grouped['age'] != 'Unknown') & (grouped['gender'] != 'Unknown')]\
            .sort_values(by='size', ascending=False).reset_index(drop=True)

        candidates = grouped[grouped['size'] > np.quantile(grouped['size'], 0.95)] # filter by 95Q
        candidates['reg_data'] = candidates['seqid'].apply(lambda x: extremes[extremes['seqid'].isin(x)])

        self.candidates = candidates











