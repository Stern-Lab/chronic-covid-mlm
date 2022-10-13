import os

import pandas as pd
import numpy as np
import argparse
import glob
import pickle

variants = {'19A':'19A',
            '19B':'19B',
            '19B+':'19A,19B',
            '19B++':'19A,19B,20A,20B,20C,20D',
            '20A':'20A',
            '20B':'20B',
            '20C':'20C',
            '20A+':'20A,20B,20C,20D',
            '20E':'20E',
            '20H':'20H',
            '20I':'20I',
            '20J':'20J',
            '21D':'21D',
            '21G':'21G',
            '21H':'21H',
            '21I':'21I',
            '21J':'21J',
            '21K':'21K',
            '21L':'21L',
            '22A':'22A',
            '22B':'22B',
            '22D':'22D',
            '21L+':'21L,22C,22D'
}

cols2keep = ['seqName', 'variant', 'nuc','aa', 'insertions','deletions', 'frameShifts']

def filter_data(df, mapper, cols2keep):
    # filter by nextclade QC
    df = df[df['qc.overallStatus'] == 'good'].fillna('')

    df['variant'] = df['clade'].apply(lambda x: x.split(' (')[0])
    df['nuc'] = df['substitutions'].apply(lambda x: x.split(','))
    df['aa'] = df['aaSubstitutions'].apply(lambda x: x.split(','))
    df = df[cols2keep]

    # add metadata information
    df['seqid'] = df['seqName'].apply(lambda x: x.split('|')[0])
    df = df[df['seqid'].isin(mapper)]
    df['date'] = df['seqid'].apply(lambda x: mapper[x]['Collection date'])
    df['date'] = df['date'].apply(lambda x: pd.to_datetime(x) if len(x.split('-')) == 3 else '')

    df = df[df['date'] != '']
    df['submission_date'] = df['seqid'].apply(lambda x: pd.to_datetime(mapper[x]['Submission date']))

    # filter out inappropriate dates
    df = df[(df['date'] != '') & (df['date'] <= df['submission_date'])]

    df['location'] = df['seqid'].apply(lambda x: mapper[x]['Location'])
    df['age'] = df['seqid'].apply(lambda x: mapper[x]['Patient age'])
    df['gender'] = df['seqid'].apply(lambda x: mapper[x]['Gender'])

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parse nextclade output data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--file', type=str, required=True, help="nextclade input file")
    parser.add_argument('--metadata-mapper', type=str, required=True, help="metadata pickle mapper")
    parser.add_argument('--out', type=str, default='./', help="output folder path")

    args = parser.parse_args()

    with open(args.metadata_mapper, 'rb') as o:
        mapper = pickle.load(o)
   
    d = pd.read_table(args.file)
    d = filter_data(d, mapper, cols2keep)
    for v in variants:
        if ',' in v:
            mask = np.any([d.variant==y for y in v.split(',')], axis=0)
        else:
            mask = d.variant==v

        variant_df = d[mask]
        out_dir = os.path.join(args.out, v)
        variant_df.to_pickle(os.path.join(out_dir, f'{os.path.basename(args.file).split(".")[0]}_{v}.pkl'))



