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
    df['date'] = df['seqid'].apply(lambda x: mapper[x]['Collection date'])
    df['date'] = df['date'].apply(lambda x: pd.to_datetime(x) if len(x.split('-')) == 3 else '')
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

    parser.add_argument('--files', type=str, required=True, help="nextclade input files")
    parser.add_argument('--metadata-mapper', type=str, required=True, help="metadata pickle mapper")
    parser.add_argument('--variant', type=str, required=True, help="Nextclade variant")
    parser.add_argument('--out', type=str, default='./', help="output folder path")

    args = parser.parse_args()

    files = glob.glob(f'{args.files}/*tsv')
    with open(args.metadata_mapper, 'rb') as o:
        mapper = pickle.load(o)
    v = variants[args.variant]

    res = []
    for f in files:
        d = pd.read_table(f)
        d = filter_data(d, mapper, cols2keep)
        if ',' in v:
            mask = np.any([d.variant==y for y in v.split(',')], axis=0)
        else:
            mask = d.variant==v

        res.append(d[mask])
    variant_df = pd.concat(res)
    variant_df.to_csv(os.path.join(args.out, f'{args.variant}.tsv'), index=False)



