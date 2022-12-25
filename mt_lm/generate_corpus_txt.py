import re
import glob
import functools

import pandas as pd
import numpy as np

syn_tokenizer = '/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/tokenizer/syn.npy'
GENEMAP = {'ORF1a':(266,13468), 'ORF3a':(25393,26220), 'ORF9b':(28284,28577),'ORF1b':(13468,21555),
           'N':(28274,29533), 'S':(21563,25384), 'E':(26245,26472), 'ORF8':(27894,28259), 'M':(26523,27191),
           'ORF7a':(27394,27759), 'ORF7b':(27756,27887), 'ORF6':(27202,27387)}


clades = ['19A','19B','20A','20B','20C','20E','20H','20I','20J','21D','21G','21H','21I',
            '21J','21K','21L','22A','22B','22D']

def flatten(l):
    return [item for sublist in l for item in sublist]

def comp(x,y):
    if x.split(':')[0] in GENEMAP:
        x_n = (int(re.findall(r'\d+', x.split(':')[1])[0])-1)*3 + GENEMAP[x.split(':')[0]][0]
    else:
        x_n = int(re.findall(r'\d+', x)[0])
    if y.split(':')[0] in GENEMAP:
        y_n = (int(re.findall(r'\d+', y.split(':')[1])[0])-1)*3 + GENEMAP[y.split(':')[0]][0]
    else:
        y_n = int(re.findall(r'\d+', y)[0])
    return x_n - y_n



syn = np.load(syn_tokenizer)
for clade in clades:
    df = pd.concat([pd.read_pickle(f) for f in
                    glob.glob(f'/sternadi/home/volume3/chronic-corona-pred/nextclade/subsets/{clade}/*.pkl')])
    if df.shape[0] == 0:
        continue
    df['syn_nuc'] = df['nuc'].apply(lambda x: list(set(x).intersection(syn)))
    df['insertions'] = df['insertions'].apply(lambda x: x.split(',') if x != '' else [])
    df['deletions'] = df['deletions'].apply(lambda x: x.split(',') if x != '' else [])
    # clean possible empty columns (e.g., no aa mutations)
    df['mt'] = df.apply(
        lambda row: [e for e in row['aa'] + row['syn_nuc'] + row['deletions'] + row['insertions'] if e != ''],
        axis=1)

    # tokenize sentences by sorting the mutation according to the appearance order on the genome
    # this includes both coding\non coding, indels and syn\non-syn mutations
    sentences = df['mt'].apply(lambda l: sorted(l, key=functools.cmp_to_key(comp))).values
    sentences = '. '.join([' '.join(s) for s in sentences])

    with open(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/data/{clade}_part.txt', 'w') as fp:
        fp.write(sentences)


