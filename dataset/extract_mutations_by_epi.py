import pickle
import functools
import re
import pandas as pd
import numpy as np
import argparse

GENEMAP = {'ORF1a':(266,13468), 'ORF3a':(25393,26220), 'ORF9b':(28284,28577),'ORF1b':(13468,21555),
           'N':(28274,29533), 'S':(21563,25384), 'E':(26245,26472), 'ORF8':(27894,28259), 'M':(26523,27191),
           'ORF7a':(27394,27759), 'ORF7b':(27756,27887), 'ORF6':(27202,27387)}

masked = pd.read_csv('/sternadi/home/volume3/chronic-corona-pred/data/pos2mask.csv')
masked = set(masked['POS'])

def mask_aa_by_codon(aa, nuc):
    if aa == ['']:
        return []
    nuc_pos = set([int(m[1:-1]) for m in nuc])
    res = []
    for mt in aa:
        start = GENEMAP[mt.split(':')[0]][0]
        pos = int(mt.split(':')[1][1:-1])
        codon = {start + pos * 3 - 1, start + pos * 3 - 2, start + pos * 3 - 3}.intersection(nuc_pos)
        if len(codon.intersection(masked)) == 0:
            res.append(mt)
    return res

def mask_del_by_pos(dels):
    if dels == []:
        return []
    res = []
    for r in dels:
        if "-" in r:
            start, end = r.split("-")
            l = set((range(int(start), int(end)+1)))
        else:
            l = {int(r)}
        if len(l.intersection(masked)) == 0:
            res.append(r)
    return res

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


def main(args):

    dtype = args.dtype

    syn = np.load('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/tokenizer/syn.npy')

    with open('/sternadi/home/volume3/chronic-corona-pred/data/GISAID/mappers/meta_mapper.pkl', 'rb') as o:
        mapper = pickle.load(o)

    name_to_epi = {mapper[epi]['Virus name']: epi for epi in mapper}

    df = pd.read_table(f'/sternadi/home/volume3/chronic-corona-pred/data/case_control/{dtype}/{dtype}.tsv')
    df = df[df['qc.overallStatus'] == 'good'].fillna('')

    df['nuc'] = df['substitutions'].apply(lambda x: x.split(','))
    df['aa'] = df['aaSubstitutions'].apply(lambda x: x.split(','))

    # remove pre-defined masked positions
    df['aa'] = df.apply(lambda row: mask_aa_by_codon(row['aa'], row['nuc']), axis=1)
    df['nuc'] = df['nuc'].apply(lambda l: [x for x in l if int(x[1:-1]) not in masked])

    df['seqid'] = df['seqName'].apply(lambda x: x.split('|')[0])
    df = df[['seqName','seqid', 'nuc','aa', 'insertions','deletions']]

    df['epi'] = df['seqid'].apply(lambda x: name_to_epi[x])
    df['syn_nuc'] = df['nuc'].apply(lambda x: list(set(x).intersection(syn)))

    # remove masked positions from deletions, insertions are ignored here.
    df['insertions'] = df['insertions'].apply(lambda x: x.split(',') if x != '' else [])
    df['deletions'] = df['deletions'].apply(lambda x: x.split(',') if x != '' else [])
    df['deletions'] = df['deletions'].apply(lambda x: mask_del_by_pos(x))

    # clean possible empty columns (e.g., no aa mutations)
    df['mt'] = df.apply(
        lambda row: [e for e in row['aa'] + row['syn_nuc'] + row['deletions'] + row['insertions'] if e != ''],
        axis=1)

    df['sentence'] = df['mt'].apply(lambda lst: ' '.join(sorted(lst, key=functools.cmp_to_key(comp))))
    epi_2_sentence = dict(df[['epi', 'sentence']].values)
    with open(f'/sternadi/home/volume3/chronic-corona-pred/data/case_control/{dtype}/epi_to_sentence.pkl', 'wb') as o:
        pickle.dump(epi_2_sentence, o)




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training SARS-CoV-2 mutational classifier')

    parser.add_argument('--dtype', default='cases',
                        help='alias for data type case or control')

    args = parser.parse_args()

    main(args)



