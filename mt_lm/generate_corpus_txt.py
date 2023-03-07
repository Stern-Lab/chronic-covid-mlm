import re
import glob
import functools

import pandas as pd
import numpy as np

import pickle

syn_tokenizer = '/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/tokenizer/syn.npy'
syn = np.load(syn_tokenizer)

GENEMAP = {'ORF1a':(266,13468), 'ORF3a':(25393,26220), 'ORF9b':(28284,28577),'ORF1b':(13468,21555),
           'N':(28274,29533), 'S':(21563,25384), 'E':(26245,26472), 'ORF8':(27894,28259), 'M':(26523,27191),
           'ORF7a':(27394,27759), 'ORF7b':(27756,27887), 'ORF6':(27202,27387)}

clades = ['19A','19B','20A','20B','20C','20E','20H','20I','20J','21D','21G','21H','21I',
            '21J','21K','21L','22A','22B','22D']

with open('/sternadi/home/volume3/chronic-corona-pred/data/GISAID/meta_mapper.pkl', 'rb') as o:
    mapper = pickle.load(o)

with open('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/few_shot/data/exclude_epis.pkl', 'rb') as fp:
    exclude_epis = pickle.load(fp)
exclude_seqid = [mapper[epi]['Virus name'] for epi in exclude_epis]

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

def divide_list(strings_list):
    result = {'ORF1a':[], 'ORF1b':[], 'S':[], 'OTHER':[]}
    current_group = []
    current_prefix = 'ORF1a'
    for string in strings_list:
        if not any([k for k in GENEMAP if k+':' in string]):
            try:
                num = int(re.findall(r'\d+', string)[0])
                prefix = [k for k,v in GENEMAP.items() if num in range(v[0], v[1]+1)][0]
            except:
                continue
        else:
            prefix = string.split(":")[0]
        if not current_prefix:
            current_prefix = prefix
            current_group.append(string)
        elif prefix == current_prefix:
            current_group.append(string)
        else:
            if current_prefix not in result:
                result['OTHER'].extend(current_group)
            else:
                result[current_prefix].extend(current_group)
            current_group = [string]
            current_prefix = prefix
    if current_group:
        if current_prefix not in result:
            result['OTHER'].extend(current_group)
        else:
            result[current_prefix].extend(current_group)
    return result



gene_corpus = {'ORF1a':[], 'ORF1b':[], 'S':[], 'OTHER':[]}

for clade in clades:
    df = pd.concat([pd.read_pickle(f) for f in
                    glob.glob(f'/sternadi/home/volume3/chronic-corona-pred/nextclade/subsets/{clade}/*.pkl')])

    if df.shape[0] == 0:
        continue

    # remove epis from validation and test
    df = df[~df['seqid'].isin(exclude_seqid)]

    # extract mutations from all types
    df['syn_nuc'] = df['nuc'].apply(lambda x: list(set(x).intersection(syn)))
    df['insertions'] = df['insertions'].apply(lambda x: x.split(',') if x != '' else [])
    df['deletions'] = df['deletions'].apply(lambda x: x.split(',') if x != '' else [])

    # clean possible empty columns (e.g., no aa mutations)
    df['mt'] = df.apply(
        lambda row: [e for e in row['aa'] + row['syn_nuc'] + row['deletions'] + row['insertions'] if e != ''],
        axis=1)
    df['mt_by_gene'] = df['mt'].apply(lambda l: divide_list(l))

    # tokenize sentences by sorting the mutation according to the appearance order on the genome
    for k in gene_corpus:
        sentences = [sorted(x[k], key=functools.cmp_to_key(comp)) for x in df['mt_by_gene'].values if x[k]!= []]
        sentences = '\n'.join([' '.join(s) for s in sentences]) + '\n'  # add new line at the end for concatenation
        gene_corpus[k] += sentences

        with open(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/gene_corpus/{k}_c_{clade}.txt', 'w') as fp:
            fp.write(sentences)


# save results to txt files
for k in gene_corpus:
    with open(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/gene_corpus/{k}.txt', 'w') as fp:
        fp.write(gene_corpus[k][0])


