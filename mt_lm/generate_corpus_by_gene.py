import re
import glob


GENEMAP = {'ORF1a':(266,13468), 'ORF3a':(25393,26220), 'ORF9b':(28284,28577),'ORF1b':(13468,21555),
           'N':(28274,29533), 'S':(21563,25384), 'E':(26245,26472), 'ORF8':(27894,28259), 'M':(26523,27191),
           'ORF7a':(27394,27759), 'ORF7b':(27756,27887), 'ORF6':(27202,27387)}

# written by chatGPT.
def divide_list(strings_list):
    result = []
    current_group = []
    current_prefix = 'ORF1a'
    for string in strings_list:
        if not any([k for k in GENEMAP if k+':' in string]):
            try:
                num = int(re.findall(r'\d+', string)[0])
                prefix = [k for k,v in GENEMAP.items() if num in range(v[0], v[1]+1)][0]
                if prefix == []:
                    continue
                string = prefix+':'+string
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
            result.append(current_group)
            current_group = [string]
            current_prefix = prefix
    if current_group:
        result.append(current_group)
    return result

genomes = []
files = glob.glob('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/data/*.txt')
for f in files:
    with open(f, 'r') as o:
        for line in o:
            genomes.extend(divide_list(line.replace('\n', '').split()))


mapper = {k:[] for k in GENEMAP}
for s in genomes:
    if len(s) == 0:
        continue
    key = s[0].split(':')[0]
    mapper[key].append(s)

for gene in mapper:
    sentences = '\n'.join([' '.join(s) for s in mapper[gene]])
    with open(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/gene_corpus/{gene}.txt', 'w') as o:
        o.write(sentences)




