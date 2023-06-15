
import pandas as pd
import numpy as np

from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline


class CustomTokenizer(BertTokenizer):
    def _tokenize(self, text):
        # replace : and - used by non-syn mutations and
        text = text.replace(':', 'X').replace('-','X')
        # tokenize by white space
        tokens = text.split()
        return tokens

folds = {'alpha':90, 'delta':120, 'omicron':150}
mapper = {'omicron':['21K', '21L', '22A', '22B', '22C'], 'alpha':['20I'], 'delta':['21J', '21I'],
          'pre_voc':['19A', '19B', '20A','20B', '20C', '20E', '20G', '20D'], 'rest':['20H', '20J', '21F', '20H', '21B']}

fold = 'omicron'
model = BertForSequenceClassification.from_pretrained(f'/sternadi/home/volume3/chronic-corona-pred/'
                                                      f'sars_cov_2_mlm/classifier/models/{fold}/checkpoint-{folds[fold]}/')

tokenizer = CustomTokenizer.from_pretrained('/sternadi/home/volume3/chronic-corona-pred/'
                                            'sars_cov_2_mlm/models/pretrained-covBERTa/', do_lower_case=False)

data = pd.read_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/'
                   'data/unknowns_clades_with_text.csv')

data = data[data['clade'].isin(mapper[fold])]
sentences = data['text'].tolist()

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None)
res = pipe(sentences[0])
data['pipe'] = data['text'].apply(lambda x: pipe(x))
data['pipe'] = data['pipe'].apply(lambda x: dict(pd.DataFrame(x[0]).values))
data['case_prob'] = data['pipe'].apply(lambda x: x['LABEL_1'])
data['control_prob'] = data['pipe'].apply(lambda x: x['LABEL_0'])
data = data.drop(columns={'pipe'})
data.to_csv(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/models/{fold}/unknowns/predictions.csv', index=False)