import argparse
import pandas as pd

import torch.nn.functional as F
from tqdm import tqdm

from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


class CustomTokenizer(BertTokenizer):
    def _tokenize(self, text):
        # replace : and - used by non-syn mutations and
        text = text.replace(':', 'X').replace('-', 'X')
        # tokenize by white space
        tokens = text.split()
        return tokens


def get_fold_data(tree, rate, df):
    fold_data = df.merge(tree, on=['id', 'label'], how='left')
    fold_data = fold_data.merge(rate, on=['id', 'label'], how='left')
    return fold_data

def stratify_by_clade_and_size(df, dist):
    samples =[]
    for k, v in dist.items():
        cur = df[(df['size'] >= k[1] - 2) & (df['size'] <= k[1] + 2)]
        if cur.shape[0] == 0:
            continue
        cur = cur[cur['clade'] == k[0]]
        if cur.shape[0] == 0:
            continue
        samples.extend(cur.sample(n=v, replace=True)['id'].values)
    return samples

def load_controls(fold):
    control = pd.read_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/data/controls_clades_with_text.csv')
    return control[control['clade'].isin(clades[fold])]

def load_tree_and_rates():
    case_tree = pd.read_pickle('/sternadi/home/volume3/chronic-corona-pred/data/case_control/tree_messures_for_casess.pkl')
    control_tree = pd.read_pickle('/sternadi/home/volume3/chronic-corona-pred/data/case_control/tree_messures_for_controls.pkl')
    case_tree['label'] = 1
    control_tree['label'] = 0
    tree = pd.concat([case_tree[['name', 'sackin_index', 'label']],
                      control_tree[['name', 'sackin_index', 'label']]]).rename(columns={'name': 'id'})

    case_rates = pd.read_csv('/sternadi/home/volume3/chronic-corona-pred/data/case_control/rates/no_sp/'
                             'cases_group_data_overall_rates.csv')
    control_rates = pd.read_csv('/sternadi/home/volume3/chronic-corona-pred/data/case_control/rates/no_sp/'
                                'controls_group_data_overall_rates.csv')
    case_rates['label'] = 1
    control_rates['label'] = 0
    rate = pd.concat([case_rates[['id', 'S', 'label']], control_rates[['id', 'S', 'label']]])
    return tree, rate



MAPPER = {'alpha':90, 'delta':120,'omicron':150}
clades = {'omicron':['21K', '21L', '22A', '22B', '22C'], 'alpha':['20I'], 'delta':['21J', '21I'],
          'pre_voc':['19A', '19B', '20A','20B', '20C', '20E', '20G', '20D'], 'rest':['20H', '20J', '21F', '20H', '21B']}


def main(args):
    def predictor(texts):
        outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
        probas = F.softmax(outputs.logits).detach().numpy()
        return probas[0][1]

    fold = args.fold
    m = args.m

    # load model and tokenizer
    model = BertForSequenceClassification.from_pretrained(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/models/'
                                                          f'{fold}/checkpoint-{MAPPER[fold]}/')
    tokenizer = CustomTokenizer.from_pretrained('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/models/pretrained-covBERTa/',
                                                do_lower_case=False)

    tree, rate = load_tree_and_rates()
    # load control_data for bootstrap
    df = pd.read_csv('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/lovocv/sampled_case_controls.csv')
    case = df[(df['group'] == fold) & (df['label']) == 1]
    control = load_controls(fold)

    control['case'] = control['text'].apply(lambda x: predictor(x))
    case['case'] = case['text'].apply(lambda x: predictor(x))

    dist = dict(case.groupby(['clade', 'size']).size())
    res = []
    for bts in tqdm(range(m)):
        ids = stratify_by_clade_and_size(control, dist)
        sampled_control = control[control['id'].isin(ids)]
        merged = pd.concat([case, sampled_control])[['id','case','label']]
        data = get_fold_data(tree, rate, merged)

        X = data.drop(columns=['id', 'label'])
        y = data['label']

        skf = StratifiedKFold(n_splits=5)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_test, y_test = X.iloc[test_index], y.iloc[test_index]

            clf = LogisticRegression(random_state=42, penalty='elasticnet',
                                           solver='saga', l1_ratio=0.5, max_iter=1000)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred)
            coefs = [(X_train.columns[i], v) for i, v in enumerate(clf.coef_[0])]
            res.append((fold, bts, i, p[1], r[1], f1[1], coefs, data.shape[0]))
    lr = pd.DataFrame(res, columns=['fold','bootstrap', 'cv_fold', 'precision', 'recall', 'f1', 'coefs', 'size'])
    lr['case'] = lr['coefs'].apply(lambda x: x[0][1])
    lr['sackin_index'] = lr['coefs'].apply(lambda x: x[1][1])
    lr['S_rate'] = lr['coefs'].apply(lambda x: x[2][1])
    lr = lr.drop(columns=['coefs'])
    lr.to_csv(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/logistic_reg/{fold}_LR_weights.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Logistic regression for feature importance')

    parser.add_argument('--fold', default='alpha', type=str, help='fold name')
    parser.add_argument('-m', default=10, type=int, help='number of control bootstrapping')
    args = parser.parse_args()
    main(args)