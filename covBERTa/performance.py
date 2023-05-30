import os
import pickle
import glob
import json
import argparse
import torch

import pandas as pd
import numpy as np

import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

from lime.lime_text import LimeTextExplainer

from sklearn.metrics import precision_recall_curve, auc, roc_curve, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

class CustomTokenizer(BertTokenizer):
    def _tokenize(self, text):
        # replace : and - used by non-syn mutations and
        text = text.replace(':', 'X').replace('-','X')
        # tokenize by white space
        tokens = text.split()
        return tokens

def _predictor(model, tokenizer, texts):
    outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
    probas = F.softmax(outputs.logits).detach().numpy()
    return probas


def get_prediction(model, tokenizer, test, fold):
    res = []
    ins = '22204:GAGCCAGAA'
    for l, txt in zip(test['label'], test['text']):
        original_txt = txt
        if fold == 'omicron':
            txt = ' '.join([m for m in txt.split() if m != ins])
        probs = _predictor(model=model, tokenizer=tokenizer, texts=txt)
        pred = np.argmax(probs[0])
        res.append((original_txt, len(txt.split()), probs[0][0], probs[0][1], pred, l))
    return pd.DataFrame(res, columns=['text', 'mts', 'control','case', 'pred', 'label'])

def weight_features(df):
    y_true = df['label'].values
    y_pred = df['pred'].values

    res = []
    for i in np.random.random(40):
        sample_weight = i * df['mt_weight'].values * (1 - i) * df['size_weight'].values
        weighted_accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        res.append((weighted_accuracy, i, 1 - i))
    acc, w1, w2 = max(res, key=lambda x: x[0])

    sample_weight = w1 * df['mt_weight'].values * w2 * df['size_weight'].values
    weighted_accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    return sample_weight, weighted_accuracy

def wrap_up_explainer(df, t):
    res_dict = {'control': dict(), 'case': dict()}
    for row in df[(df['label'] == 1) & (df['case'] >= t)].iterrows():
        row = row[1]
        ctl = row['explainer']['control']
        for m, v in ctl:
            if m in res_dict['control']:
                res_dict['control'][m][0] += np.abs(v)
                res_dict['control'][m][1] += 1
                res_dict['control'][m][2] += row['explainer']['score']
            else:
                res_dict['control'][m] = [np.abs(v), 1, row['explainer']['score']]

        cs = row['explainer']['case']
        for m, v in cs:
            if m in res_dict['case']:
                res_dict['case'][m][0] += v
                res_dict['case'][m][1] += 1
                res_dict['case'][m][2] += row['explainer']['score']
            else:
                res_dict['case'][m] = [v, 1, row['explainer']['score']]
    ctl = pd.DataFrame(res_dict['control']).T.reset_index().rename(
        columns={'index': 'mt', 0: 'score', 1: 'n_samples', 2: 'exp_score'})
    cs = pd.DataFrame(res_dict['case']).T.reset_index().rename(
        columns={'index': 'mt', 0: 'score', 1: 'n_samples', 2: 'exp_score'})
    ctl['label'] = 0
    cs['label'] = 1
    res = pd.concat([cs, ctl])
    return res


def plot_pr_curve(recall, precision, out):
    with sns.plotting_context("poster"):
        sns.lineplot(x=recall, y=precision, label=f'AUPR:{np.round(auc(recall, precision), 2)}', c='deeppink')
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(out, 'precision_recall_curve.pdf'))
        plt.close()


def plot_auc_curve(tpr, fpr, out):
    with sns.plotting_context("poster"):
        sns.lineplot(x=fpr, y=tpr, label=f'AUC:{np.round(auc(fpr, tpr), 2)}', c='deeppink')
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.ylabel('True Positive rate')
        plt.xlabel('False Positive Rate')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(out, 'roc_curve.pdf'))
        plt.close()

def main(args):

    t = args.t
    fold = args.fold
    tokenizer = CustomTokenizer.from_pretrained(
        '/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/models/pretrained-covBERTa/', do_lower_case=False)
    model = BertForSequenceClassification.from_pretrained(args.model)
    test = pd.read_csv(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/lovocv/fold_{fold}_test.csv')
    data = pd.read_csv(f'/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/lovocv/sampled_case_controls.csv')

    def predictor(texts):
        outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
        probas = F.softmax(outputs.logits).detach().numpy()
        return probas

    def explain_prediction(text, explainer, n=10):
        exp = explainer.explain_instance(text, predictor, num_features=n, num_samples=2000)
        vals = {"control": [x for x in exp.as_list() if x[1] < 0],
                "case": [x for x in exp.as_list() if x[1] >= 0],
                "score": exp.score}
        return vals


    # get predictions for all test cases
    pred = get_prediction(model, tokenizer, test, fold)
    pred = pred.merge(data[['text', 'size', 'clade']], on=['text'], how='left')

    # get weighted sum for features
    pred['mt_weight'] = pred['mts'] / pred['mts'].sum()
    pred['size_weight'] = pred['size'] / pred['size'].sum()
    pred['label_score'] = pred.apply(lambda x: x['case'] if x['label'] == 1 else x['control'], axis=1)
    sample_weight, weighted_accuracy = weight_features(pred)
    pred['sample_weight'] = sample_weight
    pred['weighted_accuracy'] = weighted_accuracy

    # save pred to csv
    out = os.path.dirname(os.path.dirname(args.model))
    os.makedirs(out, exist_ok=True)
    if args.clade == 'all':
        pred.to_csv(os.path.join(out, 'preds.csv'), index=False)

    # calc aupr, auc
    precision, recall, thresholds = precision_recall_curve(pred['label'].values, pred['label_score'].values,
                                                           sample_weight=sample_weight)
    fpr, tpr, thresholds = roc_curve(pred['label'].values, pred['label_score'].values, sample_weight=sample_weight)
    plot_pr_curve(recall, precision, out)
    plot_auc_curve(tpr, fpr, out)

    # save all performance data
    if args.clade == 'all':
        pd.DataFrame({'recall': recall, 'precision':precision, 'fold':fold}).to_csv(os.path.join(out, 'precision_recall.csv'), index=False)
        pd.DataFrame({'fpr': fpr, 'tpr':tpr, 'fold':fold}).to_csv(os.path.join(out, 'fpr_tpr.csv'), index=False)

    # run the explainer to generate top mutational candidates
    class_names = ['control', 'case']
    explainer = LimeTextExplainer(class_names=class_names, split_expression=' ')
    pred['explainer'] = pred['text'].apply(lambda x: explain_prediction(x, explainer))
    alias = ''
    if args.clade == 'BA1':
        pred = pred[pred['clade'] == '21K']
        alias = 'BA1_'
    elif args.clade == 'BA2':
        pred = pred[pred['clade'].isin(['21L', '22B', '22C'])]
        alias = 'BA2_'

    exp = wrap_up_explainer(pred, t=t)
    exp.to_csv(os.path.join(out, f'{alias}explainer.csv'), index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Performance assessment for LOCOCV')

    parser.add_argument('--model', default='/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/models/omicron/checkpoint-150/',
                        help='path to a model checkpoint')
    parser.add_argument('--fold', default='omicron', type=str, help='fold name')
    parser.add_argument('--clade', default='all', type=str, help='whether to extract explainability by clade')
    parser.add_argument('-t', default=0, type=float, help='threshold for cases to consider as reliable')
    args = parser.parse_args()
    main(args)
