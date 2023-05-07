import comet_ml
import os
import torch
import argparse
import glob
import numpy as np

import torch.nn as nn
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# this code was inspired from : https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward - get logits
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute weighted loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.5]).to('cuda'))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    cm = confusion_matrix(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }

class CustomTokenizer(BertTokenizer):
    def _tokenize(self, text):
        # replace : and - used by non-syn mutations and
        text = text.replace(':', 'X').replace('-','X')
        # tokenize by white space
        tokens = text.split()
        return tokens

tokenizer = CustomTokenizer.from_pretrained('/sternadi/home/volume3/chronic-corona-pred/'
                                            'sars_cov_2_mlm/models/pretrained-covBERTa/', do_lower_case=False)
def _encode(x):
    return tokenizer(x["text"], truncation=True, padding="max_length",
                     max_length=160, return_special_tokens_mask=True)


def main(args):

    experiment = comet_ml.Experiment(
        api_key='r0TcQHQkcqR8wVtCye8tTSAJj',
        project_name="cov-transformers-trainer",
    )
    experiment.add_tags(args.tags.split(','))

    # load dataset train and test splits
    files_map = {'train': glob.glob(f"{args.dataset}/fold_{args.fold}_train.csv"),
                 'test': glob.glob(f"{args.dataset}/fold_{args.fold}_test.csv")}
    if args.debug:
        files_map['train'], files_map['test'] = files_map['train'][:10], files_map['test'][:4]

    dataset = load_dataset("csv", data_files=files_map,
                           cache_dir='/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/HF_DATASETS_CACHE')

    # tokenizing train & test dataset
    ncpus = args.ncpu
    train_dataset = dataset["train"].map(_encode, batched=True, num_proc=ncpus)
    test_dataset = dataset["test"].map(_encode, batched=True, num_proc=ncpus)

    # remove other columns and set input_ids and attention_mask as tensors
    train_dataset.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])

    print(f"Loaded {len(train_dataset)} train samples, {len(test_dataset)} test samples")
    parameters = {'per_device_train_batch_size': 10, 'per_device_eval_batch_size':64,
                  'num_train_epochs': args.epoches}
    experiment.log_parameters(parameters)

    # configure model output path
    model_path = os.path.join(args.model_path, f'fold_{args.fold}')
    os.makedirs(model_path, exist_ok=True)


    # initialize the model with the config
    model = BertForSequenceClassification.from_pretrained(args.model)
    # Freeze the lower layers - do not want to overfit....
    for name, param in model.named_parameters():
        if name.startswith('bert.encoder'):
            param.requires_grad = False


    # Create an optimizer and only pass the parameters of the classification head to it
    optimizer = AdamW(model.classifier.parameters(), lr=5e-5, no_deprecation_warning=True)


    training_args = TrainingArguments(
        output_dir=model_path,
        evaluation_strategy="steps",
        overwrite_output_dir=True,
        num_train_epochs=args.epoches,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=64,
        logging_steps=args.logging_interval,
        save_steps=args.save_interval,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # train the model
    trainer.train()

    eval_metrics = trainer.evaluate(eval_dataset=test_dataset)
    experiment.log_metrics(eval_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training SARS-CoV-2 mutational classifier')

    # training dataset - '../'
    parser.add_argument('--dataset', default='/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/train_test',
                        help='path to a directory with .txt train and test dataset')
    parser.add_argument('--model', default='/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/models/pretrained-covBERTa/checkpoint-29000/',
                        help='path to a directory with .txt train and test dataset')
    parser.add_argument('--fold', default=0, type=int,
                        help='fold number')
    parser.add_argument('--model-path', default='/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/classifier/models/',
                        help='path to a directory to save model outputs')

    parser.add_argument('--tags', default='classify', help='tags for comet-ml experiment')

    parser.add_argument('--ncpu', type=int, default=10, help='number of cpus')

    parser.add_argument('--debug', action='store_true')



    # training parameters - 1000000
    parser.add_argument('-e', '--epoches', type=int, default=20, help='number of data epoches')
    parser.add_argument('--save-interval', type=int, default=50, help='number of step between data saving')
    parser.add_argument('--logging-interval', type=int, default=100, help='number of step between data logginh')


    args = parser.parse_args()

    main(args)
