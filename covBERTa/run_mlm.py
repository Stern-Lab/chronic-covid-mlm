import comet_ml
import os
import torch
import argparse
import glob
import numpy as np
from sklearn.metrics import accuracy_score

from mt_tokenizer import Tokenize
from datasets import load_dataset
from transformers import BertConfig, BertForMaskedLM, \
    DataCollatorForLanguageModeling, TrainingArguments, Trainer

# this code was inspired from : https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/

def main(args):

    experiment = comet_ml.Experiment(
        api_key='r0TcQHQkcqR8wVtCye8tTSAJj',
        project_name="cov-transformers-trainer",
    )
    experiment.add_tags(args.tags.split(','))

    # load dataset train and test splits
    files_map = {'train': glob.glob(f"{args.dataset}/train*csv"), 'test': glob.glob(f"{args.dataset}/test*csv")}
    if args.debug:
        files_map['train'], files_map['test'] = files_map['train'][:10], files_map['test'][:4]

    dataset = load_dataset("csv", data_files=files_map,
                           cache_dir='/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/HF_DATASETS_CACHE')

    # configure model output path
    model_path = os.path.join(args.model_path, args.save_prefix)
    os.makedirs(model_path, exist_ok=True)

    # load a tokenizer or train from scratch if needed
    vocab_size = args.vocab_size
    max_length = args.max_length
    train_test_map = {'train':[x.replace('csv', 'txt') for x in files_map['train']],
                      'test':[x.replace('csv', 'txt') for x in files_map['test']]}
    tokenizer = Tokenize(data=train_test_map, model_path=model_path,
                         vocab_size=vocab_size, max_length=max_length,
                         num_proc=args.ncpu, truncate=True)

    #if tokenizer does not exist train
    if not tokenizer.exists():
        tokenizer.train_tokenizer()
        tokenizer.save()

    tokenizer.load_pretrained()

    # encode train and test by vocab
    train_dataset, test_dataset = tokenizer.encode(dataset)
    print(f"Loaded {len(train_dataset)} train samples, {len(test_dataset)} test samples")
    parameters = {'per_device_train_batch_size': 10, 'gradient_accumulation_steps':8, 'per_device_eval_batch_size':64,
                  'num_train_epochs': args.epoches, 'mlm_probability':args.p}
    experiment.log_parameters(parameters)


    # initialize the model with the config
    model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
    model = BertForMaskedLM(config=model_config)

    # initialize data collector for LM batching
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer.tokenizer, mlm=True, mlm_probability=args.p
    )

    training_args = TrainingArguments(
        output_dir=model_path,
        evaluation_strategy="steps",
        overwrite_output_dir=True,
        num_train_epochs=args.epoches,
        auto_find_batch_size=True,
        per_device_train_batch_size=10,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=64,
        logging_steps=args.logging_interval,
        save_steps=args.save_interval,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    # train the model
    trainer.train()





if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training SARS-CoV-2 mutational word embedding')

    # training dataset
    parser.add_argument('--dataset', default='/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/gene_corpus/train_test',
                        help='path to a directory with .txt train and test dataset')
    parser.add_argument('--model-path', default='/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/models/',
                        help='path to a directory to save model outputs')
    parser.add_argument('--save-prefix', default='pretrained-covBERTa', help='path prefix for saving models')

    parser.add_argument('--vocab-size', type=int, default=38_000, help='vocabulary size')
    parser.add_argument('--max-length', type=int, default=160, help='maximal sequence length')
    parser.add_argument('-p', type=float, default=0.2, help='masking rate')
    parser.add_argument('--tags', default='mlm', help='tags for comet-ml experiment')

    parser.add_argument('--ncpu', type=int, default=10, help='masking rate')

    parser.add_argument('--debug', action='store_true')



    # training parameters - 1000000
    parser.add_argument('-e', '--epoches', type=int, default=2, help='number of data epoches')
    parser.add_argument('--save-interval', type=int, default=1000, help='number of step between data saving')
    parser.add_argument('--logging-interval', type=int, default=1000, help='number of step between data logginh')


    args = parser.parse_args()

    main(args)
