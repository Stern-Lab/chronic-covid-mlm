import argparse
import os
import pickle
import glob
import json
import comet_ml


import pandas as pd
import numpy as np
from models.CustomTransformer import CustomTransformer
from sentence_transformers import SentenceTransformer, models, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from torch import nn


def load_data(file_path, debug=False):
    n = 0
    dataset = []
    with open(file_path, 'r') as o:
        for line in o:
            # ignore header line
            if line == 'label,sentence1,sentence2\n':
                continue
            label, text1, text2 = line.strip().split(',')
            n +=1
            if 'train.csv' in file_path:
                dataset.append(InputExample(texts=[text1, text2], label=float(label)))
            else:
                dataset.append((text1, text2, float(label)))

            if debug:
                if n > 10000:
                    break
    return dataset

def main(args):

    experiment = comet_ml.Experiment(
        api_key='r0TcQHQkcqR8wVtCye8tTSAJj',
        project_name="cov-transformers-trainer",
    )
    experiment.add_tags(args.tags.split(','))

    cache_dir = '/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/HF_DATASETS_CACHE'
    word_embedding_model = CustomTransformer('covberta', max_seq_length=160, do_lower_case=False, cache_dir=cache_dir)

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256,
                               activation_function=nn.Tanh())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

    train = load_data(f"{args.dataset}/fold{args.fold}_train.csv", args.debug)
    eval = load_data(f"{args.dataset}/fold{args.fold}_test.csv")

    #Define train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train, shuffle=True, batch_size=args.batch_size)
    evaluator = evaluation.BinaryClassificationEvaluator([e[0] for e in eval],
                                                         [e[1] for e in eval],
                                                         [e[2] for e in eval],
                                                         name=f'eval_fold{args.fold}',
                                                         write_csv=True,
                                                         batch_size=args.batch_size)

    if args.loss == 'cosine':
        train_loss = losses.CosineSimilarityLoss(model)
    else:
        train_loss = losses.ContrastiveLoss(model)

    parameters = {'epoches': args.num_epoches, 'evaluation_steps':args.eval_steps, 'use_amp':False,
                  'loss': args.loss}
    experiment.log_parameters(parameters)

    #Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=args.num_epoches,
              warmup_steps=100,
              evaluator=evaluator,
              evaluation_steps=args.eval_steps,
              use_amp=False,
              checkpoint_save_steps=args.save_interval,
              checkpoint_path=args.output,
              output_path=args.output,
              )

    experiment.end()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training SARS-CoV-2 s-bert')

    # training dataset
    parser.add_argument('--dataset', default='/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/few_shot/train_test/',
                        help='path to a directory with train and test')
    # embedding model architecture
    parser.add_argument('model', nargs='?', help='pretrained model (optional)')

    parser.add_argument('--fold', type=int, default=0, help='fold number, can be 0,1 or 2')
    parser.add_argument('--out-features', type=int, default=256, help='size of the dense function')
    parser.add_argument('--loss', type=str, default='cosine', help='name of the loss function can be cosine or contrastive')

    # training parameters - 1000000
    parser.add_argument('-n', '--num-epoches', type=int, default=1, help='number of training epoches')
    parser.add_argument('--steps', type=int, default=1000, help='number of training steps per epoch')
    parser.add_argument('--eval-steps', type=int, default=500, help='number of training steps')
    # eval_steps
    parser.add_argument('--save-interval', type=int, default=10000, help='number of step between data saving')

    parser.add_argument('--batch-size', type=int, default=100, help='minibatch size')

    parser.add_argument('--output', help='output file path')
    parser.add_argument('--tags', default='s-bert', help='tags for comet-ml experiment')
    parser.add_argument('--ncpu', type=int, default=10, help='number of cpus')

    parser.add_argument('--debug', action='store_true')


    args = parser.parse_args()

    main(args)