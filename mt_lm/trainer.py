from comet_ml import Experiment

import argparse
import torch
import torch.nn as nn

import numpy as np

from torch.utils.data import DataLoader
from mt_corpus import MutationalCorpus, MaskedMutationalCorpus
from utils import collate_batch, mask_grad, save
from mlm import SkipLSTM

import gc



def main(args):
    # set the device
    device = args.device
    use_cuda = (device != -1) and torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device is: {device}")

    # define experiment
    experiment = Experiment(api_key="r0TcQHQkcqR8wVtCye8tTSAJj",
                            project_name="mt-mlm")
    experiment.add_tags(args.tags.split(','))

    dataset_path = args.dataset
    noise = args.noise
    debug = args.debug

    # load corpus - overwriting for lower memory complexity
    corpus = MutationalCorpus(dataset_path, debug, 3)
    background = torch.from_numpy(corpus.background)
    corpus = MaskedMutationalCorpus(corpus, args.p, background, noise)

    print("Processed all corpus files")

    # set training data params
    batch_size = args.batch_size
    num_steps = args.num_steps
    weight_decay = args.weight_decay
    lr = args.lr
    clip = args.clip

    masked_corpus_iter = DataLoader(corpus, batch_size=batch_size, collate_fn=collate_batch)

    ## initialize the model
    if args.model is not None:
        # load pretrained model
        model = torch.load(args.model)

    else:
        nin = len(background)
        nout = nin
        hidden_dim = args.lstm_dim
        num_layers = args.num_layers
        dropout = args.dropout

        model = SkipLSTM(nin, nout, hidden_dim, num_layers, dropout=dropout)
        # print number of trainable parameters
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of parameters in model: {pytorch_total_params}")

    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    minibatch_iterator = iter(masked_corpus_iter)

    # update experiment params
    prefix = 'train'
    parameters = {'batch_size': batch_size, 'learning_rate': lr}
    experiment.log_parameters(parameters, prefix=prefix)


    step = 0
    n = 0
    loss_estimate = 0
    acc_estimate = 0

    model.train()

    save_iter = 100
    save_interval = args.save_interval
    while save_iter <= step:
        save_iter = min(save_iter*10, save_iter+save_interval, num_steps) # next save

    # clear the cache
    gc.collect()
    torch.cuda.empty_cache()

    save_prefix = f'{args.save_prefix}_{nin}_{nout}_{hidden_dim}_{num_layers}'
    digits = int(np.floor(np.log10(num_steps))) + 1

    for i in range(step, num_steps):
        try:
            x,y = next(minibatch_iterator)
        except StopIteration:
            print(f"step {i}, no more data to load")
            save(save_prefix, model, digits, i, use_cuda)
            break

        optim.zero_grad()
        loss,correct,b = mask_grad(model, x, y, len(background), use_cuda)

        n += b
        delta = b*(loss - loss_estimate)
        loss_estimate += delta/n
        delta = correct - b*acc_estimate
        acc_estimate += delta/n

        # clip the gradients if needed
        if not np.isinf(clip):
            # only clip the RNN layers
            nn.utils.clip_grad_norm_(model.layers.parameters(), clip)

        # parameter update
        optim.step()

        experiment.log_metrics({"accuracy":acc_estimate}, step=i)
        experiment.log_metrics({"loss": loss_estimate}, step=i)
        experiment.log_metrics({"perplexity": np.exp(loss_estimate)}, step=i)

        # reset the accumlation metrics
        n = 0
        loss_estimate = 0
        acc_estimate = 0

        # save the model
        if i+1 == save_iter:
            save_iter = min(save_iter*10, save_iter+save_interval, num_steps) # next save

            if save_prefix is not None:
                model.eval()
                save_path = save_prefix + '_iter' + str(i + 1).zfill(digits) + '.sav'
                model.cpu()
                torch.save(model.state_dict(), save_path)
                if use_cuda:
                    model.cuda()

            # flip back to train mode
            model.train()

    experiment.end()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training SARS-CoV-2 mutational word embedding')

    # training dataset
    parser.add_argument('--dataset', default='/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/gene_corpus/ALL.txt',
                        help='path to a directory with .txt training dataset')
    # embedding model architecture
    parser.add_argument('model', nargs='?', help='pretrained model (optional)')

    parser.add_argument('--lstm-dim', type=int, default=512, help='hidden units of LSTMs')
    parser.add_argument('--num-layers', type=int, default=2, help='number of LSTMs layers')
    parser.add_argument('--dropout', type=float, default=0, help='dropout probability')

    # training parameters - 1000000
    parser.add_argument('-n', '--num-steps', type=int, default=1000000, help='number of training steps')
    parser.add_argument('--save-interval', type=int, default=100000, help='number of step between data saving')

    parser.add_argument('-p', type=float, default=0.15, help='masking rate')
    parser.add_argument('--batch-size', type=int, default=100, help='minibatch size')

    parser.add_argument('--weight-decay', type=float, default=0, help='L2 regularization (default: 0)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 1e-4)')
    parser.add_argument('--clip', type=float, default=np.inf, help='gradient clipping max norm (default: inf)')

    parser.add_argument('--output', help='output file path')
    parser.add_argument('--save-prefix', help='path prefix for saving models')
    parser.add_argument('--tags', default='mlm', help='tags for comet-ml experiment')
    parser.add_argument('--device', type=int, default=-2, help='compute device to use')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--noise', action='store_true')

    args = parser.parse_args()

    main(args)





