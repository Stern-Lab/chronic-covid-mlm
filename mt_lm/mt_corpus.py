import glob
import itertools
import torch
import numpy as np
import pickle
import os

from torchtext.vocab import build_vocab_from_iterator
from collections import Counter



class MutationalCorpus:
    def __init__(self, dataset, debug, n):

        sentences = []
        with open(dataset, 'r') as o:
            for line in o:
                text = line.split()
                if  n <= len(text) < 160:
                    sentences.append(text)
        if debug:
            if len(sentences) >= 10000:
                sentences = sentences[:10000]

        vocab = build_vocab_from_iterator(sentences, min_freq=10, specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])

        # save vocab for encoding
        with open(f'{dataset.replace(".txt", "_vocab.pkl")}', 'wb') as fp:
            pickle.dump(vocab, fp)


        self.x = [torch.from_numpy(np.asarray(vocab(s))) for s in sentences]

        v = [vocab(s) for s in sentences]
        counts = np.zeros(len(vocab))
        token_dist = Counter(itertools.chain(*v))
        for k, v in token_dist.items():
            counts[k] = v
        background = counts / counts.sum()

        self.background = background

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        return x.long()


# this function is taken from the prose repo: https://github.com/tbepler/prose
class MaskedMutationalCorpus:
    def __init__(self, x, p, background, noise):
        self.x = x
        self.p = p
        self.background = background
        self.noise = noise

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        p = self.p
        n = len(self.background)

        # create the random mask... i.e. which positions to infer
        mask = torch.rand(len(x), device=x.device)
        mask = (mask < p).long() # masking probability

        y = mask*x + (1-mask)*n

        # sample the masked positions from the noise distribution
        if self.noise:
            noise = torch.multinomial(self.background, len(x), replacement=True)
            x = (1-mask)*x + mask*noise

        return x, y






