import glob
import itertools
import torch
import numpy as np

from torchtext.vocab import build_vocab_from_iterator
from collections import Counter


class MutationalCorpus:
    def __init__(self, dataset, debug, n):

        fs = glob.glob(f'{dataset}/*.txt')
        sentences = []
        for f in fs:
            with open(f, 'r') as o:
                s = o.read().split('. ')
                text = [m.split() for m in s if len(m.split()) >= n]
                sentences.extend(text)
            if debug:
                if len(sentences) >= 10000:
                    sentences = sentences[:10000]
                    break

        vocab = build_vocab_from_iterator(sentences, min_freq=10, specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])

        self.vocab = vocab
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






