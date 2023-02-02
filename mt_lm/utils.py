import itertools
import torch

import numpy as np
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, PackedSequence



def collate_batch(batch):
    text, masked, lengths = [], [], []
    for _text, _masked in batch:
        text.append(_text)
        masked.append(_masked)
        lengths.append(len(_text))
    print(len(text), len(masked), len(lengths))
    order = np.argsort(lengths)[::-1] # sort in descending order for batching

    # now sort text and masked before padding
    text = np.array(text).reshape(len(text),1)[order]
    masked = np.array(masked).reshape(len(masked),1)[order]
    lengths = np.array(lengths).reshape(len(lengths),1)[order]

    text_pad = pad_sequence(text, batch_first=True)
    masked_pad = pad_sequence(masked, batch_first=True)

    text_packed = pack_padded_sequence(text_pad, lengths, batch_first=True)
    masked_packed = pack_padded_sequence(masked_pad, lengths, batch_first=True)

    return text_packed, masked_packed

def mask_grad(model, x, y, m, use_cuda):

    # unpack y
    y = y.data

    if use_cuda:
        x = PackedSequence(x.data.cuda(), x.batch_sizes)
        y = y.cuda()

    mask = (y < m)
    # check that we have noised positions...
    loss = 0
    correct = 0
    n = mask.float().sum().item()
    if n > 0:
        logits = model(x).data

        # only calculate loss for noised positions
        logits = logits[mask]
        y = y[mask]

        loss = F.cross_entropy(logits, y)

        _,y_hat = torch.max(logits, 1)

        w_loss = loss
        w_loss.backward()

        loss = loss.item()
        correct = torch.sum((y == y_hat).float()).item()

    return loss, correct, n