import os
import json
from collections import Counter

import pandas as pd
from transformers import BertTokenizer


class CustomTokenizer(BertTokenizer):
    def _tokenize(self, text):
        # replace : and - used by non-syn mutations and
        text = text.replace(':', 'X').replace('-','X')
        # tokenize by white space
        tokens = text.split()
        return tokens


class Tokenize:
    def __init__(self, data, model_path='pretrained-covBERTa', vocab_size=38_000, max_length=160,
                 num_proc=1, truncate=True):
        self.train = data['train']
        self.val = data['test']
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.truncate = truncate
        self.num_proc = num_proc

        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

        self.tokenizer = None
        self.tokens = None

    def load_pretrained(self):
        tokenizer = CustomTokenizer.from_pretrained(self.model_path, do_lower_case=False)
        self.tokenizer = tokenizer

    def exists(self):
        return os.path.exists(os.path.join(self.model_path, 'vocab.txt'))

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def train_tokenizer(self):
        train = self.train
        # create the vocabulary for the tokenizer
        res = Counter()
        for f in train:
            df = pd.read_csv(f)
            c = df['text'].apply(lambda x: x.split()).explode()
            c = Counter(c.apply(lambda s: s.replace(':', 'X').replace('-','X')))
            res.update(c)

        tokens = self.special_tokens + [t[0] for t in res.most_common(self.vocab_size - len(self.special_tokens))]
        self.tokens = tokens


    def encode(self, x, num_proc=10):
        def _encode(x):
            return self.tokenizer(x["text"], truncation=True, padding="max_length",
                         max_length=self.max_length, return_special_tokens_mask=True)

        # tokenizing train & test dataset
        train_dataset = x["train"].map(_encode, batched=True, num_proc=num_proc)
        test_dataset = x["test"].map(_encode, batched=True, num_proc=num_proc)

        # remove other columns and set input_ids and attention_mask as tensors
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        return train_dataset, test_dataset


    def save(self):
        os.makedirs(self.model_path, exist_ok=True)
        # save the tokenizer
        vocab = '\n'.join(self.tokens)
        with open(os.path.join(self.model_path, "vocab.txt"),'w') as o:
            o.write(vocab)

        with open(os.path.join(self.model_path, "config.json"), "w") as f:
            tokenizer_cfg = {
                "do_lower_case": False,
                "unk_token": "[UNK]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "mask_token": "[MASK]",
                "model_max_length": self.max_length,
                "max_len": self.max_length,
            }
            json.dump(tokenizer_cfg, f)








