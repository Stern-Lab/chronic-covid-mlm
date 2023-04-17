import os
import pandas as pd
from sklearn.model_selection import train_test_split


class TextData:
    def __init__(self, path, min=3, max=160):
        self.path = path
        self.min = min
        self.max = max
        self.lines = []

    def load_txt(self):
        with open(self.path, 'r') as o:
            for line in o:
                if self.min <= len(line.split()) < self.max:
                    self.lines.append(line.replace('\n', ''))
        print(f"Loaded {len(self.lines)} sentences")

    @staticmethod
    def split_chunks(lst, n, save_path, alias='train'):
        for i, start in enumerate(range(0, len(lst), n)):
            with open(os.path.join(save_path, f'{alias}{i}.txt'), 'w') as o:
                o.writelines(lst[start:start+n])
            df = pd.DataFrame(lst[start:start+n], columns=['text'])
            df.to_csv(os.path.join(save_path, f'{alias}{i}.csv'), index=False)

    def split_train_test(self, p=0.1, n=10000):
        train, test = train_test_split(self.lines, test_size=p)

        splits_save = os.path.join(os.path.dirname(self.path), 'train_test')
        os.makedirs(splits_save, exist_ok=True)

        # split train files
        self.split_chunks(train, n, splits_save, alias='train')
        print(f"Finished processing {len(train)} sentences into {len(train)//n} chunks")
        self.split_chunks(test, n, splits_save, alias='test')
        print(f"Finished processing {len(test)} sentences into {len(test) // n} chunks")


dataset = TextData('/sternadi/home/volume3/chronic-corona-pred/sars_cov_2_mlm/gene_corpus/ALL.txt')
dataset.load_txt()
dataset.split_train_test()













