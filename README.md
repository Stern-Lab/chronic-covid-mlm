# chronic-covid-pred


This repository contains the code supporting the paper
**Using big sequencing data to identify chronic SARS-Coronavirus-2 infections**

## Getting the data
*Note:* The data will be deposited to the Zenodo database and assigned a permanent DOI.
meanwhile, do not use the download instructions below and go to [this link](https://tinyurl.com/4jxuvbak).

Start by downloading the data files from the Zenodo database.  

1. Click on the Zenodo link at the top of the repository or use [this link](https://tinyurl.com/4jxuvbak) to download the data zip file
2. Alternatively, use the command line as follows: 
```
mkdir data
cd data

wget https://zenodo.org/record/XXXXXXX/files/data.tar.gz?download=1
tar -zxvf data.tar.gz
rm data.tar.gz
```

## Setting up the working environment
First, set up python environment and dependencies. 

:round_pushpin: This installation requires a GPU with compatible CUDA 11.7
#### using conda
```
conda env create -f environment.yml
conda activate chronic-mlm
```

The setup was tested on Python 3.10.
Versions of all required programs appear in `environment.yml`.

## Loading the model
#### Pre-train MLM
:round_pushpin:Upon acceptance, the model will be uploaded to the Hugging Face Hub :hugs:, and the instructions for usage will be updated here.</br>
Meanwhile, you may load the model using the data checkpoints provided.

```
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

class CustomTokenizer(BertTokenizer):
    def _tokenize(self, text):
        text = text.replace(':', 'X').replace('-','X')
        tokens = text.split()
        return tokens

tokenizer = CustomTokenizer.from_pretrained('data/Supplementary Dataset 6/models/pretrained-covBERTa/',
                                             do_lower_case=False)
model = BertModel.from_pretrained('data/Supplementary Dataset 6/pretrained-covBERTa/checkpoint-29000/',
                                  output_hidden_states = True)
```

#### Per variant classifier
For loading the classification models use the same tokenizer but load the latest checkpoint to `BertForSequenceClassification`:
```
model = BertModel.from_pretrained('data/Supplementary Dataset 6/classifier/omicron/checkpoint-180/',
                                  output_hidden_states = True)
```



