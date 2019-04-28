import torch
import torchtext
import spacy
import pandas as pd
import numpy as np

def _AGNews(train=True):
    NLP = spacy.load('en_core_web_sm')
    tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
    
    # Creating Field for data
    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=60, batch_first=True)
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False, batch_first=True)
    datafields = [("text", TEXT),("label", LABEL)]
    
    if train:
        filename = 'researchlib/dataset/ag_news.train'
    else:
        filename = 'researchlib/dataset/ag_news.test'
        
    # Load data from pd.DataFrame into torchtext.data.Dataset
    with open(filename, 'r') as datafile:     
        data = [line.strip().split(',', maxsplit=1) for line in datafile]
        data_text = list(map(lambda x: x[1], data))
        data_label = list(map(lambda x: int(x[0].strip()[-1])-1, data))
        full_df = pd.DataFrame({"text": data_text, "label": data_label})
    
    examples = [torchtext.data.Example.fromlist(i, datafields) for i in full_df.values.tolist()]
    datas = torchtext.data.Dataset(examples, datafields)

    TEXT.build_vocab(datas)
    vocab = TEXT.vocab
    print('VOC:', len(vocab))
    return datas