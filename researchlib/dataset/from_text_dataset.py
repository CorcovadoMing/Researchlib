import torch
from torchtext import data

def TextDataset(dataset, batch_size, train=True):
    loader = data.BucketIterator(
                dataset(train),
                batch_size=batch_size,
                sort_key=lambda x: len(x.text),
                repeat=False,
                shuffle=train)
    return loader
