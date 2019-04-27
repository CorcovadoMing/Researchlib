import torch
from torchtext import data

def FromTextDataset(dataset, batch_size, shuffle=True):
    loader = data.BucketIterator(
                dataset,
                batch_size=batch_size,
                sort_key=lambda x: len(x.text),
                repeat=False,
                shuffle=shuffle)
    return loader
