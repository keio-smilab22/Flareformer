import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from tqdm import tqdm

class SkipMissingValueBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        loader = DataLoader(dataset)
        self.dataset = dataset
        self.missing_value_indices:list = dataset.missing_value_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        if self.shuffle:
            self.sampler = RandomSampler(self.dataset)
        else:
            self.sampler = SequentialSampler(self.dataset)

    def __iter__(self):
        batch = []
        
        for idx in self.sampler:
            if idx in self.missing_value_indices:
                continue
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


