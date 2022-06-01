from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
from datasets.flare import FlareDataset
from numpy import int64
from typing import Iterator, List


class TrainBalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset: FlareDataset, n_classes: int, n_samples: int):
        print("Prepare Batch Sampler ...")
        loader = DataLoader(dataset)
        self.labels_list = []
        for x, y, idx in tqdm(loader):
            self.labels_list.append(np.argmax(y))

        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {
            label: np.where(self.labels.numpy() == label)[0]
            for label in self.labels_set
        }
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self) -> Iterator[List[int64]]:
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(
                self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][
                        self.used_label_indices_count[
                            class_
                        ]: self.used_label_indices_count[class_]
                        + self.n_samples
                    ]
                )
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(
                    self.label_to_indices[class_]
                ):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size
