from collections import defaultdict

import torch
from torch.utils.data import Sampler


class SameSizeBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        sizes: list[tuple[int, int]],
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.buckets: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
        for index, size in enumerate(sizes):
            self.buckets[size].append(index)

    def __iter__(self):
        batches = []
        for indices in self.buckets.values():
            if self.shuffle:
                perm = torch.randperm(len(indices)).tolist()
                indices = [indices[i] for i in perm]

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            perm = torch.randperm(len(batches)).tolist()
            batches = [batches[i] for i in perm]

        yield from batches

    def __len__(self):
        if self.drop_last:
            return sum(
                len(bucket) // self.batch_size for bucket in self.buckets.values()
            )
        else:
            return sum(
                (len(bucket) + self.batch_size - 1) // self.batch_size
                for bucket in self.buckets.values()
            )
