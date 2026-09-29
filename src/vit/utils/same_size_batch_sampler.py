from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

import torch
from torch.utils.data import Dataset, Sampler

ImageSize = tuple[int, int]


class SizedImageDataset(Dataset, ABC):
    """Dataset that exposes image sizes without loading all size metadata."""

    @abstractmethod
    def size_counts(self) -> Iterable[tuple[ImageSize, int]]:
        """Yield each distinct image size and its number of examples."""

    @abstractmethod
    def indices_for_size(
        self, size: ImageSize, *, shuffle: bool, generator: torch.Generator | None
    ) -> Iterable[int]:
        """Yield indices with ``size``; implementations may stream from storage."""


class SameSizeBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: SizedImageDataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        generator: torch.Generator | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

    def __iter__(self) -> Iterator[list[int]]:
        groups = list(self.dataset.size_counts())
        if self.shuffle:
            perm = torch.randperm(len(groups), generator=self.generator).tolist()
            groups = [groups[i] for i in perm]

        for size, _count in groups:
            batch = []
            for index in self.dataset.indices_for_size(
                size, shuffle=self.shuffle, generator=self.generator
            ):
                batch.append(index)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

            if batch and not self.drop_last:
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return sum(
                count // self.batch_size for _, count in self.dataset.size_counts()
            )
        return sum(
            (count + self.batch_size - 1) // self.batch_size
            for _, count in self.dataset.size_counts()
        )
