from collections.abc import Iterator
from pathlib import Path
from typing import override

import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from torchvision.transforms import v2

from vit.data.synthtiger import SynthTiger


def collate_features(batch: list[tuple[torch.Tensor, str]]):
    features, texts = zip(*batch, strict=True)
    lengths = torch.tensor([feature.shape[0] for feature in features])
    if bool((lengths == lengths[0]).all()):
        return torch.stack(features), list(texts), None
    padded = pad_sequence(list(features), batch_first=True)
    valid = torch.arange(padded.shape[1])[None, :] < lengths[:, None]
    return padded, list(texts), valid


def collate_images(batch: list[tuple[torch.Tensor, str]]):
    images, texts = zip(*batch, strict=True)
    return list(images), list(texts), None


class SplitIndexSampler(Sampler[int]):
    def __init__(self, dataset: SynthTiger, *, shuffle: bool):
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        return self.dataset.indices(shuffle=self.shuffle)

    def __len__(self) -> int:
        return len(self.dataset)


class PatchBucketBatchSampler(Sampler[list[int]]):
    """Fill batches from nearby patch counts without discarding size tails."""

    def __init__(self, dataset: SynthTiger, batch_size: int, bucket_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size

    def __iter__(self) -> Iterator[list[int]]:
        buckets: dict[int, list[int]] = {}
        for index, patch_count in self.dataset.indices_with_patch_counts(shuffle=True):
            bucket = buckets.setdefault(patch_count // self.bucket_size, [])
            bucket.append(index)
            if len(bucket) == self.batch_size:
                yield bucket
                buckets[patch_count // self.bucket_size] = []

        # Combine remainders so only the final batch is smaller than batch_size.
        batch: list[int] = []
        tail_batches: list[list[int]] = []
        for bucket_id in sorted(buckets):
            for index in buckets[bucket_id]:
                batch.append(index)
                if len(batch) == self.batch_size:
                    tail_batches.append(batch)
                    batch = []
        for index in torch.randperm(len(tail_batches)).tolist():
            yield tail_batches[index]
        if batch:
            yield batch

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class SynthTigerDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 64,
        num_workers: int = 8,
        val_fraction: float = 0.01,
        test_fraction: float = 0.01,
        split_seed: int = 42,
        load_features: bool = False,
        features_dir: Path | None = None,
        patch_bucket_size: int = 1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.split_seed = split_seed
        self.load_features = load_features
        self.patch_bucket_size = patch_bucket_size
        self.features_dir = (
            Path(features_dir)
            if features_dir is not None
            else self.data_dir.with_name(f"{self.data_dir.name}-features-fp16")
        )

        self.normalize = v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def _dataset(self, split: str) -> SynthTiger:
        return SynthTiger(
            self.data_dir / "gt.db",
            index_filter=self._split_filter(split),
            transform=None if self.load_features else self.normalize,
            features_dir=self.features_dir if self.load_features else None,
        )

    def _split_filter(self, split: str) -> tuple[str, tuple[int, ...]]:
        score = "((idx * 2654435761 + ?) % 4294967296)"
        seed_offset = (self.split_seed * 1013904223) % 2**32
        val_threshold = int(self.val_fraction * 2**32)
        test_threshold = int((self.val_fraction + self.test_fraction) * 2**32)

        if split == "val":
            return f"AND {score} < ?", (seed_offset, val_threshold)
        if split == "test":
            return (
                f"AND {score} >= ? AND {score} < ?",
                (seed_offset, val_threshold, seed_offset, test_threshold),
            )
        return f"AND {score} >= ?", (seed_offset, test_threshold)

    @override
    def setup(self, stage):
        if stage in "fit":
            self.train = self._dataset("train")

        if stage in ("fit", "validate"):
            self.val = self._dataset("val")

        if stage in "test":
            self.test = self._dataset("test")

    @override
    def train_dataloader(self) -> StatefulDataLoader:
        return self._dataloader(self.train, shuffle=True)

    @override
    def val_dataloader(self) -> StatefulDataLoader | None:
        return self._dataloader(self.val, shuffle=False)

    @override
    def test_dataloader(self) -> StatefulDataLoader:
        return self._dataloader(self.test, shuffle=False)

    def _dataloader(self, dataset: SynthTiger, *, shuffle: bool) -> StatefulDataLoader:
        batching = (
            {
                "batch_sampler": PatchBucketBatchSampler(
                    dataset, self.batch_size, self.patch_bucket_size
                )
            }
            if shuffle
            else {
                "batch_size": self.batch_size,
                "sampler": SplitIndexSampler(dataset, shuffle=False),
            }
        )
        return StatefulDataLoader(
            dataset,
            collate_fn=collate_features if self.load_features else collate_images,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            **batching,
        )
