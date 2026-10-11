from pathlib import Path
from typing import override

import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import ConcatDataset, Subset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchvision.transforms import v2

from vit.data import Rle


def collate_masks(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, list[Tensor]]:
    images, masks = zip(*batch, strict=True)
    return torch.stack(images), list(masks)


class RleDataModule(LightningDataModule):
    def __init__(
        self,
        data_dirs: list[tuple[Path, Path]],
        size: tuple[int, int] = (640, 512),
        batch_size: int = 8,
        num_workers: int = 16,
        val_fraction: float = 0.05,
        test_fraction: float = 0.02,
        split_seed: int = 42,
    ):
        super().__init__()
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.num_workers = num_workers

        ratio = size[1] / size[0]
        resize = v2.Resize(size, antialias=True)
        augment = v2.Compose(
            [
                v2.RandomResizedCrop(
                    size=size,
                    scale=(0.5, 1),
                    ratio=(ratio, ratio),
                    antialias=True,
                ),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
            ]
        )
        normalize = v2.Compose(
            [
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                v2.ToPureTensor(),
            ]
        )

        train_transform = v2.Compose([augment, normalize])
        val_transform = v2.Compose([resize, normalize])
        self.train_source = ConcatDataset(
            [
                Rle(annotations_dir, image_dir, transform=train_transform)
                for annotations_dir, image_dir in self.data_dirs
            ]
        )
        self.val_source = ConcatDataset(
            [
                Rle(annotations_dir, image_dir, transform=val_transform)
                for annotations_dir, image_dir in self.data_dirs
            ]
        )
        indices = torch.randperm(
            len(self.train_source), generator=torch.Generator().manual_seed(split_seed)
        ).tolist()
        val_count = int(val_fraction * len(indices))
        test_count = int(test_fraction * len(indices))
        train_count = len(indices) - val_count - test_count
        self.train_indices = indices[:train_count]
        self.val_indices = indices[train_count : train_count + val_count]
        self.test_indices = indices[train_count + val_count :]

    @override
    def setup(self, stage):
        if stage == "fit":
            self.train = Subset(self.train_source, self.train_indices)

        if stage in ("fit", "validate"):
            self.val = Subset(self.val_source, self.val_indices)

        if stage == "test":
            self.test = Subset(self.val_source, self.test_indices)

    @override
    def train_dataloader(self) -> StatefulDataLoader:
        return StatefulDataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_masks,
        )

    @override
    def val_dataloader(self) -> StatefulDataLoader:
        return StatefulDataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_masks,
        )

    @override
    def test_dataloader(self) -> StatefulDataLoader:
        return StatefulDataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_masks,
        )
