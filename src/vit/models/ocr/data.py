from pathlib import Path
from typing import override

from lightning import LightningDataModule
from torchdata.stateful_dataloader import StatefulDataLoader

from vit.data.synthtiger import SynthTiger
from vit.utils.same_size_batch_sampler import SameSizeBatchSampler


class SynthTigerDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 64,
        num_workers: int = 8,
        val_fraction: float = 0.01,
        test_fraction: float = 0.01,
        split_seed: int = 42,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.split_seed = split_seed

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
            self.train = SynthTiger(
                self.data_dir / "gt.db",
                index_filter=self._split_filter("train"),
            )

        if stage in ("fit", "validate"):
            self.val = SynthTiger(
                self.data_dir / "gt.db",
                index_filter=self._split_filter("val"),
            )

        if stage in "test":
            self.test = SynthTiger(
                self.data_dir / "gt.db",
                index_filter=self._split_filter("test"),
            )

    @override
    def train_dataloader(self) -> StatefulDataLoader:
        return StatefulDataLoader(
            self.train,
            batch_sampler=SameSizeBatchSampler(
                self.train,
                self.batch_size,
                shuffle=True,
            ),
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    @override
    def val_dataloader(self) -> StatefulDataLoader | None:
        return StatefulDataLoader(
            self.val,
            batch_sampler=SameSizeBatchSampler(
                self.val,
                self.batch_size,
                shuffle=False,
            ),
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    @override
    def test_dataloader(self) -> StatefulDataLoader:
        return StatefulDataLoader(
            self.test,
            batch_sampler=SameSizeBatchSampler(
                self.test,
                self.batch_size,
                shuffle=False,
            ),
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
