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
        val_data_dir: Path | None = None,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.val_data_dir = Path(val_data_dir) if val_data_dir is not None else None
        self.batch_size = batch_size
        self.num_workers = num_workers

    @override
    def setup(self, stage: str | None = None):
        if stage in (None, "fit") and self.train is None:
            self.train = SynthTiger(self.data_dir / "gt.db")

        if stage in (None, "fit", "validate"):
            self.val = SynthTiger(self.data_dir / "gt.db")

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
            ),
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
