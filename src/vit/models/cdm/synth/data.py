from pathlib import Path
from typing import override

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from vit.data import ByteCameraDepth


class ByteCameraDepthDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        depth: bool = True,
        hole_mask: bool = False,
        batch_size: int = 24,
        num_workers: int = 16,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.depth = depth
        self.hole_mask = hole_mask
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.resize = v2.Resize((320, 512), antialias=True)
        self.augment = v2.Compose(
            [
                v2.RandomChoice(
                    [
                        self.resize,
                        v2.RandomCrop((320, 512), pad_if_needed=True),
                    ],
                ),
                v2.RandomHorizontalFlip(),
            ]
        )
        self.normalize = v2.Compose(
            [
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                v2.ToPureTensor(),
            ]
        )

        scenes = sorted(
            scene.name for scene in self.data_dir.iterdir() if scene.is_dir()
        )
        assert len(scenes) == 39
        self.train_scenes = scenes[:-6]
        self.val_scenes = scenes[-6:-3]
        self.test_scenes = scenes[-3:]

    @override
    def setup(self, stage):
        if stage == "fit":
            self.train = ByteCameraDepth(
                data_dir=self.data_dir,
                scenes=self.train_scenes,
                depth=self.depth,
                hole_mask=self.hole_mask,
                transform=v2.Compose([self.augment, self.normalize]),
            )

        if stage in ("fit", "validate"):
            self.val = ByteCameraDepth(
                data_dir=self.data_dir,
                scenes=self.val_scenes,
                depth=self.depth,
                hole_mask=self.hole_mask,
                transform=v2.Compose([self.resize, self.normalize]),
            )

        if stage == "test":
            self.test = ByteCameraDepth(
                data_dir=self.data_dir,
                scenes=self.test_scenes,
                depth=self.depth,
                hole_mask=self.hole_mask,
                transform=v2.Compose([self.resize, self.normalize]),
            )

    @override
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    @override
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    @override
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )
