from pathlib import Path
from typing import override

from lightning import LightningDataModule
from torchdata.stateful_dataloader import StatefulDataLoader
from torchvision.transforms import v2

from vit.data import ByteCameraDepth

BEDROOM_SCENES = {
    "20250228_1418",
    "20250228_2158",
    "20250306_2241",
}
BATHROOM_SCENES = {
    "20250301_2338",
    "20250306_2154",
    "20250306_2207",
}
LIVING_ROOM_SCENES = {
    "20250302_0023",
    "20250306_2142",
    "20250306_2148",
}
KITCHEN_SCENES = {
    "20250302_2352",
    "20250302_2355",
    "20250302_2357",
    "20250302_2358",
    "20250306_2246",
    "20250317_2359",
    "20250318_0008",
    "20250318_0011",
    "20250318_0013",
    "20250318_0014",
    "20250318_0016",
}
MARKET_SCENES = {
    "20250307_2323",
    "20250307_2327",
    "20250307_2331",
    "20250307_2337",
    "20250307_2339",
    "20250318_0029",
    "20250318_0032",
    "20250318_0034",
}
OFFICE_SCENES = {
    "20250307_2349",
    "20250318_0021",
    "20250318_0023",
    "20250320_2320",
    "20250320_2321",
    "20250320_2323",
    "20250320_2324",
    "20250320_2326",
    "20250320_2328",
}
BREAKROOM_SCENES = {
    "20250320_2331",
    "20250320_2334",
}

SCENE_CATEGORIES = (
    BEDROOM_SCENES,
    BATHROOM_SCENES,
    LIVING_ROOM_SCENES,
    KITCHEN_SCENES,
    MARKET_SCENES,
    OFFICE_SCENES,
    BREAKROOM_SCENES,
)
ALL_SCENES = set().union(*SCENE_CATEGORIES)
assert sum(map(len, SCENE_CATEGORIES)) == len(ALL_SCENES)

TRAIN_SCENES = LIVING_ROOM_SCENES | KITCHEN_SCENES | MARKET_SCENES | OFFICE_SCENES
VAL_SCENES = BEDROOM_SCENES | BREAKROOM_SCENES
TEST_SCENES = BATHROOM_SCENES


class ByteCameraDepthDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        size: tuple[int, int] = (320, 512),
        depth: bool = True,
        holes: bool = False,
        batch_size: int = 24,
        num_workers: int = 16,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.depth = depth
        self.holes = holes
        self.batch_size = batch_size
        self.num_workers = num_workers

        ratio = size[1] / size[0]
        self.resize = v2.Resize(size, antialias=True)
        self.augment = v2.Compose(
            [
                v2.RandomResizedCrop(
                    size=size,
                    scale=(0.5, 1),
                    ratio=(ratio, ratio),
                    antialias=True,
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

        available_scenes = {
            scene.name for scene in self.data_dir.iterdir() if scene.is_dir()
        }
        if available_scenes != ALL_SCENES:
            raise ValueError(
                "scene split does not match available scenes; "
                f"\tmissing={sorted(ALL_SCENES - available_scenes)}, "
                f"\tunexpected={sorted(available_scenes - ALL_SCENES)}"
            )

        self.train_scenes = sorted(TRAIN_SCENES)
        self.val_scenes = sorted(VAL_SCENES)
        self.test_scenes = sorted(TEST_SCENES)

    @override
    def setup(self, stage):
        if stage == "fit":
            self.train = ByteCameraDepth(
                data_dir=self.data_dir,
                scenes=self.train_scenes,
                depth=self.depth,
                holes=self.holes,
                transform=v2.Compose([self.augment, self.normalize]),
            )

        if stage in ("fit", "validate"):
            self.val = ByteCameraDepth(
                data_dir=self.data_dir,
                scenes=self.val_scenes,
                depth=self.depth,
                holes=self.holes,
                transform=v2.Compose([self.resize, self.normalize]),
            )

        if stage == "test":
            self.test = ByteCameraDepth(
                data_dir=self.data_dir,
                scenes=self.test_scenes,
                depth=self.depth,
                holes=self.holes,
                transform=v2.Compose([self.resize, self.normalize]),
            )

    @override
    def train_dataloader(self) -> StatefulDataLoader:
        return StatefulDataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    @override
    def val_dataloader(self) -> StatefulDataLoader:
        return StatefulDataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    @override
    def test_dataloader(self) -> StatefulDataLoader:
        return StatefulDataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )
