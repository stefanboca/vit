import re
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import decode_image
from torchvision.transforms import v2

PNG_REGEX = re.compile(r"(color|depth)_(\d{1,4}).png")


@dataclass
class FrameInfo:
    image: Path
    depth: Path


class BCDSample(TypedDict):
    images: Tensor
    index: Tensor
    depth: NotRequired[Tensor]
    holes: NotRequired[Tensor]


class ByteCameraDepth(Dataset):
    def __init__(
        self,
        data_dir: Path,
        scenes: list[str] | None = None,
        cameras: list[str] | None = None,
        depth: bool = True,
        holes: bool = False,
        transform=None,
    ):
        super().__init__()
        self.depth = depth
        self.holes = holes
        self.transform = transform

        scene_dirs = (
            [data_dir / scene for scene in scenes]
            if scenes is not None
            else sorted(data_dir.iterdir())
        )

        if cameras is not None:
            cameras = [f"camera_{camera}" for camera in cameras]

        self.frames: list[FrameInfo] = []
        for scene_dir in scene_dirs:
            if not scene_dir.exists():
                print(f"warning: scene {scene_dir} does not exist")
                continue

            camera_dirs = (
                [scene_dir / camera for camera in cameras]
                if cameras is not None
                else sorted(scene_dir.iterdir())
            )

            for camera_dir in camera_dirs:
                if not camera_dir.exists():
                    print(f"warning: camera dir {camera_dir} does not exist")
                    continue
                images = {}
                depths = {}
                for png in camera_dir.glob("*.png"):
                    if (match := PNG_REGEX.fullmatch(png.name)) is not None:
                        ty = match.group(1)
                        idx = int(match.group(2))
                        if ty == "color":
                            images[idx] = png
                        elif ty == "depth":
                            depths[idx] = png
                self.frames.extend(
                    FrameInfo(images[idx], depths[idx])
                    for idx in sorted(set(images.keys()).intersection(depths.keys()))
                )

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index: int) -> BCDSample:
        frame = self.frames[index]

        image = v2.functional.to_dtype(decode_image(str(frame.image)), scale=True)
        sample: BCDSample = {
            "images": tv_tensors.Image(image),
            "index": torch.tensor(index),
        }

        if self.depth or self.holes:
            depth = decode_image(str(frame.depth))
            if self.depth:
                sample["depth"] = tv_tensors.Mask(depth.float())
            if self.holes:
                sample["holes"] = tv_tensors.Mask(depth == 0)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
