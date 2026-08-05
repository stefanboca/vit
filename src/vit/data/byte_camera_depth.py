from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import decode_image
from torchvision.transforms import v2


@dataclass
class FrameInfo:
    color: Path
    depth: Path


class BCDSample(TypedDict):
    color: Tensor
    depth: NotRequired[Tensor]
    hole_mask: NotRequired[Tensor]


class ByteCameraDepth(Dataset):
    def __init__(
        self,
        data_dir: Path,
        cameras: str | list[str],
        depth: bool = True,
        hole_mask: bool = False,
        transform=None,
    ):
        super().__init__()
        self.depth = depth
        self.hole_mask = hole_mask
        self.transform = transform

        if isinstance(cameras, str):
            cameras = [cameras]

        self.frames: list[FrameInfo] = []
        for session_dir in sorted(data_dir.iterdir()):
            for camera in cameras:
                camera_dir = session_dir / f"camera_{camera}"
                if not camera_dir.exists():
                    continue
                color_imgs = sorted(camera_dir.glob("color_*.png"))
                depth_imgs = sorted(camera_dir.glob("depth_*.png"))
                self.frames.extend(
                    FrameInfo(f[0], f[1]) for f in zip(color_imgs, depth_imgs)
                )

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index: int) -> BCDSample:
        frame = self.frames[index]

        color = v2.functional.to_dtype(decode_image(str(frame.color)), scale=True)
        sample: BCDSample = {"color": tv_tensors.Image(color)}

        if self.depth or self.hole_mask:
            depth = decode_image(str(frame.depth))
            if self.depth:
                sample["depth"] = tv_tensors.Mask(depth)
            if self.hole_mask:
                sample["hole_mask"] = tv_tensors.Mask((depth == 0).to(torch.long))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
