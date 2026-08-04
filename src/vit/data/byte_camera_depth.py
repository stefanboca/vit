from dataclasses import dataclass
from pathlib import Path

from icecream import ic
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import decode_image


@dataclass
class FrameInfo:
    color: Path
    depth: Path


@dataclass
class Frame:
    color: Tensor
    depth: Tensor


class ByteCameraDepth(Dataset):
    def __init__(self, data_dir: Path, cameras: str | list[str]):
        super().__init__()
        if isinstance(cameras, str):
            cameras = [cameras]

        self.frames: list[FrameInfo] = []
        for session_dir in data_dir.iterdir():
            for camera in cameras:
                camera_dir = session_dir / f"camera_{camera}"
                if not camera_dir.exists():
                    continue
                color_imgs = sorted(camera_dir.glob("color_*.png"))
                depth_imgs = sorted(camera_dir.glob("depth_*.png"))
                self.frames.extend(
                    FrameInfo(f[0], f[1]) for f in zip(color_imgs, depth_imgs)
                )
        ic(self.frames[:10])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index) -> Frame:
        frame = self.frames[index]
        color_img = decode_image(frame.color)
        depth_img = decode_image(frame.depth)
        return Frame(color_img, depth_img)
