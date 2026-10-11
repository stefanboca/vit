import json
from pathlib import Path
from typing import override

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import decode_image
from torchvision.transforms import v2


class Rle(Dataset):
    def __init__(self, annotations_dir: Path, image_dir: Path, transform=None):
        self.transform = transform
        self.image_dir = image_dir
        self.annotation_files = sorted(annotations_dir.glob("*.json"))

    def __len__(self) -> int:
        return len(self.annotation_files)

    @override
    def __getitem__(self, index: int):
        annotation_file = self.annotation_files[index]
        with open(annotation_file, "r") as f:
            annotation = json.load(f)
        image = v2.functional.to_dtype(
            decode_image(self.image_dir / annotation["image"]),
            dtype=torch.float32,
            scale=True,
        )
        _, H, W = image.shape
        masks = []
        for object in annotation["objects"]:
            assert (object["height"], object["width"]) == (H, W)
            masks.append(decode_rle(object["mask_rle"], H, W))
        mask_tensor = (
            torch.stack(masks) if masks else torch.empty((0, H, W), dtype=torch.bool)
        )
        item = tv_tensors.Image(image), tv_tensors.Mask(mask_tensor)

        if self.transform is not None:
            item = self.transform(item)

        return item


def decode_rle(rle: list[int], height: int, width: int) -> Tensor:
    pairs = torch.as_tensor(rle, dtype=torch.long).reshape(-1, 2)
    counts, values = pairs.T
    return torch.repeat_interleave(values.bool(), counts).reshape(height, width)
