import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode, v2
from tqdm import tqdm

from vit.models.seg import SegVit
from vit.models.seg.data import ImageFolderWithNames, collate_named_images


def encode_rle(mask: Tensor) -> list[int]:
    values, counts = torch.unique_consecutive(mask.bool().flatten(), return_counts=True)
    return torch.stack((counts, values.long()), dim=1).flatten().cpu().tolist()


def predict(args: Namespace):
    device = torch.device(args.device)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SegVit.load_from_checkpoint(
        args.ckpt, map_location="cpu", max_objects=args.max_objects
    ).to(device)
    model.eval()

    transform = v2.Compose(
        [
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            v2.ToPureTensor(),
        ]
    )
    dataset = ImageFolderWithNames(args.image_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_named_images,
    )

    for images, names in tqdm(dataloader):
        height, width = images.shape[-2:]
        with torch.no_grad():
            model_images = v2.functional.resize(
                images.to(device, non_blocking=True),
                [512, 640],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            predictions = [
                [
                    v2.functional.resize(
                        tv_tensors.Mask(mask),
                        [height, width],
                        interpolation=InterpolationMode.NEAREST_EXACT,
                    )
                    for mask in masks
                ]
                for masks in model.forward(model_images)
            ]

        for name, masks in zip(names, predictions, strict=True):
            assert all(mask.shape == (height, width) for mask in masks)
            path = (out_dir / name).with_suffix(".json")
            with open(path, "w") as f:
                json.dump(
                    {
                        "image": name,
                        "prompt": "",
                        "objects": [
                            {
                                "mask_rle": encode_rle(mask),
                                "height": height,
                                "width": width,
                            }
                            for mask in masks
                        ],
                    },
                    f,
                )


def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--image-dir", type=Path)
    predict_parser.add_argument("--out-dir", type=Path)
    predict_parser.add_argument("--ckpt", type=Path)
    predict_parser.add_argument("--batch-size", type=int, default=8)
    predict_parser.add_argument("--num-workers", type=int, default=16)
    predict_parser.add_argument("--max-objects", type=int, default=12)
    predict_parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    predict_parser.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
