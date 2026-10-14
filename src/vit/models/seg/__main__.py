import argparse
import json
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import pycocotools.mask
import torch
from PIL import Image
from rfdetr import RFDETRSeg2XLarge
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode, v2
from tqdm import tqdm

from vit.data.rle import decode_rle
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


def convert_to_coco(args: Namespace):
    input_dirs: list[tuple[Path, Path]] = args.input_dirs
    out_dir: Path = args.out_dir

    categories = [{"id": 1, "name": "tire"}]

    annotation_files = []
    for image_dir, annotation_dir in input_dirs:
        annotation_files.extend(
            (annotation_file, image_dir)
            for annotation_file in annotation_dir.glob("**/*.json")
        )
    annotation_files.sort()

    random.seed(42)
    random.shuffle(annotation_files)
    train_valid_split_idx = int(len(annotation_files) * 0.93)
    valid_test_split_idx = int(len(annotation_files) * 0.98)
    train = annotation_files[:train_valid_split_idx]
    valid = annotation_files[train_valid_split_idx:valid_test_split_idx]
    test = annotation_files[valid_test_split_idx:]
    for split, annotations in tqdm(
        (("train", train), ("valid", valid), ("test", test)), desc="Split"
    ):
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        dataset = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories,
        }
        annotation_id = 1
        for image_id, (annotation_file, image_dir) in enumerate(
            tqdm(annotations, desc=f"Annotations ({split})"), start=1
        ):
            with open(annotation_file, "r") as f:
                annotation = json.load(f)
            image_path = image_dir / annotation["image"]
            image_name: str = f"{image_id}_{annotation['image']}"
            (split_dir / image_name).hardlink_to(image_path)
            with Image.open(image_path) as image:
                width, height = image.size
            dataset["images"].append(
                {
                    "id": image_id,
                    "file_name": image_name,
                    "height": height,
                    "width": width,
                }
            )
            for object in annotation["objects"]:
                assert (object["height"], object["width"]) == (height, width)
                mask = decode_rle(object["mask_rle"], height, width)

                rle = pycocotools.mask.encode(np.asfortranarray(mask, dtype=np.uint8))
                area = int(pycocotools.mask.area(rle))
                if area == 0:
                    raise ValueError("instance mask must contain foreground pixels")
                bbox = pycocotools.mask.toBbox(rle).tolist()
                rle["counts"] = rle["counts"].decode("ascii")
                dataset["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "segmentation": rle,
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
        with (split_dir / "_annotations.coco.json").open("w") as f:
            json.dump(dataset, f)


def rfdetr_train(args: Namespace):
    model = RFDETRSeg2XLarge(
        resolution=504,
    )
    model.train(
        dataset_dir=args.dataset_dir,
        output_dir=args.out_dir,
        resume=args.resume,
        class_names=["tire"],
        device=torch.device(args.device),
        epochs=20,
        batch_size="auto",
        lr=1e-4,
        lr_encoder=1e-4,
        num_workers=8,
        square_resize_div_64=False,
        multi_scale=False,
        expanded_scales=False,
        progress_bar="rich",
        tensorboard=True,
        train_log_on_step=True,
        compute_train_metrics=True,
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
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    predict_parser.set_defaults(func=predict)

    convert_to_coco_parser = subparsers.add_parser("convert-to-coco")
    convert_to_coco_parser.add_argument(
        "--input-dirs",
        nargs=2,
        action="append",
        type=Path,
        metavar=("IMAGE_DIR", "ANNOTATION_DIR"),
        required=True,
    )
    convert_to_coco_parser.add_argument("--out-dir", type=Path)
    convert_to_coco_parser.set_defaults(func=convert_to_coco)

    rfdetr_train_parser = subparsers.add_parser("rfdetr-train")
    rfdetr_train_parser.add_argument("--dataset-dir", type=Path, required=True)
    rfdetr_train_parser.add_argument("--out-dir", type=Path, required=True)
    rfdetr_train_parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    rfdetr_train_parser.add_argument("--resume", type=Path)
    rfdetr_train_parser.set_defaults(func=rfdetr_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
