import argparse
import colorsys
import json
import math
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import onnx
import pycocotools.mask
import supervision as sv
import torch
from PIL import Image
from rfdetr import RFDETR, RFDETRSeg2XLarge
from rfdetr.export.prepare import prepare_export_graph
from tqdm import tqdm

from vit.data.rle import decode_rle


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


def split_coco(args: Namespace):
    ratios: tuple[float, float, float] = args.ratios
    assert all(
        math.isfinite(ratio) and ratio >= 0 for ratio in ratios
    ) and math.isclose(sum(ratios), 1.0)

    with args.annotations.open() as f:
        dataset = json.load(f)
    images = list(dataset["images"])
    image_ids = {image["id"] for image in images}
    assert len(image_ids) == len(images)
    annotations_by_image = {image_id: [] for image_id in image_ids}
    for annotation in dataset["annotations"]:
        assert annotation["image_id"] in image_ids
        annotations_by_image[annotation["image_id"]].append(annotation)

    image_dir = args.image_dir or args.annotations.parent
    image_paths = {}
    file_names = set()
    for image in images:
        file_name = Path(image["file_name"])
        file_names.add(file_name)
        image_path = (image_dir / file_name).resolve()
        assert image_path.is_file()
        image_paths[image["id"]] = image_path

    random.Random(args.seed).shuffle(images)
    train_end = int(len(images) * ratios[0])
    valid_end = int(len(images) * (ratios[0] + ratios[1]))
    splits = (
        ("train", images[:train_end]),
        ("valid", images[train_end:valid_end]),
        ("test", images[valid_end:]),
    )
    args.out_dir.mkdir(parents=True)
    for split, split_images in splits:
        split_dir = args.out_dir / split
        split_dir.mkdir()
        split_annotations = [
            annotation
            for image in split_images
            for annotation in annotations_by_image[image["id"]]
        ]
        for image in split_images:
            destination = split_dir / image["file_name"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(image_paths[image["id"]])
        split_dataset = {
            **dataset,
            "images": split_images,
            "annotations": split_annotations,
        }
        with (split_dir / "_annotations.coco.json").open("w") as f:
            json.dump(split_dataset, f)
        print(
            f"{split}: {len(split_images)} images, {len(split_annotations)} annotations"
        )


def predict(args: Namespace):
    Image.init()
    image_paths = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in Image.registered_extensions()
    )
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}")
    if args.image_dir.resolve() == args.out_dir.resolve():
        raise ValueError("--out-dir must differ from --image-dir")

    model = RFDETR.from_checkpoint(args.ckpt, device=args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_paths, desc="RF-DETR predictions"):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        detections = model.predict(image, threshold=args.threshold)
        # Spread successive instance colors around the hue wheel without repeating
        # the default palette when an image contains many instances.
        colors = []
        for index in range(max(1, len(detections))):
            rgb = colorsys.hsv_to_rgb((index * 0.61803398875) % 1, 0.75, 1)
            colors.append(sv.Color(*(round(channel * 255) for channel in rgb)))
        palette = sv.ColorPalette(colors=colors)
        annotator_kwargs = {"color": palette, "color_lookup": sv.ColorLookup.INDEX}
        labels = [
            f"{name} {confidence:.2f}"
            for name, confidence in zip(
                detections.data["class_name"],  # ty: ignore[unresolved-attribute]
                detections.confidence,  # ty: ignore[unresolved-attribute, invalid-argument-type]
                strict=True,
            )
        ]
        # Supervision's NumPy scenes use BGR; RF-DETR receives the RGB PIL image.
        scene = np.array(image)[:, :, ::-1].copy()
        scene = sv.MaskAnnotator(**annotator_kwargs).annotate(scene, detections)  # ty: ignore[invalid-argument-type]
        scene = sv.BoxAnnotator(**annotator_kwargs).annotate(scene, detections)  # ty: ignore[invalid-argument-type]
        scene = sv.LabelAnnotator(**annotator_kwargs).annotate(  # ty: ignore[invalid-argument-type]
            scene,
            detections,  # ty: ignore[invalid-argument-type]
            labels=labels,
        )
        Image.fromarray(scene[:, :, ::-1]).save(args.out_dir / image_path.name)


def train(args: Namespace):
    model = RFDETRSeg2XLarge(
        resolution=504,
    )
    model.train(
        dataset_dir=args.dataset_dir,
        output_dir=args.out_dir,
        resume=args.resume,
        device=torch.device(args.device),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-4,
        lr_encoder=1e-4,
        num_workers=args.num_workers,
        square_resize_div_64=True,
        multi_scale=False,
        expanded_scales=False,
        progress_bar="rich",
        tensorboard=True,
        train_log_on_step=True,
        compute_train_metrics=True,
    )


def export(args: Namespace):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = RFDETR.from_checkpoint(args.ckpt, device=args.device)
    export_model = model.model.model
    assert export_model is not None
    graph = prepare_export_graph(
        export_model,
        model.model_config,
        shape=(model.model.resolution, model.model.resolution),
        device=args.device,
        infer_dir=str(args.sample_image) if args.sample_image is not None else None,
        batch_size=1,
        dynamic_batch=False,
    )
    # RF-DETR's public ONNX exporter forces the legacy TorchScript path. Reuse
    # its preparation, including frozen position embeddings, then trace directly.
    export_model.export()
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    export_model.to(device=args.device, dtype=dtype)
    input_sample = graph.input_tensors.to(device=args.device, dtype=dtype)
    output_path = out_dir / f"{model.size}_{args.precision}.onnx"

    with torch.no_grad():
        torch.onnx.export(
            export_model,
            (input_sample,),
            output_path,
            input_names=list(graph.input_names),
            output_names=list(graph.output_names),
            opset_version=args.opset_version,
            dynamo=True,
            external_data=False,
            verify=args.verify,
            report=True,
            artifacts_dir=str(out_dir),
        )
    onnx.checker.check_model(str(output_path), full_check=True)
    print(f"Exported {args.precision} ONNX model to {output_path}")


def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

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

    split_coco_parser = subparsers.add_parser("split-coco")
    split_coco_parser.add_argument("--annotations", type=Path, required=True)
    split_coco_parser.add_argument("--image-dir", type=Path)
    split_coco_parser.add_argument("--out-dir", type=Path, required=True)
    split_coco_parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.94, 0.05, 0.01),
        metavar=("TRAIN", "VALID", "TEST"),
    )
    split_coco_parser.add_argument("--seed", type=int, default=42)
    split_coco_parser.set_defaults(func=split_coco)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--image-dir", type=Path, required=True)
    predict_parser.add_argument("--out-dir", type=Path, required=True)
    predict_parser.add_argument("--ckpt", type=Path, required=True)
    predict_parser.add_argument("--threshold", type=float, default=0.5)
    predict_parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    predict_parser.set_defaults(func=predict)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset-dir", type=Path, required=True)
    train_parser.add_argument("--out-dir", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument(
        "--batch-size",
        type=lambda value: value if value == "auto" else int(value),
        default="auto",
    )
    train_parser.add_argument("--num-workers", type=int, default=8)
    train_parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    train_parser.add_argument("--resume", type=Path)
    train_parser.set_defaults(func=train)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--ckpt", type=Path, required=True)
    export_parser.add_argument("--out-dir", type=Path, required=True)
    export_parser.add_argument("--opset-version", type=int, default=24)
    export_parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    export_parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    export_parser.add_argument("--sample-image", type=Path)
    export_parser.add_argument(
        "--verify", action=argparse.BooleanOptionalAction, default=True
    )
    export_parser.set_defaults(func=export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
