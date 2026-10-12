import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr
from PIL import Image
from tqdm import tqdm

from vit.data.rle import decode_rle
from vit.models.seg.model import mask_colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--annotation-dir", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--max-objects", type=int, default=12)

    args = parser.parse_args()

    rr.init("seg")
    rr.save(args.output_file)

    image_dir: Path = args.image_dir
    annotation_dir: Path = args.annotation_dir

    colors = mask_colors(args.max_objects)
    rr.log(
        "/",
        rr.AnnotationContext(
            [(0, "background", (0, 0, 0, 0))]
            + [
                (i, f"object {i}", (*color, 255))
                for i, color in enumerate(colors, start=1)
            ]
        ),
        static=True,
    )

    for sample_idx, annotation_file in enumerate(
        tqdm(sorted(annotation_dir.glob("**/*.json")))
    ):
        with annotation_file.open("r") as f:
            annotation = json.load(f)

        rr.set_time("sample", sequence=sample_idx)
        rr.log("image", rr.EncodedImage(path=image_dir / annotation["image"]))

        labels = None
        for mask_idx, object in enumerate(annotation["objects"], start=1):
            height, width = object["height"], object["width"]
            if labels is None:
                labels = np.zeros((height, width), dtype=np.uint8)
            mask = decode_rle(object["mask_rle"], height, width)
            labels[mask.cpu().numpy()] = mask_idx
        if labels is not None:
            rr.log("image/masks", rr.SegmentationImage(labels, opacity=0.6))
        else:
            rr.log("image/masks", rr.Clear(recursive=False))


if __name__ == "__main__":
    main()
