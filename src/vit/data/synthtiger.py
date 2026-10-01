import argparse
import os
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import batched
from pathlib import Path
from typing import override

import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.io import decode_image
from torchvision.transforms import v2
from tqdm import tqdm

from vit.utils.same_size_batch_sampler import (
    ImageSize,
    SameSizeBatchSampler,
    SizedImageDataset,
)

DB_BATCH_SIZE = 4096
MAX_IN_FLIGHT = 4 * DB_BATCH_SIZE
DEFAULT_OCR_BACKBONE = "vit_small_plus_patch16_dinov3"
OCR_PATCH_SIZE = 16


class SynthTiger(SizedImageDataset):
    """SynthTIGER OCR dataset backed by a SQLite ``gt.db`` index."""

    def __init__(
        self,
        db_path: Path,
        image_root: Path | None = None,
        transform=None,
        *,
        index_filter: tuple[str, tuple[int, ...]] = ("", ()),
        features_dir: Path | None = None,
    ):
        self.db_path = Path(db_path)
        self.image_root = (
            Path(image_root) if image_root is not None else self.db_path.parent
        )
        self.transform = transform
        self.index_filter = index_filter
        self.features_dir = Path(features_dir) if features_dir is not None else None

        self._db: sqlite3.Connection | None = None
        self._db_pid: int | None = None

    def __len__(self) -> int:
        where, parameters = self.index_filter
        (count,) = (
            self._connection()
            .execute(f"SELECT COUNT(*) FROM items WHERE 1 = 1 {where}", parameters)
            .fetchone()
        )
        return count

    def _connection(self) -> sqlite3.Connection:
        pid = os.getpid()
        if self._db is None or self._db_pid != pid:
            if self._db is not None:
                self._db.close()
            self._db = sqlite3.connect(
                f"{self.db_path.resolve().as_uri()}?mode=ro&immutable=1",
                uri=True,
            )
            self._db_pid = pid
        return self._db

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_db"] = None
        state["_db_pid"] = None
        return state

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_or_features, label, _path = self._get_item(index)
        return image_or_features, label

    def get_with_path(self, index: int) -> tuple[torch.Tensor, str, str]:
        """Return an image or cached features, label, and database path."""
        return self._get_item(index)

    def _get_item(self, index: int) -> tuple[torch.Tensor, str, str]:
        row = (
            self._connection()
            .execute(
                "SELECT path, label FROM items WHERE idx = ?",
                (index + 1,),
            )
            .fetchone()
        )
        if row is None:
            raise IndexError(index)

        path, label = row
        if self.features_dir is not None:
            features = torch.load(
                self.features_dir / f"{path}.pt",
                map_location="cpu",
                weights_only=True,
            )
            return features, label, path

        image = v2.functional.to_dtype(
            decode_image(str(self.image_root / path)), scale=True
        )
        if self.transform is not None:
            image = self.transform(image)
        return image, label, path

    @override
    def size_counts(self) -> Iterator[tuple[ImageSize, int]]:
        where, parameters = self.index_filter
        for width, height, count in self._connection().execute(
            "SELECT width, height, COUNT(*) FROM items "
            f"WHERE 1 = 1 {where} "
            "GROUP BY width, height ORDER BY width, height",
            parameters,
        ):
            yield (width, height), count

    def indices(self, *, shuffle: bool) -> Iterator[int]:
        where, parameters = self.index_filter
        order = "random()" if shuffle else "idx"
        for (index,) in self._connection().execute(
            f"SELECT idx FROM items WHERE 1 = 1 {where} ORDER BY {order}",
            parameters,
        ):
            yield index - 1

    def indices_with_patch_counts(self, *, shuffle: bool) -> Iterator[tuple[int, int]]:
        where, parameters = self.index_filter
        order = "random()" if shuffle else "idx"
        for index, width, height in self._connection().execute(
            f"SELECT idx, width, height FROM items WHERE 1 = 1 {where} ORDER BY {order}",
            parameters,
        ):
            yield index - 1, (width // OCR_PATCH_SIZE) * (height // OCR_PATCH_SIZE)

    @override
    def indices_for_size(
        self,
        size: ImageSize,
        *,
        shuffle: bool,
        generator: torch.Generator | None,
    ):
        # SQLite performs the ordering, so Python only holds one batch at a time.
        # SQLite's random() uses its own RNG; generator is accepted to satisfy the
        # sampler interface but cannot seed this ordering.
        order = "random()" if shuffle else "idx"
        where, split_parameters = self.index_filter
        for (index,) in self._connection().execute(
            "SELECT idx FROM items WHERE width = ? AND height = ? "
            f"{where} ORDER BY {order}",
            (*size, *split_parameters),
        ):
            yield index - 1


def process_gt_txt_line(source: Path, prefix: Path, line: str):
    delim_idx = line.find("\t")
    path = line[:delim_idx].strip()
    label = line[delim_idx:].strip()
    with Image.open(source / path) as img:
        width, height = img.size
    return str(prefix / path), label, width, height


def completed_gt_txt_rows(
    executor: ThreadPoolExecutor, source: Path, prefix: Path, lines
):
    line_iter = iter(lines)
    pending = set()

    def submit_next() -> bool:
        try:
            line = next(line_iter)
        except StopIteration:
            return False
        pending.add(executor.submit(process_gt_txt_line, source, prefix, line))
        return True

    for _ in range(MAX_IN_FLIGHT):
        if not submit_next():
            break

    while pending:
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            pending.remove(future)
            result = future.result()
            submit_next()
            yield result


def gen_db(args):
    if not args.dataset.exists():
        print("dataset directory does not exist", file=sys.stderr)
        return
    if not args.dataset.is_dir():
        print("dataset is not a directory", file=sys.stderr)
        return

    sources = (
        [(args.dataset, Path("."))]
        if (args.dataset / "gt.txt").is_file()
        else [
            (split_dir, Path(split_dir.name))
            for split_dir in sorted(args.dataset.iterdir())
            if split_dir.is_dir() and (split_dir / "gt.txt").is_file()
        ]
    )
    if not sources:
        print(
            "no gt.txt found in dataset directory or its child directories",
            file=sys.stderr,
        )
        return

    db_file: Path = args.dataset / "gt.db"
    if db_file.exists():
        db_file.unlink()
    db = sqlite3.connect(db_file)
    db.execute("PRAGMA journal_mode = OFF")
    db.execute("PRAGMA synchronous = OFF")
    db.execute("PRAGMA locking_mode = EXCLUSIVE")
    db.execute("PRAGMA temp_store = MEMORY")
    db.execute(
        "CREATE TABLE items("
        "idx INTEGER PRIMARY KEY,"
        "path TEXT,"
        "label TEXT,"
        "width INTEGER,"
        "height INTEGER"
        ");"
    )

    with db, ThreadPoolExecutor(max_workers=16) as executor:
        for source, prefix in sources:
            with (source / "gt.txt").open() as f:
                total = sum(1 for _ in f)
                f.seek(0)
                results = tqdm(
                    completed_gt_txt_rows(executor, source, prefix, f),
                    desc=f"Indexing {prefix}",
                    total=total,
                )
                for items in batched(results, DB_BATCH_SIZE):
                    db.executemany(
                        "INSERT INTO items(path, label, width, height) VALUES(?, ?, ?, ?)",
                        items,
                    )
        print("Committing DB transaction...")

    print("Creating indices...")
    db.execute("CREATE INDEX items_size_idx ON items(width, height, idx)")


class _FeatureCacheDataset(SizedImageDataset):
    """Image dataset view that samples only items missing cached features."""

    def __init__(self, dataset: SynthTiger, features_dir: Path):
        self.dataset = dataset
        self.features_dir = features_dir
        self._missing_size_counts: tuple[tuple[ImageSize, int], ...] | None = None

    def __len__(self):
        return len(self.dataset)

    @override
    def __getitem__(self, index: int):
        image, _label, path = self.dataset.get_with_path(index)
        return path, image

    @override
    def size_counts(self) -> Iterator[tuple[ImageSize, int]]:
        if self._missing_size_counts is None:
            counts: Counter[ImageSize] = Counter()
            for path, width, height in self.dataset._connection().execute(
                "SELECT path, width, height FROM items"
            ):
                feature_path = self.features_dir / f"{path}.pt"
                if not feature_path.is_file():
                    counts[(width, height)] += 1
            self._missing_size_counts = tuple(sorted(counts.items()))
        yield from self._missing_size_counts

    @override
    def indices_for_size(
        self,
        size: ImageSize,
        *,
        shuffle: bool,
        generator: torch.Generator | None,
    ) -> Iterator[int]:
        order = "random()" if shuffle else "idx"
        for index, path in self.dataset._connection().execute(
            "SELECT idx, path FROM items WHERE width = ? AND height = ? "
            f"ORDER BY {order}",
            size,
        ):
            if not (self.features_dir / f"{path}.pt").is_file():
                yield index - 1


def save_tensor_atomic(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("wb") as file:
            torch.save(tensor, file)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_feature_batch(
    paths: list[str], features: torch.Tensor, features_dir: Path
) -> int:
    for path, feature in zip(paths, features, strict=True):
        # Each batch row is a view into the full batch storage. Clone so each
        # .pt contains only that image's feature tensor.
        save_tensor_atomic(feature.clone(), features_dir / f"{path}.pt")
    return len(paths)


@torch.inference_mode()
def cache_features(args):
    data_dir: Path = args.dataset
    features_dir: Path = args.features_dir
    features_dir.mkdir(parents=True, exist_ok=True)

    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    images = SynthTiger(data_dir / "gt.db", transform=normalize)
    dataset = _FeatureCacheDataset(images, features_dir)
    missing_count = sum(count for _, count in dataset.size_counts())
    if missing_count == 0:
        print("All image features are already cached.")
        return

    loader = DataLoader(
        dataset,
        batch_sampler=SameSizeBatchSampler(dataset, args.batch_size, shuffle=False),
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    backbone = timm.create_model(args.backbone, pretrained=True).to(args.device)
    backbone.eval()

    pending = set()
    with (
        ThreadPoolExecutor(max_workers=args.writer_workers) as executor,
        tqdm(
            total=missing_count, desc="Caching backbone features", unit="image"
        ) as progress,
    ):
        for paths, images in loader:
            images = images.to(args.device, non_blocking=True)
            if images.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    features = backbone.forward_features(images)  # ty: ignore[call-non-callable]
            else:
                features = backbone.forward_features(images)  # ty: ignore[call-non-callable]
            features = features.to(device="cpu", dtype=torch.float16)

            pending.add(
                executor.submit(save_feature_batch, paths, features, features_dir)
            )
            if len(pending) >= args.max_in_flight_batches:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    progress.update(future.result())

        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                progress.update(future.result())


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    gen_db_parser = subparsers.add_parser("gen-db")
    gen_db_parser.add_argument("dataset", type=Path)
    gen_db_parser.set_defaults(func=gen_db)

    cache_parser = subparsers.add_parser("cache-features")
    cache_parser.add_argument("dataset", type=Path)
    cache_parser.add_argument("--features-dir", type=Path, default=None)
    cache_parser.add_argument("--backbone", default=DEFAULT_OCR_BACKBONE)
    cache_parser.add_argument("--batch-size", type=int, default=256)
    cache_parser.add_argument("--num-workers", type=int, default=16)
    cache_parser.add_argument("--writer-workers", type=int, default=2)
    cache_parser.add_argument("--max-in-flight-batches", type=int, default=2)
    cache_parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    cache_parser.set_defaults(func=cache_features)

    args = parser.parse_args()
    if getattr(args, "features_dir", None) is None:
        args.features_dir = args.dataset.with_name(f"{args.dataset.name}-features-fp16")
    args.func(args)


if __name__ == "__main__":
    main()
