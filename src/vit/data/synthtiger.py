import argparse
import os
import sqlite3
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import batched
from pathlib import Path
from typing import override

import torch
from PIL import Image
from torchvision.io import decode_image
from torchvision.transforms import v2
from tqdm import tqdm

from vit.utils.same_size_batch_sampler import ImageSize, SizedImageDataset

DB_BATCH_SIZE = 4096
MAX_IN_FLIGHT = 4 * DB_BATCH_SIZE


class SynthTiger(SizedImageDataset):
    """SynthTIGER OCR dataset backed by a SQLite ``gt.db`` index."""

    def __init__(self, db_path: Path, image_root: Path | None = None, transform=None):
        self.db_path = Path(db_path)
        self.image_root = (
            Path(image_root) if image_root is not None else self.db_path.parent
        )
        self.transform = transform
        self._db: sqlite3.Connection | None = None
        self._db_pid: int | None = None

    def __len__(self) -> int:
        (count,) = self._connection().execute("SELECT COUNT(*) FROM items").fetchone()
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

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        row = (
            self._connection()
            .execute("SELECT path, label FROM items WHERE idx = ?", (index + 1,))
            .fetchone()
        )
        if row is None:
            raise IndexError(index)

        path, label = row
        image = v2.functional.to_dtype(
            decode_image(str(self.image_root / path)), scale=True
        )
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @override
    def size_counts(self):
        yield from self._connection().execute(
            "SELECT width, height, COUNT(*) FROM items "
            "GROUP BY width, height ORDER BY width, height"
        )

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
        for (index,) in self._connection().execute(
            f"SELECT idx FROM items WHERE width = ? AND height = ? ORDER BY {order}",
            size,
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
        "   idx INTEGER PRIMARY KEY,"
        "   path TEXT,"
        "   label TEXT,"
        "   width INTEGER,"
        "   height INTEGER"
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


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    gen_db_parser = subparsers.add_parser("gen-db")
    gen_db_parser.add_argument("dataset", type=Path)
    gen_db_parser.set_defaults(func=gen_db)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
