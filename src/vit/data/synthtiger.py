import argparse
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def gen_db(args):
    if not args.dataset.exists():
        print("dataset directory does not exist", file=sys.stderr)
        return
    if not args.dataset.is_dir():
        print("dataset is not a directory", file=sys.stderr)
        return

    def process_line(line: str):
        delim_idx = line.find("\t")
        path = line[:delim_idx].strip()
        label = line[delim_idx:].strip()
        with Image.open(args.dataset / path) as img:
            width, height = img.size
        return str(path), label, width, height

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

    gt_file = args.dataset / "gt.txt"
    with db, gt_file.open() as f, ThreadPoolExecutor(max_workers=16) as executor:
        for items in batched(executor.map(process_line, tqdm(f.readlines())), 65_536):
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
