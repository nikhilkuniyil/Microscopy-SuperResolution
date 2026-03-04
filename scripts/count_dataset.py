from __future__ import annotations

import argparse
from pathlib import Path

EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".dib"}


def count_images(root: Path) -> int:
    return sum(
        1 for p in root.rglob("*")
        if p.suffix.lower() in EXTS and "masks" not in p.parts
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Count images per dataset subdirectory")
    parser.add_argument("--root", default="data/base", help="Root directory to scan")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Directory not found: {root}")

    total = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        count = count_images(d)
        total += count
        print(f"{d.name:<30} {count:>6} images")

    print(f"{'':->36}")
    print(f"{'TOTAL':<30} {total:>6} images")


if __name__ == "__main__":
    main()
