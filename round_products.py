#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Round MICAPS4 product grid values to integers in-place.

Reads every m4 file under the product directory, keeps the 3-line header
(title, datetime, grid parameters) unchanged, and rounds each grid value to
the nearest integer.

Usage:
    python round_products.py
    python round_products.py --product-dir <PRODUCT_DIR>

Founded in 2026-07-17
Modified in 2026-07-21
@author: yinlb
"""

import argparse
import pathlib
import sys
import typing

import arrow

from src import utils


def _is_day_folder(name: str) -> bool:
    """Check if a directory name is a YYYYMMDD day string."""
    return len(name) == 8 and name.isdigit()


def _round_file(path: pathlib.Path, encoding: str = 'GBK') -> None:
    """Round all grid values in a MICAPS4 file to integers.

    The MICAPS4 format has a fixed 3-line header followed by the grid data
    rows. Only the data rows are modified; the header is preserved exactly.
    """
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()

    if len(lines) < 3:
        print(f'Skipping {path}: file has fewer than 3 header lines')
        return

    header = lines[:3]
    data_lines = lines[3:]

    rounded_data = list()
    for line in data_lines:
        stripped = line.strip()
        if not stripped:
            continue
        values = stripped.split()
        rounded = [str(int(round(float(v)))) for v in values]
        rounded_data.append(' '.join(rounded) + '\n')

    with open(path, 'w', encoding=encoding) as f:
        f.writelines(header)
        f.writelines(rounded_data)

    print(f'Rounded {path}')


def round_products(product_dir: pathlib.Path) -> None:
    """Round all m4 files under product_dir to integers in-place."""
    if not product_dir.exists():
        print(f'Product directory does not exist: {product_dir}')
        return

    for folder in sorted(product_dir.iterdir()):
        if not folder.is_dir() or not _is_day_folder(folder.name):
            continue

        for file_path in sorted(folder.iterdir()):
            if not file_path.is_file():
                continue
            _round_file(file_path)

    print('All done.')


def main() -> None:
    """Main entry point."""
    cfg = utils.load_config(operational=True)
    default_product_dir = cfg['operational']['paths']['product_dir']
    parser = argparse.ArgumentParser(
        description='Round MICAPS4 product grid values to integers in-place.'
    )
    parser.add_argument(
        '--product-dir', '-p', default=default_product_dir,
        help='Product directory to process (default: operational config product_dir)',
    )
    args = parser.parse_args()

    product_dir = pathlib.Path(args.product_dir)
    round_products(product_dir)


if __name__ == '__main__':
    main()
