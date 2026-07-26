#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Repair old product structure to the new day-level folder layout.

Old structure:
    ops/products/<YYYYMMDD>HH/<PREFIX>-<init_time>.<NNN>

New structure:
    ops/products/<YYYYMMDD>/<init_time>.<NNN>

The configured filename prefix (m4_filename_prefix) is stripped from the
filename on disk, while the m4 title keeps the prefix.

Usage:
    python repair_products.py

Founded in 2026-07-17
Modified in 2026-07-21
@author: yinlb
"""

import pathlib
import shutil

import arrow

from src import utils


def _is_init_time_folder(name: str) -> bool:
    """Check if a directory name is a YYYYMMDDHH init-time string."""
    return len(name) == 10 and name.isdigit()


def _is_day_folder(name: str) -> bool:
    """Check if a directory name is a YYYYMMDD day string."""
    return len(name) == 8 and name.isdigit()


def _lead_from_filename(filename: str) -> int:
    """Parse lead hour from '.NNN' suffix."""
    suffix = pathlib.Path(filename).suffix
    if len(suffix) == 4 and suffix[1:].isdigit():
        return int(suffix[1:])
    raise ValueError(f'Cannot parse lead from filename: {filename}')


def _read_m4_header(path: pathlib.Path):
    """Read MICAPS4 file header lines with GBK encoding."""
    with open(path, 'r', encoding='GBK') as f:
        return f.readlines()


def _write_m4_header(path: pathlib.Path, lines):
    """Write MICAPS4 file header lines with GBK encoding."""
    with open(path, 'w', encoding='GBK') as f:
        f.writelines(lines)


def _strip_prefix(filename: str, prefix: str) -> str:
    """Remove configured filename prefix if present."""
    if prefix in filename:
        return filename.replace(prefix, '')
    return filename


def repair_products(
    product_dir: pathlib.Path, title_template: str, filename_prefix: str
) -> None:
    """Repair product directory structure and m4 titles."""
    if not product_dir.exists():
        print(f'Product directory does not exist: {product_dir}')
        return

    # Step 1: Move old init-time folders into day folders
    for folder in sorted(product_dir.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        if not _is_init_time_folder(name):
            continue

        day = name[:8]
        day_folder = product_dir / day
        day_folder.mkdir(parents=True, exist_ok=True)

        for file_path in folder.iterdir():
            if not file_path.is_file():
                continue
            new_name = _strip_prefix(file_path.name, filename_prefix)
            dst_file = day_folder / new_name
            if dst_file.exists():
                dst_file.unlink()
            shutil.move(str(file_path), str(dst_file))

        try:
            folder.rmdir()
        except OSError:
            pass
        print(f'Moved {name} -> {day}/')

    # Step 2: Strip any remaining prefixes and fix m4 titles
    for folder in sorted(product_dir.iterdir()):
        if not folder.is_dir() or not _is_day_folder(folder.name):
            continue

        for file_path in folder.iterdir():
            if not file_path.is_file():
                continue

            # Strip configured filename prefix if still present
            new_name = _strip_prefix(file_path.name, filename_prefix)
            if new_name != file_path.name:
                new_path = folder / new_name
                if new_path.exists():
                    new_path.unlink()
                shutil.move(str(file_path), str(new_path))
                file_path = new_path

            init_time = file_path.stem
            lead = _lead_from_filename(file_path.name)
            expected_title = title_template.format(
                init_time=init_time, lead=lead
            )

            lines = _read_m4_header(file_path)
            title_line = lines[0].strip()
            if not title_line.startswith('diamond 4 '):
                continue
            current_title = title_line[len('diamond 4 '):]

            if current_title == expected_title:
                continue

            lines[0] = f'diamond 4 {expected_title}\n'
            # Ensure datetime line matches init_time
            parts = lines[1].strip().split()
            dt = arrow.get(init_time, 'YYYYMMDDHH')
            parts[0] = str(dt.year)
            parts[1] = str(dt.month)
            parts[2] = str(dt.day)
            parts[3] = str(dt.hour)
            lines[1] = ' '.join(parts) + '\n'
            _write_m4_header(file_path, lines)
            print(f'Fixed title: {file_path.name}')

    print('Repair done.')


def main() -> None:
    """Main entry point."""
    cfg = utils.load_config(operational=True)
    product_dir = pathlib.Path(cfg['operational']['paths']['product_dir'])
    title_template = cfg['operational']['output']['m4_title_template']
    filename_prefix = cfg['operational']['output']['m4_filename_prefix']
    repair_products(product_dir, title_template, filename_prefix)


if __name__ == '__main__':
    main()
