#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Copy one day's MICAPS4 product files to a date range without parsing m4.

Performs a direct filesystem copy for each source file, renames it by
shifting the init-time date, and adjusts file/folder timestamps according to
the fake_timestamp configuration. Output structure:

    <product_dir>/<YYYYMMDD>/<YYYYMMDD>HH.<NNN>

Usage:
    python copy_daily_files.py <src_day> <dst_start> <dst_end> [-p <path>]
    python copy_daily_files.py <src_day> "['<dst_start>', '<dst_end>']"

Examples:
    python copy_daily_files.py 20260701 20260702 20260717
    python copy_daily_files.py 20260701 "['20260702', '20260717']"
    python copy_daily_files.py 20260701 20260702 20260717 -p <DIR>

Founded in 2026-07-17
Modified in 2026-09-30
@author: yinlb, space-bunny
"""

import argparse
import ast
import os
import pathlib
import shutil
import typing

import arrow

from src import utils


def _compute_timestamp(
    init_time: str,
    offset_hours: int,
    fake_cfg: typing.Dict,
) -> float:
    """Compute Unix timestamp for a file or folder.

    Delegates to utils.compute_fake_timestamp when fake timestamps are
    enabled; otherwise returns init_time + offset_hours exactly.

    Args:
        init_time: YYYYMMDDHH string.
        offset_hours: Timestamp offset in hours from init_time.
        fake_cfg: fake_timestamp configuration dictionary.

    Returns:
        Unix timestamp.
    """
    if fake_cfg['enabled']:
        return utils.compute_fake_timestamp(init_time, fake_cfg)
    init_dt = arrow.get(init_time, 'YYYYMMDDHH')
    init_dt = init_dt.replace(tzinfo='Asia/Shanghai')
    return init_dt.shift(hours=offset_hours).floor('hour').timestamp()


def _set_timestamp(path: pathlib.Path, ts: float) -> None:
    """Set filesystem atime/mtime for a file or folder.

    Args:
        path: Target file or folder path.
        ts: Unix timestamp to set.
    """
    os.utime(str(path), (ts, ts))


def _is_day_folder(name: str) -> bool:
    """Check if a directory name is a YYYYMMDD day string.

    Args:
        name: Directory name to check.

    Returns:
        True if name is an 8-digit day string.
    """
    return len(name) == 8 and name.isdigit()


def _init_time_from_filename(filename: str) -> str:
    """Parse YYYYMMDDHH init time from filename '...YYYYMMDDHH.NNN'.

    Args:
        filename: Product filename ending with 10 digits before suffix.

    Returns:
        YYYYMMDDHH init time string.

    Raises:
        ValueError: If no 10-digit init time is found.
    """
    stem = pathlib.Path(filename).stem
    # The last 10 characters should be YYYYMMDDHH
    if len(stem) >= 10 and stem[-10:].isdigit():
        return stem[-10:]
    raise ValueError(f'Cannot parse init time from filename: {filename}')


def _expand_day_range(start_str: str, end_str: str) -> typing.List[str]:
    """Expand a closed YYYYMMDD day range into a list of day strings.

    Args:
        start_str: Start day in YYYYMMDD.
        end_str: End day in YYYYMMDD.

    Returns:
        List of YYYYMMDD day strings from start to end inclusive.

    Raises:
        ValueError: If end day is before start day.
    """
    start = arrow.get(start_str, 'YYYYMMDD')
    end = arrow.get(end_str, 'YYYYMMDD')
    if end < start:
        raise ValueError(
            f'End day {end_str} must be equal to or after start day '
            f'{start_str}.'
        )
    days = list()
    current = start
    while current <= end:
        days.append(current.format('YYYYMMDD'))
        current = current.shift(days=1)
    return days


def _parse_dst_days(args: typing.List[str]) -> typing.List[str]:
    """Parse destination day arguments into a closed day range.

    Supports two styles:
    - Two positional arguments: <dst_start> <dst_end>
    - One list literal: "['<dst_start>', '<dst_end>']"

    Args:
        args: Remaining positional arguments.

    Returns:
        List of YYYYMMDD day strings.

    Raises:
        ValueError: If the arguments match neither supported style.
    """
    if len(args) == 1 and args[0].startswith('[') and args[0].endswith(']'):
        day_list = ast.literal_eval(args[0])
        if len(day_list) != 2:
            raise ValueError(
                'Destination list must contain exactly two days '
                '(start and end).'
            )
        return _expand_day_range(str(day_list[0]), str(day_list[1]))
    if len(args) == 2:
        return _expand_day_range(args[0], args[1])
    raise ValueError(
        'Destination days must be given as <dst_start> <dst_end> or as a '
        'two-element list literal.'
    )


def _copy_day_files(
    src_day: str,
    dst_days: typing.List[str],
    product_dir: pathlib.Path,
    fake_cfg: typing.Dict,
    offset_hours: int,
) -> None:
    """Copy source day files to target days with renamed init times.

    Args:
        src_day: Source day in YYYYMMDD.
        dst_days: Target day list in YYYYMMDD.
        product_dir: Product root directory.
        fake_cfg: fake_timestamp configuration dictionary.
        offset_hours: Timestamp offset in hours from init_time.
    """
    src_folder = product_dir / src_day
    if not src_folder.exists():
        print(f'Source folder not found: {src_folder}')
        return

    src_files = [p for p in sorted(src_folder.iterdir()) if p.is_file()]
    if not src_files:
        print(f'Source folder has no files: {src_folder}')
        return

    print(f'Source folder: {src_folder} ({len(src_files)} files)')

    for dst_day in dst_days:
        day_start = arrow.now()
        dst_folder = product_dir / dst_day
        dst_folder.mkdir(parents=True, exist_ok=True)

        for src_file in src_files:
            src_init = _init_time_from_filename(src_file.name)
            dst_init = dst_day + src_init[8:]
            dst_filename = src_file.name.replace(src_init, dst_init)
            dst_file = dst_folder / dst_filename

            shutil.copy2(str(src_file), str(dst_file))

            file_ts = _compute_timestamp(dst_init, offset_hours, fake_cfg)
            _set_timestamp(dst_file, file_ts)

        folder_ts = _compute_timestamp(dst_day + '12', offset_hours, fake_cfg)
        _set_timestamp(dst_folder, folder_ts)
        day_elapsed = (arrow.now() - day_start).total_seconds()
        print(f'  Copied {dst_day} in {utils.format_time(day_elapsed)}')


def main(argv: typing.Optional[typing.List[str]] = None) -> None:
    """Main entry: parse arguments and copy daily product files.

    Args:
        argv: Optional argument list; defaults to command-line args.
    """
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(
        description=(
            "Copy one day's MICAPS4 product files to a date range without "
            'parsing m4.'
        ),
        epilog=(
            'Usage examples:\n'
            '  python copy_daily_files.py 20260701 20260707 20260716\n'
            '  python copy_daily_files.py 20260701 '
            "\"['20260707', '20260716']\"\n"
            '  python copy_daily_files.py 20260701 20260707 20260716 '
            '-p <PRODUCT_DIR>'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('src_day', help='Source day in YYYYMMDD')
    parser.add_argument(
        '--product-dir', '-p', default=None,
        help='Output product directory (default: operational config path)',
    )
    args, remaining = parser.parse_known_args(argv)
    if len(remaining) not in (1, 2):
        parser.error(
            'Destination days must be given as <dst_start> <dst_end> or '
            'as a two-element list literal.'
        )
    src_day = args.src_day
    try:
        dst_days = _parse_dst_days(remaining)
    except ValueError as exc:
        parser.error(str(exc))

    # 2. Load configuration
    cfg = utils.load_config(operational=True)
    product_dir = pathlib.Path(
        args.product_dir or cfg['operational']['paths']['product_dir']
    )
    fake_cfg = cfg['operational']['synthetic']['fake_timestamp']
    offset_hours = int(fake_cfg['offset_hours'])
    total_start = arrow.now()
    print(f'Product directory: {product_dir}')
    print(f'Source day: {src_day}')
    print(
        f'Target days: {dst_days[0]} -> {dst_days[-1]} '
        f'({len(dst_days)} days)'
    )

    # 3. Copy target days
    print('\nCopying target days...')
    _copy_day_files(src_day, dst_days, product_dir, fake_cfg, offset_hours)

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(f'\nAll done. Total time: {utils.format_time(total_elapsed)}')


if __name__ == '__main__':
    print('Program copy_daily_files.py started')
    main()
