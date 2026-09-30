#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Clone one day's MICAPS4 products to a date range with Gaussian noise.

Reads each source m4 file once with meteva, then applies per-cell Gaussian
noise N(1, 0.01) repeatedly to generate every day in the requested target
range. Output structure:

    <product_dir>/<YYYYMMDD>/<YYYYMMDD>HH.<NNN>

The m4 file title keeps the Chinese prefix configured in m4_title_template.

Usage:
    python copy_daily_products.py <src_day> <dst_start> <dst_end> [-p <path>]
    python copy_daily_products.py <src_day> "['<dst_start>', '<dst_end>']"

Examples:
    python copy_daily_products.py 20260701 20260702 20260717
    python copy_daily_products.py 20260701 "['20260702', '20260717']"
    python copy_daily_products.py 20260701 20260702 20260717 -p <DIR>

Founded in 2026-07-17
Modified in 2026-09-30
@author: yinlb, space-bunny
"""

import argparse
import ast
import copy
import os
import pathlib
import typing

import arrow
import numpy as np
from meteva import base as meb    # type: ignore

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


def _lead_from_filename(filename: str) -> int:
    """Parse lead hour from '.NNN' suffix.

    Args:
        filename: Product filename like '<init>.001'.

    Returns:
        Lead hour as integer.

    Raises:
        ValueError: If the suffix is not a three-digit lead.
    """
    suffix = pathlib.Path(filename).suffix
    if len(suffix) == 4 and suffix[1:].isdigit():
        return int(suffix[1:])
    raise ValueError(f'Cannot parse lead from filename: {filename}')


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


def _apply_noise(grd: object, std: float = 0.01) -> object:
    """Return a copy of grd with per-cell Gaussian noise N(1, std) applied.

    Args:
        grd: meteva grid_data object (not modified).
        std: Standard deviation of the Gaussian noise. Defaults to 0.01.

    Returns:
        grid_data: A copy of grd with noised values.
    """
    new_grd = copy.deepcopy(grd)
    values = np.squeeze(new_grd.values).copy()
    noise = np.random.normal(loc=1.0, scale=std, size=values.shape)
    values = values * noise

    if new_grd.values.ndim == 6:
        new_grd.values[0, 0, 0, 0, :, :] = values
    else:
        new_grd.values = values
    return new_grd


def _check_and_fix_source(
    product_dir: pathlib.Path,
    fake_cfg: typing.Dict,
    offset_hours: int,
) -> None:
    """Check and rewrite timestamps for the source day's folders/files.

    Args:
        product_dir: Product root directory.
        fake_cfg: fake_timestamp configuration dictionary.
        offset_hours: Timestamp offset in hours from init_time.
    """
    if not fake_cfg['enabled']:
        return

    for folder in sorted(product_dir.iterdir()):
        if not folder.is_dir() or not _is_day_folder(folder.name):
            continue
        day = folder.name
        # Day folder timestamp: use noon of that day as representative
        day_ts = _compute_timestamp(day + '12', offset_hours, fake_cfg)
        _set_timestamp(folder, day_ts)

        for file_path in folder.iterdir():
            if not file_path.is_file():
                continue
            init_time = _init_time_from_filename(file_path.name)
            file_ts = _compute_timestamp(init_time, offset_hours, fake_cfg)
            _set_timestamp(file_path, file_ts)

    print('Source day timestamps checked/fixed.')


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


def _fill_missing_leads(
    lead_grids: typing.Dict[int, object],
    expected_leads: typing.List[int],
) -> typing.Dict[int, object]:
    """Fill missing leads by copying the nearest available lead grid.

    Prefers the nearest previous lead; if none exists, falls back to the
    nearest next lead. Missing leads that have no available neighbour are
    skipped.

    Args:
        lead_grids: Mapping of lead hour to grid_data (not modified).
        expected_leads: Full lead hour list to complete.

    Returns:
        Mapping of lead hour to grid_data covering expected_leads.
    """
    if not lead_grids or not expected_leads:
        return lead_grids

    available_leads = sorted(lead_grids.keys())
    complete: typing.Dict[int, object] = {}

    for lead in expected_leads:
        if lead in lead_grids:
            complete[lead] = lead_grids[lead]
            continue

        prev_leads = [cand for cand in available_leads if cand < lead]
        if prev_leads:
            src_lead = max(prev_leads)
        else:
            next_leads = [cand for cand in available_leads if cand > lead]
            if not next_leads:
                continue
            src_lead = min(next_leads)

        complete[lead] = copy.deepcopy(lead_grids[src_lead])

    return complete


def _copy_day_chain(
    src_day: str,
    dst_days: typing.List[str],
    product_dir: pathlib.Path,
    fake_cfg: typing.Dict,
    offset_hours: int,
    cfg: typing.Dict,
) -> None:
    """Copy source day to target days with chained Gaussian noise.

    Args:
        src_day: Source day in YYYYMMDD.
        dst_days: Target day list in YYYYMMDD.
        product_dir: Product root directory.
        fake_cfg: fake_timestamp configuration dictionary.
        offset_hours: Timestamp offset in hours from init_time.
        cfg: Merged configuration dictionary.
    """
    src_folder = product_dir / src_day
    if not src_folder.exists():
        print(f'Source folder not found: {src_folder}')
        return

    output_cfg = cfg['operational']['output']
    forecast_cfg = cfg['operational']['forecast']
    expected_leads = forecast_cfg['lead_hours']
    title_template = output_cfg['m4_title_template']

    # Load all source grids, grouped by init_time
    init_time_grids: typing.Dict[str, typing.Dict[int, object]] = {}
    init_base_filename: typing.Dict[str, str] = {}

    for src_file in sorted(src_folder.iterdir()):
        if not src_file.is_file():
            continue
        init_time = _init_time_from_filename(src_file.name)
        lead = _lead_from_filename(src_file.name)
        if init_time not in init_time_grids:
            init_time_grids[init_time] = {}
            init_base_filename[init_time] = src_file.name
        init_time_grids[init_time][lead] = meb.read_griddata_from_micaps4(
            str(src_file)
        )

    # Fill missing leads so the output sequence stays complete
    for init_time in init_time_grids:
        init_time_grids[init_time] = _fill_missing_leads(
            init_time_grids[init_time], expected_leads
        )

    print(
        f'Source folder: {src_folder} ({len(init_time_grids)} init times, '
        f'{len(expected_leads)} expected leads)'
    )

    for dst_day in dst_days:
        day_start = arrow.now()
        dst_folder = product_dir / dst_day
        dst_folder.mkdir(parents=True, exist_ok=True)

        for src_init, lead_grids in init_time_grids.items():
            dst_init = dst_day + src_init[8:]
            base_filename = init_base_filename[src_init]
            for lead, grd in lead_grids.items():
                lead_grids[lead] = _apply_noise(
                    grd, std=float(fake_cfg['noise_std'])
                )

                dst_filename = base_filename.replace(src_init, dst_init)
                dst_filename = dst_filename[:-4] + f'.{lead:03d}'
                dst_file = dst_folder / dst_filename

                title = title_template.format(init_time=dst_init, lead=lead)
                success = meb.write_griddata_to_micaps4(
                    da=lead_grids[lead],
                    save_path=str(dst_file),
                    title=title,
                )
                if not success:
                    print(f'  Warning: failed to write {dst_file}')
                    continue

                file_ts = _compute_timestamp(dst_init, offset_hours, fake_cfg)
                _set_timestamp(dst_file, file_ts)

        folder_ts = _compute_timestamp(dst_day + '12', offset_hours, fake_cfg)
        _set_timestamp(dst_folder, folder_ts)
        day_elapsed = (arrow.now() - day_start).total_seconds()
        print(f'  Generated {dst_day} in {utils.format_time(day_elapsed)}')


def main(argv: typing.Optional[typing.List[str]] = None) -> None:
    """Main entry: parse arguments and clone daily products.

    Args:
        argv: Optional argument list; defaults to command-line args.
    """
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(
        description=(
            'Clone one day MICAPS4 products to a date range with per-cell '
            'Gaussian noise.'
        ),
        epilog=(
            'Usage examples:\n'
            '  python copy_daily_products.py 20260701 20260707 20260716\n'
            '  python copy_daily_products.py 20260701 '
            "\"['20260707', '20260716']\"\n"
            '  python copy_daily_products.py 20260701 20260707 20260716 '
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

    # 3. Check source day timestamps
    print('\nChecking source day timestamps...')
    _check_and_fix_source(product_dir, fake_cfg, offset_hours)

    # 4. Generate target days with chained noise
    print('\nGenerating target days...')
    _copy_day_chain(
        src_day, dst_days, product_dir, fake_cfg, offset_hours, cfg
    )

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(f'\nAll done. Total time: {utils.format_time(total_elapsed)}')


if __name__ == '__main__':
    print('Program copy_daily_products.py started')
    main()
