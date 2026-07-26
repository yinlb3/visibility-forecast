#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the nearest-station map artifact for operational inference.

Computes the nearest station id for every grid cell and saves it to the
configured nearest_map_file. Run manually whenever the grid spec or the
station table changes.

Usage:
    python build_near_map.py              # grid spec from config
    python build_near_map.py 2026071700   # grid spec from that meta.json

Founded in 2026-07-26
Modified in 2026-07-26
@author: yinlb
"""

import pathlib
import sys
import typing

import arrow

from src import logger, near_map, utils


def _resolve_grid_spec(
    args: typing.Tuple[str, ...],
    cfg: typing.Dict
) -> typing.Dict:
    """Resolve grid spec from config or an existing meta.json.

    Args:
        args: Parsed time arguments (0 or 1 element).
        cfg: Merged configuration dictionary.

    Returns:
        Grid spec dict (slon/dlon/nlon/slat/dlat/nlat).
    """
    if len(args) == 0:
        logger.info('Grid spec from interpolation.target_extent config')
        return near_map.grid_spec_from_config(cfg)
    meta_path = pathlib.Path(
        cfg['operational']['paths']['intermediate_dir']
    ) / args[0] / 'meta.json'
    logger.info(f'Grid spec from {meta_path}')
    return near_map.grid_spec_from_meta(str(meta_path))


def main(args: typing.Optional[typing.Tuple[str, ...]] = None) -> None:
    """
    Main entry: load config, build and save the nearest-station map.

    Args:
        args: Optional parsed arguments; defaults to command-line args.
    """
    if args is None:
        args = utils.parse_time_args(sys.argv)
    if len(args) > 1:
        print('Usage: python build_near_map.py [YYYYMMDDHH]')
        raise SystemExit(1)

    cfg = utils.load_config(operational=True)
    op_cfg = cfg['operational']
    logger.setup_logger(
        log_dir=op_cfg['paths']['log_dir'],
        log_level=op_cfg['output']['log_level'],
        run_id=arrow.now().format('YYYYMMDDHHmmss')
    )

    sta = near_map.load_stations(cfg)
    grid = _resolve_grid_spec(args, cfg)
    nlat = grid['nlat']
    nlon = grid['nlon']
    logger.info(
        f'Building nearest-station map: {len(sta)} stations, '
        f'grid {nlat}x{nlon}'
    )
    near_id = near_map.build_near_id_map(grid, sta)
    near_map.save_near_map(
        cfg['operational']['inference']['nearest_map_file'],
        near_id, grid, sta
    )


if __name__ == '__main__':
    print('Program build_near_map.py started')
    total_start = arrow.now()

    parsed = utils.parse_time_args(sys.argv)
    main(args=parsed)

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        f'Program build_near_map.py finished, total time '
        f'{utils.format_time(total_elapsed)}'
    )
