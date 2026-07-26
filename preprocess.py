#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Operational preprocessing entry point.

Prepares forecast GRIB inputs for one or more initialization cycles,
saves intermediate arrays, and writes MICAPS4 forecast products under
the configured operational directories.

Founded in 2026-07-16
Modified in 2026-07-26
@author: yinlb
"""

import json
import os
import pathlib
import shutil
import sys
import typing
from collections import defaultdict

import arrow
import numpy as np
from meteva import base as meb

from src import grib_forecast, logger, p1_config_data as p1, utils


def _write_products_for_init(
    init_time: str,
    results: typing.List[typing.Tuple[int, typing.Any, pathlib.Path]],
    cfg: typing.Dict,
    load_obs: bool = False,
    obs_data: typing.Optional[typing.Dict] = None,
) -> None:
    """Write m4 products and intermediate arrays for one init_time.

    Args:
        init_time: YYYYMMDDHH string.
        results: List of (lead, grid_data, grib_path) tuples for this init_time.
        cfg: Merged configuration dictionary.
        load_obs: Whether observation arrays should be saved.
        obs_data: Optional dictionary with observation arrays and metadata.
    """
    paths = cfg['operational']['paths']
    region = cfg['operational']['runtime']['region']

    # Sort results by lead hour
    results = sorted(results, key=lambda x: x[0])

    # Output directories
    out_dir = pathlib.Path(paths['intermediate_dir']) / init_time
    out_dir.mkdir(parents=True, exist_ok=True)
    # Products are grouped by day: ops/products/YYYYMMDD/
    product_dir = pathlib.Path(paths['product_dir']) / init_time[:8]
    product_dir.mkdir(parents=True, exist_ok=True)

    # Write MICAPS4 products and stack forecast grids
    output_cfg = cfg['operational']['output']
    m4_template = output_cfg['m4_filename_template']
    title_template = output_cfg['m4_title_template']
    fake_cfg = cfg['operational']['synthetic']['fake_timestamp']

    display_dir_cfg = cfg['operational']['paths']['display_dir']
    display_dir = (
        pathlib.Path(display_dir_cfg)
        if display_dir_cfg and display_dir_cfg != '<DISPLAY_DIR>'
        else None
    )
    if display_dir is not None:
        display_dir.mkdir(parents=True, exist_ok=True)

    grids = list()
    m4_products = list()
    processed_leads = list()
    for lead, forecast_grd, forecast_path in results:
        m4_filename = m4_template.format(init_time=init_time, lead=lead)
        m4_title = title_template.format(init_time=init_time, lead=lead)
        m4_path = product_dir / m4_filename
        meb.write_griddata_to_micaps4(
            da=forecast_grd,
            save_path=str(m4_path),
            title=m4_title,
        )

        # Adjust filesystem timestamp
        ts = utils.compute_fake_timestamp(init_time, fake_cfg)
        os.utime(str(m4_path), (ts, ts))

        # Copy to business display directory with flat structure
        if display_dir is not None:
            display_path = display_dir / m4_filename
            shutil.copy2(str(m4_path), str(display_path))
            # Ensure copied file has the same timestamp
            os.utime(str(display_path), (ts, ts))

        grids.append(np.squeeze(forecast_grd.values))
        m4_products.append(str(m4_path))
        processed_leads.append(lead)

    forecast_stack = np.stack(grids, axis=0)
    np.save(str(out_dir / 'forecast_grid.npy'), forecast_stack)

    # Save metadata
    meta = {
        'init_time': init_time,
        'region': region,
        'forecast_shape': list(forecast_stack.shape),
        'lead_hours': cfg['operational']['forecast']['lead_hours'],
        'processed_leads': processed_leads,
        'm4_products': m4_products,
    }
    # Grid spec from the first grid's coordinates so downstream consumers
    # (e.g. inference) can rebuild the exact lat-lon grid
    first_grd = results[0][1]
    lons = first_grd.coords['lon'].values
    lats = first_grd.coords['lat'].values
    meta['grid'] = {
        'slon': float(lons[0]),
        'dlon': float(lons[1] - lons[0]),
        'nlon': int(len(lons)),
        'slat': float(lats[0]),
        'dlat': float(lats[1] - lats[0]),
        'nlat': int(len(lats)),
    }
    if load_obs and obs_data is not None:
        meta['n_stations'] = obs_data['n_stations']
        meta['vis_shape'] = obs_data['vis_shape']

    with open(out_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    # Save observation arrays if requested
    if load_obs and obs_data is not None:
        obs_data['sta'].to_csv(str(out_dir / 'sta.csv'), index=False)
        np.save(str(out_dir / 'vis.npy'), obs_data['vis'])
        np.save(str(out_dir / 'pre.npy'), obs_data['pre'])
        np.save(str(out_dir / 'rhu.npy'), obs_data['rhu'])
        np.save(str(out_dir / 'vis_grade.npy'), obs_data['vis_grade'])
        np.save(str(out_dir / 'index_mlyr.npy'), obs_data['idx_mlyr'])

    logger.info(f'Finished preprocessing {init_time}, saved to {out_dir}')


def _load_observations(cfg: typing.Dict) -> typing.Dict:
    """Load and filter observation data for research workflows.

    Args:
        cfg: Merged configuration dictionary.

    Returns:
        Dict with station table (sta), observation arrays (vis, pre,
        rhu, vis_grade), region mask (idx_mlyr), and metadata
        (n_stations, vis_shape).
    """
    data_dir = cfg['paths']['data_dir']
    region = cfg['operational']['runtime']['region']
    region_cfg = cfg['draw']['regions']

    if region == 'mlyr':
        region_provinces = tuple(region_cfg['mlyr_provinces'])
    else:
        region_provinces = tuple(region_cfg['provinces'])

    sta, idx_east_china = p1.read_sta(
        sta_path=str(pathlib.Path(data_dir) / 'sta2411.csv'),
        provinces=tuple(region_cfg['provinces'])
    )
    vis, pre, rhu = p1.load_obs(
        data_dir=data_dir,
        idx_east_china=idx_east_china
    )
    sta, vis, pre, rhu, idx_mlyr = p1.filter_region(
        sta=sta, vis=vis, pre=pre, rhu=rhu,
        region_provinces=region_provinces
    )
    vis_grade = p1.grade_visibility(vis=vis)

    return {
        'sta': sta,
        'vis': vis,
        'pre': pre,
        'rhu': rhu,
        'vis_grade': vis_grade,
        'idx_mlyr': idx_mlyr,
        'n_stations': int(np.sum(idx_mlyr)),
        'vis_shape': list(vis.shape),
    }


def main(args: typing.Optional[typing.Tuple[str, ...]] = None) -> None:
    """
    Main entry: load config, setup logger, process init times.

    Args:
        args: Optional parsed arguments; defaults to command-line args.
    """
    # 1. Parse arguments and load configuration
    if args is None:
        args = utils.parse_time_args(sys.argv)
    cfg = utils.load_config(operational=True)
    op_cfg = cfg['operational']

    # 2. Setup logger and resolve runtime parameters
    logger.setup_logger(
        log_dir=op_cfg['paths']['log_dir'],
        log_level=op_cfg['output']['log_level'],
        run_id=arrow.now().format('YYYYMMDDHHmmss')
    )
    init_times = utils.resolve_init_times(args, cfg)
    lead_hours = op_cfg['forecast']['lead_hours']
    skip_missing = bool(op_cfg['forecast']['skip_missing'])
    fallback_cfg = op_cfg['synthetic']['fallback']
    fallback_enabled = bool(fallback_cfg['enabled'])
    load_obs = bool(op_cfg['preprocess']['load_observations'])
    logger.info(f'Resolved init times: {init_times}')
    logger.info(f'Lead hours: {lead_hours}')

    # 3. Load observations once in the main process if requested
    obs_data = None
    if load_obs:
        logger.info('Loading observations')
        obs_data = _load_observations(cfg)

    # 4. Build full task list: (init_time, lead)
    tasks = list()
    for init_time in init_times:
        for lead in lead_hours:
            if skip_missing and not grib_forecast.forecast_file_exists(
                init_time, cfg, lead
            ):
                logger.warning(
                    f'Forecast GRIB missing, skipped: {init_time} f{lead:02d}'
                )
                continue
            tasks.append((init_time, lead))
    logger.info(f'Total tasks to process: {len(tasks)}')

    # 5. Load and interpolate all tasks serially
    results = list()
    for init_time, lead in tasks:
        result = grib_forecast.load_forecast_task(init_time, cfg, lead)
        _, _, grid, file_path = result
        if grid is None:
            # Missing GRIB: skip copy/read/interpolate entirely
            logger.warning(
                f'Forecast GRIB missing for {init_time} f{lead:02d}: '
                f'{file_path}'
            )
            if fallback_enabled:
                try:
                    fallback_grd = grib_forecast.generate_fallback_grid(
                        cfg, init_time, lead
                    )
                    results.append((
                        init_time, lead, fallback_grd,
                        pathlib.Path('fallback')
                    ))
                    logger.warning(
                        f'Using fallback sample data for {init_time} '
                        f'f{lead:02d}'
                    )
                except Exception as fallback_e:
                    logger.error(
                        f'Fallback generation failed for {init_time} '
                        f'f{lead:02d}: {fallback_e}'
                    )
                    if not skip_missing:
                        raise FileNotFoundError(
                            f'Forecast GRIB missing and fallback failed: '
                            f'{file_path}'
                        )
            elif not skip_missing:
                raise FileNotFoundError(
                    f'Forecast GRIB missing: {file_path}'
                )
            continue
        results.append(result)

    # 6. Group results by init_time and write products
    grouped = defaultdict(list)
    for init_time, lead, grid, file_path in results:
        grouped[init_time].append((lead, grid, file_path))
    success = 0
    failed = 0
    for init_time in init_times:
        if init_time not in grouped:
            logger.error(f'No forecast files loaded for {init_time}')
            failed += 1
            continue
        try:
            _write_products_for_init(
                init_time, grouped[init_time], cfg,
                load_obs=load_obs, obs_data=obs_data
            )
            success += 1
        except Exception as exc:
            logger.error(f'Failed to write products for {init_time}: {exc}')
            failed += 1
    logger.info(f'Preprocessing summary: {success} succeeded, {failed} failed')


if __name__ == '__main__':
    print('Program preprocess.py started')
    total_start = arrow.now()

    parsed = utils.parse_time_args(sys.argv)
    main(args=parsed)

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        f'Program preprocess.py finished, total time '
        f'{utils.format_time(total_elapsed)}'
    )
