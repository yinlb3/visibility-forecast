#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified operational pipeline: preprocess + inference in one process.

Loads GRIB forecasts, applies nearest-station PDF correction, and writes
corrected MICAPS4 products to product_dir. Intermediate .npy files are only
written when output.save_intermediate is true.

The output.copy mode controls the missing-GRIB supplement strategy:
  - 'none'    - no supplement for missing GRIB
  - 'files'   - supplement missing GRIB with existing m4 files
  - 'product' - supplement missing GRIB with synthetic data

Whether products are copied to display_dir is controlled by the
operational.paths.display_dir setting (empty string disables copying).

This module is self-contained and does not import preprocess.py or
inference.py. Those legacy scripts will be removed once documentation is
updated.

Usage:
    python pipeline.py [YYYYMMDDHH [YYYYMMDDHH]]
    python pipeline.py --utc [YYYYMMDDHH [YYYYMMDDHH]]
    python pipeline.py --bjt [YYYYMMDDHH [YYYYMMDDHH]]

--utc (default) treats times as GRIB initialization time in UTC.
--bjt treats times as product/output time in Beijing time (CST, UTC+8)
and converts them to UTC internally.

Founded in 2026-08-01
Modified in 2026-09-30
@author: yinlb, space-bunny
"""

import collections
import pathlib
import sys
import traceback
import typing

import arrow
import numpy as np
import pandas as pd
import xarray as xr

from src import forecast_prep, grib_forecast, logger
from src import model_registry, near_map
from src import postprocess, product_writer, utils


def _resolve_common_params(
    cfg: typing.Dict
) -> typing.Tuple[bool, bool, str, bool, bool]:
    """Resolve shered preprocessing/inference parameters from config.

    Args:
        cfg: Merged configuration dictionary.

    Returns:
        Tuple of (skip_missing, fallback_enabled, copy_mode,
                  save_intermediate, load_obs).
    """
    op_cfg = cfg['operational']

    skip_missing_raw = op_cfg['forecast']['skip_missing']
    if isinstance(skip_missing_raw, bool):
        skip_missing = skip_missing_raw
    else:
        skip_missing = str(skip_missing_raw).lower() != 'false'

    fallback_enabled = bool(op_cfg['synthetic']['fallback']['enabled'])
    copy_mode = str(op_cfg['output']['copy']).lower()
    save_intermediate = bool(op_cfg['output']['save_intermediate'])
    load_obs = bool(op_cfg['preprocess']['load_observations'])

    return (
        skip_missing, fallback_enabled, copy_mode, save_intermediate, load_obs
    )




def _regenerate_missing_pdfm_cache(
    init_times: typing.List[str],
    corrected_grids: typing.Dict[str, np.ndarray],
    max_lookback: int,
    lead_hours: typing.List[int],
    cfg: typing.Dict,
    sta: pd.DataFrame,
    reg: model_registry.ModelRegistry,
    near_id: np.ndarray,
    copy_mode: str,
    load_obs: bool,
    obs_data: typing.Optional[typing.Dict]
) -> typing.Dict[str, np.ndarray]:
    """Backfill missing lagged PDFM grids from GRIB when possible.

    For each target init_time, walk back up to max_lookback hours. If a
    lagged PDFM cache is missing but the original GRIB exists, load the
    full lead-hour block, run PDFM correction, save the result to cache,
    and include it in the returned dictionary. If GRIB is missing or
    regeneration fails, that lagged hour is skipped and TLE will average
    the remaining samples.

    Args:
        init_times: Target YYYYMMDDHH strings for this run.
        corrected_grids: Current run's corrected grids, keyed by init_time.
        max_lookback: Maximum hours to look back.
        lead_hours: Operational lead hours to regenerate.
        cfg: Merged configuration dictionary.
        sta: Station table.
        reg: Model registry instance.
        near_id: Nearest-station map (nlat, nlon).
        copy_mode: Output copy mode, also drives missing-GRIB supplement.
        load_obs: Whether observation arrays should be saved.
        obs_data: Optional observation dictionary.

    Returns:
        Mapping init_time -> corrected grid array including current run and
        any successfully regenerated lagged grids.
    """
    combined_grids = dict(corrected_grids)
    if not lead_hours:
        return combined_grids

    first_lead = lead_hours[0]
    missing_init_times = set()
    for init_time in init_times:
        for i in range(1, max_lookback + 1):
            lag_init = utils.shift_init_time(init_time, -i)
            if lag_init in combined_grids:
                continue
            if grib_forecast.forecast_file_exists(
                lag_init, cfg, first_lead
            ):
                missing_init_times.add(lag_init)

    if not missing_init_times:
        return combined_grids

    tasks = [
        (lag_init, lead)
        for lag_init in sorted(missing_init_times)
        for lead in lead_hours
    ]
    logger.info(
        f'Regenerating PDFM cache for {len(missing_init_times)} lagged '
        f'init times'
    )
    lag_results = forecast_prep.load_forecast_grids(
        tasks=tasks,
        cfg=cfg,
        skip_missing=True,
        copy_mode=copy_mode,
        fallback_enabled=False,
    )

    lag_grouped = collections.defaultdict(list)
    for lag_init, lead, grid, file_path in lag_results:
        lag_grouped[lag_init].append((lead, grid, file_path))

    grid_spec = near_map.grid_spec_from_config(cfg)
    cap = float(cfg['operational']['inference']['visibility_cap'])

    for lag_init in sorted(missing_init_times):
        results = lag_grouped.get(lag_init, [])
        if not results:
            continue
        try:
            forecast_stack, processed_leads = (
                forecast_prep.build_forecast_stack(
                    results, all_leads=lead_hours
                )
            )
            meta = {
                'processed_leads': processed_leads,
                'grid': grid_spec,
            }
            corrected = _run_correction(
                lag_init, forecast_stack, meta, cfg, sta, reg,
                near_id=near_id
            )
            # Preserve NaN slots for leads whose GRIB was missing.
            corrected[np.isnan(forecast_stack)] = np.nan
            corrected[corrected > cap] = cap
            combined_grids[lag_init] = corrected
            _save_pdfm_cache(lag_init, corrected, cfg)
            logger.info(f'Regenerated PDFM cache for {lag_init}')
        except Exception as exc:
            logger.warning(
                f'PDFM cache regeneration failed for {lag_init}: {exc}\n'
                f'{traceback.format_exc()}'
            )
            continue

    return combined_grids



def _process_init_time(
    init_time: str,
    results: typing.List[typing.Tuple[int, xr.DataArray, pathlib.Path]],
    cfg: typing.Dict,
    sta: pd.DataFrame,
    reg: model_registry.ModelRegistry,
    near_id: np.ndarray,
    copy_mode: str,
    save_intermediate: bool,
    load_obs: bool,
    obs_data: typing.Optional[typing.Dict]
) -> typing.Tuple[np.ndarray, typing.Dict, typing.List[int]]:
    """Run preprocessing + PDFM correction for a single init_time in memory.

    TLE averaging and product writing are hour_accessndled by the caller so that
    multiple init_times can participate in the same TLE ensemble.

    Args:
        init_time: YYYYMMDDHH string.
        results: List of (lead, grid_data, source_path) tuples.
        cfg: Merged configuration dictionary.
        sta: Station table.
        reg: Model registry instance.
        near_id: Nearest-station map (nlat, nlon).
        copy_mode: Output copy mode controlling missing-GRIB supplement.
        save_intermediate: Whether to write intermediate arrays.
        load_obs: Whether observation arrays should be saved.
        obs_data: Optional observation dictionary.

    Returns:
        Tuple of (corrected grid array, metadata dict, lead hours list).
    """
    logger.info(f'Start pipeline processing {init_time}')

    # 1. Build forecast stack and metadata
    forecast_stack, processed_leads = forecast_prep.build_forecast_stack(
        results
    )
    meta = forecast_prep.build_raw_meta(
        init_time, processed_leads, forecast_stack, results,
        cfg, load_obs, obs_data
    )
    if save_intermediate:
        forecast_prep.save_raw_intermediate(
            init_time, forecast_stack, meta, cfg, load_obs, obs_data
        )

    # 2. Correction
    corrected, model_id, leads = _run_correction(
        init_time, forecast_stack, meta, cfg, sta, reg, near_id=near_id
    )
    if save_intermediate:
        forecast_prep.save_corrected_intermediate(
            init_time, corrected, meta, leads, model_id, cfg
        )

    logger.info(f'Finished pipeline processing {init_time}')
    return corrected, meta, leads


def _beijing_to_utc(bjt_time: str) -> str:
    """Convert a YYYYMMDDHH string from Beijing time (CST, UTC+8) to UTC.

    Args:
        bjt_time: YYYYMMDDHH string in Beijing time.

    Returns:
        YYYYMMDDHH string in UTC.
    """
    bj_dt = arrow.get(bjt_time, 'YYYYMMDDHH').replace(tzinfo='Asia/Shanghai')
    return bj_dt.to('UTC').format('YYYYMMDDHH')


def _parse_pipeline_args(
    argv: typing.List[str]
) -> typing.Tuple[str, ...]:
    """Parse pipeline arguments with an optional time-mode flag.

    The first argument may be '--utc' or '--bjt'. If omitted, the arguments
    are treated as UTC input times for backward compatibility.

    Args:
        argv: sys.argv list.

    Returns:
        Tuple of validated UTC YYYYMMDDHH strings.

    Raises:
        SystemExit: If the argument count or format is invalid.
    """
    usage = (
        'Usage: python pipeline.py [--utc|--bjt] '
        '[YYYYMMDDHH] [YYYYMMDDHH]'
    )

    if len(argv) > 1 and argv[1] in ('--utc', '--bjt'):
        mode = argv[1]
        time_args = argv[2:]
    else:
        mode = '--utc'
        time_args = argv[1:]

    if len(time_args) > 2:
        print(usage)
        raise SystemExit(1)

    for arg in time_args:
        if len(arg) != 10:
            print(f'Invalid time format: {arg}. Expected YYYYMMDDHH.')
            raise SystemExit(1)
        try:
            arrow.get(arg, 'YYYYMMDDHH')
        except arrow.ParserError:
            print(f'Invalid datetime: {arg}')
            raise SystemExit(1)

    if mode == '--bjt':
        return tuple(_beijing_to_utc(t) for t in time_args)
    return tuple(time_args)


def main(args: typing.Optional[typing.Tuple[str, ...]] = None) -> None:
    """Main entry: load config, setup logger, run unified pipeline.

    Args:
        args: Optional parsed arguments; defaults to command-line args.
    """
    # 1. Parse arguments and load configuration
    if args is None:
        args = _parse_pipeline_args(sys.argv)
    cfg = utils.load_config(operational=True)
    op_cfg = cfg['operational']

    # 2. Setup logger and resolve runtime parameters once
    logger.setup_logger(
        log_dir=op_cfg['paths']['log_dir'],
        log_level=op_cfg['output']['log_level'],
        run_id=arrow.now().format('YYYYMMDDHHmmss')
    )
    init_times = utils.resolve_init_times(args, cfg)
    lead_hours = op_cfg['forecast']['lead_hours']
    skip_missing, fallback_enabled, copy_mode, save_intermediate, load_obs = \
        _resolve_common_params(cfg)

    logger.info(f'Resolved init times: {init_times}')
    logger.info(f'Lead hours: {lead_hours}')
    logger.info(f'skip_missing: {skip_missing}')
    logger.info(f'copy: {copy_mode}')
    logger.info(f'save_intermediate: {save_intermediate}')

    # 3. Load observations once if requested
    obs_data = None
    if load_obs:
        logger.info('Loading observations')
        obs_data = forecast_prep.load_observations(cfg)

    # 4. Load station table, model registry, and validate nearest-station map
    # once per run. Correction always runs because output is always corrected.
    sta = near_map.load_stations(cfg)
    reg = model_registry.ModelRegistry(
        registry_dir=op_cfg['paths']['model_registry_dir'],
        default_model_id=op_cfg['model']['default_model_id']
    )
    logger.info(f'Loaded {len(sta)} stations for correction')

    near_id = None
    if init_times:
        try:
            first_model = reg.select_model_id(init_time=init_times[0])
            first_meta = reg.get_metadata(first_model)
        except (RuntimeError, KeyError, ValueError) as exc:
            logger.error(f'Model registry check failed: {exc}')
            sys.exit(1)
        logger.info(f'Model for {init_times[0]}: {first_model}')
        if 'station_ids' not in first_meta:
            logger.warning(
                'Registry metadata has no station_ids, '
                'falling back to filtered CSV order'
            )

        # Align stations for the first model and load nearest-station map once.
        # This map is reused for all init_times in the run.
        try:
            grid = near_map.grid_spec_from_config(cfg)
            sta_aligned = _align_stations(sta, first_meta)
            near_id = near_map.load_near_map(
                op_cfg['inference']['nearest_map_file'], grid, sta_aligned
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.error(f'Nearest-station map check failed: {exc}')
            sys.exit(1)

    # 5. Build task list and load/supplement GRIB forecasts
    tasks = forecast_prep.build_tasks(
        init_times, lead_hours, cfg, skip_missing, copy_mode
    )
    logger.info(f'Total tasks to process: {len(tasks)}')
    results = forecast_prep.load_forecast_grids(
        tasks, cfg, skip_missing, copy_mode, fallback_enabled
    )

    # 6. Group by init_time and run preprocessing + PDFM correction in memory
    grouped = collections.defaultdict(list)
    for init_time, lead, grid, file_path in results:
        grouped[init_time].append((lead, grid, file_path))

    corrected_grids: typing.Dict[str, np.ndarray] = {}
    meta_map: typing.Dict[str, typing.Dict] = {}
    leads_map: typing.Dict[str, typing.List[int]] = {}

    success = 0
    failed = 0
    for init_time in init_times:
        if init_time not in grouped:
            logger.error(f'No forecast files loaded for {init_time}')
            failed += 1
            continue
        try:
            corrected, meta, leads = _process_init_time(
                init_time, grouped[init_time], cfg, sta, reg, near_id,
                copy_mode, save_intermediate, load_obs, obs_data
            )
            corrected_grids[init_time] = corrected
            meta_map[init_time] = meta
            leads_map[init_time] = leads
            postprocess.save_pdfm_cache(init_time, corrected, cfg)
            success += 1
        except Exception as exc:
            logger.error(f'Pipeline failed for {init_time}: {exc}')
            failed += 1
            if len(init_times) == 1:
                raise

    # 7. Optional time-lagged ensemble across PDFM-corrected init times
    tle_cfg = op_cfg['inference']['tle']
    tle_enabled = bool(tle_cfg['enabled'])
    if tle_enabled and corrected_grids:
        logger.info(
            'Applying time-lagged ensemble (TLE) to PDFM-corrected grids'
        )
        tle_max_lookback = int(tle_cfg['max_lookback_hours'])
        tle_fallback = bool(tle_cfg['fallback_to_pdfm'])
        # Use the lead list from the first corrected grid; all init times shere
        # the same operational lead_hours.
        first_leads = next(iter(leads_map.values()))
        # Merge the current run with cached PDFM grids from previous hours
        # so that anti-diagonal lagged samples are available for TLE.
        combined_grids = dict(corrected_grids)
        for init_time in init_times:
            cached = postprocess.load_pdfm_cache(
                init_time, tle_max_lookback, cfg
            )
            for lag_init, lag_grid in cached.items():
                if lag_init not in combined_grids:
                    combined_grids[lag_init] = lag_grid
        # Backfill missing lagged grids from GRIB when possible. If a lagged
        # init_time has no GRIB or fails, it is skipped and TLE averages the
        # remaining samples.
        combined_grids = _regenerate_missing_pdfm_cache(
            init_times=init_times,
            corrected_grids=combined_grids,
            max_lookback=tle_max_lookback,
            lead_hours=lead_hours,
            cfg=cfg,
            sta=sta,
            reg=reg,
            near_id=near_id,
            copy_mode=copy_mode,
            load_obs=load_obs,
            obs_data=obs_data,
        )
        tle_grids = postprocess.apply_equal_tle(
            combined_grids,
            leads=first_leads,
            max_lookback=tle_max_lookback,
            fallback_to_pdfm=tle_fallback,
        )
        # Only keep TLE outputs for the init_times requested in this run.
        corrected_grids = {
            init_time: tle_grids[init_time]
            for init_time in init_times
            if init_time in tle_grids
        }

    # 8. Write MICAPS4 products (PDFM or TLE) for each init_time
    for init_time in corrected_grids:
        product_writer.write_corrected_m4_products(
            init_time,
            corrected_grids[init_time],
            meta_map[init_time],
            leads_map[init_time],
            cfg,
            tle_enabled=tle_enabled,
        )

    # 9. Remove PDFM cache files older tthan cache_keep_hours
    if init_times:
        postprocess.cleanup_pdfm_cache(max(init_times), cfg)

    logger.info(f'Pipeline summary: {success} succeeded, {failed} failed')
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    print('Program pipeline.py started')
    total_start = arrow.now()

    parsed = _parse_pipeline_args(sys.argv)
    main(args=parsed)

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        f'Program pipeline.py finished, total time '
        f'{utils.format_time(total_elapsed)}'
    )
