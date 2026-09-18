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
Modified in 2026-08-17
@author: yinlb
"""

import collections
import json
import os
import pathlib
import shutil
import sys
import traceback
import typing

import arrow
import joblib
import numpy as np
import pandas as pd
import xarray as xr
from meteva import base as meb    # type: ignore

from src import grib_forecast, logger, model_registry, near_map, tle_model
from src import p1_config_data as p1, pdf_model, utils


def _resolve_common_params(
    cfg: typing.Dict
) -> typing.Tuple[bool, bool, str, bool, bool]:
    """Resolve shared preprocessing/inference parameters from config.

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


def _load_one_task(
    init_time: str,
    lead: int,
    cfg: typing.Dict,
    skip_missing: bool,
    copy_mode: str,
    fallback_enabled: bool
) -> typing.Tuple[
    typing.Optional[typing.Tuple[str, int, xr.DataArray, pathlib.Path]],
    typing.List[typing.Tuple[str, str]]
]:
    """Load one (init_time, lead) task and supplement if missing.

    Does not use the global logger because this helper may run inside a
    joblib worker process. Returns log messages for the caller to emit.

    Args:
        init_time: YYYYMMDDHH string.
        lead: Forecast lead hour.
        cfg: Merged configuration dictionary.
        skip_missing: Whether to skip leads that cannot be loaded/supplemented.
        copy_mode: Output copy mode, also drives supplement strategy.
        fallback_enabled: Whether to use legacy synthetic fallback.

    Returns:
        Tuple of (optional result tuple, list of (level, message)).
    """
    messages = list()
    result = grib_forecast.load_forecast_task(init_time, cfg, lead)
    _, _, grid, file_path = result
    if grid is not None:
        return result, messages

    messages.append((
        'WARNING',
        f'Forecast GRIB missing for {init_time} f{lead:02d}: {file_path}'
    ))
    if copy_mode in ('product', 'files'):
        try:
            supplement_grd = grib_forecast.load_supplement_grid(
                init_time, cfg, lead, copy_mode
            )
            messages.append((
                'WARNING',
                f'Using {copy_mode} supplement data for {init_time} '
                f'f{lead:02d}'
            ))
            return (
                init_time, lead, supplement_grd,
                pathlib.Path(f'{copy_mode}_supplement')
            ), messages
        except Exception as supplement_e:
            messages.append((
                'ERROR',
                f'{copy_mode} supplement failed for {init_time} '
                f'f{lead:02d}: {supplement_e}'
            ))
            if not skip_missing:
                raise FileNotFoundError(
                    f'Forecast GRIB missing and {copy_mode} '
                    f'supplement failed: {file_path}'
                )
    elif fallback_enabled:
        try:
            fallback_grd = grib_forecast.generate_fallback_grid(
                cfg, init_time, lead
            )
            messages.append((
                'WARNING',
                f'Using fallback sample data for {init_time} f{lead:02d}'
            ))
            return (
                init_time, lead, fallback_grd, pathlib.Path('fallback')
            ), messages
        except Exception as fallback_e:
            messages.append((
                'ERROR',
                f'Fallback generation failed for {init_time} '
                f'f{lead:02d}: {fallback_e}'
            ))
            if not skip_missing:
                raise FileNotFoundError(
                    f'Forecast GRIB missing and fallback failed: '
                    f'{file_path}'
                )
    elif not skip_missing:
        raise FileNotFoundError(
            f'Forecast GRIB missing: {file_path}'
        )

    return None, messages


def _load_forecast_grids(
    tasks: typing.List[typing.Tuple[str, int]],
    cfg: typing.Dict,
    skip_missing: bool,
    copy_mode: str,
    fallback_enabled: bool
) -> typing.List[typing.Tuple[str, int, xr.DataArray, pathlib.Path]]:
    """Load and interpolate GRIB forecasts in parallel, supplementing missing.

    Args:
        tasks: List of (init_time, lead) tuples to process.
        cfg: Merged configuration dictionary.
        skip_missing: Whether to skip leads that cannot be loaded/supplemented.
        copy_mode: Output copy mode, also drives supplement strategy.
        fallback_enabled: Whether to use legacy synthetic fallback.

    Returns:
        List of (init_time, lead, grid_data, source_path) tuples.
    """
    n_jobs = int(cfg['operational']['parallel']['n_jobs'])
    if n_jobs == 0:
        n_jobs = 1

    task_results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_load_one_task)(
            init_time, lead, cfg, skip_missing, copy_mode, fallback_enabled
        )
        for init_time, lead in tasks
    )

    results = list()
    for result, messages in task_results:
        for level, message in messages:
            getattr(logger, level.lower())(message)
        if result is not None:
            results.append(result)

    return results


def _build_forecast_stack(
    results: typing.List[typing.Tuple[int, xr.DataArray, pathlib.Path]],
    all_leads: typing.Optional[typing.List[int]] = None
) -> typing.Tuple[np.ndarray, typing.List[int]]:
    """Stack raw forecast grids by lead hour.

    When all_leads is given, the returned stack has one slot for every lead
    in all_leads; missing leads are filled with NaN so that lagged grids used
    by TLE keep the same shape as the current run.

    Args:
        results: List of (lead, grid_data, grib_path) tuples for one init_time.
        all_leads: Optional complete lead list. If provided, processed_leads
            equals all_leads and missing entries are NaN-filled.

    Returns:
        Tuple of (forecast_stack, processed_leads).
    """
    sorted_results = sorted(results, key=lambda x: x[0])
    if all_leads is None:
        grids = [
            np.squeeze(forecast_grd.values)
            for _, forecast_grd, _ in sorted_results
        ]
        processed_leads = [lead for lead, _, _ in sorted_results]
        return np.stack(grids, axis=0), processed_leads

    lead_to_idx = {lead: idx for idx, lead in enumerate(all_leads)}
    first_grid = np.squeeze(sorted_results[0][1].values)
    forecast_stack = np.full(
        (len(all_leads),) + first_grid.shape,
        np.nan,
        dtype=first_grid.dtype
    )
    for lead, forecast_grd, _ in sorted_results:
        if lead in lead_to_idx:
            forecast_stack[lead_to_idx[lead]] = np.squeeze(
                forecast_grd.values
            )
    return forecast_stack, list(all_leads)


def _build_raw_meta(
    init_time: str,
    processed_leads: typing.List[int],
    forecast_stack: np.ndarray,
    results: typing.List[typing.Tuple[int, xr.DataArray, pathlib.Path]],
    cfg: typing.Dict,
    load_obs: bool = False,
    obs_data: typing.Optional[typing.Dict] = None,
) -> typing.Dict:
    """Build preprocessing metadata dictionary from processed results.

    Args:
        init_time: YYYYMMDDHH string.
        processed_leads: Lead hours that were actually processed.
        forecast_stack: Stacked forecast array (n_lead, nlat, nlon).
        results: List of (lead, grid_data, grib_path) tuples.
        cfg: Merged configuration dictionary.
        load_obs: Whether observation arrays were saved.
        obs_data: Optional dictionary with observation arrays and metadata.

    Returns:
        Metadata dictionary including grid spec.
    """
    output_cfg = cfg['operational']['output']
    copy_mode = str(output_cfg['copy']).lower()

    meta = {
        'init_time': init_time,
        'region': cfg['operational']['runtime']['region'],
        'forecast_shape': list(forecast_stack.shape),
        'lead_hours': cfg['operational']['forecast']['lead_hours'],
        'processed_leads': processed_leads,
        'm4_products': list(),
        'copy': copy_mode,
    }
    # Grid spec from the first grid's coordinates so downstream consumers
    # can rebuild the exact lat-lon grid.
    first_grd = sorted(results, key=lambda x: x[0])[0][1]
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

    return meta


def _save_raw_intermediate(
    init_time: str,
    forecast_stack: np.ndarray,
    meta: typing.Dict,
    cfg: typing.Dict,
    load_obs: bool = False,
    obs_data: typing.Optional[typing.Dict] = None,
) -> None:
    """Save raw forecast intermediate arrays and metadata to disk.

    Args:
        init_time: YYYYMMDDHH string.
        forecast_stack: Stacked forecast array (n_lead, nlat, nlon).
        meta: Metadata dictionary including grid spec.
        cfg: Merged configuration dictionary.
        load_obs: Whether observation arrays should be saved.
        obs_data: Optional dictionary with observation arrays and metadata.
    """
    out_dir = pathlib.Path(
        cfg['operational']['paths']['intermediate_dir']
    ) / init_time
    os.makedirs(out_dir, exist_ok=True)

    np.save(str(out_dir / 'forecast_grid.npy'), forecast_stack)
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

    logger.info(f'Saved raw intermediates to {out_dir}')


def _align_stations(
    sta: pd.DataFrame,
    reg_meta: typing.Dict
) -> pd.DataFrame:
    """Align station table to the model's training station order.

    Reindexes by the registry metadata station_ids when available;
    otherwise falls back to the filtered CSV order with a warning.

    Args:
        sta: Station table from near_map.load_stations.
        reg_meta: Registry metadata of the selected model.

    Returns:
        pd.DataFrame: Station table in training order.

    Raises:
        ValueError: If station_ids contain ids missing from the table.
    """
    if 'station_ids' not in reg_meta:
        # Logged once at startup in main(); keep per-init log at DEBUG
        logger.debug(
            'Registry metadata has no station_ids, '
            'falling back to filtered CSV order'
        )
        return sta

    station_ids = [int(sid) for sid in reg_meta['station_ids']]
    indexed = sta.set_index('id')
    missing = [sid for sid in station_ids if sid not in indexed.index]
    if missing:
        raise ValueError(
            f'station_ids not found in station CSV: {missing[:5]} '
            f'({len(missing)} missing)'
        )
    return indexed.loc[station_ids].reset_index()


def _apply_correction(
    forecast: np.ndarray,
    models: typing.Union[pdf_model.PDF, typing.List[pdf_model.PDF]],
    near_id: np.ndarray,
    sta: pd.DataFrame
) -> np.ndarray:
    """Correct forecast grid cells with station-specific PDF models.

    Cells whose nearest station model is not fitted keep their raw
    values; NaN cells remain NaN.

    Args:
        forecast: Raw forecast array (n_lead, nlat, nlon).
        models: Single PDF instance or list of PDF per station.
        near_id: Nearest station id per grid cell (nlat, nlon).
        sta: Station table aligned to the model training order.

    Returns:
        Corrected forecast array, same shape as forecast.
    """
    if forecast.dtype == np.float32:
        corrected = forecast.copy()
    else:
        corrected = forecast.astype(np.float32).copy()
    if isinstance(models, pdf_model.PDF):
        out = models.predict(corrected)
        if out is None:
            logger.warning('PDF model is not fitted, keeping raw values')
            return corrected
        return out

    for sta_idx, station_id in enumerate(sta['id'].values):
        mask = near_id == station_id
        if not np.any(mask):
            continue
        out = models[sta_idx].predict(forecast[:, mask])
        if out is None:
            logger.warning(
                f'Model for station {station_id} is not fitted, '
                f'keeping raw values'
            )
            continue
        corrected[:, mask] = out
    return corrected


def _run_correction(
    init_time: str,
    forecast: np.ndarray,
    meta: typing.Dict,
    cfg: typing.Dict,
    sta: pd.DataFrame,
    reg: model_registry.ModelRegistry,
    near_id: typing.Optional[np.ndarray] = None
) -> typing.Tuple[np.ndarray, str, typing.List[int]]:
    """Run correction inference on in-memory forecast arrays.

    Args:
        init_time: YYYYMMDDHH string.
        forecast: Raw forecast array (n_lead, nlat, nlon).
        meta: Intermediate metadata including 'grid' and 'processed_leads'.
        cfg: Merged configuration dictionary.
        sta: Station table from near_map.load_stations.
        reg: Model registry instance.
        near_id: Optional nearest-station map (nlat, nlon). If None, it is
            loaded from the configured nearest_map_file.

    Returns:
        Tuple of (corrected array, model_id, leads).

    Raises:
        ValueError: If data dimensions or model/station counts mismatch.
    """
    leads = [int(lead) for lead in meta['processed_leads']]
    if forecast.shape[0] != len(leads):
        raise ValueError(
            f'forecast_grid lead dim {forecast.shape[0]} != '
            f'processed_leads {len(leads)} for {init_time}'
        )

    # 1. Load model and align stations to the training order
    model_id = reg.select_model_id(init_time=init_time)
    models = reg.load_model(model_id=model_id)
    logger.info(f'Loaded model {model_id}')
    sta_aligned = _align_stations(sta, reg.get_metadata(model_id))
    if isinstance(models, list) and len(models) != len(sta_aligned):
        raise ValueError(
            f'Model count {len(models)} != station count '
            f'{len(sta_aligned)}'
        )

    # 2. Correct grid cells with nearest-station models
    if near_id is None:
        near_id = near_map.load_near_map(
            cfg['operational']['inference']['nearest_map_file'],
            meta['grid'], sta_aligned
        )
    corrected = _apply_correction(forecast, models, near_id, sta_aligned)
    cap = float(cfg['operational']['inference']['visibility_cap'])
    corrected[corrected > cap] = cap

    return corrected, model_id, leads


def _save_corrected_intermediate(
    init_time: str,
    corrected: np.ndarray,
    meta: typing.Dict,
    leads: typing.List[int],
    model_id: str,
    cfg: typing.Dict
) -> None:
    """Save corrected arrays and metadata to intermediate_dir.

    Args:
        init_time: YYYYMMDDHH string.
        corrected: Corrected forecast array (n_lead, nlat, nlon).
        meta: Intermediate metadata including the grid spec.
        leads: Processed lead hours matching corrected's first axis.
        model_id: Resolved model identifier for metadata.
        cfg: Merged configuration dictionary.
    """
    out_dir = pathlib.Path(
        cfg['operational']['paths']['intermediate_dir']
    ) / init_time
    os.makedirs(out_dir, exist_ok=True)
    pred_grade = p1.grade_visibility(vis=corrected)
    np.save(str(out_dir / 'pred_vis.npy'), corrected.astype(np.float32))
    np.save(str(out_dir / 'pred_vis_grade.npy'), pred_grade)
    out_meta = {
        'init_time': init_time,
        'model_id': model_id,
        'region': meta['region'],
        'processed_leads': leads,
        'pred_shape': list(corrected.shape),
        'grid': meta['grid'],
    }
    with open(out_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(out_meta, f, indent=2)
    logger.info(f'Saved corrected arrays to {out_dir}')


def _write_corrected_m4_products(
    init_time: str,
    corrected: np.ndarray,
    meta: typing.Dict,
    leads: typing.List[int],
    cfg: typing.Dict,
    tle_enabled: bool = False,
) -> None:
    """Write corrected MICAPS4 products to product_dir.

    If display_dir is configured in operational.paths, also copy products
    to the business display directory.

    Args:
        init_time: YYYYMMDDHH string.
        corrected: Corrected forecast array (n_lead, nlat, nlon).
        meta: Intermediate metadata including the grid spec.
        leads: Processed lead hours matching corrected's first axis.
        cfg: Merged configuration dictionary.
        tle_enabled: Whether to use TLE-specific output templates.
    """
    paths = cfg['operational']['paths']
    inf_cfg = cfg['operational']['inference']
    out_cfg = cfg['operational']['output']

    if tle_enabled:
        m4_template = out_cfg.get('tle_m4_file_tmpl', '')
        title_template = out_cfg.get('tle_m4_title_tmpl', '')
    else:
        m4_template = out_cfg['corr_m4_file_tmpl']
        title_template = out_cfg['corr_m4_title_tmpl']
    if not m4_template:
        return
    effective_num = int(out_cfg['effective_num'])
    fake_cfg = cfg['operational']['synthetic']['fake_timestamp']

    # Output m4 products use Beijing time (CST, UTC+8).
    bj_init_time = utils.utc_to_beijing(init_time)

    product_dir = pathlib.Path(paths['product_dir']) / bj_init_time[:8]
    os.makedirs(product_dir, exist_ok=True)
    display_dir_cfg = paths['display_dir']
    display_dir = (
        pathlib.Path(display_dir_cfg)
        if display_dir_cfg and display_dir_cfg != '<DISPLAY_DIR>'
        else None
    )
    if display_dir is not None:
        os.makedirs(display_dir, exist_ok=True)

    grid = meta['grid']
    meb_grid = meb.grid(
        [
            grid['slon'],
            grid['slon'] + (grid['nlon'] - 1) * grid['dlon'],
            grid['dlon'],
        ],
        [
            grid['slat'],
            grid['slat'] + (grid['nlat'] - 1) * grid['dlat'],
            grid['dlat'],
        ],
    )
    for i, lead in enumerate(leads):
        grd = meb.grid_data(meb_grid, corrected[i])
        grd = grib_forecast.set_forecast_time(grd, bj_init_time, lead)
        m4_path = product_dir / m4_template.format(
            init_time=bj_init_time, lead=lead
        )
        # Explicit vmin/vmax: NaN cells break meteva's auto range calc
        meb.write_griddata_to_micaps4(
            da=grd,
            save_path=str(m4_path),
            title=title_template.format(init_time=bj_init_time, lead=lead),
            vmin=0,
            vmax=float(inf_cfg['visibility_cap']),
            effectiveNum=effective_num,
        )

        # Adjust filesystem timestamp using Beijing time.
        ts = utils.compute_fake_timestamp(bj_init_time, fake_cfg)
        os.utime(str(m4_path), (ts, ts))

        # Copy to business display directory with flat structure
        if display_dir is not None:
            display_path = display_dir / m4_path.name
            shutil.copy2(str(m4_path), str(display_path))
            os.utime(str(display_path), (ts, ts))

    logger.info(f'Saved {len(leads)} corrected m4 products to {product_dir}')


def _build_tasks(
    init_times: typing.List[str],
    lead_hours: typing.List[int],
    cfg: typing.Dict,
    skip_missing: bool,
    copy_mode: str
) -> typing.List[typing.Tuple[str, int]]:
    """Build the full (init_time, lead) task list.

    Missing GRIB files are skipped only when copy='none' and skip_missing
    is true; otherwise they are kept for supplementation.

    Args:
        init_times: List of YYYYMMDDHH strings.
        lead_hours: Lead hours to process.
        cfg: Merged configuration dictionary.
        skip_missing: Whether missing GRIB may be skipped.
        copy_mode: Output copy mode.

    Returns:
        List of (init_time, lead) tuples.
    """
    tasks = list()
    for init_time in init_times:
        for lead in lead_hours:
            file_exists = grib_forecast.forecast_file_exists(
                init_time, cfg, lead
            )
            if not file_exists and copy_mode == 'none' and skip_missing:
                logger.warning(
                    f'Forecast GRIB missing, skipped: {init_time} f{lead:02d}'
                )
                continue
            tasks.append((init_time, lead))
    return tasks


def _save_pdfm_cache(
    init_time: str,
    corrected: np.ndarray,
    cfg: typing.Dict,
) -> None:
    """Save PDFM-corrected grid to cache for future TLE runs.

    Args:
        init_time: YYYYMMDDHH string.
        corrected: Corrected grid array (n_lead, nlat, nlon).
        cfg: Merged configuration dictionary.
    """
    cache_dir_cfg = cfg['operational']['paths'].get('pdfm_cache_dir', '')
    if not cache_dir_cfg or cache_dir_cfg == '<PDFM_CACHE_DIR>':
        return
    cache_dir = pathlib.Path(cache_dir_cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'{init_time}.npy'
    np.save(str(cache_path), corrected)
    logger.info(f'Saved PDFM cache: {cache_path}')


def _load_pdfm_cache(
    init_time: str,
    max_lookback: int,
    cfg: typing.Dict,
) -> typing.Dict[str, np.ndarray]:
    """Load cached PDFM-corrected grids from previous init times.

    Loads one .npy file per lagged init_time if it exists. Missing hours are
    silently skipped; TLE fallback will use PDFM values for those.

    Args:
        init_time: Target YYYYMMDDHH string.
        max_lookback: Maximum hours to look back.
        cfg: Merged configuration dictionary.

    Returns:
        Mapping lagged_init_time -> cached grid array.
    """
    cache_dir_cfg = cfg['operational']['paths'].get('pdfm_cache_dir', '')
    if not cache_dir_cfg or cache_dir_cfg == '<PDFM_CACHE_DIR>':
        return {}
    cache_dir = pathlib.Path(cache_dir_cfg)
    cached: typing.Dict[str, np.ndarray] = {}
    for i in range(1, max_lookback + 1):
        lag_init = utils.shift_init_time(init_time, -i)
        cache_path = cache_dir / f'{lag_init}.npy'
        if cache_path.exists():
            cached[lag_init] = np.load(str(cache_path))
            logger.info(f'Loaded PDFM cache: {cache_path}')
    return cached


def _cleanup_pdfm_cache(
    latest_init_time: str,
    cfg: typing.Dict,
) -> None:
    """Remove PDFM cache files older than cache_keep_hours.

    Args:
        latest_init_time: Newest YYYYMMDDHH string processed in this run.
        cfg: Merged configuration dictionary.
    """
    tle_cfg = cfg['operational']['inference'].get('tle', {})
    keep_hours = int(tle_cfg.get('cache_keep_hours', 48))
    cache_dir_cfg = cfg['operational']['paths'].get('pdfm_cache_dir', '')
    if not cache_dir_cfg or cache_dir_cfg == '<PDFM_CACHE_DIR>':
        return
    cache_dir = pathlib.Path(cache_dir_cfg)
    if not cache_dir.exists():
        return
    latest_dt = arrow.get(latest_init_time, 'YYYYMMDDHH')
    cutoff = latest_dt.shift(hours=-keep_hours)
    removed = 0
    for cache_path in cache_dir.glob('*.npy'):
        try:
            file_init = arrow.get(cache_path.stem, 'YYYYMMDDHH')
            if file_init < cutoff:
                cache_path.unlink()
                removed += 1
        except (ValueError, OSError):
            continue
    if removed:
        logger.info(f'Removed {removed} old PDFM cache files')


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
    lag_results = _load_forecast_grids(
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
            forecast_stack, processed_leads = _build_forecast_stack(
                results, all_leads=lead_hours
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

    TLE averaging and product writing are handled by the caller so that
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
    forecast_stack, processed_leads = _build_forecast_stack(results)
    meta = _build_raw_meta(
        init_time, processed_leads, forecast_stack, results,
        cfg, load_obs, obs_data
    )
    if save_intermediate:
        _save_raw_intermediate(
            init_time, forecast_stack, meta, cfg, load_obs, obs_data
        )

    # 2. Correction
    corrected, model_id, leads = _run_correction(
        init_time, forecast_stack, meta, cfg, sta, reg, near_id=near_id
    )
    if save_intermediate:
        _save_corrected_intermediate(
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
        obs_data = _load_observations(cfg)

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
    tasks = _build_tasks(init_times, lead_hours, cfg, skip_missing, copy_mode)
    logger.info(f'Total tasks to process: {len(tasks)}')
    results = _load_forecast_grids(
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
            _save_pdfm_cache(init_time, corrected, cfg)
            success += 1
        except Exception as exc:
            logger.error(f'Pipeline failed for {init_time}: {exc}')
            failed += 1
            if len(init_times) == 1:
                raise

    # 7. Optional time-lagged ensemble across PDFM-corrected init times
    tle_cfg = op_cfg['inference'].get('tle', {})
    tle_enabled = bool(tle_cfg.get('enabled', False))
    if tle_enabled and corrected_grids:
        logger.info(
            'Applying time-lagged ensemble (TLE) to PDFM-corrected grids'
        )
        tle_max_lookback = int(tle_cfg.get('max_lookback_hours', 24))
        tle_fallback = bool(tle_cfg.get('fallback_to_pdfm', True))
        # Use the lead list from the first corrected grid; all init times share
        # the same operational lead_hours.
        first_leads = next(iter(leads_map.values()))
        # Merge the current run with cached PDFM grids from previous hours
        # so that anti-diagonal lagged samples are available for TLE.
        combined_grids = dict(corrected_grids)
        for init_time in init_times:
            cached = _load_pdfm_cache(init_time, tle_max_lookback, cfg)
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
        tle_grids = tle_model.apply_equal_tle(
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
        _write_corrected_m4_products(
            init_time,
            corrected_grids[init_time],
            meta_map[init_time],
            leads_map[init_time],
            cfg,
            tle_enabled=tle_enabled,
        )

    # 9. Remove PDFM cache files older than cache_keep_hours
    if init_times:
        _cleanup_pdfm_cache(max(init_times), cfg)

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
