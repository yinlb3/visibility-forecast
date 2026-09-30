# -*- coding: utf-8 -*-
"""
Forecast data reading and preprocessing for the operational pipeline.

Turns raw GRIB2 files into regular (n_lead, nlat, nlon) arrays:

- build_tasks lists the (init_time, lead) pairs to process.
- load_forecast_grids reads and interpolates them in parallel, applying
  the configured missing-file strategy.
- build_forecast_stack stacks the per-lead grids into one array.
- build_raw_meta and the save_* helpers write diagnostic intermediate
  files; the pipeline keeps everything in memory, so writing them is
  optional and controlled by operational.output.save_intermediate.

Observation loading for research workflows lives here too
(load_observations), since it is the other data-ingest path.

Founded in 2026-09-30
Modified in 2026-09-30
@author: yinlb, space-bunny
"""

import json
import os
import pathlib
import typing

import joblib
import numpy as np
import xarray as xr

from src import grib_forecast, logger
from src import p1_config_data as p1


# ==================== Task Planning ====================


def build_tasks(
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


# ==================== GRIB Reading and Interpolation ====================


def load_one_task(
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


def load_forecast_grids(
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
        joblib.delayed(load_one_task)(
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


def build_forecast_stack(
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


# ==================== Observations ====================


def load_observations(cfg: typing.Dict) -> typing.Dict:
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

    # 'mlyr' selects the sub-region; any other value means all of east china.
    if region == 'mlyr':
        region_provinces = tuple(region_cfg['mlyr_provinces'])
    else:
        region_provinces = tuple(region_cfg['provinces'])

    sta, idx_east_china = p1.read_stationtion(
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


# ==================== Intermediate Storage ====================
# Diagnostic artifacts only: the pipeline keeps everything in memory, so
# these files are written solely when save_intermediate is enabled.


def build_raw_meta(
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


def save_raw_intermediate(
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


def save_corrected_intermediate(
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