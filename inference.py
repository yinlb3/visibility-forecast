#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Operational inference entry point.

Reads preprocessed intermediate data, loads station-specific PDF models
from the registry, corrects each grid cell with the model of its nearest
station, and writes corrected visibility products.

Founded in 2026-07-16
Modified in 2026-07-31
@author: yinlb
"""

import json
import os
import pathlib
import shutil
import sys
import typing

import arrow
import numpy as np
import pandas as pd
from meteva import base as meb

from src import grib_forecast, logger, model_registry, near_map
from src import p1_config_data as p1, pdf_model, utils


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
    station_ids = reg_meta.get('station_ids')
    if station_ids is None:
        # Logged once at startup in main(); keep per-init log at DEBUG
        logger.debug(
            'Registry metadata has no station_ids, '
            'falling back to filtered CSV order'
        )
        return sta

    station_ids = [int(sid) for sid in station_ids]
    indexed = sta.set_index('id')
    missing = [sid for sid in station_ids if sid not in indexed.index]
    if missing:
        raise ValueError(
            f'station_ids not found in station CSV: {missing[:5]} '
            f'({len(missing)} missing)'
        )
    return indexed.loc[station_ids].reset_index()


def _load_intermediate(init_time: str, cfg: typing.Dict) -> typing.Dict:
    """Load forecast grid and metadata for one init time.

    Args:
        init_time: YYYYMMDDHH string.
        cfg: Merged configuration dictionary.

    Returns:
        Dict with 'forecast' (n_lead, nlat, nlon) and 'meta'.

    Raises:
        FileNotFoundError: If the intermediate forecast file is missing.
        KeyError: If meta.json lacks the grid spec (re-run preprocess).
    """
    inter_dir = pathlib.Path(
        cfg['operational']['paths']['intermediate_dir']
    ) / init_time
    forecast_path = inter_dir / 'forecast_grid.npy'
    meta_path = inter_dir / 'meta.json'
    if not forecast_path.exists():
        raise FileNotFoundError(
            f'Forecast grid not found: {forecast_path}'
        )
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    if 'grid' not in meta:
        raise KeyError(
            f'meta.json has no grid spec: {meta_path}. '
            f'Please re-run preprocess.py for {init_time}.'
        )
    return {'forecast': np.load(str(forecast_path)), 'meta': meta}


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


def _write_corrected(
    init_time: str,
    corrected: np.ndarray,
    meta: typing.Dict,
    leads: typing.List[int],
    model_id: str,
    cfg: typing.Dict
) -> None:
    """Write corrected npy/meta and per-lead MICAPS4 products.

    Corrected arrays are saved next to the preprocessing intermediates so
    product_dir only contains the final MICAPS4 files. The output 'copy' mode
    controls whether corrected MICAPS4 products are written:
      - 'none'    - no corrected MICAPS4 products
      - 'files'   - reserved for raw products in preprocess.py
      - 'product' - corrected m4 products + copy to display_dir

    Args:
        init_time: YYYYMMDDHH string.
        corrected: Corrected forecast array (n_lead, nlat, nlon).
        meta: Intermediate metadata including the grid spec.
        leads: Processed lead hours matching corrected's first axis.
        model_id: Resolved model identifier for metadata.
        cfg: Merged configuration dictionary.
    """
    paths = cfg['operational']['paths']
    inf_cfg = cfg['operational']['inference']
    out_cfg = cfg['operational']['output']

    # 1. Save corrected arrays and metadata under intermediate_dir/{init_time}/
    out_dir = pathlib.Path(paths['intermediate_dir']) / init_time
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_grade = p1.grade_visibility(vis=corrected)
    np.save(str(out_dir / 'pred_vis.npy'), corrected.astype(np.float32))
    np.save(str(out_dir / 'pred_vis_grade.npy'), pred_grade)
    out_meta = {
        'init_time': init_time,
        'model_id': model_id,
        'region': meta.get('region'),
        'processed_leads': leads,
        'pred_shape': list(corrected.shape),
        'grid': meta['grid'],
    }
    with open(out_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(out_meta, f, indent=2)
    logger.info(f'Saved corrected arrays to {out_dir}')

    # 2. Skip MICAPS4 products when the copy mode does not ask for corrected
    # products. 'files' mode writes raw products in preprocess.py; 'none' writes
    # no products at all.
    copy_mode = str(out_cfg.get('copy', 'product')).lower()
    if copy_mode != 'product':
        logger.info(
            f'copy={copy_mode} for {init_time}: '
            f'not writing corrected MICAPS4 products'
        )
        return

    # 3. Write corrected MICAPS4 products (empty template disables)
    m4_template = out_cfg.get('corrected_m4_filename_template', '')
    if not m4_template:
        return
    title_template = out_cfg.get(
        'corrected_m4_title_template',
        'Corrected visibility {init_time}.{lead:03d}'
    )
    fake_cfg = cfg['operational']['synthetic']['fake_timestamp']
    write_and_copy = copy_mode == 'product'

    product_dir = pathlib.Path(paths['product_dir']) / init_time[:8]
    product_dir.mkdir(parents=True, exist_ok=True)
    display_dir_cfg = paths['display_dir']
    display_dir = (
        pathlib.Path(display_dir_cfg)
        if display_dir_cfg and display_dir_cfg != '<DISPLAY_DIR>'
        else None
    )
    if write_and_copy and display_dir is not None:
        display_dir.mkdir(parents=True, exist_ok=True)

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
        grd = grib_forecast.set_forecast_time(grd, init_time, lead)
        m4_path = product_dir / m4_template.format(
            init_time=init_time, lead=lead
        )
        # Explicit vmin/vmax: NaN cells break meteva's auto range calc
        meb.write_griddata_to_micaps4(
            da=grd,
            save_path=str(m4_path),
            title=title_template.format(init_time=init_time, lead=lead),
            vmin=0,
            vmax=float(inf_cfg['visibility_cap']),
        )

        # Adjust filesystem timestamp
        ts = utils.compute_fake_timestamp(init_time, fake_cfg)
        os.utime(str(m4_path), (ts, ts))

        # Copy to business display directory with flat structure
        if write_and_copy and display_dir is not None:
            display_path = display_dir / m4_path.name
            shutil.copy2(str(m4_path), str(display_path))
            os.utime(str(display_path), (ts, ts))

    logger.info(f'Saved {len(leads)} corrected m4 products to {product_dir}')


def _inference_one(
    init_time: str,
    cfg: typing.Dict,
    sta: pd.DataFrame,
    reg: model_registry.ModelRegistry
) -> None:
    """Run correction inference for one init time.

    Args:
        init_time: YYYYMMDDHH string.
        cfg: Merged configuration dictionary.
        sta: Station table from near_map.load_stations.
        reg: Model registry instance.

    Raises:
        ValueError: If data dimensions or model/station counts mismatch.
    """
    logger.info(f'Start inference {init_time}')

    # 1. Load intermediate data and validate lead alignment
    inter = _load_intermediate(init_time, cfg)
    forecast = inter['forecast']
    meta = inter['meta']
    leads = [int(lead) for lead in meta['processed_leads']]
    if forecast.shape[0] != len(leads):
        raise ValueError(
            f'forecast_grid lead dim {forecast.shape[0]} != '
            f'processed_leads {len(leads)} for {init_time}'
        )

    # 2. Load model and align stations to the training order
    model_id = reg.select_model_id(init_time=init_time)
    models = reg.load_model(model_id=model_id)
    logger.info(f'Loaded model {model_id}')
    sta_aligned = _align_stations(sta, reg.get_metadata(model_id))
    if isinstance(models, list) and len(models) != len(sta_aligned):
        raise ValueError(
            f'Model count {len(models)} != station count '
            f'{len(sta_aligned)}'
        )

    # 3. Correct grid cells with nearest-station models
    near_id = near_map.load_near_map(
        cfg['operational']['inference']['nearest_map_file'],
        meta['grid'], sta_aligned
    )
    corrected = _apply_correction(forecast, models, near_id, sta_aligned)
    cap = float(cfg['operational']['inference']['visibility_cap'])
    corrected[corrected > cap] = cap

    # 4. Write corrected products
    _write_corrected(init_time, corrected, meta, leads, model_id, cfg)
    logger.info(f'Finished inference {init_time}')


def main(args: typing.Optional[typing.Tuple[str, ...]] = None) -> None:
    """
    Main entry: load config, setup logger, run inference.

    Args:
        args: Optional parsed arguments; defaults to command-line args.
    """
    # 1. Parse arguments and load configuration
    if args is None:
        args = utils.parse_time_args(sys.argv)
    cfg = utils.load_config(operational=True)
    op_cfg = cfg['operational']

    # 2. Setup logger and resolve init times
    logger.setup_logger(
        log_dir=op_cfg['paths']['log_dir'],
        log_level=op_cfg['output']['log_level'],
        run_id=arrow.now().format('YYYYMMDDHHmmss')
    )
    init_times = utils.resolve_init_times(args, cfg)
    logger.info(f'Resolved init times: {init_times}')

    # 3. Load station table and model registry once per run
    sta = near_map.load_stations(cfg)
    reg = model_registry.ModelRegistry(
        registry_dir=op_cfg['paths']['model_registry_dir'],
        default_model_id=op_cfg['model']['default_model_id']
    )
    logger.info(f'Loaded {len(sta)} stations for correction')

    # 4. Fail fast when no model is available for the first init time
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

    # 5. Fail fast when the nearest-station map is missing or stale
    try:
        grid = near_map.grid_spec_from_config(cfg)
        near_map.load_near_map(
            op_cfg['inference']['nearest_map_file'], grid, sta
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f'Nearest-station map check failed: {exc}')
        sys.exit(1)

    # 6. Run inference for each init time
    success = 0
    failed = 0
    for init_time in init_times:
        try:
            _inference_one(init_time, cfg, sta, reg)
            success += 1
        except Exception as exc:
            logger.error(f'Inference failed for {init_time}: {exc}')
            failed += 1
            if len(init_times) == 1:
                raise

    # 7. Summarize and set exit code
    logger.info(f'Inference summary: {success} succeeded, {failed} failed')
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    print('Program inference.py started')
    total_start = arrow.now()

    parsed = utils.parse_time_args(sys.argv)
    main(args=parsed)

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        f'Program inference.py finished, total time '
        f'{utils.format_time(total_elapsed)}'
    )
