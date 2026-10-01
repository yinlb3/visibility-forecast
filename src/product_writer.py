# -*- coding: utf-8 -*-
"""
MICAPS4 product output for the operational pipeline.

Writes the corrected visibility grid as MICAPS4 files under
operational.paths.product_dir, one file per lead hour, and optionally
copies them into the business display directory with a flat structure.

Product names and time headers use Beijing time (CST, UTC+8), matching
the operational convention of the GRIB input files.

Founded in 2026-09-30
Modified in 2026-09-30
@author: yinlb, space-bunny
"""

import os
import pathlib
import shutil
import typing

import numpy as np
from meteva import base as meb    # type: ignore

from src import grib_forecast, logger, utils


def _display_dir_from_config(cfg: typing.Dict) -> typing.Optional[pathlib.Path]:
    """Resolve the business display directory.

    Args:
        cfg: Merged configuration dictionary.

    Returns:
        pathlib.Path: Display directory, or None when copying is disabled.
    """
    display_dir_cfg = cfg['operational']['paths']['display_dir']
    if not display_dir_cfg or display_dir_cfg == '<DISPLAY_DIR>':
        return None
    return pathlib.Path(display_dir_cfg)


def write_corrected_m4_products(
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
        m4_template = out_cfg['tle_m4_file_tmpl']
        title_template = out_cfg['tle_m4_title_tmpl']
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
    product_dir.mkdir(parents=True, exist_ok=True)
    display_dir = _display_dir_from_config(cfg)
    if display_dir is not None:
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