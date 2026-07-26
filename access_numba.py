#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Founded in 2024-03-24
Modified in 2026-07-21
@author: yinlb
"""
import pathlib
import typing

import arrow
import joblib
import numpy as np
import pandas as pd
from numba import njit

from src import p1_config_data as p1
from src import utils, vis_acc


# Load access experiment configuration at import time
_ACCESS_CFG = utils.CFG['access']
_ACCESS_INPUT = _ACCESS_CFG['input_files']
_ACCESS_OUTPUT = _ACCESS_CFG['output_files']
_ACCESS_PARAMS = _ACCESS_CFG['params']

_N_HOURS = int(_ACCESS_PARAMS['n_hours'])
_VAL_DAYS = int(_ACCESS_PARAMS['val_days'])
_VISIBILITY_CAP = float(_ACCESS_PARAMS['visibility_cap'])
_SAVE_PDFM_MODELS = bool(_ACCESS_PARAMS['save_pdfm_models'])
_RUN_SCHEME_0 = bool(_ACCESS_PARAMS['run_scheme_0'])
_RUN_SCHEME_1 = bool(_ACCESS_PARAMS['run_scheme_1'])
_RUN_SCHEME_2 = bool(_ACCESS_PARAMS['run_scheme_2'])
_RUN_REGION = str(_ACCESS_PARAMS['run_region']).lower()


@njit
def _pdf_predict_numba(
    c: np.ndarray,
    pr0: np.ndarray,
    out: np.ndarray,
) -> None:
    """Numba-compiled PDF matching prediction."""
    n = pr0.size
    n_c = c.shape[0]
    for i in range(n):
        value = pr0[i]
        if np.isnan(value):
            continue
        if value < c[0, 1]:
            out[i] = c[0, 0]
            continue
        if value > c[-1, 1]:
            out[i] = c[-1, 0]
            continue
        left = 0
        right = n_c - 1
        while right - left > 1:
            mid = round((left + right) / 2)
            if value <= c[mid, 1]:
                right = mid
            else:
                left = mid
        denom = c[right, 1] - c[left, 1]
        if denom == 0.0:
            out[i] = c[right, 0]
        else:
            k = (c[right, 0] - c[left, 0]) / denom
            out[i] = c[left, 0] + k * (value - c[left, 1])


class PDF:
    """PDF matching model for visibility forecast correction."""

    def __init__(self):
        self.c = None

    def fit(self, ob: np.ndarray, pr: np.ndarray):
        """Fit PDF matching curve from obs and forecast samples."""
        ob = ob[~np.isnan(ob)][::10000]
        pr = pr[~np.isnan(pr)][::10000]
        a = np.unique(ob)
        a = np.sort(a)
        self.c = np.zeros((len(a), 2), dtype=np.float32)
        self.c[:, 0] = a
        pr = np.sort(pr)
        for i, a0 in enumerate(a):
            p0 = np.mean(ob <= a0)
            j = round(p0 * (len(pr) - 1))
            self.c[i, 1] = pr[j]

    def predict(self, pr0: np.ndarray) -> typing.Optional[np.ndarray]:
        """Apply PDF matching correction to forecast array."""
        if self.c is None:
            return None
        shape = pr0.shape
        pr0 = pr0.flatten().astype(np.float32)
        out = np.zeros_like(pr0) + np.nan
        _pdf_predict_numba(self.c, pr0, out)
        return out.reshape(shape)


def _build_region_index(
    sta_path: pathlib.Path,
    provinces_all: typing.Tuple[str, ...],
    provinces_region: typing.Tuple[str, ...],
) -> np.ndarray:
    """Build region mask for stations already filtered to east china.

    The input numpy arrays (vis1183_ob.npy, vis_gjz.npy) already contain
    only east-china stations (1183), so we only need the MLYR subset mask
    of length 1183.
    """
    sta_all = pd.read_csv(str(sta_path), low_memory=False, encoding='utf-8')
    # Filter to east china first, matching the input arrays
    idx_all = None
    for province in provinces_all:
        mask = sta_all.loc[:, 'province'] == province
        idx_all = mask if idx_all is None else idx_all | mask
    sta = sta_all.loc[idx_all].copy()
    sta.reset_index(drop=True, inplace=True)

    # Build MLYR mask on filtered stations
    n_east = len(sta)
    idx_region = np.zeros(n_east, dtype=np.bool_)
    for i in range(n_east):
        if sta.loc[i, 'province'] in provinces_region:
            idx_region[i] = True
    return idx_region


def _load_and_prepare(
    data_dir: pathlib.Path,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                  np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load obs/forecast, mask missing values, and split train/val."""
    ob_east_china = np.load(str(data_dir / _ACCESS_INPUT['observation_npy']))[:, :, 1:, :]
    pr_east_china = np.load(str(data_dir / _ACCESS_INPUT['forecast_npy']))[:1461, :, 1:, :]
    ob_east_china = utils.mask_missing(ob_east_china)
    pr_east_china = utils.mask_missing(pr_east_china)

    cfg_regions = utils.CFG['draw']['regions']
    sta_path = data_dir / _ACCESS_INPUT['station_csv']
    idx_mlyr = _build_region_index(
        sta_path,
        tuple(cfg_regions['provinces']),
        tuple(cfg_regions['mlyr_provinces']),
    )

    ob_mlyr = ob_east_china[..., idx_mlyr]
    pr_mlyr = pr_east_china[..., idx_mlyr]

    train_ob = ob_east_china[:-_VAL_DAYS, ...]
    train_pr = pr_east_china[:-_VAL_DAYS, ...]
    val_ob = ob_east_china[-_VAL_DAYS:, ...]
    val_pr = pr_east_china[-_VAL_DAYS:, ...]
    train_ob_mlyr = ob_mlyr[:-_VAL_DAYS, ...]
    train_pr_mlyr = pr_mlyr[:-_VAL_DAYS, ...]
    val_ob_mlyr = ob_mlyr[-_VAL_DAYS:, ...]
    val_pr_mlyr = pr_mlyr[-_VAL_DAYS:, ...]

    return (train_ob, train_pr, val_ob, val_pr,
            train_ob_mlyr, train_pr_mlyr, val_ob_mlyr, val_pr_mlyr)


def _apply_tle_to_4d(
    arr: np.ndarray,
    n_hours: int = _N_HOURS,
) -> np.ndarray:
    """Apply TLE to 4D array (days, n_hours, lead, stations) and return 4D."""
    shape = arr.shape
    arr_3d = np.reshape(arr, (-1, n_hours, shape[-1]))
    arr_3d = utils.apply_tle_equal_weight(arr_3d, n_hours)
    return np.reshape(arr_3d, shape)


def _save_npy(path: pathlib.Path, arr: np.ndarray) -> None:
    """Save numpy array as float32 to path."""
    np.save(str(path), arr.astype(np.float32))


def _print_metrics(acc: vis_acc.VisAcc) -> None:
    """Print standard verification metrics."""
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts_ge())
    print(acc.get_far_ge())
    print(acc.get_mar_ge())


def _verify_predictions(val_ob: np.ndarray, pred_path: pathlib.Path) -> None:
    """Load predictions, cap at visibility cap, and print metrics."""
    pred = np.load(str(pred_path))
    pred[pred >= _VISIBILITY_CAP] = _VISIBILITY_CAP
    acc = vis_acc.VisAcc(val_ob, pred)
    _print_metrics(acc)


def _run_scheme_overall(
    train_ob: np.ndarray,
    train_pr: np.ndarray,
    val_pr: np.ndarray,
    model_path: pathlib.Path,
    pred_path: pathlib.Path,
    region: str,
) -> None:
    """Run scheme 0: overall PDF matching."""
    time_arrow = arrow.now()
    model = PDF()
    model.fit(train_ob.flatten(), train_pr.flatten())
    elapsed = (arrow.now() - time_arrow).total_seconds()
    print(f'Scheme 0 {region} {utils.format_time(elapsed)}')
    if _SAVE_PDFM_MODELS:
        joblib.dump(model, str(model_path))

    pred_pdfm = model.predict(val_pr)
    pred_tle = _apply_tle_to_4d(pred_pdfm)
    _save_npy(pred_path, pred_tle)


def _run_scheme_hour(
    train_ob: np.ndarray,
    train_pr: np.ndarray,
    val_pr: np.ndarray,
    model_path: pathlib.Path,
    pred_path: pathlib.Path,
    region: str,
) -> None:
    """Run scheme 1: hour-specific PDF matching."""
    pred_pdfm = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    time_arrow = arrow.now()
    models = list()
    for i in range(_N_HOURS):
        model = PDF()
        model.fit(
            train_ob[:, :, i, :].flatten(),
            train_pr[:, :, i, :].flatten(),
        )
        models.append(model)
    elapsed = (arrow.now() - time_arrow).total_seconds()
    print(f'Scheme 1 {region} {utils.format_time(elapsed)}')
    if _SAVE_PDFM_MODELS:
        joblib.dump(models, str(model_path))

    for i in range(_N_HOURS):
        pred_pdfm[:, :, i, :] = models[i].predict(val_pr[:, :, i, :])
    pred_tle = _apply_tle_to_4d(pred_pdfm)
    _save_npy(pred_path, pred_tle)


def _run_scheme_station(
    train_ob: np.ndarray,
    train_pr: np.ndarray,
    val_pr: np.ndarray,
    model_path: pathlib.Path,
    pred_pdfm_path: pathlib.Path,
    pred_tle_path: pathlib.Path,
    region: str,
) -> None:
    """Run scheme 2: station-specific PDF matching."""
    pred_pdfm = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    time_arrow = arrow.now()
    models = list()
    n_stations = val_pr.shape[-1]
    for i in range(n_stations):
        model = PDF()
        model.fit(
            train_ob[:, :, :, i].flatten(),
            train_pr[:, :, :, i].flatten(),
        )
        models.append(model)
    elapsed = (arrow.now() - time_arrow).total_seconds()
    print(f'Scheme 2 {region} {utils.format_time(elapsed)}')
    if _SAVE_PDFM_MODELS:
        joblib.dump(models, str(model_path))

    for i in range(n_stations):
        pred_pdfm[:, :, :, i] = models[i].predict(val_pr[:, :, :, i])
    pred_tle = _apply_tle_to_4d(pred_pdfm)
    _save_npy(pred_pdfm_path, pred_pdfm)
    _save_npy(pred_tle_path, pred_tle)


def main() -> None:
    """Run PDF matching correction experiment."""
    data_dir = pathlib.Path(utils.CFG['paths']['data_dir'])
    out_cfg = _ACCESS_OUTPUT

    _run_all = _RUN_REGION in ('all', 'both')
    _run_mlyr = _RUN_REGION in ('mlyr', 'both')

    # 1. Load and prepare data
    (train_ob, train_pr, val_ob, val_pr,
     train_ob_mlyr, train_pr_mlyr, val_ob_mlyr, val_pr_mlyr) = \
        _load_and_prepare(data_dir)

    # Save mlyr validation subsets for downstream use
    if _run_mlyr:
        _save_npy(data_dir / 'vis_ob_mlyr.npy', val_ob_mlyr)
        _save_npy(data_dir / 'vis_pr_mlyr.npy', val_pr_mlyr)

    # 2. Scheme 0: overall PDF matching
    pdfm_models = out_cfg['pdfm_models']
    corrected = out_cfg['corrected_forecasts']
    if _RUN_SCHEME_0:
        if _run_all:
            _run_scheme_overall(
                train_ob, train_pr, val_pr,
                data_dir / pdfm_models['pdfm0'],
                data_dir / corrected['pred0'],
                'east_china',
            )
        if _run_mlyr:
            _run_scheme_overall(
                train_ob_mlyr, train_pr_mlyr, val_pr_mlyr,
                data_dir / pdfm_models['pdfm0_mlyr'],
                data_dir / corrected['pred0_mlyr'],
                'mlyr',
            )

    # 3. Scheme 1: hour-specific PDF matching
    if _RUN_SCHEME_1:
        if _run_all:
            _run_scheme_hour(
                train_ob, train_pr, val_pr,
                data_dir / pdfm_models['pdfm1'],
                data_dir / corrected['pred1'],
                'east_china',
            )
        if _run_mlyr:
            _run_scheme_hour(
                train_ob_mlyr, train_pr_mlyr, val_pr_mlyr,
                data_dir / pdfm_models['pdfm1_mlyr'],
                data_dir / corrected['pred1_mlyr'],
                'mlyr',
            )

    # 4. Scheme 2: station-specific PDF matching
    if _RUN_SCHEME_2:
        if _run_all:
            _run_scheme_station(
                train_ob, train_pr, val_pr,
                data_dir / pdfm_models['pdfm2'],
                data_dir / corrected['pdfm'],
                data_dir / corrected['pred2'],
                'east_china',
            )
        if _run_mlyr:
            _run_scheme_station(
                train_ob_mlyr, train_pr_mlyr, val_pr_mlyr,
                data_dir / pdfm_models['pdfm2_mlyr'],
                data_dir / corrected['pdfm_mlyr'],
                data_dir / corrected['pred2_mlyr'],
                'mlyr',
            )

    # 5. Compute raw forecast TLE
    if _run_all:
        raw_tle = _apply_tle_to_4d(val_pr)
        _save_npy(data_dir / corrected['tle'], raw_tle)
    if _run_mlyr:
        raw_tle_mlyr = _apply_tle_to_4d(val_pr_mlyr)
        _save_npy(data_dir / corrected['tle_mlyr'], raw_tle_mlyr)

    # 6. Verification metrics
    if _run_all:
        print('****** nwp ******')
        _print_metrics(vis_acc.VisAcc(val_ob, val_pr))
        _verify_predictions(val_ob, data_dir / corrected['pred2'])
        _verify_predictions(val_ob, data_dir / corrected['pdfm'])
        _verify_predictions(val_ob, data_dir / corrected['tle'])

    if _run_mlyr:
        print('*' * 20)
        _print_metrics(vis_acc.VisAcc(val_ob_mlyr, val_pr_mlyr))
        _verify_predictions(val_ob_mlyr, data_dir / corrected['pred2_mlyr'])
        _verify_predictions(val_ob_mlyr, data_dir / corrected['pdfm_mlyr'])
        _verify_predictions(val_ob_mlyr, data_dir / corrected['tle_mlyr'])


if __name__ == '__main__':
    print('Program access.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    elapsed_str = utils.format_time(total_elapsed)
    print(f'Program access.py finished, total time {elapsed_str}')
