#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Founded in 2024-07-10
Modified in 2026-07-21
@author: yinlb
"""
import pathlib

import arrow
import numpy as np
from numba import njit

from src import utils, vis_acc


# Load temporal-lead experiment configuration at import time
_TL_CFG = utils.CFG['tl']
_TL_PARAMS = _TL_CFG['params']
_TL_INPUT = _TL_CFG['input_files']
_TL_OUTPUT = _TL_CFG['output_files']
_N_HOURS = int(_TL_PARAMS['n_hours'])
_N_RAW_STATIONS = int(_TL_PARAMS['n_raw_stations'])
_N_SUBSET_STATIONS = int(_TL_PARAMS['n_subset_stations'])
_VAL_DAYS = int(_TL_PARAMS['val_days'])
_TLE_START_IDX = int(_TL_PARAMS['tle_start_idx'])
_TLE_END_IDX = int(_TL_PARAMS['tle_end_idx'])
_TLE_WINDOW_HOURS = int(_TL_PARAMS['tle_window_hours'])
_VISIBILITY_CAP = float(_TL_PARAMS['visibility_cap'])


def _compute_tle_weights(
    ref_pr_window: np.ndarray,
    ref_ob_window: np.ndarray,
) -> np.ndarray:
    """Compute TLE weights for one time step.

    Processes all init hours in one shot using numpy reductions.

    Args:
        ref_pr_window: Forecast reference window,
            shape (window, n_hours, n_stations).
        ref_ob_window: Observation reference window,
            shape (window, n_hours, n_stations).

    Returns:
        weights: Array of shape (5, n_hours).
    """
    weights = np.zeros((5, _N_HOURS), dtype=np.float32) + np.nan

    # Strategy 0: equal weight
    weights[0, :] = 1.0

    # Strategy 1: correlation between the first two rows of ref_pr_window
    # Matches np.corrcoef(val_pr_, val_ob_)[0, 1] for 2D inputs (rowvar=True).
    pr0 = ref_pr_window[0, :, :]  # (n_hours, n_stations)
    pr1 = ref_pr_window[1, :, :]  # (n_hours, n_stations)
    pr0_mean = np.mean(pr0, axis=1, keepdims=True)
    pr1_mean = np.mean(pr1, axis=1, keepdims=True)
    pr0_std = np.std(pr0, axis=1, keepdims=True)
    pr1_std = np.std(pr1, axis=1, keepdims=True)
    cov = np.mean((pr0 - pr0_mean) * (pr1 - pr1_mean), axis=1, keepdims=True)
    weights[1, :] = (cov / (pr0_std * pr1_std)).squeeze()

    # Strategy 2: inverse MAE
    mae = np.mean(
        np.abs(ref_pr_window - ref_ob_window),
        axis=(0, 2),
    )
    weights[2, :] = 1 + 1 / mae

    # Strategy 3: inverse RMSE
    rmse = np.mean(
        (ref_pr_window - ref_ob_window) ** 2,
        axis=(0, 2),
    ) ** 0.5
    weights[3, :] = 1 + 1 / rmse

    # Strategy 4: inverse MRE
    # Original code selects elements where pr+ob != 0 and computes the mean
    # of |pr-ob|/(pr+ob). NaNs in selected elements propagate to the mean.
    sum_arr = ref_pr_window + ref_ob_window
    mask = sum_arr != 0
    ratio = np.abs(ref_pr_window - ref_ob_window) / np.where(mask, sum_arr, 1.0)
    valid_count = np.sum(mask, axis=(0, 2))
    ratio_sum = np.sum(np.where(mask, ratio, 0.0), axis=(0, 2))
    has_nan_valid = np.any(mask & np.isnan(ratio), axis=(0, 2))
    mre = np.where(has_nan_valid, np.nan, ratio_sum / valid_count)
    weights[4, :] = 1 + 1 / mre

    return weights


@njit
def _apply_tle_step_numba(
    input_slice: np.ndarray,
    weights: np.ndarray,
    n_hours: int,
) -> np.ndarray:
    """Apply TLE averaging for a single time step (Numba-compiled).

    Args:
        input_slice: Forecast window covering [i-n_hours+1, i+n_hours-1],
            shape (2*n_hours-1, n_hours, n_stations).
        weights: TLE weights, shape (5, n_hours).
        n_hours: Number of init hours (from config).

    Returns:
        output: TLE-merged forecasts for this time step,
            shape (5, n_hours, n_stations).
    """
    n_stations = input_slice.shape[2]
    output = np.zeros((5, n_hours, n_stations), dtype=np.float32) + np.nan
    center = n_hours - 1

    for j in range(n_hours):
        for s in range(5):
            denom = 0.0
            for k in range(j, n_hours):
                denom += weights[s, k]
            if denom == 0.0:
                continue
            for station in range(n_stations):
                vis = 0.0
                for k in range(j, n_hours):
                    t_idx = center + j - k
                    vis += weights[s, k] * input_slice[t_idx, k, station]
                output[s, j, station] = vis / denom

    return output


def _apply_tle(
    input_array: np.ndarray,
    ref_pr: np.ndarray,
    ref_ob: np.ndarray,
    start_idx: int = None,
    end_idx: int = None,
) -> np.ndarray:
    """Apply TLE averaging using Numba for the inner loops.

    Args:
        input_array: Input forecast array, shape (T, n_hours, n_stations).
        ref_pr: Forecast reference array, shape (T, n_hours, n_stations).
        ref_ob: Observation reference array, shape (T, n_hours, n_stations).
        start_idx: Start index for the TLE loop. Defaults to config.
        end_idx: End index for the TLE loop. Defaults to config.

    Returns:
        output: TLE-merged arrays, shape (5, T, n_hours, n_stations).
    """
    if start_idx is None:
        start_idx = _TLE_START_IDX
    if end_idx is None:
        end_idx = _TLE_END_IDX

    output = np.zeros((5,) + input_array.shape, dtype=np.float32) + np.nan
    half_window = _N_HOURS - 1

    for i in range(start_idx, end_idx):
        weights = _compute_tle_weights(
            ref_pr[i - _TLE_WINDOW_HOURS: i - _TLE_WINDOW_HOURS + _N_HOURS, :, :],
            ref_ob[i - _TLE_WINDOW_HOURS: i - _TLE_WINDOW_HOURS + _N_HOURS, :, :],
        )  # (5, n_hours)
        # Local window covers [i-n_hours+1, i+n_hours-1]
        input_slice = input_array[i - half_window: i + half_window + 1, :, :]
        output[:, i, :, :] = _apply_tle_step_numba(input_slice, weights, _N_HOURS)

    return output


def _print_metrics(acc: vis_acc.VisAcc) -> None:
    """Print standard verification metrics from a VisAcc object.

    Args:
        acc: VisAcc instance with obs and forecast data.
    """
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts_ge())
    print(acc.get_far_ge())
    print(acc.get_mar_ge())


def _verify_tle(
    name: str,
    val_ob: np.ndarray,
    pred_tle: np.ndarray,
) -> None:
    """Cap predictions and print metrics for all strategies.

    Args:
        name: Label printed before each strategy block.
        val_ob: Observation array.
        pred_tle: TLE predictions, shape (5, T, n_hours, n_stations).
    """
    for s in range(pred_tle.shape[0]):
        print(f'****** {name}{s} ******')
        pred = pred_tle[s].copy()
        pred[pred >= _VISIBILITY_CAP] = _VISIBILITY_CAP
        acc = vis_acc.VisAcc(val_ob, pred)
        _print_metrics(acc)


def main() -> None:
    """Run temporal-lead experiment and verification."""
    # 1. Load observation and forecast data
    data_dir = pathlib.Path(utils.CFG['paths']['data_dir'])
    ob = np.load(str(data_dir / _TL_INPUT['observation_npy']))[:, :, 1:, :]
    pr = np.load(str(data_dir / _TL_INPUT['forecast_npy']))[:, :, 1:, :]
    ob = utils.mask_missing(ob)
    pr = utils.mask_missing(pr)

    # 2. Select station subset and split validation set
    index_sta08 = np.load(str(data_dir / _TL_INPUT['station_index_npy']))
    print(np.sum(index_sta08))
    val_ob = ob[-_VAL_DAYS:, :, :, index_sta08]
    val_pr = pr[-_VAL_DAYS:, :, :, index_sta08]
    val_ob = np.reshape(val_ob, (-1, _N_HOURS, _N_SUBSET_STATIONS))
    val_pr = np.reshape(val_pr, (-1, _N_HOURS, _N_SUBSET_STATIONS))
    del ob, pr

    # 3. Verify raw NWP forecast
    print('****** nwp ******')
    acc = vis_acc.VisAcc(val_ob, val_pr)
    _print_metrics(acc)

    # 4. Compute TLE on raw NWP forecast
    pred_tle = _apply_tle(val_pr, val_pr, val_ob)
    for s in range(5):
        tle_name = _TL_OUTPUT['tle_template'].format(s=s)
        np.save(str(data_dir / tle_name), pred_tle[s])

    # 5. Reload TLE files and verify (including pre-existing tle5)
    pred_tle = np.stack([
        np.load(str(data_dir / _TL_OUTPUT['tle_template'].format(s=0))),
        np.load(str(data_dir / _TL_OUTPUT['tle_template'].format(s=1))),
        np.load(str(data_dir / _TL_OUTPUT['tle_template'].format(s=2))),
        np.load(str(data_dir / _TL_OUTPUT['tle_template'].format(s=3))),
        np.load(str(data_dir / _TL_OUTPUT['tle_template'].format(s=4))),
        np.load(str(data_dir / _TL_INPUT['pre_existing_tle'])),
    ])
    _verify_tle('tle', val_ob, pred_tle)

    # 6. Verify PDFM forecast
    pred_pdfm = np.load(str(data_dir / _TL_INPUT['pdfm_input_npy']))
    pred_pdfm = np.reshape(pred_pdfm, (-1, _N_HOURS, _N_SUBSET_STATIONS))
    print('****** pdfm ******')
    acc = vis_acc.VisAcc(val_ob, pred_pdfm)
    _print_metrics(acc)

    # 7. Compute TLE on PDFM forecast
    pred_pdfm_tle = _apply_tle(pred_pdfm, val_pr, val_ob)
    for s in range(5):
        tle_name = _TL_OUTPUT['pdfm_tle_template'].format(s=s)
        np.save(str(data_dir / tle_name), pred_pdfm_tle[s])

    # 8. Reload PDFM-TLE files and verify (including pre-existing tle5)
    pred_pdfm_tle = np.stack([
        np.load(str(data_dir / _TL_OUTPUT['pdfm_tle_template'].format(s=0))),
        np.load(str(data_dir / _TL_OUTPUT['pdfm_tle_template'].format(s=1))),
        np.load(str(data_dir / _TL_OUTPUT['pdfm_tle_template'].format(s=2))),
        np.load(str(data_dir / _TL_OUTPUT['pdfm_tle_template'].format(s=3))),
        np.load(str(data_dir / _TL_OUTPUT['pdfm_tle_template'].format(s=4))),
        np.load(str(data_dir / _TL_INPUT['pre_existing_pdfm_tle'])),
    ])
    _verify_tle('pdfm-tle', val_ob, pred_pdfm_tle)


if __name__ == '__main__':
    print('Program tl.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(f'Program tl.py finished, total time: {utils.format_time(total_elapsed)}')
