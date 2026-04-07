# -*- coding: utf-8 -*-
"""
Forecast data loading, overall verification and station-level metrics module.

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""

import os
import pathlib
import typing

import numpy as np
import pandas as pd

from src import vis_acc

VisAcc = vis_acc.VisAcc


def load_forecast_data(
    data_dir: str,
    index_cjzxy: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Load last 365 days (2020-2023) obs and CMA-SH-WARR forecast data.

    Args:
        data_dir (str): Data root directory.
        index_cjzxy (np.ndarray): Middle-lower Yangtze station filter index.

    Returns:
        tuple: (vis_ob, cma_sh_warr), arrays of shape (-1, 24, 502).
    """
    # 1. Load obs data (last 365 days, skip first lead)
    path = str(pathlib.Path(data_dir) / 'vis1183_ob.npy')
    vis_ob = np.load(path)[-365:, :, 1:, index_cjzxy]
    vis_ob = np.reshape(vis_ob, (-1, 24, 502))
    vis_ob[vis_ob >= 999990] = np.nan  # Missing marker
    vis_ob[vis_ob >= 30000] = 30000    # Cap at 30000m

    # 2. Load CMA-SH-WARR forecast data
    pr_path = str(pathlib.Path(data_dir) / 'vis1183_pr.npy')
    cma_sh_warr = np.load(pr_path)[-365:, :, 1:, index_cjzxy]
    cma_sh_warr = np.reshape(cma_sh_warr, (-1, 24, 502))
    cma_sh_warr[cma_sh_warr >= 30000] = 30000

    return vis_ob, cma_sh_warr


def load_experiment_preds(data_dir: str) -> typing.Tuple[np.ndarray, ...]:
    """
    Load 5 PDFM-TLE experiment forecast datasets.

    Args:
        data_dir (str): Data root directory.

    Returns:
        tuple: 5 forecast arrays (pred_pdfm_tle0~4).
    """
    preds = list()
    for i in range(5):
        path = str(pathlib.Path(data_dir) / f'vis_gjz_pdfm_tle{i}_cjzxy.npy')
        pred = np.load(path)
        pred = np.reshape(pred, (-1, 24, 502))
        pred[pred >= 30000] = 30000
        preds.append(pred)
    return tuple(preds)


def _fmt_arr(arr: np.ndarray) -> str:
    """Format numpy array with 4 decimal places."""
    return np.array2string(
        np.array(arr), precision=4, separator=' ', suppress_small=True
    )


def print_overall_metrics(
    vis_ob: np.ndarray,
    pred: np.ndarray,
    name: str
) -> None:
    """
    Calculate and print overall verification metrics for a scheme.

    Args:
        vis_ob (np.ndarray): Observation array.
        pred (np.ndarray): Forecast array.
        name (str): Scheme name for output.
    """
    acc = VisAcc(vis_ob, pred)
    print(f'[Overall Metrics] {name}')
    print(f'  R    = {acc.get_r():.4f}')
    print(f'  MAE  = {acc.get_mae():.4f} m')
    print(f'  RMSE = {acc.get_rmse():.4f} m')
    print(f'  MRE  = {acc.get_mre():.4f}')
    print(f'  TS2  = {_fmt_arr(acc.get_ts2())}')
    print(f'  ETS2 = {_fmt_arr(acc.get_ets2())}')
    print(f'  HSS2 = {_fmt_arr(acc.get_hss2())}')
    print(f'  TSS2 = {_fmt_arr(acc.get_tss2())}')
    print(f'  FAR2 = {_fmt_arr(acc.get_far2())}')
    print(f'  MAR2 = {_fmt_arr(acc.get_mar2())}')
    print(f'  POD2 = {_fmt_arr(acc.get_pod2())}')


def calc_and_save_station_metrics(
    vis_ob: np.ndarray,
    cma_sh_warr: np.ndarray,
    pred_pdfm_tle2: np.ndarray,
    output_dir: str
) -> None:
    """
    Calculate station-level verification metrics and save to CSV.

    Computes corr, mae, rmse, mre, and ts1~ts6 for each of the 502 stations
    for both CMA-SH-WARR and PDFM-TLE.

    Args:
        vis_ob (np.ndarray): Observation array.
        cma_sh_warr (np.ndarray): CMA-SH-WARR forecast array.
        pred_pdfm_tle2 (np.ndarray): PDFM-TLE forecast array.
        output_dir (str): Output directory path.
    """
    # 1. Prepare output directory
    csv_dir = pathlib.Path(output_dir) / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)

    # 2. Init metrics containers for all stations
    metrics: typing.Dict[str, typing.Dict[str, list]] = {
        'corr': {'sta': [], 'CMA-SH-WARR': [], 'PDFM-TLE': []},
        'mae': {'sta': [], 'CMA-SH-WARR': [], 'PDFM-TLE': []},
        'rmse': {'sta': [], 'CMA-SH-WARR': [], 'PDFM-TLE': []},
        'mre': {'sta': [], 'CMA-SH-WARR': [], 'PDFM-TLE': []},
        'ts1': {'sta': [], 'CMA-SH-WARR': [], 'PDFM-TLE': []},
        'ts2': {'sta': [], 'CMA-SH-WARR': [], 'PDFM-TLE': []},
        'ts3': {'sta': [], 'CMA-SH-WARR': [], 'PDFM-TLE': []},
        'ts4': {'sta': [], 'CMA-SH-WARR': [], 'PDFM-TLE': []},
        'ts5': {'sta': [], 'CMA-SH-WARR': [], 'PDFM-TLE': []},
        'ts6': {'sta': [], 'CMA-SH-WARR': [], 'PDFM-TLE': []},
    }

    # 3. Loop through each station and calc metrics
    n_sta = vis_ob.shape[2]
    for i in range(n_sta):
        # 3.1 Record station index
        for key in metrics:
            metrics[key]['sta'].append(i)

        # 3.2 Calc CMA-SH-WARR metrics
        acc_nwp = VisAcc(vis_ob[:, :, i], cma_sh_warr[:, :, i])
        metrics['corr']['CMA-SH-WARR'].append(acc_nwp.get_r())
        metrics['mae']['CMA-SH-WARR'].append(acc_nwp.get_mae())
        metrics['rmse']['CMA-SH-WARR'].append(acc_nwp.get_rmse())
        metrics['mre']['CMA-SH-WARR'].append(acc_nwp.get_mre())
        ts = acc_nwp.get_ts2()
        metrics['ts1']['CMA-SH-WARR'].append(ts[0])
        metrics['ts2']['CMA-SH-WARR'].append(ts[1])
        metrics['ts3']['CMA-SH-WARR'].append(ts[2])
        metrics['ts4']['CMA-SH-WARR'].append(ts[3])
        metrics['ts5']['CMA-SH-WARR'].append(ts[4])
        metrics['ts6']['CMA-SH-WARR'].append(ts[5])

        # 3.3 Calc PDFM-TLE metrics
        acc = VisAcc(vis_ob[:, :, i], pred_pdfm_tle2[:, :, i])
        metrics['corr']['PDFM-TLE'].append(acc.get_r())
        metrics['mae']['PDFM-TLE'].append(acc.get_mae())
        metrics['rmse']['PDFM-TLE'].append(acc.get_rmse())
        metrics['mre']['PDFM-TLE'].append(acc.get_mre())
        ts = acc.get_ts2()
        metrics['ts1']['PDFM-TLE'].append(ts[0])
        metrics['ts2']['PDFM-TLE'].append(ts[1])
        metrics['ts3']['PDFM-TLE'].append(ts[2])
        metrics['ts4']['PDFM-TLE'].append(ts[3])
        metrics['ts5']['PDFM-TLE'].append(ts[4])
        metrics['ts6']['PDFM-TLE'].append(ts[5])

    # 4. Save all metrics to CSV files
    for key, data in metrics.items():
        pd.DataFrame(data).to_csv(
            str(csv_dir / f'vis_sta_{key}.csv'), index=False)
    # Rename ts1~ts6 files to match expected names with '+' suffix
    for i in range(1, 7):
        old_path = str(csv_dir / f'vis_sta_ts{i}.csv')
        new_path = str(csv_dir / f'vis_sta_ts{i}+.csv')
        if os.path.exists(old_path):
            os.replace(old_path, new_path)
