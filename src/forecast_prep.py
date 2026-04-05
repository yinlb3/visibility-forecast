# -*- coding: utf-8 -*-
"""
Forecast data loading and overall verification module.

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""

from typing import Tuple

import numpy as np

from src.vis_acc import VisAcc


def load_forecast_data(
    data_dir: str,
    index_cjzxy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load last 365 days (2020-2023) obs and CMA-SH-WARR forecast data.

    Args:
        data_dir (str): Data root directory.
        index_cjzxy (np.ndarray): Middle-lower Yangtze station filter index.

    Returns:
        tuple: (vis_ob, cma_sh_warr), arrays of shape (-1, 24, 502).
    """
    vis_ob = np.load(rf'{data_dir}\vis1183_ob.npy')[-365:, :, 1:, index_cjzxy]
    vis_ob = np.reshape(vis_ob, (-1, 24, 502))
    vis_ob[vis_ob >= 999990] = np.nan
    vis_ob[vis_ob >= 30000] = 30000

    cma_sh_warr = np.load(rf'{data_dir}\vis1183_pr.npy')[-365:, :, 1:, index_cjzxy]
    cma_sh_warr = np.reshape(cma_sh_warr, (-1, 24, 502))
    cma_sh_warr[cma_sh_warr >= 30000] = 30000

    return vis_ob, cma_sh_warr


def load_experiment_preds(data_dir: str) -> Tuple[np.ndarray, ...]:
    """
    Load 5 PDFM-TLE experiment forecast datasets.

    Args:
        data_dir (str): Data root directory.

    Returns:
        tuple: 5 forecast arrays (pred_pdfm_tle0~4).
    """
    preds = []
    for i in range(5):
        path = rf'{data_dir}\vis_gjz_pdfm_tle{i}_cjzxy.npy'
        pred = np.load(path)
        pred = np.reshape(pred, (-1, 24, 502))
        pred[pred >= 30000] = 30000
        preds.append(pred)
    return tuple(preds)


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
        name (str): Scheme name for printing.
    """
    acc = VisAcc(vis_ob, pred)
    print(f'[print_overall_metrics] {name}')
    print(f'  R: {acc.get_r()}')
    print(f'  MAE: {acc.get_mae()}')
    print(f'  RMSE: {acc.get_rmse()}')
    print(f'  MRE: {acc.get_mre()}')
    print(f'  TS2: {acc.get_ts2()}')
    print(f'  ETS2: {acc.get_ets2()}')
    print(f'  HSS2: {acc.get_hss2()}')
    print(f'  TSS2: {acc.get_tss2()}')
    print(f'  FAR2: {acc.get_far2()}')
    print(f'  MAR2: {acc.get_mar2()}')
    print(f'  POD2: {acc.get_pod2()}')
