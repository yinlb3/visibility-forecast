# -*- coding: utf-8 -*-
"""
预报数据加载与整体检验模块.

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
    加载 2020-2023 年最后 365 天的观测与 CMA-SH-WARR 预报数据.

    Args:
        data_dir (str): 数据根目录.
        index_cjzxy (np.ndarray): 长江中下游站点筛选索引.

    Returns:
        tuple: (vis_ob, cma_sh_warr), 均为形状 (-1, 24, 502) 的数组.
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
    加载 5 组 PDFM-TLE 试验预报数据.

    Args:
        data_dir (str): 数据根目录.

    Returns:
        tuple: 5 个预报数组 (pred_pdfm_tle0~4).
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
    计算并打印某方案的整体检验指标.

    Args:
        vis_ob (np.ndarray): 观测数组.
        pred (np.ndarray): 预报数组.
        name (str): 方案名称, 用于打印标识.
    """
    acc = VisAcc(vis_ob, pred)
    print(name)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_ets2())
    print(acc.get_hss2())
    print(acc.get_tss2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print(acc.get_pod2())
