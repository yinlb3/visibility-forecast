# -*- coding: utf-8 -*-
"""
2024 年独立样本个例分析模块 (阶段 9).

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""

import numpy as np
import arrow

from src.vis_acc import VisAcc


def load_2024_preds(data_dir: str) -> tuple:
    """
    9.1 加载 2024 年 PDFM/TLE 独立样本预报数据.

    Returns:
        tuple: (pred_pdfm2, pred_tle2).
    """
    pred_pdfm2 = np.load(rf'{data_dir}\vis_gjz_pdfm2_cjzxy.npy')
    pred_pdfm2 = np.reshape(pred_pdfm2, shape=(-1, 24, 502))
    pred_pdfm2[pred_pdfm2 >= 30000] = 30000

    pred_tle2 = np.load(rf'{data_dir}\vis_gjz_tle2_cjzxy.npy')
    pred_tle2 = np.reshape(pred_tle2, shape=(-1, 24, 502))
    pred_tle2[pred_tle2 >= 30000] = 30000

    return pred_pdfm2, pred_tle2


def print_2024_overall_metrics(vis_ob: np.ndarray, pred_pdfm2: np.ndarray, pred_tle2: np.ndarray) -> None:
    """Print overall verification metrics for PDFM and TLE."""
    acc = VisAcc(vis_ob, pred_pdfm2)
    print('[print_2024_overall_metrics] PDFM')
    print(f'  R={acc.get_r()}, MAE={acc.get_mae()}, RMSE={acc.get_rmse()}, MRE={acc.get_mre()}')
    print(f'  TS2={acc.get_ts2()}, FAR2={acc.get_far2()}, MAR2={acc.get_mar2()}')

    acc = VisAcc(vis_ob, pred_tle2)
    print('[print_2024_overall_metrics] TLE')
    print(f'  R={acc.get_r()}, MAE={acc.get_mae()}, RMSE={acc.get_rmse()}, MRE={acc.get_mre()}')
    print(f'  TS2={acc.get_ts2()}, FAR2={acc.get_far2()}, MAR2={acc.get_mar2()}')


def load_2024_eval_data(data_dir: str, index_cjzxy: np.ndarray) -> tuple:
    """Load 2024 obs and forecast data for case evaluation."""
    cma_sh_warr = np.load(rf'{data_dir}\vis_gjz_2024.npy')
    cma_sh_warr = np.reshape(cma_sh_warr[:, 1:, index_cjzxy], shape=(-1, 24, 502))
    cma_sh_warr[cma_sh_warr >= 30000] = 30000

    pred_pdfm2 = np.load(rf'{data_dir}\vis_gjz_pdfm2_cjzxy_2024.npy')
    pred_pdfm2 = np.reshape(pred_pdfm2, shape=(-1, 24, 502))
    pred_pdfm2[pred_pdfm2 >= 30000] = 30000

    return cma_sh_warr, pred_pdfm2


def align_forecast_times(vis_ob: np.ndarray, cma_sh_warr: np.ndarray, pred_pdfm2: np.ndarray) -> tuple:
    """
    Apply time offset to 24 forecast init times to align with obs.
    """
    n_days = vis_ob.shape[0]  # 动态获取天数（365或366）
    vis_ob_ = np.zeros_like(vis_ob) + np.nan
    pred_pdfm2_ = np.zeros_like(pred_pdfm2) + np.nan
    cma_sh_warr_ = np.zeros_like(cma_sh_warr) + np.nan
    for i in range(24):
        # 使用实际数据长度，避免硬编码8783
        src_len = n_days - 1 - i
        if src_len > 0:
            vis_ob_[i + 1:, i, :] = vis_ob[:src_len, i, :]
            pred_pdfm2_[i + 1:, i, :] = pred_pdfm2[:src_len, i, :]
            cma_sh_warr_[i + 1:, i, :] = cma_sh_warr[:src_len, i, :]
    return vis_ob_, cma_sh_warr_, pred_pdfm2_


def analyze_case_studies(
    vis_ob_: np.ndarray,
    cma_sh_warr_: np.ndarray,
    pred_pdfm2_: np.ndarray
) -> None:
    """
    9.2 计算 18 组个例各阶段及合并后的 TS4+.
    """
    date_list = [
        ['2024-01-02', '2024-01-13', '2024-01-03'],
        ['2024-01-29', '2024-02-01', '2024-01-30'],
        ['2024-02-08', '2024-02-11', '2024-02-10'],
        ['2024-03-02', '2024-03-08', '2024-03-04'],
        ['2024-03-11', '2024-03-18', '2024-03-14'],
        ['2024-03-23', '2024-03-28', '2024-03-27'],
        ['2024-03-31', '2024-04-02', '2024-04-01'],
        ['2024-04-06', '2024-04-08', '2024-04-08'],
        ['2024-04-11', '2024-04-29', '2024-04-14'],
        ['2024-05-02', '2024-05-06', '2024-05-05'],
        ['2024-06-09', '2024-06-12', '2024-06-10'],
        ['2024-06-18', '2024-06-21', '2024-06-20'],
        ['2024-06-24', '2024-07-01', '2024-06-27'],
        ['2024-07-11', '2024-07-13', '2024-07-11'],
        ['2024-10-14', '2024-10-18', '2024-10-14'],
        ['2024-10-27', '2024-10-29', '2024-10-28'],
        ['2024-11-10', '2024-11-12', '2024-11-10'],
        ['2024-12-04', '2024-12-07', '2024-12-05']
    ]
    ind = np.zeros(shape=(8784, 4), dtype=np.bool_)

    for d in date_list:
        i = round((arrow.get(d[0]) - arrow.get('2024')).total_seconds() / 3600)
        j = round((arrow.get(d[1]) - arrow.get('2024')).total_seconds() / 3600)
        k = round((arrow.get(d[2]) - arrow.get('2024')).total_seconds() / 3600)
        # Output TS4+ for CMA-SH-WARR and PDFM-TLE at each phase (all, start, end, peak)
        print(
            VisAcc(vis_ob_[i: j + 24, :, :], cma_sh_warr_[i: j + 24, :, :]).get_ts2()[3],
            VisAcc(vis_ob_[i: i + 24, :, :], cma_sh_warr_[i: i + 24, :, :]).get_ts2()[3],
            VisAcc(vis_ob_[j: j + 24, :, :], cma_sh_warr_[j: j + 24, :, :]).get_ts2()[3],
            VisAcc(vis_ob_[k: k + 24, :, :], cma_sh_warr_[k: k + 24, :, :]).get_ts2()[3],
            VisAcc(vis_ob_[i: j + 24, :, :], pred_pdfm2_[i: j + 24, :, :]).get_ts2()[3],
            VisAcc(vis_ob_[i: i + 24, :, :], pred_pdfm2_[i: i + 24, :, :]).get_ts2()[3],
            VisAcc(vis_ob_[j: j + 24, :, :], pred_pdfm2_[j: j + 24, :, :]).get_ts2()[3],
            VisAcc(vis_ob_[k: k + 24, :, :], pred_pdfm2_[k: k + 24, :, :]).get_ts2()[3]
        )
        ind[i: j + 24, 0] = True
        ind[i: i + 24, 1] = True
        ind[j: j + 24, 2] = True
        ind[k: k + 24, 3] = True

    print(
        VisAcc(vis_ob_[ind[:, 0], :, :], cma_sh_warr_[ind[:, 0], :, :]).get_ts2()[3],
        VisAcc(vis_ob_[ind[:, 1], :, :], cma_sh_warr_[ind[:, 1], :, :]).get_ts2()[3],
        VisAcc(vis_ob_[ind[:, 2], :, :], cma_sh_warr_[ind[:, 2], :, :]).get_ts2()[3],
        VisAcc(vis_ob_[ind[:, 3], :, :], cma_sh_warr_[ind[:, 3], :, :]).get_ts2()[3],
        VisAcc(vis_ob_[ind[:, 0], :, :], pred_pdfm2_[ind[:, 0], :, :]).get_ts2()[3],
        VisAcc(vis_ob_[ind[:, 1], :, :], pred_pdfm2_[ind[:, 1], :, :]).get_ts2()[3],
        VisAcc(vis_ob_[ind[:, 2], :, :], pred_pdfm2_[ind[:, 2], :, :]).get_ts2()[3],
        VisAcc(vis_ob_[ind[:, 3], :, :], pred_pdfm2_[ind[:, 3], :, :]).get_ts2()[3]
    )
