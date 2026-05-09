#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 3.5: Operational application case study evaluation result output module.

Founded in 2026-04-14
Modified in 2026-04-15
@author: yinlb
"""

import pathlib
import typing

import arrow
import numpy as np

from src import vis_acc

VisAcc = vis_acc.VisAcc


def _fmt_arr(arr: np.ndarray) -> str:
    """Format numpy array with 4 decimal places for display."""
    return np.array2string(
        np.array(arr), precision=4, separator=' ', suppress_small=True
    )


def load_2024_obs(data_dir: str, idx_mlyr: np.ndarray) -> np.ndarray:
    """Load 2024 observation data for case study."""
    vis_ob = np.load(str(pathlib.Path(data_dir) / 'vis1183_ob_2024.npy'))
    vis_ob = np.reshape(vis_ob[:, 1:, idx_mlyr], shape=(-1, 24, 502))
    # Data cleaning: mark missing values as NaN, cap visibility at 30000m
    vis_ob[vis_ob >= 999990] = np.nan
    vis_ob[vis_ob >= 30000] = 30000
    return vis_ob


def load_2024_preds(data_dir: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Load 2024 PDFM/TLE prediction data for overall metrics."""
    path = pathlib.Path(data_dir) / 'vis_gjz_pdfm2_mlyr_2024.npy'
    pred_pdfm2 = np.load(str(path))
    pred_pdfm2 = np.reshape(pred_pdfm2, shape=(-1, 24, 502))
    # Cap forecast visibility at 30000m to match obs range
    pred_pdfm2[pred_pdfm2 >= 30000] = 30000

    path = pathlib.Path(data_dir) / 'vis_gjz_tle2_mlyr_2024.npy'
    pred_tle2 = np.load(str(path))
    pred_tle2 = np.reshape(pred_tle2, shape=(-1, 24, 502))
    pred_tle2[pred_tle2 >= 30000] = 30000

    return pred_pdfm2, pred_tle2


def print_2024_overall_metrics(
    vis_ob: np.ndarray, pred_pdfm2: np.ndarray, pred_tle2: np.ndarray
) -> None:
    """Print overall verification metrics for PDFM and TLE."""
    # 1. Print PDFM overall metrics
    acc = VisAcc(vis_ob, pred_pdfm2)
    print('[2024 Overall Metrics] PDFM')
    print(f'  R={acc.get_r():.4f}, MAE={acc.get_mae():.1f} m, '
          f'RMSE={acc.get_rmse():.1f} m, MRE={acc.get_mre():.4f}')
    print(f'  TS_GE={_fmt_arr(acc.get_ts_ge())}, '
          f'FAR_GE={_fmt_arr(acc.get_far_ge())}, '
          f'MAR_GE={_fmt_arr(acc.get_mar_ge())}')

    # 2. Print TLE overall metrics
    acc = VisAcc(vis_ob, pred_tle2)
    print('[2024 Overall Metrics] TLE')
    print(f'  R={acc.get_r():.4f}, MAE={acc.get_mae():.1f} m, '
          f'RMSE={acc.get_rmse():.1f} m, MRE={acc.get_mre():.4f}')
    print(f'  TS_ge={_fmt_arr(acc.get_ts_ge())}, '
          f'FAR_ge={_fmt_arr(acc.get_far_ge())}, '
          f'MAR_ge={_fmt_arr(acc.get_mar_ge())}')


def load_2024_eval_data(
    data_dir: str, idx_mlyr: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Load 2024 obs and forecast data for case evaluation."""
    cma_sh_warr = np.load(str(pathlib.Path(data_dir) / 'vis_gjz_2024.npy'))
    cma_sh_warr = np.reshape(cma_sh_warr[:, 1:, idx_mlyr], shape=(-1, 24, 502))
    cma_sh_warr[cma_sh_warr >= 30000] = 30000

    path = pathlib.Path(data_dir) / 'vis_gjz_pdfm2_mlyr_2024.npy'
    pred_pdfm2 = np.load(str(path))
    pred_pdfm2 = np.reshape(pred_pdfm2, shape=(-1, 24, 502))
    pred_pdfm2[pred_pdfm2 >= 30000] = 30000

    return cma_sh_warr, pred_pdfm2


def align_forecast_times(
    vis_ob: np.ndarray, cma_sh_warr: np.ndarray, pred_pdfm2: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply time offset to 24 forecast init times to align with obs."""
    # Apply time offset to align 24 forecast init times with obs timestamps
    vis_ob_ = np.zeros_like(vis_ob) + np.nan
    pred_pdfm2_ = np.zeros_like(pred_pdfm2) + np.nan
    cma_sh_warr_ = np.zeros_like(cma_sh_warr) + np.nan
    for i in range(24):
        src_len_vis = vis_ob.shape[0] - 1 - i
        if src_len_vis > 0:
            vis_ob_[i + 1:, i, :] = vis_ob[:src_len_vis, i, :]
        src_len_cma = cma_sh_warr.shape[0] - 1 - i
        if src_len_cma > 0:
            cma_sh_warr_[i + 1:, i, :] = cma_sh_warr[:src_len_cma, i, :]
        src_len_pred = pred_pdfm2.shape[0] - 1 - i
        if src_len_pred > 0:
            pred_pdfm2_[i + 1:, i, :] = pred_pdfm2[:src_len_pred, i, :]
    return vis_ob_, cma_sh_warr_, pred_pdfm2_


def analyze_case_studies(
    vis_ob_: np.ndarray,
    cma_sh_warr_: np.ndarray,
    pred_pdfm2_: np.ndarray
) -> None:
    """Analyze 18 case studies and calc TS4+ for all phases and merged."""
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
        ['2024-12-04', '2024-12-07', '2024-12-05'],
    ]
    # 1. Define 18 case studies with start, end, and peak dates
    case_idx = np.zeros(shape=(8784, 4), dtype=np.bool_)

    for idx, dt in enumerate(date_list, 1):
        base = arrow.get('2024')
        i = round((arrow.get(d[0]) - base).total_seconds() / 3600)
        j = round((arrow.get(d[1]) - base).total_seconds() / 3600)
        k = round((arrow.get(d[2]) - base).total_seconds() / 3600)
        # 2. Extract four phases: all period, start, end, and peak
        ob_all = vis_ob_[i: j + 24, :, :]
        fcst_all = cma_sh_warr_[i: j + 24, :, :]
        ts_ge4_cma_all = VisAcc(ob_all, fcst_all).get_ts_ge()[3]
        ob_s = vis_ob_[i: i + 24, :, :]
        fcst_s = cma_sh_warr_[i: i + 24, :, :]
        ts_ge4_cma_start = VisAcc(ob_s, fcst_s).get_ts_ge()[3]
        ob_e = vis_ob_[j: j + 24, :, :]
        fcst_e = cma_sh_warr_[j: j + 24, :, :]
        ts_ge4_cma_end = VisAcc(ob_e, fcst_e).get_ts_ge()[3]
        ob_p = vis_ob_[k: k + 24, :, :]
        fcst_p = cma_sh_warr_[k: k + 24, :, :]
        ts_ge4_cma_peak = VisAcc(ob_p, fcst_p).get_ts_ge()[3]
        p_ob_all = vis_ob_[i: j + 24, :, :]
        p_fcst_all = pred_pdfm2_[i: j + 24, :, :]
        ts_ge4_pdfm_all = VisAcc(p_ob_all, p_fcst_all).get_ts_ge()[3]
        p_ob_s = vis_ob_[i: i + 24, :, :]
        p_fcst_s = pred_pdfm2_[i: i + 24, :, :]
        ts_ge4_pdfm_start = VisAcc(p_ob_s, p_fcst_s).get_ts_ge()[3]
        p_ob_e = vis_ob_[j: j + 24, :, :]
        p_fcst_e = pred_pdfm2_[j: j + 24, :, :]
        ts_ge4_pdfm_end = VisAcc(p_ob_e, p_fcst_e).get_ts_ge()[3]
        p_ob_p = vis_ob_[k: k + 24, :, :]
        p_fcst_p = pred_pdfm2_[k: k + 24, :, :]
        ts_ge4_pdfm_peak = VisAcc(p_ob_p, p_fcst_p).get_ts_ge()[3]
        case_str = f'Case {idx:02d} [{d[0]}~{d[1]}, pk={d[2]}] '
        cma_str = f'CMA=[a:{ts_ge4_cma_all:.4f}, s:{ts_ge4_cma_start:.4f}, '
        cma_str += f'e:{ts_ge4_cma_end:.4f}, p:{ts_ge4_cma_peak:.4f}]  '
        pdfm_str = f'PDFM=[a:{ts_ge4_pdfm_all:.4f},s:{ts_ge4_pdfm_start:.4f},'
        pdfm_str += f'e:{ts_ge4_pdfm_end:.4f}, p:{ts_ge4_pdfm_peak:.4f}]'
        print(case_str + cma_str + pdfm_str)
        case_idx[i: j + 24, 0] = True
        case_idx[i: i + 24, 1] = True
        case_idx[j: j + 24, 2] = True
        case_idx[k: k + 24, 3] = True

    # 3. Calc merged statistics across all cases by phase
    idx0 = case_idx[:, 0]
    ob0, fcst0 = vis_ob_[idx0, :, :], cma_sh_warr_[idx0, :, :]
    ts_ge4_cma_m_all = VisAcc(ob0, fcst0).get_ts_ge()[3]
    idx1 = case_idx[:, 1]
    ob1, fcst1 = vis_ob_[idx1, :, :], cma_sh_warr_[idx1, :, :]
    ts_ge4_cma_m_start = VisAcc(ob1, fcst1).get_ts_ge()[3]
    idx2 = case_idx[:, 2]
    ob2, fcst2 = vis_ob_[idx2, :, :], cma_sh_warr_[idx2, :, :]
    ts_ge4_cma_m_end = VisAcc(ob2, fcst2).get_ts_ge()[3]
    idx3 = case_idx[:, 3]
    ob3, fcst3 = vis_ob_[idx3, :, :], cma_sh_warr_[idx3, :, :]
    ts_ge4_cma_m_peak = VisAcc(ob3, fcst3).get_ts_ge()[3]
    p_ob0, p_fcst0 = vis_ob_[idx0, :, :], pred_pdfm2_[idx0, :, :]
    ts_ge4_pdfm_m_all = VisAcc(p_ob0, p_fcst0).get_ts_ge()[3]
    p_ob1, p_fcst1 = vis_ob_[idx1, :, :], pred_pdfm2_[idx1, :, :]
    ts_ge4_pdfm_m_start = VisAcc(p_ob1, p_fcst1).get_ts_ge()[3]
    p_ob2, p_fcst2 = vis_ob_[idx2, :, :], pred_pdfm2_[idx2, :, :]
    ts_ge4_pdfm_m_end = VisAcc(p_ob2, p_fcst2).get_ts_ge()[3]
    p_ob3, p_fcst3 = vis_ob_[idx3, :, :], pred_pdfm2_[idx3, :, :]
    ts_ge4_pdfm_m_peak = VisAcc(p_ob3, p_fcst3).get_ts_ge()[3]
    merged_str = f'Merged cases  '
    cma_s = f'a:{ts_ge4_cma_m_all:.4f}, s:{ts_ge4_cma_m_start:.4f}, '
    cma_s += f'e:{ts_ge4_cma_m_end:.4f}, p:{ts_ge4_cma_m_peak:.4f}'
    cma_str = f'CMA=[{cma_s}]  '
    pdfm_s = f'a:{ts_ge4_pdfm_m_all:.4f}, s:{ts_ge4_pdfm_m_start:.4f}, '
    pdfm_s += f'e:{ts_ge4_pdfm_m_end:.4f}, p:{ts_ge4_pdfm_m_peak:.4f}'
    pdfm_str = f'PDFM=[{pdfm_s}]'
    print(merged_str + cma_str + pdfm_str)
