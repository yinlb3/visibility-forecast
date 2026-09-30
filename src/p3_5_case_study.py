#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 3.5: Operational application case study evaluation result output module.

Founded in 2026-04-14
Modified in 2026-09-30
@author: yinlb, space-bunny
"""

import pathlib
import typing
from collections import defaultdict

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
        ['2024-01-02', '2024-01-13', '2024-01-03', 2],
        ['2024-01-29', '2024-02-01', '2024-01-30', 3],
        ['2024-02-08', '2024-02-11', '2024-02-10', 2],
        ['2024-03-02', '2024-03-08', '2024-03-04', 2],
        ['2024-03-11', '2024-03-18', '2024-03-14', 2],
        ['2024-03-23', '2024-03-28', '2024-03-27', 1],
        ['2024-03-31', '2024-04-02', '2024-04-01', 3],
        ['2024-04-06', '2024-04-08', '2024-04-08', 1],
        ['2024-04-11', '2024-04-29', '2024-04-14', 1],
        ['2024-05-02', '2024-05-06', '2024-05-05', 3],
        ['2024-06-24', '2024-07-01', '2024-06-27', 4],
        ['2024-07-11', '2024-07-13', '2024-07-11', 1],
        ['2024-10-14', '2024-10-18', '2024-10-14', 1],
        ['2024-10-27', '2024-10-29', '2024-10-28', 1],
        ['2024-11-10', '2024-11-12', '2024-11-10', 2],
        ['2024-12-04', '2024-12-07', '2024-12-05', 2],
    ]
    # 1. Define 16 case studies with start, end, peak dates and category
    # case_idx[:, 0~3] marks hours belonging to all/start/end/peak phases.
    # cat_data collects phase arrays by (category, phase, variable) for later
    # category-level aggregation.
    case_idx = np.zeros(shape=(8784, 4), dtype=np.bool_)
    cat_data = defaultdict(list)

    for idx, dt in enumerate(date_list, 1):
        cat = dt[3]
        base = arrow.get('2024')
        # Convert dates to hour indices relative to 2024-01-01 00:00 UTC.
        # Slice end uses j+24 to include the full end day (24 hours).
        i = round((arrow.get(dt[0]) - base).total_seconds() / 3600)
        j = round((arrow.get(dt[1]) - base).total_seconds() / 3600)
        k = round((arrow.get(dt[2]) - base).total_seconds() / 3600)
        # 2. Extract four phases: all period, start, end, and peak
        ob_entire = vis_ob_[i: j + 24, :, :]
        fcst_entire = cma_sh_warr_[i: j + 24, :, :]
        ts_ge4_cma_all = VisAcc(ob_entire, fcst_entire).get_ts_ge()[3]
        ob_onset = vis_ob_[i: i + 24, :, :]
        fcst_onset = cma_sh_warr_[i: i + 24, :, :]
        ts_ge4_cma_start = VisAcc(ob_onset, fcst_onset).get_ts_ge()[3]
        ob_termination = vis_ob_[j: j + 24, :, :]
        fcst_termination = cma_sh_warr_[j: j + 24, :, :]
        ts_ge4_cma_end = VisAcc(ob_termination, fcst_termination).get_ts_ge()[3]
        ob_severe = vis_ob_[k: k + 24, :, :]
        fcst_severe = cma_sh_warr_[k: k + 24, :, :]
        ts_ge4_cma_peak = VisAcc(ob_severe, fcst_severe).get_ts_ge()[3]
        p_ob_entire = vis_ob_[i: j + 24, :, :]
        p_fcst_entire = pred_pdfm2_[i: j + 24, :, :]
        ts_ge4_pdfm_all = VisAcc(p_ob_entire, p_fcst_entire).get_ts_ge()[3]
        p_ob_onset = vis_ob_[i: i + 24, :, :]
        p_fcst_onset = pred_pdfm2_[i: i + 24, :, :]
        ts_ge4_pdfm_start = VisAcc(p_ob_onset, p_fcst_onset).get_ts_ge()[3]
        p_ob_termination = vis_ob_[j: j + 24, :, :]
        p_fcst_termination = pred_pdfm2_[j: j + 24, :, :]
        ts_ge4_pdfm_end = VisAcc(
            p_ob_termination, p_fcst_termination
        ).get_ts_ge()[3]
        p_ob_severe = vis_ob_[k: k + 24, :, :]
        p_fcst_severe = pred_pdfm2_[k: k + 24, :, :]
        ts_ge4_pdfm_peak = VisAcc(p_ob_severe, p_fcst_severe).get_ts_ge()[3]
        case_str = f'Case {idx:02d} [{dt[0]}~{dt[1]}, pk={dt[2]}] '
        cma_str = f'CMA=[a:{ts_ge4_cma_all:.4f}, s:{ts_ge4_cma_start:.4f}, '
        cma_str += f'e:{ts_ge4_cma_end:.4f}, p:{ts_ge4_cma_peak:.4f}]  '
        pdfm_str = f'PDFM=[a:{ts_ge4_pdfm_all:.4f},s:{ts_ge4_pdfm_start:.4f},'
        pdfm_str += f'e:{ts_ge4_pdfm_end:.4f}, p:{ts_ge4_pdfm_peak:.4f}]'
        print(case_str + cma_str + pdfm_str)
        case_idx[i: j + 24, 0] = True
        case_idx[i: i + 24, 1] = True
        case_idx[j: j + 24, 2] = True
        case_idx[k: k + 24, 3] = True
        cat_data[(cat, 0, 'ob')].append(ob_entire)
        cat_data[(cat, 0, 'cma')].append(fcst_entire)
        cat_data[(cat, 0, 'pdfm')].append(p_fcst_entire)
        cat_data[(cat, 1, 'ob')].append(ob_onset)
        cat_data[(cat, 1, 'cma')].append(fcst_onset)
        cat_data[(cat, 1, 'pdfm')].append(p_fcst_onset)
        cat_data[(cat, 2, 'ob')].append(ob_termination)
        cat_data[(cat, 2, 'cma')].append(fcst_termination)
        cat_data[(cat, 2, 'pdfm')].append(p_fcst_termination)
        cat_data[(cat, 3, 'ob')].append(ob_severe)
        cat_data[(cat, 3, 'cma')].append(fcst_severe)
        cat_data[(cat, 3, 'pdfm')].append(p_fcst_severe)

    # 3. Calc merged statistics across all cases by phase.
    # Merging concatenates the same phase of every case, so each of the
    # four masks below collects one phase over the whole 2024 case set.
    idx_entire = case_idx[:, 0]
    ob_merged, fcst_merged = (
        vis_ob_[idx_entire, :, :], cma_sh_warr_[idx_entire, :, :]
    )
    ts_ge4_cma_merged_entire = VisAcc(ob_merged, fcst_merged).get_ts_ge()[3]
    idx_onset = case_idx[:, 1]
    ob_merged, fcst_merged = (
        vis_ob_[idx_onset, :, :], cma_sh_warr_[idx_onset, :, :]
    )
    ts_ge4_cma_merged_onset = VisAcc(ob_merged, fcst_merged).get_ts_ge()[3]
    idx_termination = case_idx[:, 2]
    ob_merged, fcst_merged = (
        vis_ob_[idx_termination, :, :], cma_sh_warr_[idx_termination, :, :]
    )
    ts_ge4_cma_merged_termination = VisAcc(
        ob_merged, fcst_merged
    ).get_ts_ge()[3]
    idx_severe = case_idx[:, 3]
    ob_merged, fcst_merged = (
        vis_ob_[idx_severe, :, :], cma_sh_warr_[idx_severe, :, :]
    )
    ts_ge4_cma_merged_severe = VisAcc(ob_merged, fcst_merged).get_ts_ge()[3]
    p_ob_merged, p_fcst_merged = (
        vis_ob_[idx_entire, :, :], pred_pdfm2_[idx_entire, :, :]
    )
    ts_ge4_pdfm_merged_entire = VisAcc(
        p_ob_merged, p_fcst_merged
    ).get_ts_ge()[3]
    p_ob_merged, p_fcst_merged = (
        vis_ob_[idx_onset, :, :], pred_pdfm2_[idx_onset, :, :]
    )
    ts_ge4_pdfm_merged_onset = VisAcc(
        p_ob_merged, p_fcst_merged
    ).get_ts_ge()[3]
    p_ob_merged, p_fcst_merged = (
        vis_ob_[idx_termination, :, :], pred_pdfm2_[idx_termination, :, :]
    )
    ts_ge4_pdfm_merged_termination = VisAcc(
        p_ob_merged, p_fcst_merged
    ).get_ts_ge()[3]
    p_ob_merged, p_fcst_merged = (
        vis_ob_[idx_severe, :, :], pred_pdfm2_[idx_severe, :, :]
    )
    ts_ge4_pdfm_merged_severe = VisAcc(
        p_ob_merged, p_fcst_merged
    ).get_ts_ge()[3]
    merged_str = 'Merged cases  '
    cma_s = f'a:{ts_ge4_cma_merged_entire:.4f}, '
    cma_s += f's:{ts_ge4_cma_merged_onset:.4f}, '
    cma_s += f'e:{ts_ge4_cma_merged_termination:.4f}, '
    cma_s += f'p:{ts_ge4_cma_merged_severe:.4f}'
    cma_str = f'CMA=[{cma_s}]  '
    pdfm_s = f'a:{ts_ge4_pdfm_merged_entire:.4f}, '
    pdfm_s += f's:{ts_ge4_pdfm_merged_onset:.4f}, '
    pdfm_s += f'e:{ts_ge4_pdfm_merged_termination:.4f}, '
    pdfm_s += f'p:{ts_ge4_pdfm_merged_severe:.4f}'
    pdfm_str = f'PDFM=[{pdfm_s}]'
    print(merged_str + cma_str + pdfm_str)

    # 4. Calc statistics by category
    for cat in sorted({k[0] for k in cat_data.keys()}):
        ob_entire_cat = np.concatenate(cat_data[(cat, 0, 'ob')])
        cma_all_cat = np.concatenate(cat_data[(cat, 0, 'cma')])
        pdfm_all_cat = np.concatenate(cat_data[(cat, 0, 'pdfm')])
        ts_cma_all = VisAcc(ob_entire_cat, cma_all_cat).get_ts_ge()[3]
        ts_pdfm_all = VisAcc(ob_entire_cat, pdfm_all_cat).get_ts_ge()[3]
        ob_onset_cat = np.concatenate(cat_data[(cat, 1, 'ob')])
        cma_s_cat = np.concatenate(cat_data[(cat, 1, 'cma')])
        pdfm_s_cat = np.concatenate(cat_data[(cat, 1, 'pdfm')])
        ts_cma_s = VisAcc(ob_onset_cat, cma_s_cat).get_ts_ge()[3]
        ts_pdfm_s = VisAcc(ob_onset_cat, pdfm_s_cat).get_ts_ge()[3]
        ob_termination_cat = np.concatenate(cat_data[(cat, 2, 'ob')])
        cma_e_cat = np.concatenate(cat_data[(cat, 2, 'cma')])
        pdfm_e_cat = np.concatenate(cat_data[(cat, 2, 'pdfm')])
        ts_cma_e = VisAcc(ob_termination_cat, cma_e_cat).get_ts_ge()[3]
        ts_pdfm_e = VisAcc(ob_termination_cat, pdfm_e_cat).get_ts_ge()[3]
        ob_severe_cat = np.concatenate(cat_data[(cat, 3, 'ob')])
        cma_p_cat = np.concatenate(cat_data[(cat, 3, 'cma')])
        pdfm_p_cat = np.concatenate(cat_data[(cat, 3, 'pdfm')])
        ts_cma_p = VisAcc(ob_severe_cat, cma_p_cat).get_ts_ge()[3]
        ts_pdfm_p = VisAcc(ob_severe_cat, pdfm_p_cat).get_ts_ge()[3]
        cat_str = f'Category {cat}  '
        cma_s = f'a:{ts_cma_all:.4f}, s:{ts_cma_s:.4f}, '
        cma_s += f'e:{ts_cma_e:.4f}, p:{ts_cma_p:.4f}'
        cma_str = f'CMA=[{cma_s}]  '
        pdfm_s = f'a:{ts_pdfm_all:.4f}, s:{ts_pdfm_s:.4f}, '
        pdfm_s += f'e:{ts_pdfm_e:.4f}, p:{ts_pdfm_p:.4f}'
        pdfm_str = f'PDFM=[{pdfm_s}]'
        def _gain(t_pdfm: float, t_cma: float) -> float:
            """Compute percentage gainovement of PDFM over CMA."""
            return ((t_pdfm - t_cma) / t_cma * 100
                    if t_cma != 0 else np.nan)
        gain_all = _gain(ts_pdfm_all, ts_cma_all)
        gain_s = _gain(ts_pdfm_s, ts_cma_s)
        gain_e = _gain(ts_pdfm_e, ts_cma_e)
        gain_p = _gain(ts_pdfm_p, ts_cma_p)
        gain_str = (f'Gain=[a:{gain_all:.2f}%, s:{gain_s:.2f}%, '
                    f'e:{gain_e:.2f}%, p:{gain_p:.2f}%]')
        print(cat_str + cma_str + pdfm_str + '  ' + gain_str)
