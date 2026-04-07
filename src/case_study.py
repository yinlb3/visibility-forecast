# -*- coding: utf-8 -*-
"""
2024 independent case study analysis module (Stage 9).

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""

import pathlib

import arrow
import numpy as np

from src import vis_acc

VisAcc = vis_acc.VisAcc


def load_2024_obs(data_dir: str, index_cjzxy: np.ndarray) -> np.ndarray:
    """Load 2024 observation data for case study."""
    # 1. Load raw obs data and reshape to (time, lead, station)
    vis_ob = np.load(str(pathlib.Path(data_dir) / 'vis1183_ob_2024.npy'))
    vis_ob = np.reshape(vis_ob[:, 1:, index_cjzxy], shape=(-1, 24, 502))
    # 2. Apply QC: mark missing values and cap at 30000m
    vis_ob[vis_ob >= 999990] = np.nan
    vis_ob[vis_ob >= 30000] = 30000
    return vis_ob


def load_2024_preds(data_dir: str) -> tuple:
    """
    Load 2024 PDFM/TLE prediction data for overall metrics.

    Args:
        data_dir (str): Data root directory.

    Returns:
        tuple: (pred_pdfm2, pred_tle2).
    """
    path = pathlib.Path(data_dir) / 'vis_gjz_pdfm2_cjzxy_2024.npy'
    pred_pdfm2 = np.load(str(path))
    pred_pdfm2 = np.reshape(pred_pdfm2, shape=(-1, 24, 502))
    pred_pdfm2[pred_pdfm2 >= 30000] = 30000

    path = pathlib.Path(data_dir) / 'vis_gjz_tle2_cjzxy_2024.npy'
    pred_tle2 = np.load(str(path))
    pred_tle2 = np.reshape(pred_tle2, shape=(-1, 24, 502))
    pred_tle2[pred_tle2 >= 30000] = 30000

    return pred_pdfm2, pred_tle2


def _fmt_arr(arr: np.ndarray) -> str:
    """Format numpy array with 4 decimal places for display."""
    return np.array2string(
        np.array(arr), precision=4, separator=' ', suppress_small=True
    )


def print_2024_overall_metrics(
    vis_ob: np.ndarray, pred_pdfm2: np.ndarray, pred_tle2: np.ndarray
) -> None:
    """
    Print overall verification metrics for PDFM and TLE.

    Args:
        vis_ob (np.ndarray): Observation array.
        pred_pdfm2 (np.ndarray): PDFM prediction array.
        pred_tle2 (np.ndarray): TLE prediction array.
    """
    acc = VisAcc(vis_ob, pred_pdfm2)
    print('[2024 Overall Metrics] PDFM')
    r = acc.get_r()
    mae = acc.get_mae()
    rmse = acc.get_rmse()
    mre = acc.get_mre()
    print(f'  R={r:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MRE={mre:.4f}')
    ts2_str = _fmt_arr(acc.get_ts2())
    far2_str = _fmt_arr(acc.get_far2())
    mar2_str = _fmt_arr(acc.get_mar2())
    print(f'  TS2={ts2_str}, FAR2={far2_str}, MAR2={mar2_str}')

    acc = VisAcc(vis_ob, pred_tle2)
    print('[2024 Overall Metrics] TLE')
    r = acc.get_r()
    mae = acc.get_mae()
    rmse = acc.get_rmse()
    mre = acc.get_mre()
    print(f'  R={r:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MRE={mre:.4f}')
    ts2_str = _fmt_arr(acc.get_ts2())
    far2_str = _fmt_arr(acc.get_far2())
    mar2_str = _fmt_arr(acc.get_mar2())
    print(f'  TS2={ts2_str}, FAR2={far2_str}, MAR2={mar2_str}')


def load_2024_eval_data(data_dir: str, index_cjzxy: np.ndarray) -> tuple:
    """
    Load 2024 obs and forecast data for case evaluation.

    Args:
        data_dir (str): Data root directory.
        index_cjzxy (np.ndarray): CJZXY station filter index.

    Returns:
        tuple: (cma_sh_warr, pred_pdfm2).
    """
    cma_sh_warr = np.load(str(pathlib.Path(data_dir) / 'vis_gjz_2024.npy'))
    cma_sh_warr = np.reshape(
        cma_sh_warr[:, 1:, index_cjzxy], shape=(-1, 24, 502)
    )
    cma_sh_warr[cma_sh_warr >= 30000] = 30000

    path = pathlib.Path(data_dir) / 'vis_gjz_pdfm2_cjzxy_2024.npy'
    pred_pdfm2 = np.load(str(path))
    pred_pdfm2 = np.reshape(pred_pdfm2, shape=(-1, 24, 502))
    pred_pdfm2[pred_pdfm2 >= 30000] = 30000

    return cma_sh_warr, pred_pdfm2


def align_forecast_times(
    vis_ob: np.ndarray, cma_sh_warr: np.ndarray, pred_pdfm2: np.ndarray
) -> tuple:
    """
    Apply time offset to 24 forecast init times to align with obs.
    Each array may have a different first dimension (e.g. 365 vs 366 days).
    """
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
    """
    Analyze 18 case studies and calc TS4+ for all phases and merged.

    Args:
        vis_ob_ (np.ndarray): Observation array with aligned time.
        cma_sh_warr_ (np.ndarray): CMA-SH-WARR forecast with aligned time.
        pred_pdfm2_ (np.ndarray): PDFM prediction with aligned time.
    """
    # 1. Define 18 case study periods [start, end, peak]
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
    # 2. Init index array for merging cases
    ind = np.zeros(shape=(8784, 4), dtype=np.bool_)

    # 3. Process each case: calc TS4+ for all phases
    for idx, d in enumerate(date_list, 1):
        base = arrow.get('2024')
        i = round((arrow.get(d[0]) - base).total_seconds() / 3600)
        j = round((arrow.get(d[1]) - base).total_seconds() / 3600)
        k = round((arrow.get(d[2]) - base).total_seconds() / 3600)
        ob_all = vis_ob_[i: j + 24, :, :]
        fcst_all = cma_sh_warr_[i: j + 24, :, :]
        cma_all = VisAcc(ob_all, fcst_all).get_ts2()[3]
        ob_s = vis_ob_[i: i + 24, :, :]
        fcst_s = cma_sh_warr_[i: i + 24, :, :]
        cma_start = VisAcc(ob_s, fcst_s).get_ts2()[3]
        ob_e = vis_ob_[j: j + 24, :, :]
        fcst_e = cma_sh_warr_[j: j + 24, :, :]
        cma_end = VisAcc(ob_e, fcst_e).get_ts2()[3]
        ob_p = vis_ob_[k: k + 24, :, :]
        fcst_p = cma_sh_warr_[k: k + 24, :, :]
        cma_peak = VisAcc(ob_p, fcst_p).get_ts2()[3]
        p_ob_all = vis_ob_[i: j + 24, :, :]
        p_fcst_all = pred_pdfm2_[i: j + 24, :, :]
        pdfm_all = VisAcc(p_ob_all, p_fcst_all).get_ts2()[3]
        p_ob_s = vis_ob_[i: i + 24, :, :]
        p_fcst_s = pred_pdfm2_[i: i + 24, :, :]
        pdfm_start = VisAcc(p_ob_s, p_fcst_s).get_ts2()[3]
        p_ob_e = vis_ob_[j: j + 24, :, :]
        p_fcst_e = pred_pdfm2_[j: j + 24, :, :]
        pdfm_end = VisAcc(p_ob_e, p_fcst_e).get_ts2()[3]
        p_ob_p = vis_ob_[k: k + 24, :, :]
        p_fcst_p = pred_pdfm2_[k: k + 24, :, :]
        pdfm_peak = VisAcc(p_ob_p, p_fcst_p).get_ts2()[3]
        # 4. Print case results and update merge indices
        case_str = f'Case {idx:02d} [{d[0]}~{d[1]}, pk={d[2]}] '
        cma_str = f'CMA=[a:{cma_all:.4f}, s:{cma_start:.4f}, '
        cma_str += f'e:{cma_end:.4f}, p:{cma_peak:.4f}]  '
        pdfm_str = f'PDFM=[a:{pdfm_all:.4f},s:{pdfm_start:.4f},'
        pdfm_str += f'e:{pdfm_end:.4f}, p:{pdfm_peak:.4f}]'
        print(case_str + cma_str + pdfm_str)
        ind[i: j + 24, 0] = True  # All period
        ind[i: i + 24, 1] = True  # Start phase
        ind[j: j + 24, 2] = True  # End phase
        ind[k: k + 24, 3] = True  # Peak phase

    # 5. Calc merged metrics across all cases
    ob0, fcst0 = vis_ob_[ind[:, 0], :, :], cma_sh_warr_[ind[:, 0], :, :]
    cma_m_all = VisAcc(ob0, fcst0).get_ts2()[3]
    ob1, fcst1 = vis_ob_[ind[:, 1], :, :], cma_sh_warr_[ind[:, 1], :, :]
    cma_m_start = VisAcc(ob1, fcst1).get_ts2()[3]
    ob2, fcst2 = vis_ob_[ind[:, 2], :, :], cma_sh_warr_[ind[:, 2], :, :]
    cma_m_end = VisAcc(ob2, fcst2).get_ts2()[3]
    ob3, fcst3 = vis_ob_[ind[:, 3], :, :], cma_sh_warr_[ind[:, 3], :, :]
    cma_m_peak = VisAcc(ob3, fcst3).get_ts2()[3]
    p_ob0, p_fcst0 = vis_ob_[ind[:, 0], :, :], pred_pdfm2_[ind[:, 0], :, :]
    pdfm_m_all = VisAcc(p_ob0, p_fcst0).get_ts2()[3]
    p_ob1, p_fcst1 = vis_ob_[ind[:, 1], :, :], pred_pdfm2_[ind[:, 1], :, :]
    pdfm_m_start = VisAcc(p_ob1, p_fcst1).get_ts2()[3]
    p_ob2, p_fcst2 = vis_ob_[ind[:, 2], :, :], pred_pdfm2_[ind[:, 2], :, :]
    pdfm_m_end = VisAcc(p_ob2, p_fcst2).get_ts2()[3]
    p_ob3, p_fcst3 = vis_ob_[ind[:, 3], :, :], pred_pdfm2_[ind[:, 3], :, :]
    pdfm_m_peak = VisAcc(p_ob3, p_fcst3).get_ts2()[3]
    merged_str = f'Merged cases  '
    cma_s = f'a:{cma_m_all:.4f}, s:{cma_m_start:.4f}, '
    cma_s += f'e:{cma_m_end:.4f}, p:{cma_m_peak:.4f}'
    cma_str = f'CMA=[{cma_s}]  '
    pdfm_s = f'a:{pdfm_m_all:.4f}, s:{pdfm_m_start:.4f}, '
    pdfm_s += f'e:{pdfm_m_end:.4f}, p:{pdfm_m_peak:.4f}'
    pdfm_str = f'PDFM=[{pdfm_s}]'
    print(merged_str + cma_str + pdfm_str)
