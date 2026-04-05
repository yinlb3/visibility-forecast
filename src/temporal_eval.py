# -*- coding: utf-8 -*-
"""
时效分析与站点/类型检验模块.

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.vis_acc import VisAcc


def _create_group(keys: Tuple[str, ...]) -> Dict[str, dict]:
    """Create 10 empty dicts for metrics group."""
    return {k: {key: list() for key in keys} for k in
            ('corr', 'mae', 'rmse', 'mre', 'ts1', 'ts2', 'ts3', 'ts4', 'ts5', 'ts6')}


def _append_metrics(acc: VisAcc, group: Dict[str, dict], name: str) -> None:
    """Append VisAcc results to a metric group."""
    ts = acc.get_ts2()
    group['corr'][name].append(acc.get_r())
    group['mae'][name].append(acc.get_mae())
    group['rmse'][name].append(acc.get_rmse())
    group['mre'][name].append(acc.get_mre())
    group['ts1'][name].append(ts[0])
    group['ts2'][name].append(ts[1])
    group['ts3'][name].append(ts[2])
    group['ts4'][name].append(ts[3])
    group['ts5'][name].append(ts[4])
    group['ts6'][name].append(ts[5])


def build_fhour_index(n_days: int = 8760, n_hours: int = 24) -> np.ndarray:
    """
    构建预报时效索引数组.

    fhour_ind[i, j] is actual forecast time (UTC) for day i, lead j.

    Args:
        n_days (int): 起报日数,默认 8760.
        n_hours (int): 每日小时数,默认 24.

    Returns:
        np.ndarray: 预报时效索引数组.
    """
    fhour_ind = np.zeros((n_days, n_hours), dtype=np.int_)
    for i in range(n_days):
        for j in range(n_hours):
            fhour_ind[i, j] = (i + j) % n_hours
    return fhour_ind


def calc_temporal_metrics(
    vis_ob: np.ndarray,
    cma_sh_warr: np.ndarray,
    pred_pdfm_tle0: np.ndarray,
    pred_pdfm_tle2: np.ndarray,
    output_dir: str
) -> np.ndarray:
    """
    计算起报时次/预报时效/预报时间的检验指标并保存.

    Args:
        vis_ob (np.ndarray): 观测数组.
        cma_sh_warr (np.ndarray): CMA-SH-WARR 预报数组.
        pred_pdfm_tle0 (np.ndarray): PDFM-TLE 试验 0.
        pred_pdfm_tle2 (np.ndarray): PDFM-TLE 试验 2.
        output_dir (str): 输出目录.

    Returns:
        np.ndarray: hour_access 四维数组, 形状为 (4, 24, 24, 10).
    """
    fhour_ind = build_fhour_index()
    vt = _create_group(('vt', 'CMA-SH-WARR', 'PDFM-TLE'))
    shour = _create_group(('shour', 'CMA-SH-WARR', 'PDFM-TLE'))
    fhour = _create_group(('fhour', 'CMA-SH-WARR', 'PDFM-TLE'))
    hour_access = np.zeros((4, 24, 24, 10), dtype=np.float32) + np.nan

    for i in range(24):
        for g, idx_name, idx_val in ((vt, 'vt', i + 1),
                                      (shour, 'shour', i),
                                      (fhour, 'fhour', i)):
            for k in g:
                g[k][idx_name].append(idx_val)

        acc_nwp = VisAcc(vis_ob[:, i, :], cma_sh_warr[:, i, :])
        _append_metrics(acc_nwp, vt, 'CMA-SH-WARR')
        acc = VisAcc(vis_ob[:, i, :], pred_pdfm_tle2[:, i, :])
        _append_metrics(acc, vt, 'PDFM-TLE')

        acc_nwp = VisAcc(vis_ob[i::24, :, :], cma_sh_warr[i::24, :, :])
        _append_metrics(acc_nwp, shour, 'CMA-SH-WARR')
        acc = VisAcc(vis_ob[i::24, :, :], pred_pdfm_tle0[i::24, :, :])
        _append_metrics(acc, shour, 'PDFM-TLE')

        acc_nwp = VisAcc(vis_ob[fhour_ind == i], cma_sh_warr[fhour_ind == i])
        _append_metrics(acc_nwp, fhour, 'CMA-SH-WARR')
        acc = VisAcc(vis_ob[fhour_ind == i, :], pred_pdfm_tle2[fhour_ind == i, :])
        _append_metrics(acc, fhour, 'PDFM-TLE')

        for j in range(24):
            acc_nwp = VisAcc(vis_ob[i::24, j, :], cma_sh_warr[i::24, j, :])
            hour_access[0, i, j, 0] = acc_nwp.get_r()
            hour_access[0, i, j, 1] = acc_nwp.get_mae()
            hour_access[0, i, j, 2] = acc_nwp.get_rmse()
            hour_access[0, i, j, 3] = acc_nwp.get_mre()
            hour_access[0, i, j, 4:] = acc_nwp.get_ts2()
            acc = VisAcc(vis_ob[i::24, j, :], pred_pdfm_tle2[i::24, j, :])
            hour_access[1, i, j, 0] = acc.get_r()
            hour_access[1, i, j, 1] = acc.get_mae()
            hour_access[1, i, j, 2] = acc.get_rmse()
            hour_access[1, i, j, 3] = acc.get_mre()
            hour_access[1, i, j, 4:] = acc.get_ts2()
            acc_nwp = VisAcc(vis_ob[i::24, j, :], pred_pdfm_tle2[i::24, j, :])
            hour_access[2, i, j, 0] = acc_nwp.get_r()
            hour_access[2, i, j, 1] = acc_nwp.get_mae()
            hour_access[2, i, j, 2] = acc_nwp.get_rmse()
            hour_access[2, i, j, 3] = acc_nwp.get_mre()
            hour_access[2, i, j, 4:] = acc_nwp.get_ts2()
            acc = VisAcc(vis_ob[i::24, j, :], pred_pdfm_tle2[i::24, j, :])
            hour_access[3, i, j, 0] = acc.get_r()
            hour_access[3, i, j, 1] = acc.get_mae()
            hour_access[3, i, j, 2] = acc.get_rmse()
            hour_access[3, i, j, 3] = acc.get_mre()
            hour_access[3, i, j, 4:] = acc.get_ts2()

    import os
    csv_dir = rf'{output_dir}\csv'
    os.makedirs(csv_dir, exist_ok=True)
    pd.DataFrame(vt['ts4']).to_csv(rf'{csv_dir}\vis_vt_ts4+.csv', index=False)
    pd.DataFrame(fhour['ts4']).to_csv(rf'{csv_dir}\vis_fhour_ts4+.csv', index=False)
    np.save(rf'{csv_dir}\hour_access.npy', hour_access)
    return hour_access


def load_v_type(data_dir: str, index_cjzxy: np.ndarray) -> np.ndarray:
    """
    加载能见度类型数据.

    Args:
        data_dir (str): 数据根目录.
        index_cjzxy (np.ndarray): 长江中下游站点筛选索引.

    Returns:
        np.ndarray: 能见度类型数组.
    """
    v_type = np.load(rf'{data_dir}\v_type.npy')
    v_type = np.reshape(v_type[-365:, :, :, index_cjzxy], (-1, 24, 502))
    return v_type


def calc_type_metrics(
    vis_ob: np.ndarray,
    cma_sh_warr: np.ndarray,
    pred_pdfm_tle0: np.ndarray,
    v_type: np.ndarray
) -> Dict[str, pd.DataFrame]:
    """
    按能见度类型计算检验指标.

    Args:
        vis_ob (np.ndarray): 观测数组.
        cma_sh_warr (np.ndarray): CMA-SH-WARR 预报数组.
        pred_pdfm_tle0 (np.ndarray): PDFM-TLE 试验 0.
        v_type (np.ndarray): 能见度类型掩码数组.

    Returns:
        dict: 各指标对应的 DataFrame.
    """
    d = _create_group(('type', 'CMA-SH-WARR', 'PDFM-TLE'))
    for i in range(3):
        for k in d:
            d[k]['type'].append(i)
        acc_nwp = VisAcc(vis_ob[v_type == i], cma_sh_warr[v_type == i])
        _append_metrics(acc_nwp, d, 'CMA-SH-WARR')
        acc = VisAcc(vis_ob[v_type == i], pred_pdfm_tle0[v_type == i])
        _append_metrics(acc, d, 'PDFM-TLE')

    print(f'[print_metrics] CMA-SH-WARR: corr={d["corr"]["CMA-SH-WARR"]:.4f}, mae={d["mae"]["CMA-SH-WARR"]:.4f}, rmse={d["rmse"]["CMA-SH-WARR"]:.4f}, mre={d["mre"]["CMA-SH-WARR"]:.4f}')
    print(f'[print_metrics] PDFM-TLE: corr={d["corr"]["PDFM-TLE"]:.4f}, mae={d["mae"]["PDFM-TLE"]:.4f}, rmse={d["rmse"]["PDFM-TLE"]:.4f}, mre={d["mre"]["PDFM-TLE"]:.4f}')
    print(f'[print_metrics] CMA-SH-WARR TS: ts1={d["ts1"]["CMA-SH-WARR"]:.4f}, ts2={d["ts2"]["CMA-SH-WARR"]:.4f}, ts3={d["ts3"]["CMA-SH-WARR"]:.4f}, ts4={d["ts4"]["CMA-SH-WARR"]:.4f}, ts5={d["ts5"]["CMA-SH-WARR"]:.4f}, ts6={d["ts6"]["CMA-SH-WARR"]:.4f}')
    print(f'[print_metrics] PDFM-TLE TS: ts1={d["ts1"]["PDFM-TLE"]:.4f}, ts2={d["ts2"]["PDFM-TLE"]:.4f}, ts3={d["ts3"]["PDFM-TLE"]:.4f}, ts4={d["ts4"]["PDFM-TLE"]:.4f}, ts5={d["ts5"]["PDFM-TLE"]:.4f}, ts6={d["ts6"]["PDFM-TLE"]:.4f}')

    return {k: pd.DataFrame(d[k]) for k in d}


def save_type_results(dfs: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """Save visibility type metrics to CSV."""
    import os
    csv_dir = rf'{output_dir}\csv'
    os.makedirs(csv_dir, exist_ok=True)
    dfs['corr'].to_csv(rf'{csv_dir}\vis_type_corr.csv', index=False)
    dfs['mae'].to_csv(rf'{csv_dir}\vis_type_mae.csv', index=False)
    dfs['rmse'].to_csv(rf'{csv_dir}\vis_type_rmse.csv', index=False)
    dfs['mre'].to_csv(rf'{csv_dir}\vis_type_mre.csv', index=False)
    for i in range(1, 7):
        dfs[f'ts{i}'].to_csv(rf'{csv_dir}\vis_type_ts{i}+.csv', index=False)


def calc_sta_metrics(
    vis_ob: np.ndarray,
    cma_sh_warr: np.ndarray,
    pred_pdfm_tle2: np.ndarray
) -> Dict[str, pd.DataFrame]:
    """
    按站点计算检验指标.

    Args:
        vis_ob (np.ndarray): 观测数组, shape (..., n_sta).
        cma_sh_warr (np.ndarray): CMA-SH-WARR 预报数组.
        pred_pdfm_tle2 (np.ndarray): PDFM-TLE 试验 2.

    Returns:
        dict: 各指标对应的 DataFrame.
    """
    n_sta = vis_ob.shape[-1]  # 从数组形状获取站点数
    d = _create_group(('sta', 'CMA-SH-WARR', 'PDFM-TLE'))
    for i in range(n_sta):
        for k in d:
            d[k]['sta'].append(i)
        acc_nwp = VisAcc(vis_ob[:, :, i], cma_sh_warr[:, :, i])
        _append_metrics(acc_nwp, d, 'CMA-SH-WARR')
        acc = VisAcc(vis_ob[:, :, i], pred_pdfm_tle2[:, :, i])
        _append_metrics(acc, d, 'PDFM-TLE')
    return {k: pd.DataFrame(d[k]) for k in d}


def save_sta_results(dfs: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """保存站点检验指标到 CSV."""
    import os
    csv_dir = rf'{output_dir}\csv'
    os.makedirs(csv_dir, exist_ok=True)
    dfs['corr'].to_csv(rf'{csv_dir}\vis_sta_corr.csv', index=False)
    dfs['mae'].to_csv(rf'{csv_dir}\vis_sta_mae.csv', index=False)
    dfs['rmse'].to_csv(rf'{csv_dir}\vis_sta_rmse.csv', index=False)
    dfs['mre'].to_csv(rf'{csv_dir}\vis_sta_mre.csv', index=False)
    for i in range(1, 7):
        dfs[f'ts{i}'].to_csv(rf'{csv_dir}\vis_sta_ts{i}+.csv', index=False)
