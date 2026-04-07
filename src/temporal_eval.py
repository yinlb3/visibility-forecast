# -*- coding: utf-8 -*-
"""
Temporal analysis and station/type verification module.

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""

import pathlib
import typing

import numpy as np
import pandas as pd

from src import vis_acc

VisAcc = vis_acc.VisAcc


def _create_group(keys: typing.Tuple[str, ...]) -> typing.Dict[str, dict]:
    """
    Create 10 empty dicts for metrics group.

    Args:
        keys (typing.Tuple[str, ...]): Metric names tuple.

    Returns:
        typing.Dict[str, dict]: Dict of empty metric dicts.
    """
    return {k: {key: list() for key in keys} for k in
            ('corr', 'mae', 'rmse', 'mre', 'ts1', 'ts2',
             'ts3', 'ts4', 'ts5', 'ts6')}


def _fmt_list(lst: list) -> str:
    """
    Format a list of floats with 4 decimal places.

    Args:
        lst (list): List of floats to format.

    Returns:
        str: Formatted string like '[0.1234, 0.5678]'.
    """
    return '[' + ', '.join(f'{float(x):.4f}' for x in lst) + ']'


def _append_metrics(
    acc: VisAcc, group: typing.Dict[str, dict], name: str
) -> None:
    """
    Append VisAcc results to a metric group.

    Args:
        acc (VisAcc): VisAcc object with calculated metrics.
        group (typing.Dict[str, dict]): Metric group dict to append to.
        name (str): Name key for the metric group.
    """
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
    Build forecast lead time index array.

    fhour_ind[i, j] is actual forecast time (UTC) for day i, lead j.

    Args:
        n_days (int): Number of forecast days, default 8760.
        n_hours (int): Hours per day, default 24.

    Returns:
        np.ndarray: Forecast lead time index array.
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
    Calc init time/lead time/forecast time metrics and save.

    Args:
        vis_ob (np.ndarray): Observation array.
        cma_sh_warr (np.ndarray): CMA-SH-WARR forecast array.
        pred_pdfm_tle0 (np.ndarray): PDFM-TLE experiment 0.
        pred_pdfm_tle2 (np.ndarray): PDFM-TLE experiment 2.
        output_dir (str): Output directory.

    Returns:
        np.ndarray: hour_access 4D array, shape (4, 24, 24, 10).
    """
    # 1. Build forecast hour index and init metric groups
    fhour_ind = build_fhour_index()
    vt = _create_group(('vt', 'CMA-SH-WARR', 'PDFM-TLE'))
    shour = _create_group(('shour', 'CMA-SH-WARR', 'PDFM-TLE'))
    fhour = _create_group(('fhour', 'CMA-SH-WARR', 'PDFM-TLE'))
    hour_access = np.zeros((4, 24, 24, 10), dtype=np.float32) + np.nan

    # 2. Loop through 24 forecast lead times
    for i in range(24):
        # 2.1 Record index values for each metric group
        for g, idx_name, idx_val in ((vt, 'vt', i + 1),
                                      (shour, 'shour', i),
                                      (fhour, 'fhour', i)):
            for k in g:
                g[k][idx_name].append(idx_val)

        # 2.2 Calc metrics by lead time (vt group)
        acc_nwp = VisAcc(vis_ob[:, i, :], cma_sh_warr[:, i, :])
        _append_metrics(acc_nwp, vt, 'CMA-SH-WARR')
        acc = VisAcc(vis_ob[:, i, :], pred_pdfm_tle2[:, i, :])
        _append_metrics(acc, vt, 'PDFM-TLE')

        # 2.3 Calc metrics by init hour (shour group)
        acc_nwp = VisAcc(vis_ob[i::24, :, :], cma_sh_warr[i::24, :, :])
        _append_metrics(acc_nwp, shour, 'CMA-SH-WARR')
        acc = VisAcc(vis_ob[i::24, :, :], pred_pdfm_tle0[i::24, :, :])
        _append_metrics(acc, shour, 'PDFM-TLE')

        # 2.4 Calc metrics by forecast hour (fhour group)
        acc_nwp = VisAcc(vis_ob[fhour_ind == i], cma_sh_warr[fhour_ind == i])
        _append_metrics(acc_nwp, fhour, 'CMA-SH-WARR')
        ob_slice = vis_ob[fhour_ind == i, :]
        pr_slice = pred_pdfm_tle2[fhour_ind == i, :]
        acc = VisAcc(ob_slice, pr_slice)
        _append_metrics(acc, fhour, 'PDFM-TLE')

        # 2.5 Calc 2D heatmap metrics (init hour x lead hour)
        for j in range(24):
            # CMA-SH-WARR metrics
            acc_nwp = VisAcc(vis_ob[i::24, j, :], cma_sh_warr[i::24, j, :])
            hour_access[0, i, j, 0] = acc_nwp.get_r()
            hour_access[0, i, j, 1] = acc_nwp.get_mae()
            hour_access[0, i, j, 2] = acc_nwp.get_rmse()
            hour_access[0, i, j, 3] = acc_nwp.get_mre()
            hour_access[0, i, j, 4:] = acc_nwp.get_ts2()
            # PDFM-TLE metrics
            acc = VisAcc(vis_ob[i::24, j, :], pred_pdfm_tle2[i::24, j, :])
            hour_access[1, i, j, 0] = acc.get_r()
            hour_access[1, i, j, 1] = acc.get_mae()
            hour_access[1, i, j, 2] = acc.get_rmse()
            hour_access[1, i, j, 3] = acc.get_mre()
            hour_access[1, i, j, 4:] = acc.get_ts2()
            # Additional experiments (3, 4)
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

    # 3. Save results to CSV and NPY
    csv_dir = pathlib.Path(output_dir) / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(vt['ts4']).to_csv(
        str(csv_dir / 'vis_vt_ts4+.csv'), index=False)
    pd.DataFrame(fhour['ts4']).to_csv(
        str(csv_dir / 'vis_fhour_ts4+.csv'), index=False)
    np.save(str(csv_dir / 'hour_access.npy'), hour_access)
    return hour_access


def load_v_type(data_dir: str, index_cjzxy: np.ndarray) -> np.ndarray:
    """
    Load visibility type data.

    Args:
        data_dir (str): Data root directory.
        index_cjzxy (np.ndarray): Middle-lower Yangtze station filter index.

    Returns:
        np.ndarray: Visibility type array.
    """
    v_type = np.load(str(pathlib.Path(data_dir) / 'v_type.npy'))
    v_type = np.reshape(v_type[-365:, :, :, index_cjzxy], (-1, 24, 502))
    return v_type


def calc_type_metrics(
    vis_ob: np.ndarray,
    cma_sh_warr: np.ndarray,
    pred_pdfm_tle0: np.ndarray,
    v_type: np.ndarray
) -> typing.Dict[str, pd.DataFrame]:
    """
    Calc metrics by visibility type.

    Args:
        vis_ob (np.ndarray): Observation array.
        cma_sh_warr (np.ndarray): CMA-SH-WARR forecast array.
        pred_pdfm_tle0 (np.ndarray): PDFM-TLE experiment 0.
        v_type (np.ndarray): Visibility type mask array.

    Returns:
        dict: DataFrames for each metric.
    """
    d = _create_group(('type', 'CMA-SH-WARR', 'PDFM-TLE'))
    for i in range(3):
        for k in d:
            d[k]['type'].append(i)
        acc_nwp = VisAcc(vis_ob[v_type == i], cma_sh_warr[v_type == i])
        _append_metrics(acc_nwp, d, 'CMA-SH-WARR')
        acc = VisAcc(vis_ob[v_type == i], pred_pdfm_tle0[v_type == i])
        _append_metrics(acc, d, 'PDFM-TLE')

    print('[Type Metrics] CMA-SH-WARR')
    c_cma = _fmt_list(d['corr']['CMA-SH-WARR'])
    a_cma = _fmt_list(d['mae']['CMA-SH-WARR'])
    r_cma = _fmt_list(d['rmse']['CMA-SH-WARR'])
    m_cma = _fmt_list(d['mre']['CMA-SH-WARR'])
    print(f'  corr={c_cma}, mae={a_cma}, rmse={r_cma}, mre={m_cma}')
    t1 = _fmt_list(d['ts1']['CMA-SH-WARR'])
    t2 = _fmt_list(d['ts2']['CMA-SH-WARR'])
    t3 = _fmt_list(d['ts3']['CMA-SH-WARR'])
    t4 = _fmt_list(d['ts4']['CMA-SH-WARR'])
    t5 = _fmt_list(d['ts5']['CMA-SH-WARR'])
    t6 = _fmt_list(d['ts6']['CMA-SH-WARR'])
    ts_str = f'ts1={t1}, ts2={t2}, ts3={t3}, ts4={t4}, ts5={t5}, ts6={t6}'
    print(f'  TS  = {ts_str}')
    print('[Type Metrics] PDFM-TLE')
    c_pdfm = _fmt_list(d['corr']['PDFM-TLE'])
    a_pdfm = _fmt_list(d['mae']['PDFM-TLE'])
    r_pdfm = _fmt_list(d['rmse']['PDFM-TLE'])
    m_pdfm = _fmt_list(d['mre']['PDFM-TLE'])
    print(f'  corr={c_pdfm}, mae={a_pdfm}, rmse={r_pdfm}, mre={m_pdfm}')
    t1p = _fmt_list(d['ts1']['PDFM-TLE'])
    t2p = _fmt_list(d['ts2']['PDFM-TLE'])
    t3p = _fmt_list(d['ts3']['PDFM-TLE'])
    t4p = _fmt_list(d['ts4']['PDFM-TLE'])
    t5p = _fmt_list(d['ts5']['PDFM-TLE'])
    t6p = _fmt_list(d['ts6']['PDFM-TLE'])
    ts_str = (
        f'ts1={t1p}, ts2={t2p}, ts3={t3p}, ts4={t4p}, ts5={t5p}, ts6={t6p}'
    )
    print(f'  TS  = {ts_str}')

    return {k: pd.DataFrame(d[k]) for k in d}


def save_type_results(
    dfs: typing.Dict[str, pd.DataFrame], output_dir: str
) -> None:
    """Save visibility type metrics to CSV."""
    import os
    csv_dir = pathlib.Path(output_dir) / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    dfs['corr'].to_csv(str(csv_dir / 'vis_type_corr.csv'), index=False)
    dfs['mae'].to_csv(str(csv_dir / 'vis_type_mae.csv'), index=False)
    dfs['rmse'].to_csv(str(csv_dir / 'vis_type_rmse.csv'), index=False)
    dfs['mre'].to_csv(str(csv_dir / 'vis_type_mre.csv'), index=False)
    for i in range(1, 7):
        dfs[f'ts{i}'].to_csv(
            str(csv_dir / f'vis_type_ts{i}+.csv'), index=False)



