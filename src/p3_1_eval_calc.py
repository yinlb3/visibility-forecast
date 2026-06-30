#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 3.1: Evaluation result calculation module.

Founded in 2026-04-14
Modified in 2026-06-30
@author: yinlb
"""

import os
import pathlib
import typing

import numpy as np
import pandas as pd
import joblib

from src import vis_acc

VisAcc = vis_acc.VisAcc


# ==================== Forecast overall / station metrics ====================

def _fmt_arr(arr: np.ndarray) -> str:
    """Format numpy array with 4 decimal places."""
    return np.array2string(
        np.array(arr), precision=4, separator=' ', suppress_small=True
    )


def format_overall_metrics(
    vis_ob: np.ndarray, pred: np.ndarray, name: str
) -> str:
    """
    Calculate overall verification metrics and return formatted string.

    Args:
        vis_ob (np.ndarray): Observed visibility array.
        pred (np.ndarray): Predicted visibility array.
        name (str): Forecast scheme name for display.

    Returns:
        str: Multi-line formatted string with R, MAE, RMSE, MRE,
        TS, FAR, MAR, POD.
    """
    acc = VisAcc(vis_ob, pred)
    lines = [
        f'[Overall Metrics] {name}',
        f'  R    = {acc.get_r():.4f}',
        f'  MAE  = {acc.get_mae():.1f} m',
        f'  RMSE = {acc.get_rmse():.1f} m',
        f'  MRE  = {acc.get_mre():.4f}',
        f'  TS_ge  = {_fmt_arr(acc.get_ts_ge())}',
        f'  FAR_ge = {_fmt_arr(acc.get_far_ge())}',
        f'  MAR_ge = {_fmt_arr(acc.get_mar_ge())}',
        f'  POD_ge = {_fmt_arr(acc.get_pod_ge())}',
    ]
    return '\n'.join(lines)


def _calc_one_station(
    i: int,
    ob_i: np.ndarray,
    cma_i: np.ndarray,
    pdfm_i: np.ndarray
) -> typing.Dict[str, typing.Union[int, float]]:
    """
    Calc metrics for a single station (parallel worker).

    Args:
        i (int): Station index.
        ob_i (np.ndarray): Observed visibility at this station.
        cma_i (np.ndarray): CMA-SH-WARR forecast at this station.
        pdfm_i (np.ndarray): PDFM-TLE forecast at this station.

    Returns:
        typing.Dict[str, typing.Union[int, float]]: Dict with station index and
        verification metrics (corr, mae, rmse, mre, ts1-6) for both schemes.
    """
    acc_nwp = VisAcc(ob_i, cma_i)
    acc = VisAcc(ob_i, pdfm_i)
    return {
        'sta': i,
        'corr_cma': acc_nwp.get_r(),
        'mae_cma': acc_nwp.get_mae(),
        'rmse_cma': acc_nwp.get_rmse(),
        'mre_cma': acc_nwp.get_mre(),
        'ts1_cma': acc_nwp.get_ts_ge()[0],
        'ts2_cma': acc_nwp.get_ts_ge()[1],
        'ts3_cma': acc_nwp.get_ts_ge()[2],
        'ts4_cma': acc_nwp.get_ts_ge()[3],
        'ts5_cma': acc_nwp.get_ts_ge()[4],
        'ts6_cma': acc_nwp.get_ts_ge()[5],
        'corr_pdfm': acc.get_r(),
        'mae_pdfm': acc.get_mae(),
        'rmse_pdfm': acc.get_rmse(),
        'mre_pdfm': acc.get_mre(),
        'ts1_pdfm': acc.get_ts_ge()[0],
        'ts2_pdfm': acc.get_ts_ge()[1],
        'ts3_pdfm': acc.get_ts_ge()[2],
        'ts4_pdfm': acc.get_ts_ge()[3],
        'ts5_pdfm': acc.get_ts_ge()[4],
        'ts6_pdfm': acc.get_ts_ge()[5],
    }


def calc_and_save_station_metrics(
    vis_ob: np.ndarray,
    cma_sh_warr: np.ndarray,
    pred_pdfm_tle2: np.ndarray,
    output_dir: str
) -> None:
    """
    Calculate station-level verification metrics and save to CSV.

    Args:
        vis_ob (np.ndarray): Observed visibility array.
        cma_sh_warr (np.ndarray): CMA-SH-WARR forecast array.
        pred_pdfm_tle2 (np.ndarray): PDFM-TLE forecast array (scheme 3).
        output_dir (str): Output directory path.
    """
    csv_dir = pathlib.Path(output_dir) / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)

    metrics: typing.Dict[str, typing.Dict[str, list]] = {
        'corr': {'sta': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()},
        'mae': {'sta': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()},
        'rmse': {'sta': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()},
        'mre': {'sta': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()},
        'ts1': {'sta': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()},
        'ts2': {'sta': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()},
        'ts3': {'sta': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()},
        'ts4': {'sta': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()},
        'ts5': {'sta': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()},
        'ts6': {'sta': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()},
    }

    # Parallel station-level metrics calculation
    n_sta = vis_ob.shape[2]
    results = joblib.Parallel(n_jobs=-1, backend='loky')(
        joblib.delayed(_calc_one_station)(
            i, vis_ob[:, :, i], cma_sh_warr[:, :, i], pred_pdfm_tle2[:, :, i]
        )
        for i in range(n_sta)
    )

    # Aggregate parallel results into metric groups
    sta_ids = [res['sta'] for res in results]
    for key in metrics:
        metrics[key]['sta'] = sta_ids

    for res in results:
        metrics['corr']['CMA-SH-WARR'].append(res['corr_cma'])
        metrics['mae']['CMA-SH-WARR'].append(res['mae_cma'])
        metrics['rmse']['CMA-SH-WARR'].append(res['rmse_cma'])
        metrics['mre']['CMA-SH-WARR'].append(res['mre_cma'])
        metrics['ts1']['CMA-SH-WARR'].append(res['ts1_cma'])
        metrics['ts2']['CMA-SH-WARR'].append(res['ts2_cma'])
        metrics['ts3']['CMA-SH-WARR'].append(res['ts3_cma'])
        metrics['ts4']['CMA-SH-WARR'].append(res['ts4_cma'])
        metrics['ts5']['CMA-SH-WARR'].append(res['ts5_cma'])
        metrics['ts6']['CMA-SH-WARR'].append(res['ts6_cma'])
        metrics['corr']['PDFM-TLE'].append(res['corr_pdfm'])
        metrics['mae']['PDFM-TLE'].append(res['mae_pdfm'])
        metrics['rmse']['PDFM-TLE'].append(res['rmse_pdfm'])
        metrics['mre']['PDFM-TLE'].append(res['mre_pdfm'])
        metrics['ts1']['PDFM-TLE'].append(res['ts1_pdfm'])
        metrics['ts2']['PDFM-TLE'].append(res['ts2_pdfm'])
        metrics['ts3']['PDFM-TLE'].append(res['ts3_pdfm'])
        metrics['ts4']['PDFM-TLE'].append(res['ts4_pdfm'])
        metrics['ts5']['PDFM-TLE'].append(res['ts5_pdfm'])
        metrics['ts6']['PDFM-TLE'].append(res['ts6_pdfm'])

    # Save metrics to CSV and rename ts files to ts+ notation
    for key, data in metrics.items():
        df = pd.DataFrame(data)
        out_path = str(csv_dir / f'vis_sta_{key}.csv')
        df.to_csv(out_path, index=False)
    for i in range(1, 7):
        old_path = str(csv_dir / f'vis_sta_ts{i}.csv')
        new_path = str(csv_dir / f'vis_sta_ts{i}+.csv')
        if os.path.exists(old_path):
            os.replace(old_path, new_path)
    for i in range(1, 7):
        metrics[f'ts{i}+'] = metrics.pop(f'ts{i}')
    return {k: pd.DataFrame(val) for k, val in metrics.items()}


def load_station_metrics(output_dir: str) -> typing.Dict[str, pd.DataFrame]:
    """
    Load pre-calculated station-level metrics from CSV files.

    Args:
        output_dir (str): Directory containing the csv/ subfolder.

    Returns:
        typing.Dict[str, pd.DataFrame]: Dict with keys
        corr/mae/rmse/mre/ts1+/.../ts6+.

    Raises:
        FileNotFoundError: If any expected CSV file is missing.
    """
    csv_dir = pathlib.Path(output_dir) / 'csv'
    metrics = {}
    metric_names = ['corr', 'mae', 'rmse', 'mre']
    for key in metric_names:
        file_path = csv_dir / f'vis_sta_{key}.csv'
        if not file_path.exists():
            raise FileNotFoundError(
                f'Station metrics file not found: {file_path}'
            )
        metrics[key] = pd.read_csv(str(file_path), low_memory=False)
    for i in range(1, 7):
        file_path = csv_dir / f'vis_sta_ts{i}+.csv'
        if not file_path.exists():
            raise FileNotFoundError(
                f'Station metrics file not found: {file_path}'
            )
        metrics[f'ts{i}+'] = pd.read_csv(str(file_path), low_memory=False)
    return metrics


# ==================== Weather type metrics ====================

def load_weather_type(data_dir: str, idx_mlyr: np.ndarray) -> np.ndarray:
    """
    Load weather type data and reshape to match forecast.

    Args:
        data_dir (str): Directory containing weather_type.npy.
        idx_mlyr (np.ndarray): Index array for Middle-Lower Yangtze
        River stations.

    Returns:
        np.ndarray: Reshaped weather type array of shape (-1, 24, 502).
    """
    path = pathlib.Path(data_dir) / 'weather_type.npy'
    weather_type = np.load(str(path), mmap_mode='r')
    wt_slice = weather_type[1096:1461, ..., idx_mlyr]
    val_wt = np.reshape(wt_slice, shape=(-1, 24, 502))
    return val_wt


def calc_weather_type_metrics(
    vis_ob: np.ndarray,
    preds: typing.Tuple[np.ndarray, ...],
    val_wt: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Calc quantitative and grade metrics by weather type.

    Args:
        vis_ob (np.ndarray): Observed visibility array.
        preds (typing.Tuple[np.ndarray, ...]): Tuple of predicted arrays.
        val_wt (np.ndarray): Weather type mask array.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: qem (quantitative metrics) and
        cem (categorical metrics by grade).
    """
    # qem shape: (4 metrics, n_pred schemes, 3 weather types)
    # cem shape: (6 grades, 4 metrics, n_pred schemes, 3 weather types)
    # val_wt encodes weather type as 1=precip, 2=fog, 3=haze.
    n_pred = len(preds)
    qem = np.zeros((4, n_pred, 3), dtype=np.float32) + np.nan
    cem = np.zeros((6, 4, n_pred, 3), dtype=np.float32) + np.nan
    for i in range(3):
        mask = val_wt == i + 1
        ob_masked = vis_ob[mask]
        for j, pred in enumerate(preds):
            pr_masked = pred[mask]
            acc = VisAcc(ob_masked, pr_masked)
            qem[0, j, i] = acc.get_r()
            qem[1, j, i] = acc.get_mae()
            qem[2, j, i] = acc.get_rmse()
            qem[3, j, i] = acc.get_mre()
            cem[:, 0, j, i] = acc.get_ts_ge()
            cem[:, 1, j, i] = acc.get_far_ge()
            cem[:, 2, j, i] = acc.get_mar_ge()
            cem[:, 3, j, i] = acc.get_pod_ge()
    return qem, cem


def save_weather_type_metrics(
    qem: np.ndarray, cem: np.ndarray, output_dir: str
) -> None:
    """
    Save weather type metrics (qem, cem) to NPY files.

    Args:
        qem (np.ndarray): Quantitative metrics array.
        cem (np.ndarray): Categorical metrics array.
        output_dir (str): Output directory path.
    """
    csv_dir = pathlib.Path(output_dir) / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(csv_dir / 'qem.npy'), qem)
    np.save(str(csv_dir / 'cem.npy'), cem)
    print(f'[Weather Type] Saved qem and cem to {csv_dir}')


def load_weather_type_metrics(
    output_dir: str
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Load weather type metrics (qem, cem) from NPY files.

    Args:
        output_dir (str): Directory containing the csv/ subfolder.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: qem and cem arrays.

    Raises:
        FileNotFoundError: If qem.npy or cem.npy is missing.
    """
    csv_dir = pathlib.Path(output_dir) / 'csv'
    qem_path = csv_dir / 'qem.npy'
    cem_path = csv_dir / 'cem.npy'
    if not qem_path.exists():
        raise FileNotFoundError(f'Weather type metrics not found: {qem_path}')
    if not cem_path.exists():
        raise FileNotFoundError(f'Weather type metrics not found: {cem_path}')
    qem = np.load(str(qem_path))
    cem = np.load(str(cem_path))
    print(f'[Weather Type] Loaded qem and cem from {csv_dir}')
    return qem, cem


# ==================== Temporal / type metrics ====================

def _create_group(keys: typing.Tuple[str, ...]) -> typing.Dict[str, dict]:
    """Create empty dicts for metrics group."""
    return {k: {key: list() for key in keys} for k in (
        'corr', 'mae', 'rmse', 'mre',
        'ts1', 'ts2', 'ts3', 'ts4', 'ts5', 'ts6'
    )}


def _fmt_list(lst: list) -> str:
    """Format a list of floats with 4 decimal places."""
    return '[' + ', '.join(f'{float(x):.4f}' for x in lst) + ']'


def _fmt_list_m(lst: list) -> str:
    """Format a list of meter-based metrics with 1 decimal place."""
    return '[' + ', '.join(f'{float(x):.1f}' for x in lst) + ']'


def _append_metrics(
    acc: VisAcc, group: typing.Dict[str, dict], name: str
) -> None:
    """Append VisAcc results to a metric group."""
    group_new = group.copy()
    group_new['corr'][name].append(acc.get_r())
    group_new['mae'][name].append(acc.get_mae())
    group_new['rmse'][name].append(acc.get_rmse())
    group_new['mre'][name].append(acc.get_mre())
    ts = acc.get_ts_ge()
    group_new['ts1'][name].append(ts[0])
    group_new['ts2'][name].append(ts[1])
    group_new['ts3'][name].append(ts[2])
    group_new['ts4'][name].append(ts[3])
    group_new['ts5'][name].append(ts[4])
    group_new['ts6'][name].append(ts[5])

    return group_new


def build_fhour_index(n_days: int = 8760, n_hours: int = 24) -> np.ndarray:
    """
    Build forecast lead time index array.

    Args:
        n_days (int): Number of days in forecast period. Defaults to 8760.
        n_hours (int): Number of hours per day. Defaults to 24.

    Returns:
        np.ndarray: 2D index array of shape (n_days, n_hours).
    """
    fhour_ind = (
        np.arange(n_days)[:, None] + np.arange(n_hours)[None, :]
    ) % n_hours
    return fhour_ind


def _calc_temporal_block(
    i: int,
    vis_ob: np.ndarray,
    cma_sh_warr: np.ndarray,
    pred_pdfm_tle: np.ndarray,
    fhour_ind: np.ndarray
) -> typing.Dict[str, typing.Union[int, float]]:
    """
    Calc metrics for one init-hour block (parallel worker).

    Args:
        i (int): Block index (init hour).
        vis_ob (np.ndarray): Observed visibility array.
        cma_sh_warr (np.ndarray): CMA-SH-WARR forecast array.
        pred_pdfm_tle (np.ndarray): PDFM-TLE forecast array.
        fhour_ind (np.ndarray): Forecast lead time index array.

    Returns:
        typing.Dict[str, typing.Union[int, float]]: Dict with block index and
        verification metrics for visibility type, start hour, and forecast hour.
    """
    # Initialize metric groups:
    # vt=visibility type, shour=start hour, fhour=forecast hour
    vt = _create_group(('vt', 'CMA-SH-WARR', 'PDFM-TLE'))
    shour = _create_group(('shour', 'CMA-SH-WARR', 'PDFM-TLE'))
    fhour = _create_group(('fhour', 'CMA-SH-WARR', 'PDFM-TLE'))
    # ha: hour_access array (2 schemes, 24 hours, 10 metrics)
    ha = np.zeros((2, 24, 10), dtype=np.float32) + np.nan

    # Metrics by visibility type (all hours for this init type)
    for k in vt:
        vt[k]['vt'].append(i)
    acc_nwp = VisAcc(vis_ob[:, i, :], cma_sh_warr[:, i, :])
    vt = _append_metrics(acc_nwp, vt, 'CMA-SH-WARR')
    acc = VisAcc(vis_ob[:, i, :], pred_pdfm_tle[:, i, :])
    vt = _append_metrics(acc, vt, 'PDFM-TLE')

    # Metrics by start hour (same init time across days)
    for k in shour:
        shour[k]['shour'].append(i)
    acc_nwp = VisAcc(vis_ob[i::24, :, :], cma_sh_warr[i::24, :, :])
    shour = _append_metrics(acc_nwp, shour, 'CMA-SH-WARR')
    acc = VisAcc(vis_ob[i::24, :, :], pred_pdfm_tle[i::24, :, :])
    shour = _append_metrics(acc, shour, 'PDFM-TLE')

    # Metrics by forecast lead hour
    for k in fhour:
        fhour[k]['fhour'].append(i)
    acc_nwp = VisAcc(vis_ob[fhour_ind == i], cma_sh_warr[fhour_ind == i])
    fhour = _append_metrics(acc_nwp, fhour, 'CMA-SH-WARR')
    ob_slice = vis_ob[fhour_ind == i, :]
    pr_slice = pred_pdfm_tle[fhour_ind == i, :]
    acc = VisAcc(ob_slice, pr_slice)
    fhour = _append_metrics(acc, fhour, 'PDFM-TLE')

    # Hour-access matrix:
    # metrics for each start hour x forecast hour combination
    for j in range(24):
        acc_nwp = VisAcc(vis_ob[i::24, j, :], cma_sh_warr[i::24, j, :])
        ha[0, j, 0] = acc_nwp.get_r()
        ha[0, j, 1] = acc_nwp.get_mae()
        ha[0, j, 2] = acc_nwp.get_rmse()
        ha[0, j, 3] = acc_nwp.get_mre()
        ha[0, j, 4:] = acc_nwp.get_ts_ge()

        acc = VisAcc(vis_ob[i::24, j, :], pred_pdfm_tle[i::24, j, :])
        ha[1, j, 0] = acc.get_r()
        ha[1, j, 1] = acc.get_mae()
        ha[1, j, 2] = acc.get_rmse()
        ha[1, j, 3] = acc.get_mre()
        ha[1, j, 4:] = acc.get_ts_ge()

    return {'vt': vt, 'shour': shour, 'fhour': fhour, 'ha': ha}


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
        vis_ob (np.ndarray): Observed visibility array.
        cma_sh_warr (np.ndarray): CMA-SH-WARR forecast array.
        pred_pdfm_tle0 (np.ndarray): PDFM-TLE forecast array (scheme 1).
        pred_pdfm_tle2 (np.ndarray): PDFM-TLE forecast array (scheme 3).
        output_dir (str): Output directory path.

    Returns:
        np.ndarray: hour_access array of shape (2, 24, 24, 10).
    """
    fhour_ind = build_fhour_index()
    vt_group = _create_group(('vt', 'CMA-SH-WARR', 'PDFM-TLE'))
    shour_group = _create_group(('shour', 'CMA-SH-WARR', 'PDFM-TLE'))
    fhour_group = _create_group(('fhour', 'CMA-SH-WARR', 'PDFM-TLE'))
    # hour_access dims: (2 schemes, 24 init hours, 24 lead hours, 10 metrics)
    hour_access = np.zeros((2, 24, 24, 10), dtype=np.float32) + np.nan

    # Parallel calculation over 24 init-hour blocks
    blocks = joblib.Parallel(n_jobs=-1, backend='loky')(
        joblib.delayed(_calc_temporal_block)(
            i, vis_ob, cma_sh_warr, pred_pdfm_tle2, fhour_ind
        )
        for i in range(24)
    )

    # Merge parallel block results into final groups and hour_access array
    for i, blk in enumerate(blocks):
        for k in vt_group:
            vt_group[k]['vt'].extend(blk['vt'][k]['vt'])
            vt_group[k]['CMA-SH-WARR'].extend(blk['vt'][k]['CMA-SH-WARR'])
            vt_group[k]['PDFM-TLE'].extend(blk['vt'][k]['PDFM-TLE'])
        for k in shour_group:
            shour_group[k]['shour'].extend(blk['shour'][k]['shour'])
            shour_group[k]['CMA-SH-WARR'].extend(blk['shour'][k]['CMA-SH-WARR'])
            shour_group[k]['PDFM-TLE'].extend(blk['shour'][k]['PDFM-TLE'])
        for k in fhour_group:
            fhour_group[k]['fhour'].extend(blk['fhour'][k]['fhour'])
            fhour_group[k]['CMA-SH-WARR'].extend(blk['fhour'][k]['CMA-SH-WARR'])
            fhour_group[k]['PDFM-TLE'].extend(blk['fhour'][k]['PDFM-TLE'])
        hour_access[:, i, :, :] = blk['ha']

    csv_dir = pathlib.Path(output_dir) / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    df_vt_ts4 = pd.DataFrame(vt_group['ts4'])
    df_fhour_ts4 = pd.DataFrame(fhour_group['ts4'])
    df_vt_ts4.to_csv(str(csv_dir / 'vis_vt_ts4+.csv'), index=False)
    df_fhour_ts4.to_csv(str(csv_dir / 'vis_fhour_ts4+.csv'), index=False)
    np.save(str(csv_dir / 'hour_access.npy'), hour_access)
    return hour_access, df_vt_ts4, df_fhour_ts4


def load_v_type(data_dir: str, idx_mlyr: np.ndarray) -> np.ndarray:
    """
    Load visibility type data.

    Args:
        data_dir (str): Directory containing v_type.npy.
        idx_mlyr (np.ndarray): Index array for MLYR stations.

    Returns:
        np.ndarray: Reshaped visibility type array.
    """
    v_type = np.load(str(pathlib.Path(data_dir) / 'v_type.npy'))
    v_type = np.reshape(v_type[-365:, :, :, idx_mlyr], (-1, 24, 502))
    return v_type


def calc_type_metrics(
    vis_ob: np.ndarray,
    cma_sh_warr: np.ndarray,
    pred_pdfm_tle0: np.ndarray,
    v_type: np.ndarray
) -> typing.Tuple[typing.Dict[str, pd.DataFrame], str]:
    """
    Calc metrics by visibility type and return formatted output string.

    Args:
        vis_ob (np.ndarray): Observed visibility array.
        cma_sh_warr (np.ndarray): CMA-SH-WARR forecast array.
        pred_pdfm_tle0 (np.ndarray): PDFM-TLE forecast array (scheme 1).
        v_type (np.ndarray): Visibility type mask array.

    Returns:
        typing.Tuple[typing.Dict[str, pd.DataFrame], str]: DataFrame dict and
        formatted output string.
    """
    # Calculate metrics by visibility type (3 categories)
    dfs = _create_group(('type', 'CMA-SH-WARR', 'PDFM-TLE'))
    for i in range(3):
        for k in dfs:
            dfs[k]['type'].append(i)
        acc_nwp = VisAcc(vis_ob[v_type == i], cma_sh_warr[v_type == i])
        dfs = _append_metrics(acc_nwp, dfs, 'CMA-SH-WARR')
        acc = VisAcc(vis_ob[v_type == i], pred_pdfm_tle0[v_type == i])
        dfs = _append_metrics(acc, dfs, 'PDFM-TLE')

    # Build formatted output string for console display
    lines = list()
    lines.append('[Type Metrics] CMA-SH-WARR')
    c_cma = _fmt_list(dfs['corr']['CMA-SH-WARR'])
    a_cma = _fmt_list_m(dfs['mae']['CMA-SH-WARR'])
    r_cma = _fmt_list_m(dfs['rmse']['CMA-SH-WARR'])
    m_cma = _fmt_list(dfs['mre']['CMA-SH-WARR'])
    lines.append(f'  corr={c_cma}, mae={a_cma} m, rmse={r_cma} m, mre={m_cma}')
    t1 = _fmt_list(dfs['ts1']['CMA-SH-WARR'])
    t2 = _fmt_list(dfs['ts2']['CMA-SH-WARR'])
    t3 = _fmt_list(dfs['ts3']['CMA-SH-WARR'])
    t4 = _fmt_list(dfs['ts4']['CMA-SH-WARR'])
    t5 = _fmt_list(dfs['ts5']['CMA-SH-WARR'])
    t6 = _fmt_list(dfs['ts6']['CMA-SH-WARR'])
    ts_str = f'ts1={t1}, ts2={t2}, ts3={t3}, ts4={t4}, ts5={t5}, ts6={t6}'
    lines.append(f'  TS  = {ts_str}')
    lines.append('[Type Metrics] PDFM-TLE')
    c_pdfm = _fmt_list(dfs['corr']['PDFM-TLE'])
    a_pdfm = _fmt_list_m(dfs['mae']['PDFM-TLE'])
    r_pdfm = _fmt_list_m(dfs['rmse']['PDFM-TLE'])
    m_pdfm = _fmt_list(dfs['mre']['PDFM-TLE'])
    lines.append(f'  corr={c_pdfm}, mae={a_pdfm}, rmse={r_pdfm}, mre={m_pdfm}')
    t1p = _fmt_list(dfs['ts1']['PDFM-TLE'])
    t2p = _fmt_list(dfs['ts2']['PDFM-TLE'])
    t3p = _fmt_list(dfs['ts3']['PDFM-TLE'])
    t4p = _fmt_list(dfs['ts4']['PDFM-TLE'])
    t5p = _fmt_list(dfs['ts5']['PDFM-TLE'])
    t6p = _fmt_list(dfs['ts6']['PDFM-TLE'])
    ts_str = f'ts1={t1p}, ts2={t2p}, ts3={t3p}, ts4={t4p}, ts5={t5p}, ts6={t6p}'
    lines.append(f'  TS  = {ts_str}')

    return {k: pd.DataFrame(dfs[k]) for k in dfs}, '\n'.join(lines)


def save_type_results(
    dfs: typing.Dict[str, pd.DataFrame], output_dir: str
) -> None:
    """
    Save visibility type metrics to CSV.

    Args:
        dfs (typing.Dict[str, pd.DataFrame]): Dict of DataFrames by metric.
        output_dir (str): Output directory path.
    """
    csv_dir = pathlib.Path(output_dir) / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    dfs['corr'].to_csv(str(csv_dir / 'vis_type_corr.csv'), index=False)
    dfs['mae'].to_csv(str(csv_dir / 'vis_type_mae.csv'), index=False)
    dfs['rmse'].to_csv(str(csv_dir / 'vis_type_rmse.csv'), index=False)
    dfs['mre'].to_csv(str(csv_dir / 'vis_type_mre.csv'), index=False)
    for i in range(1, 7):
        dfs[f'ts{i}'].to_csv(str(csv_dir / f'vis_type_ts{i}+.csv'), index=False)


def load_temporal_metrics(output_dir: str) -> np.ndarray:
    """
    Load pre-calculated temporal metrics (hour_access) from NPY file.

    Args:
        output_dir (str): Directory containing the csv/ subfolder.

    Returns:
        np.ndarray: hour_access array.

    Raises:
        FileNotFoundError: If hour_access.npy is missing.
    """
    csv_dir = pathlib.Path(output_dir) / 'csv'
    file_path = csv_dir / 'hour_access.npy'
    if not file_path.exists():
        raise FileNotFoundError(f'Temporal metrics not found: {file_path}')
    return np.load(str(file_path))


def load_type_results(output_dir: str) -> typing.Dict[str, pd.DataFrame]:
    """
    Load pre-calculated visibility type metrics from CSV files.

    Args:
        output_dir (str): Directory containing the csv/ subfolder.

    Returns:
        typing.Dict[str, pd.DataFrame]: Dict of DataFrames by metric.

    Raises:
        FileNotFoundError: If any expected CSV file is missing.
    """
    csv_dir = pathlib.Path(output_dir) / 'csv'
    metrics = {}
    metric_names = ['corr', 'mae', 'rmse', 'mre']
    for key in metric_names:
        file_path = csv_dir / f'vis_type_{key}.csv'
        if not file_path.exists():
            raise FileNotFoundError(f'Type metrics file not found: {file_path}')
        metrics[key] = pd.read_csv(str(file_path), low_memory=False)
    for i in range(1, 7):
        file_path = csv_dir / f'vis_type_ts{i}+.csv'
        if not file_path.exists():
            raise FileNotFoundError(f'Type metrics file not found: {file_path}')
        metrics[f'ts{i}'] = pd.read_csv(str(file_path), low_memory=False)
    return metrics
