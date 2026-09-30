#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 1: Configuration and data loading module.

Founded in 2026-04-14
Modified in 2026-09-30
@author: yinlb, space-bunny
"""

import os
import pathlib
import typing

import numpy as np
import pandas as pd

from src import vis_acc

THRES = vis_acc.THRES


def _save_stage1_cache(
    sta: pd.DataFrame,
    vis: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    vis_grade: np.ndarray,
    month_ind: np.ndarray,
    idx_mlyr: np.ndarray,
    cache_dir: str,
) -> None:
    """Save stage 1 intermediate results for skipping."""
    os.makedirs(cache_dir, exist_ok=True)
    sta.to_csv(str(pathlib.Path(cache_dir) / 'sta.csv'), index=False)
    np.save(str(pathlib.Path(cache_dir) / 'vis.npy'), vis)
    np.save(str(pathlib.Path(cache_dir) / 'pre.npy'), pre)
    np.save(str(pathlib.Path(cache_dir) / 'rhu.npy'), rhu)
    np.save(str(pathlib.Path(cache_dir) / 'vis_grade.npy'), vis_grade)
    np.save(str(pathlib.Path(cache_dir) / 'month_ind.npy'), month_ind)
    np.save(str(pathlib.Path(cache_dir) / 'index_mlyr.npy'), idx_mlyr)


def _load_stage1_cache(
    cache_dir: str,
) -> typing.Tuple[
    pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """Load stage 1 intermediate results."""
    sta = pd.read_csv(str(pathlib.Path(cache_dir) / 'sta.csv'))
    vis = np.load(str(pathlib.Path(cache_dir) / 'vis.npy'))
    pre = np.load(str(pathlib.Path(cache_dir) / 'pre.npy'))
    rhu = np.load(str(pathlib.Path(cache_dir) / 'rhu.npy'))
    vis_grade = np.load(str(pathlib.Path(cache_dir) / 'vis_grade.npy'))
    month_ind = np.load(str(pathlib.Path(cache_dir) / 'month_ind.npy'))
    idx_mlyr = np.load(str(pathlib.Path(cache_dir) / 'index_mlyr.npy'))
    return sta, vis, pre, rhu, vis_grade, month_ind, idx_mlyr


def read_station(sta_path: str, provinces: tuple) -> tuple:
    """Read station info and filter by province."""
    sta = pd.read_csv(sta_path, low_memory=False, encoding='utf-8')
    sta = sta.sort_values(by=['id'])
    sta.reset_index(drop=True, inplace=True)
    # Build OR-chain boolean mask for target provinces
    idx_east_china = None
    for province in provinces:
        if idx_east_china is None:
            idx_east_china = sta.loc[:, 'province'] == province
        else:
            idx_east_china |= sta.loc[:, 'province'] == province
    sta = sta.loc[idx_east_china]
    sta.reset_index(drop=True, inplace=True)
    return sta, idx_east_china.values


def load_obs(data_dir: str, idx_east_china: np.ndarray) -> tuple:
    """Load vis/precip/RH obs data and apply QC."""
    # Visibility: replace missing values, cap at 30000m
    vis_path = str(pathlib.Path(data_dir) / 'vis20-23.npy')
    vis = np.load(vis_path)[:, idx_east_china]
    vis[vis >= 999990] = np.nan
    vis[vis >= 30000] = 30000
    # Precipitation: filter extreme values
    pre_path = str(pathlib.Path(data_dir) / 'pre20-23.npy')
    pre = np.load(pre_path)[:, idx_east_china]
    pre[pre >= 200] = np.nan
    pre[pre >= 30000] = 30000
    # Relative humidity: clamp to [0, 100]
    rhu_path = str(pathlib.Path(data_dir) / 'rhu20-23.npy')
    rhu = np.load(rhu_path)[:, idx_east_china]
    rhu[rhu >= 999990] = np.nan
    rhu[rhu > 100] = 100
    rhu[rhu < 0] = 0
    return vis, pre, rhu


def filter_region(
    sta: pd.DataFrame, vis: np.ndarray, pre: np.ndarray, rhu: np.ndarray,
    region_provinces: tuple
) -> tuple:
    """Secondary filter stations by target provinces."""
    n_sta = len(sta)
    # Mark stations belonging to target region provinces
    idx_mlyr = np.zeros(n_sta, dtype=np.bool_)
    for i in range(n_sta):
        if sta.loc[i, 'province'] in region_provinces:
            idx_mlyr[i] = True
    sta = sta.loc[idx_mlyr].copy()
    sta = sta.reset_index(drop=True)
    sta = sta.astype({'data0': float})
    # Subset obs arrays to region stations
    vis = vis[:, idx_mlyr]
    pre = pre[:, idx_mlyr]
    rhu = rhu[:, idx_mlyr]
    return sta, vis, pre, rhu, idx_mlyr


def grade_visibility(vis: np.ndarray) -> np.ndarray:
    """Grade visibility into six levels."""
    vis_grade = np.zeros_like(vis, dtype=np.int_) - 1
    vis_grade[~np.isnan(vis)] = 0
    # Assign higher grade for lower visibility (worse conditions)
    for i, thr in enumerate(THRES):
        vis_grade[vis < thr] = i + 1
    return vis_grade


def build_month_index(start_year: int, n_hours: int) -> np.ndarray:
    """Build hourly-to-month index array."""
    dates = pd.date_range(start=str(start_year), periods=n_hours, freq='h')
    return dates.month.to_numpy()


def load_forecast_data(
    data_dir: str, idx_mlyr: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Load last 365 days obs and CMA-SH-WARR forecast data."""
    # Load obs: slice last 365 days, reshape to (days, 24h, stations)
    path = str(pathlib.Path(data_dir) / 'vis1183_ob.npy')
    vis_ob = np.load(path)[-365:, :, 1:, idx_mlyr]
    vis_ob = np.reshape(vis_ob, (-1, 24, 502))
    vis_ob[vis_ob >= 999990] = np.nan
    vis_ob[vis_ob >= 30000] = 30000

    # Load CMA-SH-WARR forecast with same reshaping
    pr_path = str(pathlib.Path(data_dir) / 'vis1183_pr.npy')
    cma_sh_warr = np.load(pr_path)[-365:, :, 1:, idx_mlyr]
    cma_sh_warr = np.reshape(cma_sh_warr, (-1, 24, 502))
    cma_sh_warr[cma_sh_warr >= 30000] = 30000
    return vis_ob, cma_sh_warr


def load_experiment_preds(data_dir: str) -> typing.Tuple[np.ndarray, ...]:
    """Load 5 PDFM-TLE experiment forecast datasets."""
    preds = list()
    # Iterate over 5 experiment configurations (TLE0 to TLE4)
    for i in range(5):
        path = str(pathlib.Path(data_dir) / f'vis_gjz_pdfm_tle{i}_mlyr.npy')
        pred = np.load(path)
        pred = np.reshape(pred, (-1, 24, 502))
        pred[pred >= 30000] = 30000
        preds.append(pred)
    return tuple(preds)
