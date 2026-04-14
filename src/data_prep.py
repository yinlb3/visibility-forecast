# -*- coding: utf-8 -*-
"""
Observation data preparation and preprocessing module.

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""

import pathlib

import arrow
import numpy as np
import pandas as pd

from src import vis_acc

THRES = vis_acc.THRES


def read_sta(sta_path: str, provinces: tuple) -> tuple:
    """
    Read station info and filter by province.

    Args:
        sta_path (str): Station CSV file path.
        provinces (tuple): Provinces to include.

    Returns:
        tuple: (Filtered station DataFrame, initial filter bool index).
    """
    # 1. Load station data and sort by ID
    sta = pd.read_csv(filepath_or_buffer=sta_path, low_memory=False, encoding='utf-8')
    sta = sta.sort_values(by=['id'])
    sta.reset_index(drop=True, inplace=True)

    # 2. Build province filter mask (OR logic for multiple provinces)
    # idx_east_china: Index for East China stations (中国东部站点索引)
    idx_east_china = None
    for province in provinces:
        if idx_east_china is None:
            idx_east_china = sta.loc[:, 'province'] == province
        else:
            idx_east_china |= sta.loc[:, 'province'] == province

    # 3. Apply filter and reset index
    sta = sta.loc[idx_east_china]
    sta.reset_index(drop=True, inplace=True)
    return sta, idx_east_china.values


def load_obs(data_dir: str, idx_east_china: np.ndarray) -> tuple:
    """
    Load vis/precip/RH obs data and apply QC.

    Args:
        data_dir (str): Data directory path, no trailing slash.
        idx_east_china (np.ndarray): Index for East China stations.

    Returns:
        tuple: (vis, pre, rhu), all QC-ed numpy arrays.
    """
    # 1. Load visibility data and apply QC (capped at 30000m)
    vis = np.load(str(pathlib.Path(data_dir) / 'vis20-23.npy'))[:, idx_east_china]
    vis[vis >= 999990] = np.nan  # Missing value marker
    vis[vis >= 30000] = 30000    # Cap at 30000m

    # 2. Load precipitation data and apply QC
    pre = np.load(str(pathlib.Path(data_dir) / 'pre20-23.npy'))[:, idx_east_china]
    pre[pre >= 200] = np.nan     # Extreme precip as missing
    pre[pre >= 30000] = 30000

    # 3. Load relative humidity and apply range limit [0, 100]
    rhu = np.load(str(pathlib.Path(data_dir) / 'rhu20-23.npy'))[:, idx_east_china]
    rhu[rhu >= 999990] = np.nan
    rhu[rhu > 100] = 100
    rhu[rhu < 0] = 0

    return vis, pre, rhu


def filter_region(
    sta: pd.DataFrame,
    vis: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    region_provinces: tuple
) -> tuple:
    """
    Secondary filter stations by target provinces.

    Args:
        sta (pd.DataFrame): Station info table.
        vis (np.ndarray): Visibility array.
        pre (np.ndarray): Precipitation array.
        rhu (np.ndarray): Relative humidity array.
        region_provinces (tuple): Target province list.

    Returns:
        tuple: (Filtered sta, vis, pre, rhu, secondary filter bool index).
    """
    n_sta = len(sta)
    # idx_mlyr: Index for MLYR (Middle-Lower Yangtze River, 长江中下游区域索引)
    idx_mlyr = np.zeros(n_sta, dtype=np.bool_)
    for i in range(n_sta):
        if sta.loc[i, 'province'] in region_provinces:
            idx_mlyr[i] = True

    sta = sta.loc[idx_mlyr]
    sta.reset_index(drop=True, inplace=True)
    sta = sta.astype({'data0': float})

    vis = vis[:, idx_mlyr]
    pre = pre[:, idx_mlyr]
    rhu = rhu[:, idx_mlyr]

    return sta, vis, pre, rhu, idx_mlyr


def grade_visibility(vis: np.ndarray) -> np.ndarray:
    """
    Grade visibility into six levels.

    Args:
        vis (np.ndarray): Obs visibility array.

    Returns:
        np.ndarray: Graded int array, -1 for missing.
    """
    vis_grade = np.zeros_like(vis, dtype=np.int_) - 1
    vis_grade[~np.isnan(vis)] = 0
    for i, t in enumerate(THRES):
        vis_grade[vis < t] = i + 1
    return vis_grade


def build_month_index(start_year: int, n_hours: int) -> np.ndarray:
    """
    Build hourly-to-month index array.

    Args:
        start_year (int): Start year.
        n_hours (int): Total hours.

    Returns:
        np.ndarray: Month index array of length n_hours.
    """
    month_ind = np.zeros(shape=n_hours, dtype=np.int_)
    base = arrow.get(str(start_year))
    for i in range(n_hours):
        month_ind[i] = base.shift(hours=i).datetime.month
    return month_ind
