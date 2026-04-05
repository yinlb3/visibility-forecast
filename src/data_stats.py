# -*- coding: utf-8 -*-
"""
Observation data statistics and output module.

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""

from typing import Tuple

import numpy as np
import pandas as pd


def build_month_stats(
    vis_grade: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    month_ind: np.ndarray,
    thres: Tuple[float, ...]
) -> dict:
    """
    Calc monthly freq of low visibility by grade.

    Args:
        vis_grade (np.ndarray): Visibility grade array.
        pre (np.ndarray): Precipitation array.
        rhu (np.ndarray): Relative humidity array.
        month_ind (np.ndarray): Hourly month index.
        thres (tuple): Visibility grade thresholds.

    Returns:
        dict: Monthly stats dict.
    """
    df_month = {'month': list()}
    for i in range(len(thres)):
        a = np.copy(vis_grade)
        a[pre == 0] = -1
        b = np.copy(vis_grade)
        b[(pre > 0) | (rhu < 80)] = -1
        c = np.copy(vis_grade)
        c[(pre < 0) | (rhu >= 80)] = -1
        for j in range(12):
            if i == 0:
                df_month['month'].append(j + 1)
            if j == 0:
                df_month[str(i + 1)] = list()
                df_month[str(i + 1) + 'pre'] = list()
                df_month[str(i + 1) + 'fog'] = list()
                df_month[str(i + 1) + 'haze'] = list()
            valid = np.sum(vis_grade[month_ind == j + 1, :] >= 0)
            df_month[str(i + 1)].append(np.sum(vis_grade[month_ind == j + 1, :] >= i + 1) / valid)
            df_month[str(i + 1) + 'pre'].append(np.sum(a[month_ind == j + 1, :] >= i + 1) / valid)
            df_month[str(i + 1) + 'fog'].append(np.sum(b[month_ind == j + 1, :] >= i + 1) / valid)
            df_month[str(i + 1) + 'haze'].append(np.sum(c[month_ind == j + 1, :] >= i + 1) / valid)
    return df_month


def build_hour_stats(
    vis_grade: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    thres: Tuple[float, ...]
) -> dict:
    """
    按小时统计各级低能见度出现频率.

    Args:
        vis_grade (np.ndarray): 能见度分级数组.
        pre (np.ndarray): 降水数组.
        rhu (np.ndarray): 相对湿度数组.
        thres (tuple): 能见度分级阈值.

    Returns:
        dict: 小时统计结果字典.
    """
    df_hour = {'hour': list()}
    for i in range(len(thres)):
        a = np.copy(vis_grade)
        a[pre == 0] = -1
        b = np.copy(vis_grade)
        b[(pre > 0) | (rhu < 80)] = -1
        c = np.copy(vis_grade)
        c[(pre < 0) | (rhu >= 80)] = -1
        for j in range(24):
            if i == 0:
                df_hour['hour'].append(j)
            if j == 0:
                df_hour[str(i + 1)] = list()
                df_hour[str(i + 1) + 'pre'] = list()
                df_hour[str(i + 1) + 'fog'] = list()
                df_hour[str(i + 1) + 'haze'] = list()
            valid = np.sum(vis_grade[j::24, :] >= 0)
            df_hour[str(i + 1)].append(np.sum(vis_grade[j::24, :] >= i + 1) / valid)
            df_hour[str(i + 1) + 'pre'].append(np.sum(a[j::24, :] >= i + 1) / valid)
            df_hour[str(i + 1) + 'fog'].append(np.sum(b[j::24, :] >= i + 1) / valid)
            df_hour[str(i + 1) + 'haze'].append(np.sum(c[j::24, :] >= i + 1) / valid)
    return df_hour


def build_sta_stats(
    sta: pd.DataFrame,
    vis: np.ndarray,
    vis_grade: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    thres: Tuple[float, ...]
) -> dict:
    """
    Calc station-wise freq and mean visibility.

    Args:
        sta (pd.DataFrame): Station info table.
        vis (np.ndarray): Raw visibility array.
        vis_grade (np.ndarray): Visibility grade array.
        pre (np.ndarray): Precipitation array.
        rhu (np.ndarray): Relative humidity array.
        thres (tuple): Visibility grade thresholds.

    Returns:
        dict: Station stats dict.
    """
    df_sta = {
        'sta_id': sta.loc[:, 'id'],
        'lon': sta.loc[:, 'lon'],
        'lat': sta.loc[:, 'lat'],
        'alti': sta.loc[:, 'alti'],
        'n': list(),
        'mean': list(),
        'lvpe': list(),
        'lvfe': list(),
        'lvhe': list()
    }
    for i in range(len(thres)):
        a = np.copy(vis_grade)
        a[pre == 0] = -1
        b = np.copy(vis_grade)
        b[(pre > 0) | (rhu < 80)] = -1
        c = np.copy(vis_grade)
        c[(pre < 0) | (rhu >= 80)] = -1
        for j in range(len(sta)):
            if j == 0:
                df_sta[str(i + 1)] = list()
                df_sta[str(i + 1) + 'pre'] = list()
                df_sta[str(i + 1) + 'fog'] = list()
                df_sta[str(i + 1) + 'haze'] = list()
            valid = np.sum(vis_grade[:, j] >= 0)
            df_sta[str(i + 1)].append(np.sum(vis_grade[:, j] >= i + 1) / valid)
            df_sta[str(i + 1) + 'pre'].append(np.sum(a[:, j] >= i + 1) / valid)
            df_sta[str(i + 1) + 'fog'].append(np.sum(b[:, j] >= i + 1) / valid)
            df_sta[str(i + 1) + 'haze'].append(np.sum(c[:, j] >= i + 1) / valid)
            if i == 0:
                df_sta['n'].append(valid)
                df_sta['mean'].append(np.nanmean(vis[:, j]))
                index_e = pre[:, j] > 0
                df_sta['lvpe'].append(np.nanmean(vis[:, j][index_e]))
                index_e = (pre[:, j] == 0) & (rhu[:, j] >= 80)
                df_sta['lvfe'].append(np.nanmean(vis[:, j][index_e]))
                index_e = (pre[:, j] == 0) & (rhu[:, j] < 80)
                df_sta['lvhe'].append(np.nanmean(vis[:, j][index_e]))
    return df_sta


def save_obs_stats(
    df_month: dict,
    df_hour: dict,
    df_sta: dict,
    output_dir: str
) -> None:
    """
    Output month/hour/sta stats to CSV.

    Args:
        df_month (dict): Monthly stats dict.
        df_hour (dict): Hourly stats dict.
        df_sta (dict): Station stats dict.
        output_dir (str): Output directory path.
    """
    import os
    csv_dir = rf'{output_dir}\csv'
    os.makedirs(csv_dir, exist_ok=True)
    pd.DataFrame(df_month).to_csv(path_or_buf=rf'{csv_dir}\vis_month.csv', index=False)
    pd.DataFrame(df_hour).to_csv(path_or_buf=rf'{csv_dir}\vis_hour.csv', index=False)
    pd.DataFrame(df_sta).to_csv(rf'{csv_dir}\vis_sta.csv', index=False)
