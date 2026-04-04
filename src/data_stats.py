# -*- coding: utf-8 -*-
"""
观测数据统计与输出模块.

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
    按月统计各级低能见度出现频率.

    Args:
        vis_grade (np.ndarray): 能见度分级数组.
        pre (np.ndarray): 降水数组.
        rhu (np.ndarray): 相对湿度数组.
        month_ind (np.ndarray): 逐小时月份索引.
        thres (tuple): 能见度分级阈值.

    Returns:
        dict: 月统计结果字典.
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
    按站点统计各级低能见度出现频率及平均能见度.

    Args:
        sta (pd.DataFrame): 站点信息表.
        vis (np.ndarray): 能见度原始数组.
        vis_grade (np.ndarray): 能见度分级数组.
        pre (np.ndarray): 降水数组.
        rhu (np.ndarray): 相对湿度数组.
        thres (tuple): 能见度分级阈值.

    Returns:
        dict: 站点统计结果字典.
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
    将月、时、站统计结果输出为 CSV.

    Args:
        df_month (dict): 月统计字典.
        df_hour (dict): 小时统计字典.
        df_sta (dict): 站点统计字典.
        output_dir (str): 输出目录路径.
    """
    pd.DataFrame(df_month).to_csv(path_or_buf=rf'{output_dir}\vis_month.csv', index=False)
    pd.DataFrame(df_hour).to_csv(path_or_buf=rf'{output_dir}\vis_hour.csv', index=False)
    pd.DataFrame(df_sta).to_csv(rf'{output_dir}\vis_sta.csv', index=False)
