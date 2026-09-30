#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 2.1: Distribution feature calculation module.

Founded in 2026-04-14
Modified in 2026-09-30
@author: yinlb, space-bunny
"""

import pathlib
import typing

import numpy as np
import pandas as pd


def build_month_stats(
    vis_grade: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    month_ind: np.ndarray,
    thres: typing.Tuple[float, ...]
) -> dict:
    """Calc monthly freq of low visibility by grade."""
    # Create weather-type masks:
    # mask_pre = precip events (pre > 0)
    # mask_fog = fog (pre==0 & rhu>=80)
    # mask_haze = haze (pre==0 & rhu<80)
    mask_pre = np.copy(vis_grade)
    mask_pre[pre == 0] = -1
    mask_fog = np.copy(vis_grade)
    mask_fog[(pre > 0) | (rhu < 80)] = -1
    mask_haze = np.copy(vis_grade)
    mask_haze[(pre < 0) | (rhu >= 80)] = -1

    df_month = {'month': list()}
    for i in range(len(thres)):
        df_month[str(i + 1)] = list()
        df_month[str(i + 1) + 'pre'] = list()
        df_month[str(i + 1) + 'fog'] = list()
        df_month[str(i + 1) + 'haze'] = list()

    # Calculate frequency per month and grade for overall and each weather type
    for j in range(12):
        df_month['month'].append(j + 1)
        mask = month_ind == j + 1
        valid = np.sum(vis_grade[mask, :] >= 0)
        grade_month = vis_grade[mask, :]
        mask_pre_month = mask_pre[mask, :]
        mask_fog_month = mask_fog[mask, :]
        mask_haze_month = mask_haze[mask, :]
        for i in range(len(thres)):
            grade = i + 1
            df_month[str(grade)].append(
                np.sum(grade_month >= grade) / valid
            )
            df_month[str(grade) + 'pre'].append(
                np.sum(mask_pre_month >= grade) / valid
            )
            df_month[str(grade) + 'fog'].append(
                np.sum(mask_fog_month >= grade) / valid
            )
            df_month[str(grade) + 'haze'].append(
                np.sum(mask_haze_month >= grade) / valid
            )
    return df_month


def build_hour_stats(
    vis_grade: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    thres: typing.Tuple[float, ...]
) -> dict:
    """Calc hourly frequency of low visibility by grade."""
    # Same weather-type masks as build_month_stats
    mask_pre = np.copy(vis_grade)
    mask_pre[pre == 0] = -1
    mask_fog = np.copy(vis_grade)
    mask_fog[(pre > 0) | (rhu < 80)] = -1
    mask_haze = np.copy(vis_grade)
    mask_haze[(pre < 0) | (rhu >= 80)] = -1

    df_hour = {'hour': list()}
    for i in range(len(thres)):
        df_hour[str(i + 1)] = list()
        df_hour[str(i + 1) + 'pre'] = list()
        df_hour[str(i + 1) + 'fog'] = list()
        df_hour[str(i + 1) + 'haze'] = list()

    # Calculate frequency per hour (using stride 24) and grade
    for j in range(24):
        df_hour['hour'].append(j)
        valid = np.sum(vis_grade[j::24, :] >= 0)
        vg_h = vis_grade[j::24, :]
        a_h = mask_pre[j::24, :]
        b_h = mask_fog[j::24, :]
        c_h = mask_haze[j::24, :]
        for i in range(len(thres)):
            grade = i + 1
            df_hour[str(grade)].append(np.sum(vg_h >= grade) / valid)
            df_hour[str(grade) + 'pre'].append(np.sum(a_h >= grade) / valid)
            df_hour[str(grade) + 'fog'].append(np.sum(b_h >= grade) / valid)
            df_hour[str(grade) + 'haze'].append(np.sum(c_h >= grade) / valid)
    return df_hour


def build_sta_stats(
    sta: pd.DataFrame,
    vis: np.ndarray,
    vis_grade: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    thres: typing.Tuple[float, ...]
) -> dict:
    """Calc station-wise freq and mean visibility."""
    n_sta = len(sta)
    # Weather-type masks: precip, fog, haze
    mask_pre = np.copy(vis_grade)
    mask_pre[pre == 0] = -1
    mask_fog = np.copy(vis_grade)
    mask_fog[(pre > 0) | (rhu < 80)] = -1
    mask_haze = np.copy(vis_grade)
    mask_haze[(pre < 0) | (rhu >= 80)] = -1

    valid_all = np.sum(vis_grade >= 0, axis=0)

    df_sta = {
        'sta_id': sta.loc[:, 'id'],
        'lon': sta.loc[:, 'lon'],
        'lat': sta.loc[:, 'lat'],
        'alti': sta.loc[:, 'alti'],
        'n': valid_all,
        'mean': np.nanmean(vis, axis=0),
        'lvpe': list(),
        'lvfe': list(),
        'lvhe': list()
    }

    # Compute mean visibility for low-vis events by weather type per station
    # lvpe = low visibility precip event mean, lvfe = fog, lvhe = haze
    for j in range(n_sta):
        df_sta['lvpe'].append(np.nanmean(vis[:, j][pre[:, j] > 0]))
        index_e = (pre[:, j] == 0) & (rhu[:, j] >= 80)
        df_sta['lvfe'].append(np.nanmean(vis[:, j][index_e]))
        index_e = (pre[:, j] == 0) & (rhu[:, j] < 80)
        df_sta['lvhe'].append(np.nanmean(vis[:, j][index_e]))

    for i in range(len(thres)):
        grade = str(i + 1)
        df_sta[grade] = np.sum(vis_grade >= i + 1, axis=0) / valid_all
        df_sta[grade + 'pre'] = np.sum(mask_pre >= i + 1, axis=0) / valid_all
        df_sta[grade + 'fog'] = np.sum(mask_fog >= i + 1, axis=0) / valid_all
        df_sta[grade + 'haze'] = np.sum(mask_haze >= i + 1, axis=0) / valid_all

    return df_sta


def save_obs_stats(
    df_month: dict,
    df_hour: dict,
    df_sta: dict,
    output_dir: str
) -> None:
    """Output month/hour/sta stats to CSV."""
    csv_dir = pathlib.Path(output_dir) / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(df_month).to_csv(
        path_or_buf=str(csv_dir / 'vis_month.csv'), index=False
    )
    pd.DataFrame(df_hour).to_csv(
        path_or_buf=str(csv_dir / 'vis_hour.csv'), index=False
    )
    pd.DataFrame(df_sta).to_csv(str(csv_dir / 'vis_sta.csv'), index=False)
