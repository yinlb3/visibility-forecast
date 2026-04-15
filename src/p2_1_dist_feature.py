#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 2.1: Distribution feature calculation module.

Founded in 2026-04-14
Modified in 2026-04-15
@author: yinlb
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
    a = np.copy(vis_grade)
    a[pre == 0] = -1
    b = np.copy(vis_grade)
    b[(pre > 0) | (rhu < 80)] = -1
    c = np.copy(vis_grade)
    c[(pre < 0) | (rhu >= 80)] = -1

    df_month = {'month': list()}
    for i in range(len(thres)):
        df_month[str(i + 1)] = list()
        df_month[str(i + 1) + 'pre'] = list()
        df_month[str(i + 1) + 'fog'] = list()
        df_month[str(i + 1) + 'haze'] = list()

    for j in range(12):
        df_month['month'].append(j + 1)
        mask = month_ind == j + 1
        valid = np.sum(vis_grade[mask, :] >= 0)
        vg = vis_grade[mask, :]
        a_m = a[mask, :]
        b_m = b[mask, :]
        c_m = c[mask, :]
        for i in range(len(thres)):
            grade = i + 1
            df_month[str(grade)].append(np.sum(vg >= grade) / valid)
            df_month[str(grade) + 'pre'].append(np.sum(a_m >= grade) / valid)
            df_month[str(grade) + 'fog'].append(np.sum(b_m >= grade) / valid)
            df_month[str(grade) + 'haze'].append(np.sum(c_m >= grade) / valid)
    return df_month


def build_hour_stats(
    vis_grade: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    thres: typing.Tuple[float, ...]
) -> dict:
    """Calc hourly frequency of low visibility by grade."""
    a = np.copy(vis_grade)
    a[pre == 0] = -1
    b = np.copy(vis_grade)
    b[(pre > 0) | (rhu < 80)] = -1
    c = np.copy(vis_grade)
    c[(pre < 0) | (rhu >= 80)] = -1

    df_hour = {'hour': list()}
    for i in range(len(thres)):
        df_hour[str(i + 1)] = list()
        df_hour[str(i + 1) + 'pre'] = list()
        df_hour[str(i + 1) + 'fog'] = list()
        df_hour[str(i + 1) + 'haze'] = list()

    for j in range(24):
        df_hour['hour'].append(j)
        valid = np.sum(vis_grade[j::24, :] >= 0)
        vg_h = vis_grade[j::24, :]
        a_h = a[j::24, :]
        b_h = b[j::24, :]
        c_h = c[j::24, :]
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
    a = np.copy(vis_grade)
    a[pre == 0] = -1
    b = np.copy(vis_grade)
    b[(pre > 0) | (rhu < 80)] = -1
    c = np.copy(vis_grade)
    c[(pre < 0) | (rhu >= 80)] = -1

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

    for j in range(n_sta):
        df_sta['lvpe'].append(np.nanmean(vis[:, j][pre[:, j] > 0]))
        index_e = (pre[:, j] == 0) & (rhu[:, j] >= 80)
        df_sta['lvfe'].append(np.nanmean(vis[:, j][index_e]))
        index_e = (pre[:, j] == 0) & (rhu[:, j] < 80)
        df_sta['lvhe'].append(np.nanmean(vis[:, j][index_e]))

    for i in range(len(thres)):
        grade = str(i + 1)
        df_sta[grade] = np.sum(vis_grade >= i + 1, axis=0) / valid_all
        df_sta[grade + 'pre'] = np.sum(a >= i + 1, axis=0) / valid_all
        df_sta[grade + 'fog'] = np.sum(b >= i + 1, axis=0) / valid_all
        df_sta[grade + 'haze'] = np.sum(c >= i + 1, axis=0) / valid_all

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
