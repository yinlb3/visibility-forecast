#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Founded in 2023-11-19
Modified in 2026-07-21
@author: yinlb
"""
import pathlib
import typing

import arrow
import numpy as np
import pandas as pd
from meteva import base as meb

from src import utils


# Load vis_grade configuration at import time
_VIS_GRADE_CFG = utils.CFG['vis_grade']
_VIS_GRADE_INPUT = _VIS_GRADE_CFG['input_files']
_VIS_GRADE_OUTPUT = _VIS_GRADE_CFG['output_files']
_VIS_GRADE_PARAMS = _VIS_GRADE_CFG['params']

_START_YEAR = int(_VIS_GRADE_PARAMS['start_year'])
_END_YEAR = int(_VIS_GRADE_PARAMS['end_year'])
_N_RAW_STATIONS = int(_VIS_GRADE_PARAMS['n_raw_stations'])
_GRADE_THRESHOLDS = list(_VIS_GRADE_PARAMS['grade_thresholds'])
_MISSING_RATE_THRESHOLD = float(_VIS_GRADE_PARAMS['missing_rate_threshold'])
_EXCLUDED_STATION_INDEX = int(_VIS_GRADE_PARAMS['excluded_station_index'])
_DAILY_OBS_PER_DAY = int(_VIS_GRADE_PARAMS['daily_obs_per_day'])
_MAP_EXTEND = list(_VIS_GRADE_PARAMS['map_extend'])
_PLOT_CONFIGS = list(_VIS_GRADE_PARAMS['plot_configs'])


def _grade_visibility(vis: np.ndarray) -> np.ndarray:
    """Convert visibility values to grade indices (0-6).

    Missing values remain -1. Lower visibility receives a higher grade index.

    Args:
        vis: Visibility array with NaN for missing values.

    Returns:
        Grade array with -1 for missing values.
    """
    vis_grade = np.zeros_like(vis, dtype=np.int_) - 1
    vis_grade[~np.isnan(vis)] = 0
    for grade, threshold in enumerate(_GRADE_THRESHOLDS, start=1):
        vis_grade[vis < threshold] = grade
    return vis_grade


def _daily_max_grade(vis_grade: np.ndarray) -> np.ndarray:
    """Compute daily maximum visibility grade from sub-daily observations.

    Args:
        vis_grade: Sub-daily grade array shaped
            (n_sub_daily * n_days, n_stations).

    Returns:
        Daily maximum grade array shaped (n_days, n_stations).
    """
    n_hours_total, n_stations = vis_grade.shape
    n_days = n_hours_total // _DAILY_OBS_PER_DAY
    vis_grade_day = np.zeros((n_days, n_stations), dtype=np.int_) - 1
    for i in range(n_days):
        start = _DAILY_OBS_PER_DAY * i
        end = _DAILY_OBS_PER_DAY * (i + 1)
        vis_grade_day[i, :] = np.max(vis_grade[start:end, :], axis=0)
    return vis_grade_day


def _yearly_grade_days(
    vis_grade_day: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Count days per grade for each year and compute missing rate.

    Args:
        vis_grade_day: Daily grade array shaped (n_days, n_stations).

    Returns:
        Tuple of (year_days, missing_rate):
            year_days: Array shaped (n_years, n_stations, n_grades).
            missing_rate: Array shaped (n_years, n_stations).
    """
    n_days, n_stations = vis_grade_day.shape
    n_years = _END_YEAR - _START_YEAR + 1
    n_grades = len(_GRADE_THRESHOLDS) + 1
    missing_rate = np.zeros((n_years, n_stations)) + np.nan
    year_days = np.zeros((n_years, n_stations, n_grades), dtype=np.int_)

    day_index = 0
    for year_offset, year in enumerate(range(_START_YEAR, _END_YEAR + 1)):
        n_days_in_year = 366 if year % 4 == 0 else 365
        year_slice = vis_grade_day[day_index:day_index + n_days_in_year, :]
        missing = np.sum(year_slice == -1, axis=0)
        missing_rate[year_offset, :] = missing / n_days_in_year
        for grade in range(n_grades):
            year_days[year_offset, :, grade] = np.sum(year_slice == grade, axis=0)
        day_index += n_days_in_year

    return year_days, missing_rate


def _select_stations(missing_rate: np.ndarray) -> np.ndarray:
    """Select stations with low missing rate, excluding a known bad station.

    Args:
        missing_rate: Array shaped (n_years, n_stations).

    Returns:
        Boolean mask of selected stations.
    """
    n_years = missing_rate.shape[0]
    sta_index = np.sum(missing_rate < _MISSING_RATE_THRESHOLD, axis=0) == n_years
    sta_index[_EXCLUDED_STATION_INDEX] = False
    return sta_index


def _build_sta_data(
    sta97: pd.DataFrame,
    year_days: np.ndarray,
    sta_index: np.ndarray,
) -> pd.DataFrame:
    """Build meteva station data with multi-grade low-visibility day counts.

    Args:
        sta97: DataFrame of selected station metadata.
        year_days: Array shaped (n_years, n_selected_stations, n_grades).
        sta_index: Boolean mask of selected stations.

    Returns:
        Meteva station data DataFrame.
    """
    sta97_sorted = sta97.sort_values(by='台站号')
    sta97_sorted = sta97_sorted.reset_index(drop=True)
    sta83 = sta97_sorted.loc[sta_index].copy()
    sta83 = sta83.reset_index(drop=True)

    mean_days = np.mean(year_days, axis=0)
    sta83.loc[:, '<1000'] = np.sum(mean_days[:, -4:], axis=1)
    sta83.loc[:, '<500'] = np.sum(mean_days[:, -3:], axis=1)
    sta83.loc[:, '<200'] = np.sum(mean_days[:, -2:], axis=1)
    sta83.loc[:, '<50'] = mean_days[:, -1]
    sta83.loc[:, 'level'] = 0
    sta83.loc[:, 'time'] = arrow.get(str(_START_YEAR)).datetime
    sta83.loc[:, 'dtime'] = 0

    cols = ['level', 'time', 'dtime', '台站号', '经度', '纬度']
    cols += ['<1000', '<500', '<200', '<50']
    sta83 = sta83.loc[:, cols]
    cols = ['level', 'time', 'dtime', 'id', 'lon', 'lat']
    cols += ['<1000', '<500', '<200', '<50']
    return meb.sta_data(sta83, columns=cols)


def _plot_grade_maps(sta83: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """Plot spatial distribution maps for each visibility grade threshold.

    Args:
        sta83: Meteva station data DataFrame with low-visibility day counts.
        output_dir: Directory to save output images.
    """
    base_cols = ['level', 'time', 'dtime', 'id', 'lon', 'lat']
    for config in _PLOT_CONFIGS:
        column = config['column']
        filename = config['filename']
        title = config['title']
        clevs = list(config['clevs'])
        cols = base_cols + [column]
        save_path = output_dir / filename
        meb.scatter_sta(
            sta83.loc[:, cols],
            save_path=str(save_path),
            dpi=300,
            title=[title],
            cmap=meb.cmaps.hour,
            map_extend=_MAP_EXTEND,
            clevs=clevs,
        )


def main() -> None:
    """Run visibility grade analysis and plot spatial distributions."""
    data_dir = pathlib.Path(utils.CFG['paths']['data_dir'])
    wind_data_dir = pathlib.Path(utils.CFG['paths']['wind_data_dir'])
    output_dir = pathlib.Path(utils.CFG['paths']['output_dir']) / \
        _VIS_GRADE_OUTPUT['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load visibility data and station metadata
    vis = np.load(str(data_dir / _VIS_GRADE_INPUT['visibility_npy']))
    sta = meb.read_station(meb.station_国家站)
    csv_path = wind_data_dir / _VIS_GRADE_INPUT['station_csv']
    sta97 = pd.read_csv(str(csv_path), encoding='gb2312', low_memory=False)

    # 2. Select stations present in the selected-station metadata table
    sta97_ids = sta97.loc[:, '台站号'].to_list()
    index = np.zeros(_N_RAW_STATIONS, dtype=np.bool_)
    for i in range(_N_RAW_STATIONS):
        if sta.loc[i, 'id'] in sta97_ids:
            index[i] = True
    vis = vis[:, index]

    # 3. Convert to visibility grades and compute daily maximum grade
    vis_grade = _grade_visibility(vis)
    vis_grade_day = _daily_max_grade(vis_grade)

    # 4. Count grade days per year and select high-quality stations
    year_days, missing_rate = _yearly_grade_days(vis_grade_day)
    sta_index = _select_stations(missing_rate)
    year_days = year_days[:, sta_index, :]

    # 5. Build station data and plot maps
    sta83 = _build_sta_data(sta97, year_days, sta_index)
    print(sta83.min())
    print(sta83.max())
    _plot_grade_maps(sta83, output_dir)


if __name__ == '__main__':
    print('Program vis_grade.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(f'Program vis_grade.py finished, total time: {utils.format_time(total_elapsed)}')
