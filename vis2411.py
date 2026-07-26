#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Founded in 2024-10-05
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


# Load vis2411 configuration at import time
_VIS2411_CFG = utils.CFG['vis2411']
_VIS2411_INPUT = _VIS2411_CFG['input_files']
_VIS2411_OUTPUT = _VIS2411_CFG['output_files']
_VIS2411_PARAMS = _VIS2411_CFG['params']

_START_YEAR = int(_VIS2411_PARAMS['start_year'])
_N_MONTHS = int(_VIS2411_PARAMS['n_months'])
_N_STATIONS = int(_VIS2411_PARAMS['n_stations'])
_N_TOTAL_HOURS = int(_VIS2411_PARAMS['n_total_hours'])
_STATION_ID_COL = str(_VIS2411_PARAMS['station_id_column'])
_TIME_COL = str(_VIS2411_PARAMS['time_column'])
_VIS_LEGACY_COL = str(_VIS2411_PARAMS['vis_column_legacy'])
_VIS_MODERN_COL = str(_VIS2411_PARAMS['vis_column_modern'])
_MODERN_SPLIT_YEAR = int(_VIS2411_PARAMS['modern_split_year'])


def _clean_station_id(df: pd.DataFrame, month_index: int) -> pd.DataFrame:
    """Remove rows with non-digit station IDs for the last two months.

    Some late-month CSVs contain malformed station IDs. Drop them before
    converting the column to integer.

    Args:
        df: Monthly visibility DataFrame.
        month_index: Zero-based month index within the extraction period.

    Returns:
        Cleaned DataFrame.
    """
    if month_index < _N_MONTHS - 2:
        return df
    index = df.loc[:, _STATION_ID_COL].astype(str).str.isdigit()
    df = df.loc[index].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def _month_hour_bounds(
    month_arrow: arrow.Arrow,
    base_arrow: arrow.Arrow,
) -> typing.Tuple[int, int]:
    """Compute hour indices for the start/end of a month.

    Args:
        month_arrow: Arrow object pointing to the first moment of the month.
        base_arrow: Arrow object pointing to the extraction start time.

    Returns:
        Tuple of (start_hour_index, end_hour_index).
    """
    left = round((month_arrow - base_arrow).total_seconds() / 3600)
    right = round((month_arrow.shift(months=1) - base_arrow).total_seconds() / 3600)
    return left, right


def _select_vis_column(month_arrow: arrow.Arrow) -> str:
    """Return the visibility column name for a given month.

    The column name changed after the modern split year.

    Args:
        month_arrow: Arrow object pointing to the month.

    Returns:
        Visibility column name.
    """
    if month_arrow < arrow.get(str(_MODERN_SPLIT_YEAR)):
        return _VIS_LEGACY_COL
    return _VIS_MODERN_COL


def _fill_station_data(
    vis: np.ndarray,
    df: pd.DataFrame,
    station_meta: pd.DataFrame,
    month_arrow: arrow.Arrow,
    base_arrow: arrow.Arrow,
) -> None:
    """Fill visibility data for one month into the output array.

    For each station, hourly observations are placed on the regular hourly
    grid. If the month is complete, resample is used; otherwise observations
    are placed individually by timestamp.

    Args:
        vis: Output visibility array shaped (n_total_hours, n_stations).
        df: Monthly visibility DataFrame.
        station_meta: National station metadata DataFrame.
        month_arrow: Arrow object for the current month.
        base_arrow: Arrow object for the extraction start time.
    """
    left, right = _month_hour_bounds(month_arrow, base_arrow)
    vis_col = _select_vis_column(month_arrow)

    for station_idx in range(_N_STATIONS):
        station_id = station_meta.loc[station_idx, 'id']
        df0 = df.loc[df.loc[:, _STATION_ID_COL] == station_id].copy()
        if len(df0) == 0:
            continue

        df0.sort_values(by=[_TIME_COL], inplace=True)
        df0.reset_index(drop=True, inplace=True)

        if len(df0) == right - left:
            # Complete hourly data: resample to enforce regular hourly grid.
            df0.loc[:, 'time'] = pd.to_datetime(df0.loc[:, _TIME_COL], utc=True)
            df0.set_index('time', inplace=True)
            df0 = df0.resample('H').first()
            vis[left:right, station_idx] = df0.loc[:, vis_col].values
        else:
            # Incomplete data: place each observation by its exact timestamp.
            for k in range(len(df0)):
                obs_arrow = arrow.get(df0.loc[k, _TIME_COL])
                hour_idx = round((obs_arrow - base_arrow).total_seconds() / 3600)
                vis[hour_idx, station_idx] = df0.loc[k, vis_col]


def main() -> None:
    """Read national station visibility CSVs and build the vis.npy array."""
    data_dir = pathlib.Path(utils.CFG['paths']['data_dir'])
    station_meta = meb.read_station(meb.station_国家站)

    vis = np.zeros((_N_TOTAL_HOURS, _N_STATIONS), dtype=np.float32) + np.nan
    base_arrow = arrow.get(str(_START_YEAR))
    csv_template = str(_VIS2411_INPUT['csv_template'])

    for month_index in range(_N_MONTHS):
        month_arrow = base_arrow.shift(months=month_index)
        yyyymm = month_arrow.format('YYYYMM')
        csv_path = data_dir / csv_template.format(yyyymm=yyyymm)
        df = pd.read_csv(str(csv_path), low_memory=False)
        df = _clean_station_id(df, month_index)
        df = df.astype({_STATION_ID_COL: int})

        _fill_station_data(vis, df, station_meta, month_arrow, base_arrow)
        print(yyyymm)

    output_path = data_dir / _VIS2411_OUTPUT['visibility_npy']
    np.save(str(output_path), vis)


if __name__ == '__main__':
    print('Program vis2411.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(f'Program vis2411.py finished, total time: {utils.format_time(total_elapsed)}')
