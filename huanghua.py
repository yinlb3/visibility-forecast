#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Founded in 2024-10-23
Modified in 2026-07-21
@author: yinlb
"""
import pathlib

import arrow
import numpy as np
import pandas as pd

from src import utils


# Load huanghua configuration at import time
_HUANGHUA_CFG = utils.CFG['huanghua']
_HUANGHUA_INPUT = _HUANGHUA_CFG['input_files']
_HUANGHUA_OUTPUT = _HUANGHUA_CFG['output_files']
_HUANGHUA_PARAMS = _HUANGHUA_CFG['params']

_START_YEAR = int(_HUANGHUA_PARAMS['start_year'])
_END_YEAR = int(_HUANGHUA_PARAMS['end_year'])
_N_TOTAL_DAYS = int(_HUANGHUA_PARAMS['n_total_days'])
_N_HOURS = int(_HUANGHUA_PARAMS['n_hours'])
_N_VARIABLES = int(_HUANGHUA_PARAMS['n_variables'])
_SHEET_NAMES = list(_HUANGHUA_PARAMS['sheet_names'])
_CONTINUOUS_SHEETS = set(_HUANGHUA_PARAMS['continuous_sheets'])
_FILENAME_TEMPLATES = list(_HUANGHUA_INPUT['filename_templates'])


def _get_n_days(year: int, month: int) -> int:
    """Return the number of days in a given year and month.

    Args:
        year: Year.
        month: Month (1-12).

    Returns:
        Number of days in the month.
    """
    if month in (4, 6, 9, 11):
        return 30
    if month == 2:
        if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):
            return 29
        return 28
    return 31


def _find_monthly_file(
    base_dir: pathlib.Path,
    year: int,
    month: int,
) -> pathlib.Path:
    """Find the first existing monthly logbook Excel file.

    Three filename patterns are tried in order.

    Args:
        base_dir: Base directory for logbook files.
        year: Year.
        month: Month.

    Returns:
        Path to an existing Excel file.

    Raises:
        FileNotFoundError: If none of the candidate files exist.
    """
    for template in _FILENAME_TEMPLATES:
        candidate = base_dir / template.format(year=year, month=month)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f'No monthly logbook found for {year}-{month:02d} in {base_dir}'
    )


def _read_sheet(
    file_path: pathlib.Path,
    sheet_name: str,
) -> pd.DataFrame:
    """Read a sheet from the monthly logbook Excel file.

    Args:
        file_path: Path to the Excel file.
        sheet_name: Name of the sheet to read.

    Returns:
        DataFrame of the sheet content.
    """
    return pd.read_excel(str(file_path), sheet_name=sheet_name)


def _fill_three_blocks(
    data: np.ndarray,
    df: pd.DataFrame,
    day_offset: int,
    n_days: int,
    variable_index: int,
) -> None:
    """Fill data from a sheet arranged in three day blocks.

    Layout: days 1-10 at rows 2-11, days 11-20 at rows 14-23,
    remaining days at rows 25 onward.

    Args:
        data: Output array shaped (n_total_days, n_hours, n_variables).
        df: DataFrame read from the sheet.
        day_offset: Starting day index in the output array.
        n_days: Number of days in the month.
        variable_index: Index of the variable in the output array.
    """
    end10 = min(10, n_days)
    end20 = min(20, n_days)
    data[day_offset:day_offset + end10, :, variable_index] = \
        df.iloc[2:2 + end10, 1:1 + _N_HOURS].values
    if end20 > 10:
        data[day_offset + 10:day_offset + end20, :, variable_index] = \
            df.iloc[14:14 + (end20 - 10), 1:1 + _N_HOURS].values
    if n_days > 20:
        start_row = 26
        n_remaining = n_days - 20
        data[day_offset + 20:day_offset + n_days, :, variable_index] = \
            df.iloc[start_row:start_row + n_remaining, 1:1 + _N_HOURS].values


def _fill_continuous(
    data: np.ndarray,
    df: pd.DataFrame,
    day_offset: int,
    n_days: int,
    variable_index: int,
) -> None:
    """Fill data from a sheet with continuous daily rows.

    Layout: all days at rows 2 onward.

    Args:
        data: Output array shaped (n_total_days, n_hours, n_variables).
        df: DataFrame read from the sheet.
        day_offset: Starting day index in the output array.
        n_days: Number of days in the month.
        variable_index: Index of the variable in the output array.
    """
    data[day_offset:day_offset + n_days, :, variable_index] = \
        df.iloc[2:2 + n_days, 1:1 + _N_HOURS].values


def _fill_month(
    data: np.ndarray,
    file_path: pathlib.Path,
    day_offset: int,
    n_days: int,
) -> None:
    """Fill one month of data for all variables.

    Args:
        data: Output array shaped (n_total_days, n_hours, n_variables).
        file_path: Path to the monthly logbook Excel file.
        day_offset: Starting day index in the output array.
        n_days: Number of days in the month.
    """
    for variable_index, sheet_name in enumerate(_SHEET_NAMES):
        df = _read_sheet(file_path, sheet_name)
        if sheet_name in _CONTINUOUS_SHEETS:
            _fill_continuous(data, df, day_offset, n_days, variable_index)
        else:
            _fill_three_blocks(data, df, day_offset, n_days, variable_index)


def main() -> None:
    """Read Huanghua airport monthly logbooks and build the meteogram array."""
    data_dir = pathlib.Path(utils.CFG['paths']['data_dir'])
    base_dir = data_dir / _HUANGHUA_INPUT['base_dir']

    data = np.zeros((_N_TOTAL_DAYS, _N_HOURS, _N_VARIABLES), dtype=np.float32) + np.nan
    day_offset = 0

    for year in range(_START_YEAR, _END_YEAR):
        for month in range(1, 13):
            n_days = _get_n_days(year, month)
            file_path = _find_monthly_file(base_dir, year, month)
            _fill_month(data, file_path, day_offset, n_days)
            day_offset += n_days
            print(f'{year}-{month:02d}')

    output_path = data_dir / _HUANGHUA_INPUT['base_dir'] / _HUANGHUA_OUTPUT['meteogram_npy']
    np.save(str(output_path), data)


if __name__ == '__main__':
    print('Program huanghua.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(f'Program huanghua.py finished, total time: {utils.format_time(total_elapsed)}')
