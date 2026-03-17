#!user/bin.python3

"""
Founded in 2024-10-23
Modified in 2024-10-23
@author: yinlb
"""
import os
import sys
import typing

import arrow
import numpy as np
import pandas as pd


def get_n_days(year: int, month: int) -> int:
    n_days = 31
    if month in (4, 6, 9, 11):
        n_days = 30
    elif month == 2:
        if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):
            n_days = 29
        else:
            n_days = 28
    return n_days


def format_time(second: float, is_abbreviation: bool = False) -> str:
    r"""Format time.

    :param second: A float number representing the number of seconds.
    :param is_abbreviation: A boolean variable representing whether processing to abbreviation.
        The default value is False.
    :return: A sequence of strings representing the time. For example: '43.5 seconds'
    :raise ValueError: The value of input parameter 'second' is wrong.
    """
    if second < 0:
        raise ValueError('The input parameter "second" cannot be negative.')
    elif is_abbreviation:
        if second <= 60:
            time_str = str(second) + 's'
        elif second <= 3600:
            time_str = str(second / 60) + 'm'
        else:
            time_str = str(second / 3600) + 'h'
    else:
        if second <= 1:
            time_str = str(second) + ' second'
        elif second <= 60:
            time_str = str(second) + ' seconds'
        elif second <= 3600:
            time_str = str(second / 60) + 'minutes'
        else:
            time_str = str(second / 3600) + 'hours'

    return time_str


def main() -> None:
    data = np.zeros((4017, 24, 8), dtype=np.float32) + np.nan
    n = 0
    for year in range(2013, 2024):
        for month in range(1, 13):
            n_days = get_n_days(year, month)
            filepath1 = fr'D:\data\历年月总簿（2013-2023）\{year}年\{year}EXCEL\{year}-{month:02d}月总簿.xls'
            filepath2 = fr'D:\data\历年月总簿（2013-2023）\{year}年\{year}EXCEL\MZGHA{year}{month:02d}.xls'
            filepath3 = fr'D:\data\历年月总簿（2013-2023）\{year}年\{year}EXCEL\MZHGA{year}{month:02d}.xls'
            if os.path.exists(filepath1):
                df = pd.read_excel(filepath1, sheet_name='场面气压')
            elif os.path.exists(filepath2):
                df = pd.read_excel(filepath2, sheet_name='场面气压')
            else:
                df = pd.read_excel(filepath3, sheet_name='场面气压')
            data[n: n + 10, :, 0] = df.iloc[2:12, 1:25]
            data[n + 10: n + 20, :, 0] = df.iloc[14:24, 1:25]
            data[n + 20: n + n_days, :, 0] = df.iloc[26: 26 + n_days - 20, 1:25]
            if os.path.exists(filepath1):
                df = pd.read_excel(filepath1, sheet_name='修正海平面气压')
            elif os.path.exists(filepath2):
                df = pd.read_excel(filepath2, sheet_name='修正海平面气压')
            else:
                df = pd.read_excel(filepath3, sheet_name='修正海平面气压')
            data[n: n + 10, :, 1] = df.iloc[2:12, 1:25]
            data[n + 10: n + 20, :, 1] = df.iloc[14:24, 1:25]
            data[n + 20: n + n_days, :, 1] = df.iloc[26: 26 + n_days - 20, 1:25]
            if os.path.exists(filepath1):
                df = pd.read_excel(filepath1, sheet_name='温度')
            elif os.path.exists(filepath2):
                df = pd.read_excel(filepath2, sheet_name='温度')
            else:
                df = pd.read_excel(filepath3, sheet_name='温度')
            data[n: n + 10, :, 2] = df.iloc[2:12, 1:25]
            data[n + 10: n + 20, :, 2] = df.iloc[14:24, 1:25]
            data[n + 20: n + n_days, :, 2] = df.iloc[26: 26 + n_days - 20, 1:25]
            if os.path.exists(filepath1):
                df = pd.read_excel(filepath1, sheet_name='相对湿度')
            elif os.path.exists(filepath2):
                df = pd.read_excel(filepath2, sheet_name='相对湿度')
            else:
                df = pd.read_excel(filepath3, sheet_name='相对湿度')
            data[n: n + 10, :, 3] = df.iloc[2:12, 1:25]
            data[n + 10: n + 20, :, 3] = df.iloc[14:24, 1:25]
            data[n + 20: n + n_days, :, 3] = df.iloc[26: 26 + n_days - 20, 1:25]
            if os.path.exists(filepath1):
                df = pd.read_excel(filepath1, sheet_name='露点温度')
            elif os.path.exists(filepath2):
                df = pd.read_excel(filepath2, sheet_name='露点温度')
            else:
                df = pd.read_excel(filepath3, sheet_name='露点温度')
            data[n: n + n_days, :, 4] = df.iloc[2: 2 + n_days, 1:25]
            if os.path.exists(filepath1):
                df = pd.read_excel(filepath1, sheet_name='总云量')
            elif os.path.exists(filepath2):
                df = pd.read_excel(filepath2, sheet_name='总云量')
            else:
                df = pd.read_excel(filepath3, sheet_name='总云量')
            data[n: n + 10, :, 5] = df.iloc[2:12, 1:25]
            data[n + 10: n + 20, :, 5] = df.iloc[14:24, 1:25]
            data[n + 20: n + n_days, :, 5] = df.iloc[26: 26 + n_days - 20, 1:25]
            if os.path.exists(filepath1):
                df = pd.read_excel(filepath1, sheet_name='低云量')
            elif os.path.exists(filepath2):
                df = pd.read_excel(filepath2, sheet_name='低云量')
            else:
                df = pd.read_excel(filepath3, sheet_name='低云量')
            data[n: n + 10, :, 6] = df.iloc[2:12, 1:25]
            data[n + 10: n + 20, :, 6] = df.iloc[14:24, 1:25]
            data[n + 20: n + n_days, :, 6] = df.iloc[26: 26 + n_days - 20, 1:25]
            if os.path.exists(filepath1):
                df = pd.read_excel(filepath1, sheet_name='主导能见度')
            elif os.path.exists(filepath2):
                df = pd.read_excel(filepath2, sheet_name='主导能见度')
            else:
                df = pd.read_excel(filepath3, sheet_name='主导能见度')
            data[n: n + n_days, :, 7] = df.iloc[2: 2 + n_days, 1:25]
            n += n_days
    np.save(r'D:\data\历年月总簿（2013-2023）\data.npy', data)


if __name__ == '__main__':
    print('The program "huanghua.py" is beginning.')
    start = arrow.now()

    main()

    end = arrow.now()
    running_time = (end - start).total_seconds()

    print('The program "huanghua.py" runs out in {:s}.'.format(format_time(running_time)))
