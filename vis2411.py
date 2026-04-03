#!user/bin.python3

"""
Founded in 2024-10-05
Modified in 2024-10-05
@author: yinlb
"""
import os
import sys

import arrow
import numpy as np
import pandas as pd
from meteva import base as meb


def format_time(second: float, is_abbreviation: bool = False) -> str:
    r"""Format time.

    :param second: A float number representing the number of seconds.
    :param is_abbreviation: A boolean variable representing whether processing to abbreviation.
        The default value is False.
    :return: A sequence of strings representing the time. For example: '43.5 seconds'
    :raise ValueError: The value of input parameter 'second' is wrong.
    """
    if second < 0:
        raise ValueError('The input parameter \'second\' cannot be negative.')
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
    sta0 = meb.read_station(meb.station_国家站)

    vis = np.zeros((639912, 2411), dtype=np.float32) + np.nan
    for i in range(876):
        time_arrow = arrow.get('1951').shift(months=i)
        yyyymm = time_arrow.format('YYYYMM')
        filepath = fr'D:\data\vis\{yyyymm}.csv'
        df = pd.read_csv(filepath, low_memory=False)
        if i >= 874:
            index = np.ones(len(df), dtype=np.bool_)
            for j in range(len(df)):
                if not df.loc[j, 'Station_Id_C'].isdigit():
                    index[j] = False
            df = df.loc[index]
            df.reset_index(drop=True, inplace=True)
        df = df.astype({'Station_Id_C': int})
        for j in range(2411):
            left = round((time_arrow - arrow.get('1951')).total_seconds() / 3600)
            right = round((time_arrow.shift(months=1) - arrow.get('1951')).total_seconds() / 3600)
            df0 = df.loc[df.loc[:, 'Station_Id_C'] == sta0.loc[j, 'id']]
            df0.sort_values(by=['Datetime'], inplace=True)
            df0.reset_index(drop=True, inplace=True)
            if len(df0) == right - left:
                df0.loc[:, 'time'] = pd.to_datetime(df0.loc[:, 'Datetime'], utc=True)
                df0.set_index('time', inplace=True)
                df0 = df0.resample('H').first()
                if time_arrow < arrow.get('2016'):
                    vis[left:right, j] = df0.loc[:, 'VIS_HOR_10MI']
                else:
                    vis[left:right, j] = df0.loc[:, 'VIS_Min']
            else:
                for k in range(len(df0)):
                    ind_k = round((arrow.get(df0.loc[k, 'Datetime']) - arrow.get('1951')).total_seconds() / 3600)
                    if time_arrow < arrow.get('2016'):
                        vis[ind_k, j] = df0.loc[k, 'VIS_HOR_10MI']
                    else:
                        vis[ind_k, j] = df0.loc[k, 'VIS_Min']
        print(yyyymm)
    np.save(r'D:\data\vis\vis.npy', vis)


if __name__ == '__main__':
    print('Program vis2411.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(f'Program vis2411.py finished, total time: {format_time(total_elapsed)}')
