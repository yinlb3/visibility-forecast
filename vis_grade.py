#!user/bin.python3

"""
Founded in 2023-11-19
Modified in 2023-11-19
@author: yinlb
"""
import os
import sys

import arrow
import numpy as np
import pandas as pd
from meteva import base as meb
from matplotlib import pyplot as plt


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
    vis = np.load(r'E:\vis\vis_grade.npy')
    sta = meb.read_station(meb.station_国家站)
    sta97 = pd.read_csv(r'D:\Project\wind\国家气象观测站.csv', encoding='gb2312', low_memory=False)
    index = np.zeros(2410, dtype=np.bool_)
    for i in range(2410):
        if sta.loc[i, 'id'] in sta97.loc[:, '台站号'].to_list():
            index[i] = True
    # for i, j in enumerate(np.where(index)[0]):
    #     if i == 38 or i == 68:
    #         print(sta.loc[j, 'id'])
    vis = vis[:, index]
    vis_grade = np.zeros_like(vis, dtype=np.int_) - 1
    vis_grade[~np.isnan(vis)] = 0
    vis_grade[vis < 10000] = 1
    vis_grade[vis < 2000] = 2
    vis_grade[vis < 1000] = 3
    vis_grade[vis < 500] = 4
    vis_grade[vis < 200] = 5
    vis_grade[vis < 50] = 6
    # np.save(r'E:\vis\vis_grade97.npy', vis_grade)

    vis_grade_day = np.zeros((15706, 97), dtype=np.int_) - 1
    for i in range(15706):
        for j in range(97):
            vis_grade_day[i, j] = np.max(vis_grade[4 * i: 4 * (i + 1), j])
    # np.save(r'E:\vis\vis_grade97_day.npy', vis_grade_day)

    c = np.zeros((34, 97)) + np.nan
    vis_grade_year_days = np.zeros((34, 97, 7), dtype=np.int_)
    i = 0
    for year in range(1980, 2014):
        n_days = 366 if year % 4 == 0 else 365
        c[year - 1980, :] = np.sum(vis_grade_day[i: i + n_days, :] == -1, axis=0) / n_days
        for j in range(7):
            vis_grade_year_days[year - 1980, :, j] = np.sum(vis_grade_day[i: i + n_days, :] == j, axis=0)
        i += n_days
    sta_index = np.sum(c < 0.02, axis=0) == 34
    sta_index[54] = False
    vis_grade_year_days = vis_grade_year_days[:, sta_index]
    # np.save(r'E:\vis\vis_grade97_year_days.npy', vis_grade_year_days)

    sta97.sort_values(by='台站号', inplace=True)
    sta97.reset_index(drop=True, inplace=True)
    sta83 = sta97.loc[sta_index]
    sta83.reset_index(drop=True, inplace=True)
    e = np.mean(vis_grade_year_days, axis=0)
    sta83.loc[:, '<1000'] = np.sum(e[:, -4:], axis=1)
    sta83.loc[:, '<500'] = np.sum(e[:, -3:], axis=1)
    sta83.loc[:, '<200'] = np.sum(e[:, -2:], axis=1)
    sta83.loc[:, '<50'] = e[:, -1]
    sta83.loc[:, 'level'] = 0
    sta83.loc[:, 'time'] = arrow.get('1980').datetime
    sta83.loc[:, 'dtime'] = 0
    sta83 = sta83.loc[:, ['level', 'time', 'dtime', '台站号', '经度', '纬度', '<1000', '<500', '<200', '<50']]
    sta83 = meb.sta_data(sta83, columns=['level', 'time', 'dtime', 'id', 'lon', 'lat', '<1000', '<500', '<200', '<50'])
    print(sta83.min())
    print(sta83.max())
    meb.scatter_sta(sta83.loc[:, ['level', 'time', 'dtime', 'id', 'lon', 'lat', '<1000']],
                    save_path=r'E:\vis\1000.png', dpi=300, title=['能见度不足1000米天数'], cmap=meb.cmaps.hour,
                    map_extend=[108.65, 114.4, 24.5, 30.25], clevs=[0, 8, 16, 24, 32, 40, 48, 56, 64])
    meb.scatter_sta(sta83.loc[:, ['level', 'time', 'dtime', 'id', 'lon', 'lat', '<500']],
                    save_path=r'E:\vis\500.png', dpi=300, title=['能见度不足500米天数'], cmap=meb.cmaps.hour,
                    map_extend=[108.65, 114.4, 24.5, 30.25], clevs=[0, 6, 12, 18, 24, 30, 36, 42, 48])
    meb.scatter_sta(sta83.loc[:, ['level', 'time', 'dtime', 'id', 'lon', 'lat', '<200']],
                    save_path=r'E:\vis\200.png', dpi=300, title=['能见度不足200米天数'], cmap=meb.cmaps.hour,
                    map_extend=[108.65, 114.4, 24.5, 30.25], clevs=[0, 4, 8, 12, 16, 20, 24, 28, 32])
    meb.scatter_sta(sta83.loc[:, ['level', 'time', 'dtime', 'id', 'lon', 'lat', '<50']],
                    save_path=r'E:\vis\50.png', dpi=300, title=['能见度不足50米天数'], cmap=meb.cmaps.hour,
                    map_extend=[108.65, 114.4, 24.5, 30.25], clevs=[0, 3, 6, 9, 12, 15, 18, 21, 24])


if __name__ == '__main__':
    print('The program "vis_grade.py" is beginning.')
    start = arrow.now()

    main()

    end = arrow.now()
    running_time = (end - start).total_seconds()

    print('The program "vis_grade.py" runs out in {:s}.'.format(format_time(running_time)))
