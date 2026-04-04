# -*- coding: utf-8 -*-
"""
观测数据准备与预处理模块.

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""

import arrow
import numpy as np
import pandas as pd

from src.vis_acc import THRES


def read_sta(sta_path: str, provinces: tuple) -> tuple:
    """
    读取站点信息并按省份筛选.

    Args:
        sta_path (str): 站点 CSV 文件路径.
        provinces (tuple): 参与分析的省级行政区列表.

    Returns:
        tuple: (筛选后的站点 DataFrame, 初始筛选布尔索引数组).
    """
    sta = pd.read_csv(filepath_or_buffer=sta_path, low_memory=False)
    sta = sta.sort_values(by=['id'])
    sta.reset_index(drop=True, inplace=True)

    index_zgdb = None
    for province in provinces:
        if index_zgdb is None:
            index_zgdb = sta.loc[:, 'province'] == province
        else:
            index_zgdb |= sta.loc[:, 'province'] == province

    sta = sta.loc[index_zgdb]
    sta.reset_index(drop=True, inplace=True)
    return sta, index_zgdb.values


def load_obs(data_dir: str, index_zgdb: np.ndarray) -> tuple:
    """
    加载能见度、降水、相对湿度观测数据并进行质控.

    Args:
        data_dir (str): 数据目录路径,末尾不带斜杠.
        index_zgdb (np.ndarray): 初始筛选站点的布尔索引.

    Returns:
        tuple: (vis, pre, rhu),均为经质控后的 numpy 数组.
    """
    vis = np.load(rf'{data_dir}\vis20-23.npy')[:, index_zgdb]
    vis[vis >= 999990] = np.nan
    vis[vis >= 30000] = 30000

    pre = np.load(rf'{data_dir}\pre20-23.npy')[:, index_zgdb]
    pre[pre >= 200] = np.nan
    pre[pre >= 30000] = 30000

    rhu = np.load(rf'{data_dir}\rhu20-23.npy')[:, index_zgdb]
    rhu[rhu >= 999990] = np.nan
    rhu[rhu > 100] = 100
    rhu[rhu < 0] = 0

    return vis, pre, rhu


def filter_region(
    sta: pd.DataFrame,
    vis: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    region_provinces: tuple
) -> tuple:
    """
    按目标省级行政区二次筛选站点及对应要素数组.

    Args:
        sta (pd.DataFrame): 站点信息表.
        vis (np.ndarray): 能见度数组.
        pre (np.ndarray): 降水数组.
        rhu (np.ndarray): 相对湿度数组.
        region_provinces (tuple): 目标省份列表.

    Returns:
        tuple: (筛选后的 sta, vis, pre, rhu, 二次筛选布尔索引).
    """
    n_sta = len(sta)
    index_region = np.zeros(n_sta, dtype=np.bool_)
    for i in range(n_sta):
        if sta.loc[i, 'province'] in region_provinces:
            index_region[i] = True

    sta = sta.loc[index_region]
    sta.reset_index(drop=True, inplace=True)
    sta = sta.astype({'data0': float})

    vis = vis[:, index_region]
    pre = pre[:, index_region]
    rhu = rhu[:, index_region]

    return sta, vis, pre, rhu, index_region


def grade_visibility(vis: np.ndarray) -> np.ndarray:
    """
    对能见度进行六级分级.

    Args:
        vis (np.ndarray): 能见度观测数组.

    Returns:
        np.ndarray: 分级后的整数数组,缺测为 -1.
    """
    vis_grade = np.zeros_like(vis, dtype=np.int_) - 1
    vis_grade[~np.isnan(vis)] = 0
    for i, t in enumerate(THRES):
        vis_grade[vis < t] = i + 1
    return vis_grade


def build_month_index(start_year: int, n_hours: int) -> np.ndarray:
    """
    构建逐小时对应的月份编号索引.

    Args:
        start_year (int): 起始年份.
        n_hours (int): 总小时数.

    Returns:
        np.ndarray: 长度为 n_hours 的月份索引数组.
    """
    month_ind = np.zeros(shape=n_hours, dtype=np.int_)
    base = arrow.get(str(start_year))
    for i in range(n_hours):
        month_ind[i] = base.shift(hours=i).datetime.month
    return month_ind
