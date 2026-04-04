# -*- coding: utf-8 -*-
"""
空间分布可视化模块 (阶段 8.3).

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""

import numpy as np
import pandas as pd
from meteva import base as meb
from scipy import stats


def _get_north_china_mask(sta: pd.DataFrame, n_sta: int = 502) -> np.ndarray:
    """返回华北站点掩码."""
    mask = np.zeros(n_sta, dtype=np.bool_)
    provinces = {'河南省', '山东省', '河北省', '北京市', '天津市', '山西省'}
    for i in range(n_sta):
        if sta.loc[i, 'province'] in provinces:
            mask[i] = True
    return mask


def plot_sta_ts4_maps(sta: pd.DataFrame, output_dir: str) -> None:
    """
    8.3 绘制站点级 TS4+ 与改善率空间分布图.
    """
    df_sta = pd.read_csv(rf'{output_dir}\vis_sta_ts4+.csv', low_memory=False)
    print(np.min(df_sta.loc[:, 'CMA-SH-WARR']), np.max(df_sta.loc[:, 'CMA-SH-WARR']))
    print(np.min(df_sta.loc[:, 'PDFM-TLE']), np.max(df_sta.loc[:, 'PDFM-TLE']))
    print(stats.pearsonr(sta.loc[:, 'lon'], df_sta.loc[:, 'CMA-SH-WARR'])[0])
    print(stats.pearsonr(sta.loc[:, 'lon'], df_sta.loc[:, 'PDFM-TLE'])[0])
    print(stats.pearsonr(sta.loc[:, 'lat'], df_sta.loc[:, 'CMA-SH-WARR'])[0])
    print(stats.pearsonr(sta.loc[:, 'lat'], df_sta.loc[:, 'PDFM-TLE'])[0])
    print(stats.pearsonr(sta.loc[:, 'alti'], df_sta.loc[:, 'CMA-SH-WARR'])[0])
    print(stats.pearsonr(sta.loc[:, 'alti'], df_sta.loc[:, 'PDFM-TLE'])[0])

    cmap, clevs = meb.def_cmap_clevs(
        meb.def_cmap_clevs(meb.cmaps.ts, vmin=0, vmax=0.7)[0],
        clevs=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    )

    sta0 = sta.loc[:, ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')]
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'CMA-SH-WARR']
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='max',
        title=[''],
        save_path=rf'{output_dir}\sta_ts4+_nwp.png',
        dpi=800
    )
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='max',
        title=[''],
        save_path=rf'{output_dir}\sta_ts4+_nwp.pdf',
        dpi=800
    )

    index_in = _get_north_china_mask(sta)
    mean_in = np.mean(df_sta.loc[index_in, 'CMA-SH-WARR'])
    mean_out = np.mean(df_sta.loc[~index_in, 'CMA-SH-WARR'])
    print(f'CMA-SH-WARR: {mean_in}, {mean_out}')

    sta0 = sta.loc[:, ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')]
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'PDFM-TLE']
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='max',
        title=[''],
        save_path=rf'{output_dir}\sta_ts4+.png',
        dpi=800
    )
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='max',
        title=[''],
        save_path=rf'{output_dir}\sta_ts4+.pdf',
        dpi=800
    )


def plot_rmse_improvement_map(sta: pd.DataFrame, output_dir: str) -> None:
    """
    8.4 绘制 RMSE 改善率空间分布图.
    """
    df_sta = pd.read_csv(rf'{output_dir}\vis_sta_rmse.csv', low_memory=False)
    index_in = _get_north_china_mask(sta)
    mean_in = np.mean(df_sta.loc[index_in, 'PDFM-TLE'])
    mean_out = np.mean(df_sta.loc[~index_in, 'PDFM-TLE'])
    print(f'PDFM-TLE: {mean_in}, {mean_out}')

    rmse_before = np.array(df_sta.loc[:, 'CMA-SH-WARR'])
    rmse_after = np.array(df_sta.loc[:, 'PDFM-TLE'])
    rmse_improvement = (rmse_before - rmse_after) / rmse_before * 100
    print(np.min(rmse_improvement), np.max(rmse_improvement))

    cmap, clevs = meb.def_cmap_clevs(
        meb.def_cmap_clevs(meb.cmaps.me, vmin=-40, vmax=60)[0],
        clevs=[-40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
    )
    sta0 = sta.loc[:, ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')]
    sta0.loc[:, 'data0'] = rmse_improvement
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='both',
        title=[''],
        save_path=rf'{output_dir}\sta_mre_improvement.png',
        dpi=800
    )
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='both',
        title=[''],
        save_path=rf'{output_dir}\sta_mre_improvement.pdf',
        dpi=800
    )
    print(np.mean(rmse_improvement))
