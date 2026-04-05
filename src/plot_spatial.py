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


def _get_north_china_mask(sta: pd.DataFrame) -> np.ndarray:
    """Return North China station mask."""
    n_sta = len(sta)
    mask = np.zeros(n_sta, dtype=np.bool_)
    provinces = {'河南省', '山东省', '河北省', '北京市', '天津市', '山西省'}
    sta_reset = sta.reset_index(drop=True)  # 重置索引为 0-n
    for i in range(n_sta):
        if sta_reset.loc[i, 'province'] in provinces:
            mask[i] = True
    return mask


def plot_sta_ts4_maps(sta: pd.DataFrame, output_dir: str) -> None:
    """
    8.3 绘制站点级 TS4+ 与改善率空间分布图.
    """
    df_sta = pd.read_csv(rf'{output_dir}\csv\vis_sta_ts4+.csv', low_memory=False)
    print(f'[plot_sta_ts4_maps] CMA-SH-WARR TS4+ range: {np.min(df_sta.loc[:, "CMA-SH-WARR"])} ~ {np.max(df_sta.loc[:, "CMA-SH-WARR"])}')
    print(f'[plot_sta_ts4_maps] PDFM-TLE TS4+ range: {np.min(df_sta.loc[:, "PDFM-TLE"])} ~ {np.max(df_sta.loc[:, "PDFM-TLE"])}')
    print(f'[plot_sta_ts4_maps] Corr(lon, CMA-SH-WARR): {stats.pearsonr(sta.loc[:, "lon"], df_sta.loc[:, "CMA-SH-WARR"])[0]}')
    print(f'[plot_sta_ts4_maps] Corr(lon, PDFM-TLE): {stats.pearsonr(sta.loc[:, "lon"], df_sta.loc[:, "PDFM-TLE"])[0]}')
    print(f'[plot_sta_ts4_maps] Corr(lat, CMA-SH-WARR): {stats.pearsonr(sta.loc[:, "lat"], df_sta.loc[:, "CMA-SH-WARR"])[0]}')
    print(f'[plot_sta_ts4_maps] Corr(lat, PDFM-TLE): {stats.pearsonr(sta.loc[:, "lat"], df_sta.loc[:, "PDFM-TLE"])[0]}')
    print(f'[plot_sta_ts4_maps] Corr(alti, CMA-SH-WARR): {stats.pearsonr(sta.loc[:, "alti"], df_sta.loc[:, "CMA-SH-WARR"])[0]}')
    print(f'[plot_sta_ts4_maps] Corr(alti, PDFM-TLE): {stats.pearsonr(sta.loc[:, "alti"], df_sta.loc[:, "PDFM-TLE"])[0]}')

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
    print(f'[plot_rmse_improvement_map] CMA-SH-WARR (North China/Other): {mean_in}, {mean_out}')

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
    df_sta = pd.read_csv(rf'{output_dir}\csv\vis_sta_rmse.csv', low_memory=False)
    index_in = _get_north_china_mask(sta)
    mean_in = np.mean(df_sta.loc[index_in, 'PDFM-TLE'])
    mean_out = np.mean(df_sta.loc[~index_in, 'PDFM-TLE'])
    print(f'[plot_rmse_improvement_map] PDFM-TLE (North China/Other): {mean_in}, {mean_out}')

    rmse_before = np.array(df_sta.loc[:, 'CMA-SH-WARR'])
    rmse_after = np.array(df_sta.loc[:, 'PDFM-TLE'])
    rmse_improvement = (rmse_before - rmse_after) / rmse_before * 100
    print(f'[plot_rmse_improvement_map] RMSE improvement range: {np.min(rmse_improvement)} ~ {np.max(rmse_improvement)}')

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
    print(f'[plot_rmse_improvement_map] RMSE improvement mean: {np.mean(rmse_improvement)}')
