#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 2.4: Spatial distribution feature plotting module.

Founded in 2026-04-14
Modified in 2026-10-01
@author: yinlb, space-bunny
"""

import pathlib
import typing

import numpy as np
import pandas as pd
from meteva import base as meb    # type: ignore

from src import utils


def _scatter_sta_pair(
    sta0: pd.DataFrame,
    save_path: str,
    cmap: object,
    clevs: object,
    cfg: typing.Dict,
) -> None:
    """Helper: save station scatter in all configured formats."""
    utils.save_station_scatter(sta0, save_path, cmap, clevs, cfg)


def _calc_region_mean(
    df_sta: pd.DataFrame,
    col: str,
    sta: pd.DataFrame,
    provinces: tuple
) -> typing.Tuple[float, float]:
    """Helper: Calc mean inside/outside specified provinces."""
    index_in = np.zeros(len(sta), dtype=np.bool_)
    for i in range(len(sta)):
        if sta.loc[i, 'province'] in provinces:
            index_in[i] = True
    mean_in = np.mean(df_sta.loc[index_in, col])
    mean_out = np.mean(df_sta.loc[~index_in, col])
    return mean_in, mean_out


def plot_sta_frequency_maps(
    sta: pd.DataFrame,
    df_sta: pd.DataFrame,
    output_dir: str,
    cfg: typing.Dict
) -> None:
    """Plot low visibility freq spatial map."""
    plot_cfg = cfg['draw']['plot']['sta_frequency_maps']
    prefix = '[plot_sta_frequency_maps]'
    cols = ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')

    # 1. Plot low visibility event (LVE) frequency map
    lve_cmap = plot_cfg['lve_cmap']
    lve_vmin = plot_cfg['lve_vmin']
    lve_vmax = plot_cfg['lve_vmax']
    lve_clevs = plot_cfg['lve_clevs']
    # Nested def_cmap_clevs: build cmap from range, then attach discrete levels
    cmap0, clevs0 = meb.def_cmap_clevs(
        getattr(meb.cmaps, lve_cmap), vmin=lve_vmin, vmax=lve_vmax)
    cmap, clevs = meb.def_cmap_clevs(cmap0, clevs=lve_clevs)
    sta0 = sta.loc[:, cols].copy()
    sta0.loc[:, 'data0'] = df_sta.loc[:, '1']
    _scatter_sta_pair(
        sta0, str(pathlib.Path(output_dir) / 'sta_1+'), cmap, clevs, cfg)
    mean_in, mean_out = _calc_region_mean(df_sta, '1', sta, ('湖南省',))
    print(f'{prefix} LVE (in/out Hunan): {mean_in:.4f}, {mean_out:.4f}')

    # 2. Plot low visibility precip event (LVPE) frequency map
    lvpe_cmap = plot_cfg['lvpe_cmap']
    lvpe_vmin = plot_cfg['lvpe_vmin']
    lvpe_vmax = plot_cfg['lvpe_vmax']
    lvpe_clevs = plot_cfg['lvpe_clevs']
    cmap0, clevs0 = meb.def_cmap_clevs(
        getattr(meb.cmaps, lvpe_cmap), vmin=lvpe_vmin, vmax=lvpe_vmax)
    cmap, clevs = meb.def_cmap_clevs(cmap0, clevs=lvpe_clevs)
    sta0.loc[:, 'data0'] = df_sta.loc[:, '1pre']
    _scatter_sta_pair(
        sta0, str(pathlib.Path(output_dir) / 'sta_1+_pre'), cmap, clevs, cfg)
    prov = ('湖南省', '江西省', '浙江省')
    mean_in, mean_out = _calc_region_mean(df_sta, '1pre', sta, prov)
    lvpe_str = f'{prefix} LVPE (in/out Hunan/Jiangxi/Zhejiang): '
    lvpe_str += f'{mean_in:.4f}, {mean_out:.4f}'
    print(lvpe_str)

    # 3. Plot low visibility fog event (LVFE) frequency map
    lvfe_cmap = plot_cfg['lvfe_cmap']
    lvfe_vmin = plot_cfg['lvfe_vmin']
    lvfe_vmax = plot_cfg['lvfe_vmax']
    lvfe_clevs = plot_cfg['lvfe_clevs']
    cmap0, clevs0 = meb.def_cmap_clevs(
        getattr(meb.cmaps, lvfe_cmap), vmin=lvfe_vmin, vmax=lvfe_vmax)
    cmap, clevs = meb.def_cmap_clevs(cmap0, clevs=lvfe_clevs)
    sta0.loc[:, 'data0'] = df_sta.loc[:, '1fog']
    _scatter_sta_pair(
        sta0, str(pathlib.Path(output_dir) / 'sta_1+_fog'), cmap, clevs, cfg)
    mean_in, mean_out = _calc_region_mean(df_sta, '1fog', sta, ('湖南省', '江苏省'))
    lvfe_str = f'{prefix} LVFE (in/out Hunan/Jiangsu): '
    print(f'{lvfe_str}{mean_in:.4f}, {mean_out:.4f}')

    # 4. Plot low visibility haze event (LVHE) frequency map
    sta0.loc[:, 'data0'] = df_sta.loc[:, '1haze']
    _scatter_sta_pair(
        sta0, str(pathlib.Path(output_dir) / 'sta_1+_haze'), cmap, clevs, cfg)
    mean_in, mean_out = _calc_region_mean(df_sta, '1haze', sta, ('湖南省',))
    print(f'{prefix} LVHE (in/out Hunan): {mean_in:.4f}, {mean_out:.4f}')


def plot_sta_mean_maps(
    sta: pd.DataFrame,
    df_sta: pd.DataFrame,
    output_dir: str,
    cfg: typing.Dict
) -> None:
    """Plot mean visibility spatial map."""
    plot_cfg = cfg['draw']['plot']['sta_mean_maps']
    cmap_name = plot_cfg['cmap']
    vmin = plot_cfg['vmin']
    vmax = plot_cfg['vmax']
    clevs = plot_cfg['clevs']
    cmap0, clevs0 = meb.def_cmap_clevs(
        getattr(meb.cmaps, cmap_name), vmin=vmin, vmax=vmax)
    cmap, clevs = meb.def_cmap_clevs(cmap0, clevs=clevs)
    cols = ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')
    sta0 = sta.loc[:, cols].copy()

    # 1. Plot mean visibility spatial map
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'mean'] / 1000
    _scatter_sta_pair(
        sta0, str(pathlib.Path(output_dir) / 'sta_ob_mean'), cmap, clevs, cfg)
    provinces = ('湖南省', '江苏省', '浙江省')
    mean_in, mean_out = _calc_region_mean(df_sta, 'mean', sta, provinces)
    prefix = '[plot_sta_mean_maps]'
    vis_str = f'{prefix} VIS (in/out Hunan/Jiangsu/Zhejiang): '
    print(f'{vis_str}{mean_in:.4f}, {mean_out:.4f}')

    # 2. Plot LVPE mean visibility map
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'lvpe'] / 1000
    _scatter_sta_pair(
        sta0,
        str(pathlib.Path(output_dir) / 'sta_ob_lvpe_mean'),
        cmap, clevs, cfg
    )

    # 3. Plot LVFE mean visibility map
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'lvfe'] / 1000
    _scatter_sta_pair(
        sta0,
        str(pathlib.Path(output_dir) / 'sta_ob_lvfe_mean'),
        cmap, clevs, cfg
    )

    # 4. Plot LVHE mean visibility map
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'lvhe'] / 1000
    _scatter_sta_pair(
        sta0,
        str(pathlib.Path(output_dir) / 'sta_ob_lvhe_mean'),
        cmap, clevs, cfg
    )
