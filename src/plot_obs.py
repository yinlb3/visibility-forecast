# -*- coding: utf-8 -*-
"""
Observation data visualization module.

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""

import gc
import pathlib
import typing

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from meteva import base as meb

from src import utils


def plot_obs_pies(
    vis_grade: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    thres: typing.Tuple[float, ...],
    output_dir: str,
    cfg: typing.Optional[typing.Dict[str, typing.Any]] = None
) -> None:
    """
    Plot weather type proportion pie charts by grade.

    Args:
        vis_grade (np.ndarray): Visibility grade array.
        pre (np.ndarray): Precipitation array.
        rhu (np.ndarray): Relative humidity array.
        thres (typing.Tuple[float, ...]): Visibility grade thresholds.
        output_dir (str): Output directory path.
        cfg (Optional[Dict]): Configuration dict for output formats.
    """
    # 1. Loop through each visibility grade
    for i in range(len(thres)):
        # 1.1 Create mask for current grade with valid pre/rhu
        index = (vis_grade == i + 1) & ~np.isnan(pre) & ~np.isnan(rhu)
        # 1.2 Count samples by weather type
        # Classify: precip > 0; fog = no precip + RH>=80%;
        # haze = no precip + RH<80%
        a = np.sum((pre > 0) & index)              # Precipitation events
        b = np.sum((pre == 0) & (rhu >= 80) & index)  # Fog events
        c = np.sum((pre == 0) & (rhu < 80) & index)   # Haze events
        abc = np.array([a, b, c])
        pct = abc / np.sum(abc) * 100
        prefix = '[plot_obs_pies]'
        pct_str = f'[{pct[0]:.2f}, {pct[1]:.2f}, {pct[2]:.2f}]'
        print(f'{prefix} Grade {i+1} type %: {pct_str}')
        # 1.3 Plot and save pie chart
        fig, ax = plt.subplots(figsize=(4, 4), dpi=800)
        labels = ('降水', '雾', '霾')
        ax.pie(
            x=(a, b, c), labels=labels, autopct='%.2f%%', startangle=90
        )
        base_path = pathlib.Path(output_dir) / f'pie_{i + 1}'
        if cfg:
            utils.save_figure(fig, base_path, cfg, bbox_inches='tight', dpi=800)
        else:
            # Fallback: save png and pdf
            fig.savefig(str(base_path.with_suffix('.png')),
                        bbox_inches='tight', dpi=800)
            fig.savefig(str(base_path.with_suffix('.pdf')),
                        bbox_inches='tight', dpi=800)
        plt.close(fig)
        del fig, ax
        gc.collect()


def plot_obs_violin_box(
    vis: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    output_dir: str,
    cfg: typing.Optional[typing.Dict[str, typing.Any]] = None
) -> None:
    """
    Plot violin/box for vis < 500m events.

    Args:
        vis (np.ndarray): Visibility array.
        pre (np.ndarray): Precipitation array.
        rhu (np.ndarray): Relative humidity array.
        output_dir (str): Output directory path.
    """
    # 1. Create masks for vis < 500m (grade 4+) by weather type
    # mask_lv4plus: Mask for visibility grade 4+ (vis < 500m, 4级及以上低能见度)
    mask_lv4plus = vis < 500
    # mask_lv4plus_pre: Mask for grade 4+ with precipitation (降水型4级及以上低能见度)
    mask_lv4plus_pre = (vis < 500) & (pre > 0)
    # mask_lv4plus_fog: Mask for grade 4+ fog events (雾型4级及以上低能见度)
    mask_lv4plus_fog = (vis < 500) & (pre == 0) & (rhu >= 80)
    # mask_lv4plus_haze: Mask for grade 4+ haze events (霾型4级及以上低能见度)
    mask_lv4plus_haze = (vis < 500) & (pre == 0) & (rhu < 80)
    # 2. Prepare data dict (convert to km)
    data = {
        'Overall': vis[mask_lv4plus] / 1000,
        'Precip': vis[mask_lv4plus_pre] / 1000,
        'Fog': vis[mask_lv4plus_fog] / 1000,
        'Haze': vis[mask_lv4plus_haze] / 1000
    }

    fig, ax = plt.subplots(figsize=(5, 5), dpi=800)
    sns.violinplot(data=data, color='skyblue')  # Plot violin
    ax.set_xlabel('低能见度事件类型')
    ax.set_ylabel('能见度 (km) ')
    base_path = pathlib.Path(output_dir) / 'violinplot_ob'
    if cfg:
        utils.save_figure(fig, base_path, cfg, bbox_inches='tight', dpi=800)
    else:
        fig.savefig(str(base_path.with_suffix('.png')),
                    bbox_inches='tight', dpi=800)
        fig.savefig(str(base_path.with_suffix('.pdf')),
                    bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()

    fig, ax = plt.subplots(figsize=(5, 5), dpi=800)
    sns.boxplot(
        data=data,
        color='black',
        width=0.1,
        showfliers=False,
        medianprops={'color': 'white'}
    )
    ax.set_xlabel('低能见度事件类型')
    ax.set_ylabel('能见度 (km) ')
    base_path = pathlib.Path(output_dir) / 'boxplot_ob'
    if cfg:
        utils.save_figure(fig, base_path, cfg, bbox_inches='tight', dpi=800)
    else:
        fig.savefig(str(base_path.with_suffix('.png')),
                    bbox_inches='tight', dpi=800)
        fig.savefig(str(base_path.with_suffix('.pdf')),
                    bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()


def _plot_stack_bar(
    ax,
    x: np.ndarray,
    pre: np.ndarray,
    fog: np.ndarray,
    haze: np.ndarray,
    color: bool
) -> None:
    """
    Helper: Plot color or B&W stacked bars.

    Args:
        ax: Matplotlib axis object.
        x (np.ndarray): X positions.
        pre (np.ndarray): Precipitation values.
        fog (np.ndarray): Fog values.
        haze (np.ndarray): Haze values.
        color (bool): Use color or B&W.
    """
    # Use tab10 color scheme for color version, B&W for print version
    if color:
        # Blue for precip, orange for fog, green for haze (tab10 palette)
        c_pre = (31 / 255, 119 / 255, 180 / 255)
        c_fog = (255 / 255, 127 / 255, 14 / 255)
        c_haze = (44 / 255, 160 / 255, 44 / 255)
        # Stack bars: precip at bottom, fog in middle, haze on top
        ax.bar(x=x, height=pre, width=0.4, color=c_pre, label='Precip')
        ax.bar(
            x=x, height=fog, bottom=pre, width=0.4, color=c_fog, label='Fog'
        )
        ax.bar(x=x, height=haze, bottom=pre + fog, width=0.4, color=c_haze,
               label='Haze')
    else:
        # B&W version: solid black for precip, hatched for fog, white for haze
        ax.bar(x=x, height=pre, width=0.4, color='black',
               edgecolor='black', label='Precip')
        ax.bar(x=x, height=fog, bottom=pre, width=0.4, color='white',
               edgecolor='black', hatch='///', label='Fog')
        ax.bar(x=x, height=haze, bottom=pre + fog, width=0.4,
               color='white', edgecolor='black', label='Haze')


def plot_monthly_bars(
    df_month: pd.DataFrame,
    output_dir: str,
    cfg: typing.Optional[typing.Dict[str, typing.Any]] = None
) -> None:
    """
    Plot monthly prob stacked bars (color and B&W).

    Args:
        df_month (pd.DataFrame): Monthly statistics DataFrame.
        output_dir (str): Output directory path.
    """
    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    _plot_stack_bar(
        ax,
        np.linspace(start=1, stop=12, num=12),
        df_month.loc[:, '1pre'].values,
        df_month.loc[:, '1fog'].values,
        df_month.loc[:, '1haze'].values,
        color=True
    )
    ax.set_xlim((0, 13))
    ax.set_xticks(range(1, 13), [f'{x}月' for x in range(1, 13)])
    ax.set_ylim((0, 1.0))
    ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
    ax.set_xlabel('月份')
    ax.set_ylabel('概率')
    ax.legend()
    base_path = pathlib.Path(output_dir) / 'month_1+'
    if cfg:
        utils.save_figure(fig, base_path, cfg, bbox_inches='tight', dpi=800)
    else:
        fig.savefig(str(base_path.with_suffix('.png')),
                    bbox_inches='tight', dpi=800)
        fig.savefig(str(base_path.with_suffix('.pdf')),
                    bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()

    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    _plot_stack_bar(
        ax,
        np.linspace(start=1, stop=12, num=12),
        df_month.loc[:, '1pre'].values,
        df_month.loc[:, '1fog'].values,
        df_month.loc[:, '1haze'].values,
        color=False
    )
    ax.set_xlim((0, 13))
    ax.set_xticks(range(1, 13), [f'{x}月' for x in range(1, 13)])
    ax.set_ylim((0, 1.0))
    ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
    ax.set_xlabel('月份')
    ax.set_ylabel('概率')
    ax.legend()
    base_path = pathlib.Path(output_dir) / 'month_1+_bw'
    if cfg:
        utils.save_figure(fig, base_path, cfg, bbox_inches='tight', dpi=800)
    else:
        fig.savefig(str(base_path.with_suffix('.png')),
                    bbox_inches='tight', dpi=800)
        fig.savefig(str(base_path.with_suffix('.pdf')),
                    bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()

    lve = np.array(df_month.loc[:, '1haze'] + df_month.loc[:, '1pre'] +
                   df_month.loc[:, '1fog'])
    prefix = '[plot_monthly_bars]'
    max_month, max_val = np.argmax(lve) + 1, np.max(lve)
    min_month, min_val = np.argmin(lve) + 1, np.min(lve)
    print(f'{prefix} LVE max month: m={max_month}, value={max_val:.4f}')
    print(f'{prefix} LVE min month: m={min_month}, value={min_val:.4f}')


def plot_monthly_violins(
    vis: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    month_ind: np.ndarray,
    output_dir: str,
    cfg: typing.Optional[typing.Dict[str, typing.Any]] = None
) -> None:
    """
    Plot monthly visibility violin by weather type.

    Args:
        vis (np.ndarray): Visibility array.
        pre (np.ndarray): Precipitation array.
        rhu (np.ndarray): Relative humidity array.
        month_ind (np.ndarray): Month index array.
        output_dir (str): Output directory path.
    """
    conditions = [
        (vis < 10000, 'boxplot_ob_month'),
        ((vis < 10000) & (pre > 0), 'boxplot_ob_month_pre'),
        ((vis < 10000) & (pre == 0) & (rhu >= 80), 'boxplot_ob_month_fog'),
        ((vis < 10000) & (pre == 0) & (rhu < 80), 'boxplot_ob_month_haze')
    ]
    for cond, fname in conditions:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
        d = dict()
        for i in range(12):
            vis_ob = vis[month_ind == i + 1, :]
            idx = cond[month_ind == i + 1, :]
            d[f'{i + 1}'] = vis_ob[idx] / 1000
        sns.violinplot(data=d, color='skyblue')
        ax.set_xlabel('月份')
        ax.set_ylabel('能见度(km)')
        base_path = pathlib.Path(output_dir) / fname
        if cfg:
            utils.save_figure(fig, base_path, cfg, bbox_inches='tight', dpi=800)
        else:
            fig.savefig(str(base_path.with_suffix('.png')),
                        bbox_inches='tight', dpi=800)
            fig.savefig(str(base_path.with_suffix('.pdf')),
                        bbox_inches='tight', dpi=800)
        if fname == 'boxplot_ob_month_haze':
            plt.close('all')
        else:
            plt.close(fig)
        del fig, ax
        gc.collect()


def plot_hourly_bars(
    df_hour: pd.DataFrame,
    output_dir: str,
    cfg: typing.Optional[typing.Dict[str, typing.Any]] = None
) -> None:
    """
    Plot hourly prob stacked bars (color and B&W).

    Args:
        df_hour (pd.DataFrame): Hourly statistics DataFrame.
        output_dir (str): Output directory path.
    """
    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    _plot_stack_bar(
        ax,
        np.linspace(start=0, stop=23, num=24),
        df_hour.loc[:, '1pre'].values,
        df_hour.loc[:, '1fog'].values,
        df_hour.loc[:, '1haze'].values,
        color=True
    )
    ax.set_xlim((-1, 24))
    ax.set_xticks(range(24), [f'{x:02d}:00' for x in range(24)])
    ax.set_ylim((0, 1.0))
    ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
    ax.set_xlabel('时间 (UTC) ')
    ax.set_ylabel('概率')
    ax.legend()
    base_path = pathlib.Path(output_dir) / 'hour_1+'
    if cfg:
        utils.save_figure(fig, base_path, cfg, bbox_inches='tight', dpi=800)
    else:
        fig.savefig(str(base_path.with_suffix('.png')),
                    bbox_inches='tight', dpi=800)
        fig.savefig(str(base_path.with_suffix('.pdf')),
                    bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()

    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    _plot_stack_bar(
        ax,
        np.linspace(start=0, stop=23, num=24),
        df_hour.loc[:, '1pre'].values,
        df_hour.loc[:, '1fog'].values,
        df_hour.loc[:, '1haze'].values,
        color=False
    )
    ax.set_xlim((-1, 24))
    ax.set_xticks(range(24), [f'{x:02d}:00' for x in range(24)])
    ax.set_ylim((0, 1.0))
    ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
    ax.set_xlabel('时间 (UTC) ')
    ax.set_ylabel('概率')
    ax.legend()
    base_path = pathlib.Path(output_dir) / 'hour_1+_bw'
    if cfg:
        utils.save_figure(fig, base_path, cfg, bbox_inches='tight', dpi=800)
    else:
        fig.savefig(str(base_path.with_suffix('.png')),
                    bbox_inches='tight', dpi=800)
        fig.savefig(str(base_path.with_suffix('.pdf')),
                    bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()

    lve = np.array(df_hour.loc[:, '1haze'] + df_hour.loc[:, '1pre'] +
                   df_hour.loc[:, '1fog'])
    prefix = '[plot_hourly_bars]'
    print(f'{prefix} LVE max hour: h={np.argmax(lve)}, value={np.max(lve):.4f}')
    print(f'{prefix} LVE min hour: h={np.argmin(lve)}, value={np.min(lve):.4f}')


def plot_hourly_violins(
    vis: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    output_dir: str,
    cfg: typing.Optional[typing.Dict[str, typing.Any]] = None
) -> None:
    """
    Plot hourly visibility violin by weather type.

    Args:
        vis (np.ndarray): Visibility array.
        pre (np.ndarray): Precipitation array.
        rhu (np.ndarray): Relative humidity array.
        output_dir (str): Output directory path.
    """
    conditions = [
        (vis < 10000, 'boxplot_ob_hour'),
        ((vis < 10000) & (pre > 0), 'boxplot_ob_hour_pre'),
        ((vis < 10000) & (pre == 0) & (rhu >= 80), 'boxplot_ob_hour_fog'),
        ((vis < 10000) & (pre == 0) & (rhu < 80), 'boxplot_ob_hour_haze')
    ]
    for cond, fname in conditions:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
        d = dict()
        if fname == 'boxplot_ob_hour':
            for i in range(24):
                idx = cond[i::24, :]
                vis_ob = vis[i::24, :]
                d[f'{i}:00'] = vis_ob[idx] / 1000
        else:
            for i in range(24):
                vis_ob = vis[i::24, :]
                idx = cond[i::24, :]
                d[f'{i}:00'] = vis_ob[idx] / 1000
        sns.violinplot(data=d, color='skyblue')
        ax.set_xlabel('小时(UTC)')
        ax.set_ylabel('能见度(km)')
        base_path = pathlib.Path(output_dir) / fname
        if cfg:
            utils.save_figure(fig, base_path, cfg, bbox_inches='tight', dpi=800)
        else:
            fig.savefig(str(base_path.with_suffix('.png')),
                        bbox_inches='tight', dpi=800)
            fig.savefig(str(base_path.with_suffix('.pdf')),
                        bbox_inches='tight', dpi=800)
        plt.close(fig)
        del fig, ax
        gc.collect()


def _scatter_sta_pair(sta0: pd.DataFrame, save_path: str, cmap, clevs) -> None:
    """
    Helper: Plot station scatter (png+pdf).

    Args:
        sta0 (pd.DataFrame): Station DataFrame.
        save_path (str): Save path prefix.
        cmap: Color map.
        clevs: Color levels.
    """
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='max',
        title=[''],
        save_path=str(pathlib.Path(save_path).with_suffix('.png')),
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
        save_path=str(pathlib.Path(save_path).with_suffix('.pdf')),
        dpi=800
    )


def _calc_region_mean(
    df_sta: pd.DataFrame,
    col: str,
    sta: pd.DataFrame,
    provinces: tuple
) -> typing.Tuple[float, float]:
    """
    Helper: Calc mean inside/outside specified provinces.

    Args:
        df_sta (pd.DataFrame): Station statistics DataFrame.
        col (str): Column name to calc mean.
        sta (pd.DataFrame): Station info DataFrame.
        provinces (tuple): Provinces to calc mean inside.

    Returns:
        Tuple[float, float]: (mean_in, mean_out).
    """
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
    output_dir: str
) -> None:
    """
    Plot low visibility freq spatial map.

    Args:
        sta (pd.DataFrame): Station info DataFrame.
        df_sta (pd.DataFrame): Station statistics DataFrame.
        output_dir (str): Output directory path.
    """
    prefix = '[plot_sta_frequency_maps]'
    cols = ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')
    # LVE
    cmap0, clevs0 = meb.def_cmap_clevs(meb.cmaps.ts, vmin=0, vmax=0.9)
    clevs_lve = [0, 0.09, 0.18, 0.27, 0.36, 0.45,
                 0.54, 0.63, 0.72, 0.81, 0.9]
    cmap, clevs = meb.def_cmap_clevs(cmap0, clevs=clevs_lve)
    sta0 = sta.loc[:, cols]
    sta0.loc[:, 'data0'] = df_sta.loc[:, '1']
    _scatter_sta_pair(
        sta0, str(pathlib.Path(output_dir) / 'sta_1+'), cmap, clevs)
    mean_in, mean_out = _calc_region_mean(df_sta, '1', sta, ('湖南省',))
    print(f'{prefix} LVE (in/out Hunan): {mean_in:.4f}, {mean_out:.4f}')

    # LVPE
    cmap0, clevs0 = meb.def_cmap_clevs(meb.cmaps.ts, vmin=0, vmax=0.2)
    clevs_lvpe = [0, 0.02, 0.04, 0.06, 0.08, 0.1,
                  0.12, 0.14, 0.16, 0.18, 0.2]
    cmap, clevs = meb.def_cmap_clevs(cmap0, clevs=clevs_lvpe)
    sta0.loc[:, 'data0'] = df_sta.loc[:, '1pre']
    _scatter_sta_pair(
        sta0, str(pathlib.Path(output_dir) / 'sta_1+_pre'), cmap, clevs)
    prov = ('湖南省', '江西省', '浙江省')
    mean_in, mean_out = _calc_region_mean(df_sta, '1pre', sta, prov)
    lvpe_str = f'{prefix} LVPE (in/out Hunan/Jiangxi/Zhejiang): '
    lvpe_str += f'{mean_in:.4f}, {mean_out:.4f}'
    print(lvpe_str)

    # LVFE
    cmap0, clevs0 = meb.def_cmap_clevs(meb.cmaps.ts, vmin=0, vmax=40)
    clevs_lvfe = [0, 0.04, 0.08, 0.12, 0.16, 0.2,
                  0.24, 0.28, 0.32, 0.36, 0.4]
    cmap, clevs = meb.def_cmap_clevs(cmap0, clevs=clevs_lvfe)
    sta0.loc[:, 'data0'] = df_sta.loc[:, '1fog']
    _scatter_sta_pair(
        sta0, str(pathlib.Path(output_dir) / 'sta_1+_fog'), cmap, clevs)
    mean_in, mean_out = _calc_region_mean(df_sta, '1fog', sta, ('湖南省', '江苏省'))
    lvfe_str = f'{prefix} LVFE (in/out Hunan/Jiangsu): '
    print(f'{lvfe_str}{mean_in:.4f}, {mean_out:.4f}')

    # LVHE
    sta0.loc[:, 'data0'] = df_sta.loc[:, '1haze']
    _scatter_sta_pair(
        sta0, str(pathlib.Path(output_dir) / 'sta_1+_haze'), cmap, clevs)
    mean_in, mean_out = _calc_region_mean(df_sta, '1haze', sta, ('湖南省',))
    print(f'{prefix} LVHE (in/out Hunan): {mean_in:.4f}, {mean_out:.4f}')


def plot_sta_mean_maps(
    sta: pd.DataFrame,
    df_sta: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Plot mean visibility spatial map.

    Args:
        sta (pd.DataFrame): Station info DataFrame.
        df_sta (pd.DataFrame): Station statistics DataFrame.
        output_dir (str): Output directory path.
    """
    cmap0, clevs0 = meb.def_cmap_clevs(meb.cmaps.vis, vmin=0, vmax=30)
    clevs_vis = [0, 1, 2, 4, 7, 10, 15, 20, 25, 30]
    cmap, clevs = meb.def_cmap_clevs(cmap0, clevs=clevs_vis)
    cols = ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')
    sta0 = sta.loc[:, cols]

    # mean
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'mean'] / 1000
    _scatter_sta_pair(
        sta0, str(pathlib.Path(output_dir) / 'sta_ob_mean'), cmap, clevs)
    provinces = ('湖南省', '江苏省', '浙江省')
    mean_in, mean_out = _calc_region_mean(df_sta, 'mean', sta, provinces)
    prefix = '[plot_sta_mean_maps]'
    vis_str = f'{prefix} VIS (in/out Hunan/Jiangsu/Zhejiang): '
    print(f'{vis_str}{mean_in:.4f}, {mean_out:.4f}')

    # lvpe
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'lvpe'] / 1000
    _scatter_sta_pair(
        sta0, str(pathlib.Path(output_dir) / 'sta_ob_lvpe_mean'), cmap, clevs)

    # lvfe
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'lvfe'] / 1000
    _scatter_sta_pair(
        sta0, str(pathlib.Path(output_dir) / 'sta_ob_lvfe_mean'), cmap, clevs)

    # lvhe
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'lvhe'] / 1000
    _scatter_sta_pair(
        sta0, str(pathlib.Path(output_dir) / 'sta_ob_lvhe_mean'), cmap, clevs)
