# -*- coding: utf-8 -*-
"""
观测数据可视化模块.

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""

from typing import Tuple

import gc
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from meteva import base as meb


def plot_obs_pies(
    vis_grade: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    thres: Tuple[float, ...],
    output_dir: str
) -> None:
    """绘制各级低能见度事件的天气类型占比饼图."""
    for i in range(len(thres)):
        index = (vis_grade == i + 1) & ~np.isnan(pre) & ~np.isnan(rhu)
        a = np.sum((pre > 0) & index)
        b = np.sum((pre == 0) & (rhu >= 80) & index)
        c = np.sum((pre == 0) & (rhu < 80) & index)
        abc = np.array([a, b, c])
        print(abc / np.sum(abc) * 100)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=800)
        ax.pie(x=(a, b, c), labels=('降水', '雾', '霾'), autopct='%.2f%%', startangle=90)
        fig.savefig(fname=rf'{output_dir}\pie_{i + 1}.png', bbox_inches='tight', dpi=800)
        fig.savefig(fname=rf'{output_dir}\pie_{i + 1}.pdf', bbox_inches='tight', dpi=800)
        plt.close(fig)
        del fig, ax
        gc.collect()


def plot_obs_violin_box(
    vis: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    output_dir: str
) -> None:
    """绘制低能见度事件(vis < 500m)的小提琴图与箱线图."""
    index0 = vis < 500
    index1 = (vis < 500) & (pre > 0)
    index2 = (vis < 500) & (pre == 0) & (rhu >= 80)
    index3 = (vis < 500) & (pre == 0) & (rhu < 80)
    data = {
        '整体': vis[index0] / 1000,
        '降水类': vis[index1] / 1000,
        '雾类': vis[index2] / 1000,
        '霾类': vis[index3] / 1000
    }

    fig, ax = plt.subplots(figsize=(5, 5), dpi=800)
    sns.violinplot(data=data, color='skyblue')
    ax.set_xlabel('低能见度事件类型')
    ax.set_ylabel('能见度 (km) ')
    plt.savefig(rf'{output_dir}\violinplot_ob.png', bbox_inches='tight', dpi=800)
    plt.savefig(rf'{output_dir}\violinplot_ob.pdf', bbox_inches='tight', dpi=800)
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
    plt.savefig(rf'{output_dir}\boxplot_ob.png', bbox_inches='tight', dpi=800)
    plt.savefig(rf'{output_dir}\boxplot_ob.pdf', bbox_inches='tight', dpi=800)
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
    """辅助函数:绘制彩色或黑白堆叠柱状图."""
    if color:
        ax.bar(x=x, height=pre, width=0.4, color=(31 / 255, 119 / 255, 180 / 255), label='降水类')
        ax.bar(x=x, height=fog, bottom=pre, width=0.4, color=(255 / 255, 127 / 255, 14 / 255), label='雾类')
        ax.bar(x=x, height=haze, bottom=pre + fog, width=0.4, color=(44 / 255, 160 / 255, 44 / 255), label='霾类')
    else:
        ax.bar(x=x, height=pre, width=0.4, color='black', edgecolor='black', label='降水类')
        ax.bar(x=x, height=fog, bottom=pre, width=0.4, color='white', edgecolor='black', hatch='///', label='雾类')
        ax.bar(x=x, height=haze, bottom=pre + fog, width=0.4, color='white', edgecolor='black', label='霾类')


def plot_monthly_bars(
    df_month: pd.DataFrame,
    output_dir: str
) -> None:
    """绘制月份概率堆叠柱状图(彩色版与黑白版)."""
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
    fig.savefig(fname=rf'{output_dir}\month_1+.png', bbox_inches='tight', dpi=800)
    fig.savefig(fname=rf'{output_dir}\month_1+.pdf', bbox_inches='tight', dpi=800)
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
    fig.savefig(fname=rf'{output_dir}\month_1+_bw.png', bbox_inches='tight', dpi=800)
    fig.savefig(fname=rf'{output_dir}\month_1+_bw.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()

    lve = np.array(df_month.loc[:, '1haze'] + df_month.loc[:, '1pre'] + df_month.loc[:, '1fog'])
    print(np.argmax(lve) + 1, np.max(lve))
    print(np.argmin(lve) + 1, np.min(lve))


def plot_monthly_violins(
    vis: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    month_ind: np.ndarray,
    output_dir: str
) -> None:
    """绘制按月份与天气类型分类的能见度小提琴图."""
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
            d[f'{i + 1}月'] = vis_ob[idx] / 1000
        sns.violinplot(data=d, color='skyblue')
        ax.set_xlabel('月份')
        ax.set_ylabel('能见度(km)')
        fig.savefig(rf'{output_dir}\{fname}.png', bbox_inches='tight', dpi=800)
        fig.savefig(rf'{output_dir}\{fname}.pdf', bbox_inches='tight', dpi=800)
        if fname == 'boxplot_ob_month_haze':
            plt.close('all')
        else:
            plt.close(fig)
        del fig, ax
        gc.collect()


def plot_hourly_bars(
    df_hour: pd.DataFrame,
    output_dir: str
) -> None:
    """绘制小时概率堆叠柱状图(彩色版与黑白版)."""
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
    fig.savefig(fname=rf'{output_dir}\hour_1+.png', bbox_inches='tight', dpi=800)
    fig.savefig(fname=rf'{output_dir}\hour_1+.pdf', bbox_inches='tight', dpi=800)
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
    fig.savefig(fname=rf'{output_dir}\hour_1+_bw.png', bbox_inches='tight', dpi=800)
    fig.savefig(fname=rf'{output_dir}\hour_1+_bw.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()

    lve = np.array(df_hour.loc[:, '1haze'] + df_hour.loc[:, '1pre'] + df_hour.loc[:, '1fog'])
    print(np.argmax(lve), np.max(lve))
    print(np.argmin(lve), np.min(lve))


def plot_hourly_violins(
    vis: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    output_dir: str
) -> None:
    """绘制按小时与天气类型分类的能见度小提琴图."""
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
        fig.savefig(rf'{output_dir}\{fname}.png', bbox_inches='tight', dpi=800)
        fig.savefig(rf'{output_dir}\{fname}.pdf', bbox_inches='tight', dpi=800)
        plt.close(fig)
        del fig, ax
        gc.collect()


def _scatter_sta_pair(sta0: pd.DataFrame, save_path: str, cmap, clevs) -> None:
    """辅助函数:绘制站点散点图(png+pdf)."""
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='max',
        title=[''],
        save_path=rf'{save_path}.png',
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
        save_path=rf'{save_path}.pdf',
        dpi=800
    )


def _calc_region_mean(
    df_sta: pd.DataFrame,
    col: str,
    sta: pd.DataFrame,
    provinces: tuple
) -> Tuple[float, float]:
    """辅助函数:计算指定省份内外的平均值."""
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
    """绘制低能见度频率站点空间分布图(LVE/LVPE/LVFE/LVHE)."""
    # LVE
    cmap, clevs = meb.def_cmap_clevs(
        meb.def_cmap_clevs(meb.cmaps.ts, vmin=0, vmax=0.9)[0],
        clevs=[0, 0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72, 0.81, 0.9]
    )
    sta0 = sta.loc[:, ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')]
    sta0.loc[:, 'data0'] = df_sta.loc[:, '1']
    _scatter_sta_pair(sta0, rf'{output_dir}\sta_1+', cmap, clevs)
    mean_in, mean_out = _calc_region_mean(df_sta, '1', sta, ('湖南省',))
    print(f'LVE: {mean_in}, {mean_out}')

    # LVPE
    cmap, clevs = meb.def_cmap_clevs(
        meb.def_cmap_clevs(meb.cmaps.ts, vmin=0, vmax=0.2)[0],
        clevs=[0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    )
    sta0.loc[:, 'data0'] = df_sta.loc[:, '1pre']
    _scatter_sta_pair(sta0, rf'{output_dir}\sta_1+_pre', cmap, clevs)
    mean_in, mean_out = _calc_region_mean(df_sta, '1pre', sta, ('湖南省', '江西省', '浙江省'))
    print(f'LVPE: {mean_in}, {mean_out}')

    # LVFE
    cmap, clevs = meb.def_cmap_clevs(
        meb.def_cmap_clevs(meb.cmaps.ts, vmin=0, vmax=40)[0],
        clevs=[0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4]
    )
    sta0.loc[:, 'data0'] = df_sta.loc[:, '1fog']
    _scatter_sta_pair(sta0, rf'{output_dir}\sta_1+_fog', cmap, clevs)
    mean_in, mean_out = _calc_region_mean(df_sta, '1fog', sta, ('湖南省', '江苏省'))
    print(f'LVFE: {mean_in}, {mean_out}')

    # LVHE
    sta0.loc[:, 'data0'] = df_sta.loc[:, '1haze']
    _scatter_sta_pair(sta0, rf'{output_dir}\sta_1+_haze', cmap, clevs)
    mean_in, mean_out = _calc_region_mean(df_sta, '1haze', sta, ('湖南省',))
    print(f'LVHE: {mean_in}, {mean_out}')


def plot_sta_mean_maps(
    sta: pd.DataFrame,
    df_sta: pd.DataFrame,
    output_dir: str
) -> None:
    """绘制平均能见度站点空间分布图(mean/lvpe/lvfe/lvhe)."""
    cmap, clevs = meb.def_cmap_clevs(
        meb.def_cmap_clevs(meb.cmaps.vis, vmin=0, vmax=30)[0],
        clevs=[0, 1, 2, 4, 7, 10, 15, 20, 25, 30]
    )
    sta0 = sta.loc[:, ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')]

    # mean
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'mean'] / 1000
    _scatter_sta_pair(sta0, rf'{output_dir}\sta_ob_mean', cmap, clevs)
    mean_in, mean_out = _calc_region_mean(df_sta, 'mean', sta, ('湖南省', '江苏省', '浙江省'))
    print(f'VIS: {mean_in}, {mean_out}')

    # lvpe
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'lvpe'] / 1000
    _scatter_sta_pair(sta0, rf'{output_dir}\sta_ob_lvpe_mean', cmap, clevs)

    # lvfe
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'lvfe'] / 1000
    _scatter_sta_pair(sta0, rf'{output_dir}\sta_ob_lvfe_mean', cmap, clevs)

    # lvhe
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'lvhe'] / 1000
    _scatter_sta_pair(sta0, rf'{output_dir}\sta_ob_lvhe_mean', cmap, clevs)


def show_cmap_legend(output_dir: str) -> None:
    """输出色标图例."""
    cmap, clevs = meb.def_cmap_clevs(meb.cmaps.ts)
    meb.tool.color_tools.show_cmap_clev(cmap, clevs, save_path=rf'{output_dir}\000.png')
    meb.tool.color_tools.show_cmap_clev(cmap, clevs, save_path=rf'{output_dir}\000.pdf')
