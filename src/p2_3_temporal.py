#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 2.3: Temporal distribution feature plotting module.

Founded in 2026-04-14
Modified in 2026-04-14
@author: yinlb
"""

import gc
import pathlib
import typing

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src import utils


def _plot_stack_bar(
    ax,
    x: np.ndarray,
    pre: np.ndarray,
    fog: np.ndarray,
    haze: np.ndarray,
    color: bool,
    bar_width: float = 0.4,
    color_precip: typing.Tuple[float, ...] = (0.122, 0.467, 0.706),
    color_fog: typing.Tuple[float, ...] = (1.0, 0.498, 0.055),
    color_haze: typing.Tuple[float, ...] = (0.173, 0.627, 0.173),
    edgecolor: str = 'black',
    hatch: str = '///'
) -> None:
    """Helper: Plot color or B&W stacked bars."""
    if color:
        ax.bar(x=x, height=pre, width=bar_width, color=color_precip, label='Precip')
        ax.bar(x=x, height=fog, bottom=pre, width=bar_width, color=color_fog, label='Fog')
        ax.bar(x=x, height=haze, bottom=pre + fog, width=bar_width, color=color_haze, label='Haze')
    else:
        ax.bar(x=x, height=pre, width=bar_width, color='black', edgecolor=edgecolor, label='Precip')
        ax.bar(x=x, height=fog, bottom=pre, width=bar_width, color='white', edgecolor=edgecolor, hatch=hatch, label='Fog')
        ax.bar(x=x, height=haze, bottom=pre + fog, width=bar_width, color='white', edgecolor=edgecolor, label='Haze')


def plot_monthly_bars(
    df_month: pd.DataFrame,
    output_dir: str,
    cfg: typing.Dict
) -> None:
    """Plot monthly prob stacked bars (color and B&W)."""
    plot_cfg = cfg['plot']['monthly_bars']
    figsize = plot_cfg['figsize']
    dpi = plot_cfg['dpi']
    bar_width = plot_cfg['bar_width']
    color_precip = tuple(plot_cfg['color_precip'])
    color_fog = tuple(plot_cfg['color_fog'])
    color_haze = tuple(plot_cfg['color_haze'])
    xlim = plot_cfg['xlim']
    ylim = plot_cfg['ylim']
    yticks = plot_cfg['yticks']

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _plot_stack_bar(
        ax, np.linspace(start=1, stop=12, num=12),
        df_month.loc[:, '1pre'].values,
        df_month.loc[:, '1fog'].values,
        df_month.loc[:, '1haze'].values,
        color=True, bar_width=bar_width,
        color_precip=color_precip, color_fog=color_fog, color_haze=color_haze
    )
    ax.set_xlim(tuple(xlim))
    ax.set_xticks(range(1, 13), [f'{x}月' for x in range(1, 13)])
    ax.set_ylim(tuple(ylim))
    ax.set_yticks(tuple(yticks))
    ax.set_xlabel('月份')
    ax.set_ylabel('概率')
    ax.legend()
    base_path = pathlib.Path(output_dir) / 'month_1+'
    utils.save_figure(fig, base_path, cfg, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    del fig, ax
    gc.collect()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _plot_stack_bar(
        ax, np.linspace(start=1, stop=12, num=12),
        df_month.loc[:, '1pre'].values,
        df_month.loc[:, '1fog'].values,
        df_month.loc[:, '1haze'].values,
        color=False, bar_width=bar_width
    )
    ax.set_xlim(tuple(xlim))
    ax.set_xticks(range(1, 13), [f'{x}月' for x in range(1, 13)])
    ax.set_ylim(tuple(ylim))
    ax.set_yticks(tuple(yticks))
    ax.set_xlabel('月份')
    ax.set_ylabel('概率')
    ax.legend()
    base_path = pathlib.Path(output_dir) / 'month_1+_bw'
    utils.save_figure(fig, base_path, cfg, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    del fig, ax
    gc.collect()

    lve = np.array(df_month.loc[:, '1haze'] + df_month.loc[:, '1pre'] + df_month.loc[:, '1fog'])
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
    cfg: typing.Dict
) -> None:
    """Plot monthly visibility violin by weather type."""
    plot_cfg = cfg['plot']['monthly_violins']
    figsize = plot_cfg['figsize']
    dpi = plot_cfg['dpi']
    color = plot_cfg['color']
    conditions = [
        (vis < 10000, 'boxplot_ob_month'),
        ((vis < 10000) & (pre > 0), 'boxplot_ob_month_pre'),
        ((vis < 10000) & (pre == 0) & (rhu >= 80), 'boxplot_ob_month_fog'),
        ((vis < 10000) & (pre == 0) & (rhu < 80), 'boxplot_ob_month_haze')
    ]
    for cond, fname in conditions:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        d = dict()
        for i in range(12):
            vis_ob = vis[month_ind == i + 1, :]
            idx = cond[month_ind == i + 1, :]
            d[f'{i + 1}'] = vis_ob[idx] / 1000
        sns.violinplot(data=d, color=color)
        ax.set_xlabel('月份')
        ax.set_ylabel('能见度(km)')
        base_path = pathlib.Path(output_dir) / fname
        utils.save_figure(fig, base_path, cfg, bbox_inches='tight', dpi=dpi)
        if fname == 'boxplot_ob_month_haze':
            plt.close('all')
        else:
            plt.close(fig)
        del fig, ax
        gc.collect()


def plot_hourly_bars(
    df_hour: pd.DataFrame,
    output_dir: str,
    cfg: typing.Dict
) -> None:
    """Plot hourly prob stacked bars (color and B&W)."""
    plot_cfg = cfg['plot']['hourly_bars']
    figsize = plot_cfg['figsize']
    dpi = plot_cfg['dpi']
    bar_width = plot_cfg['bar_width']
    color_precip = tuple(plot_cfg['color_precip'])
    color_fog = tuple(plot_cfg['color_fog'])
    color_haze = tuple(plot_cfg['color_haze'])
    xlim = plot_cfg['xlim']
    ylim = plot_cfg['ylim']
    yticks = plot_cfg['yticks']

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _plot_stack_bar(
        ax, np.linspace(start=0, stop=23, num=24),
        df_hour.loc[:, '1pre'].values,
        df_hour.loc[:, '1fog'].values,
        df_hour.loc[:, '1haze'].values,
        color=True, bar_width=bar_width,
        color_precip=color_precip, color_fog=color_fog, color_haze=color_haze
    )
    ax.set_xlim(tuple(xlim))
    ax.set_xticks(range(24), [f'{x:02d}:00' for x in range(24)])
    ax.set_ylim(tuple(ylim))
    ax.set_yticks(tuple(yticks))
    ax.set_xlabel('时间 (UTC) ')
    ax.set_ylabel('概率')
    ax.legend()
    base_path = pathlib.Path(output_dir) / 'hour_1+'
    utils.save_figure(fig, base_path, cfg, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    del fig, ax
    gc.collect()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _plot_stack_bar(
        ax, np.linspace(start=0, stop=23, num=24),
        df_hour.loc[:, '1pre'].values,
        df_hour.loc[:, '1fog'].values,
        df_hour.loc[:, '1haze'].values,
        color=False, bar_width=bar_width
    )
    ax.set_xlim(tuple(xlim))
    ax.set_xticks(range(24), [f'{x:02d}:00' for x in range(24)])
    ax.set_ylim(tuple(ylim))
    ax.set_yticks(tuple(yticks))
    ax.set_xlabel('时间 (UTC) ')
    ax.set_ylabel('概率')
    ax.legend()
    base_path = pathlib.Path(output_dir) / 'hour_1+_bw'
    utils.save_figure(fig, base_path, cfg, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    del fig, ax
    gc.collect()

    lve = np.array(df_hour.loc[:, '1haze'] + df_hour.loc[:, '1pre'] + df_hour.loc[:, '1fog'])
    prefix = '[plot_hourly_bars]'
    print(f'{prefix} LVE max hour: h={np.argmax(lve)}, value={np.max(lve):.4f}')
    print(f'{prefix} LVE min hour: h={np.argmin(lve)}, value={np.min(lve):.4f}')


def plot_hourly_violins(
    vis: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    output_dir: str,
    cfg: typing.Dict
) -> None:
    """Plot hourly visibility violin by weather type."""
    plot_cfg = cfg['plot']['hourly_violins']
    figsize = plot_cfg['figsize']
    dpi = plot_cfg['dpi']
    color = plot_cfg['color']
    conditions = [
        (vis < 10000, 'boxplot_ob_hour'),
        ((vis < 10000) & (pre > 0), 'boxplot_ob_hour_pre'),
        ((vis < 10000) & (pre == 0) & (rhu >= 80), 'boxplot_ob_hour_fog'),
        ((vis < 10000) & (pre == 0) & (rhu < 80), 'boxplot_ob_hour_haze')
    ]
    for cond, fname in conditions:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
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
        sns.violinplot(data=d, color=color)
        ax.set_xlabel('小时(UTC)')
        ax.set_ylabel('能见度(km)')
        base_path = pathlib.Path(output_dir) / fname
        utils.save_figure(fig, base_path, cfg, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        del fig, ax
        gc.collect()
