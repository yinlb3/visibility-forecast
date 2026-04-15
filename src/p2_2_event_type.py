#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 2.2: Event type proportion output and plotting module.

Founded in 2026-04-14
Modified in 2026-04-14
@author: yinlb
"""

import gc
import pathlib
import typing

import numpy as np
from matplotlib import pyplot as plt

from src import utils


def plot_obs_pies(
    vis_grade: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    thres: typing.Tuple[float, ...],
    output_dir: str,
    cfg: typing.Dict
) -> None:
    """
    Plot weather type proportion pie charts by grade.
    """
    plot_cfg = cfg['plot']['obs_pies']
    figsize = plot_cfg['figsize']
    dpi = plot_cfg['dpi']
    for i in range(len(thres)):
        index = (vis_grade == i + 1) & ~np.isnan(pre) & ~np.isnan(rhu)
        a = np.sum((pre > 0) & index)
        b = np.sum((pre == 0) & (rhu >= 80) & index)
        c = np.sum((pre == 0) & (rhu < 80) & index)
        abc = np.array([a, b, c])
        pct = abc / np.sum(abc) * 100
        prefix = '[plot_obs_pies]'
        pct_str = f'[{pct[0]:.2f}, {pct[1]:.2f}, {pct[2]:.2f}]'
        print(f'{prefix} Grade {i+1} type %: {pct_str}')
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        labels = ('降水', '雾', '霾')
        ax.pie(x=(a, b, c), labels=labels, autopct='%.2f%%', startangle=90)
        base_path = pathlib.Path(output_dir) / f'pie_{i + 1}'
        utils.save_figure(fig, base_path, cfg, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        del fig, ax
        gc.collect()


def plot_obs_violin_box(
    vis: np.ndarray,
    pre: np.ndarray,
    rhu: np.ndarray,
    output_dir: str,
    cfg: typing.Dict
) -> None:
    """
    Plot violin/box for vis < 500m events.
    """
    import seaborn as sns

    plot_cfg = cfg['plot']['obs_violin_box']
    mask_lv4plus = vis < 500
    mask_lv4plus_pre = (vis < 500) & (pre > 0)
    mask_lv4plus_fog = (vis < 500) & (pre == 0) & (rhu >= 80)
    mask_lv4plus_haze = (vis < 500) & (pre == 0) & (rhu < 80)
    data = {
        'Overall': vis[mask_lv4plus] / 1000,
        'Precip': vis[mask_lv4plus_pre] / 1000,
        'Fog': vis[mask_lv4plus_fog] / 1000,
        'Haze': vis[mask_lv4plus_haze] / 1000
    }

    violin_figsize = plot_cfg['violin_figsize']
    violin_color = plot_cfg['violin_color']
    dpi = plot_cfg['dpi']
    fig, ax = plt.subplots(figsize=violin_figsize, dpi=dpi)
    sns.violinplot(data=data, color=violin_color)
    ax.set_xlabel('低能见度事件类型')
    ax.set_ylabel('能见度 (km) ')
    base_path = pathlib.Path(output_dir) / 'violinplot_ob'
    utils.save_figure(fig, base_path, cfg, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    del fig, ax
    gc.collect()

    box_figsize = plot_cfg['box_figsize']
    box_color = plot_cfg['box_color']
    box_width = plot_cfg['box_width']
    fig, ax = plt.subplots(figsize=box_figsize, dpi=dpi)
    sns.boxplot(
        data=data, color=box_color, width=box_width,
        showfliers=False, medianprops={'color': 'white'}
    )
    ax.set_xlabel('低能见度事件类型')
    ax.set_ylabel('能见度 (km) ')
    base_path = pathlib.Path(output_dir) / 'boxplot_ob'
    utils.save_figure(fig, base_path, cfg, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    del fig, ax
    gc.collect()
