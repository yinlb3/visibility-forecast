#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 3.4: Ablation experiment result output module.

Founded in 2026-04-14
Modified in 2026-04-15
@author: yinlb
"""

import gc
import os
import pathlib
import typing

import numpy as np
from matplotlib import pyplot as plt

from src import utils


def plot_vis_cdf(
    vis_ob: np.ndarray,
    cma_sh_warr: np.ndarray,
    pred_pdfm_tle0: np.ndarray,
    output_dir: str,
    cfg: typing.Dict
) -> None:
    """Calc and plot visibility CDF."""
    plot_cfg = cfg['plot']['vis_cdf']
    vis_values = np.arange(-1, 30001, 1, dtype=np.float32)
    cdf = np.zeros((3, vis_values.size), dtype=np.float32)

    # Filter to valid triple-matched samples
    mask_ob = ~np.isnan(vis_ob)
    mask_nwp = ~np.isnan(cma_sh_warr)
    mask_pr = ~np.isnan(pred_pdfm_tle0)
    index = mask_ob & mask_nwp & mask_pr
    ob = vis_ob[index]
    nwp = cma_sh_warr[index]
    pr = pred_pdfm_tle0[index]

    # Build CDF for each dataset: sort → searchsorted → normalize
    ob_sorted = np.sort(ob)
    nwp_sorted = np.sort(nwp)
    pr_sorted = np.sort(pr)
    cdf[0, :] = np.searchsorted(ob_sorted, vis_values, side='right')
    cdf[0, :] /= ob.size
    cdf[1, :] = np.searchsorted(nwp_sorted, vis_values, side='right')
    cdf[1, :] /= nwp.size
    cdf[2, :] = np.searchsorted(pr_sorted, vis_values, side='right')
    cdf[2, :] /= pr.size

    figsize = plot_cfg['figsize']
    dpi = plot_cfg['dpi']
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    color_ob = plot_cfg['color_ob']
    color_cma = plot_cfg['color_cma']
    color_pdfm = plot_cfg['color_pdfm']
    ax.plot(vis_values / 1000, cdf[0, :], '-', c=color_ob, label='实况')
    ax.plot(
        vis_values / 1000, cdf[1, :],
        '-', c=color_cma, label='CMA-SH3-WARR'
    )
    ax.plot(vis_values / 1000, cdf[2, :], '-', c=color_pdfm, label='PDFM-TLE')
    xlim = plot_cfg['xlim']
    xticks = plot_cfg['xticks']
    ylim = plot_cfg['ylim']
    yticks = plot_cfg['yticks']
    ax.set_xlim(tuple(xlim))
    ax.set_xticks(tuple(xticks))
    ax.set_ylim(tuple(ylim))
    ax.set_yticks(tuple(yticks))
    ax.set_xlabel('能见度 (km) ')
    ax.set_ylabel('累积概率')
    ax.legend()
    utils.save_figure(
        fig, pathlib.Path(output_dir) / 'vis_cdf',
        cfg, bbox_inches='tight', dpi=dpi
    )
    plt.close(fig)
    del fig, ax
    gc.collect()


def plot_weather_type_eval_bw(
    qem: np.ndarray,
    filename: str,
    max_y: float,
    output_dir: str,
    cfg: typing.Dict,
    unit: str = ''
) -> None:
    """Plot weather-type metric comparison bars (B&W style)."""
    plot_cfg = cfg['plot']['weather_type_eval_bw']
    os.makedirs(name=output_dir, exist_ok=True)
    figsize = plot_cfg['figsize']
    dpi = plot_cfg['dpi']
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x_pos = np.linspace(start=1, stop=6, num=6)
    bar_width = plot_cfg['bar_width']
    color_precip = plot_cfg['color_precip']
    color_fog = plot_cfg['color_fog']
    color_haze = plot_cfg['color_haze']
    edgecolor = plot_cfg['edgecolor']
    hatch = plot_cfg['hatch']
    # Plot grouped bars with offset x-positions for 3 weather types
    bars1 = ax.bar(
        x=x_pos - 0.2, height=qem[:, 0] / max_y, width=bar_width,
        color=color_precip, edgecolor=edgecolor, label='降水类'
    )
    bars2 = ax.bar(
        x=x_pos, height=qem[:, 1] / max_y, width=bar_width,
        color=color_fog, edgecolor=edgecolor, hatch=hatch, label='雾类'
    )
    bars3 = ax.bar(
        x=x_pos + 0.2, height=qem[:, 2] / max_y, width=bar_width,
        color=color_haze, edgecolor=edgecolor, label='霾类'
    )
    # Add value labels on top of each bar
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            height = 0 if height < 0 else height
            val = height * max_y
            label = f'{val:.0f}' if unit else f'{val:.2f}'
            ax.text(
                x=bar.get_x() + bar.get_width() / 2.,
                y=height + 0.02, s=label,
                ha='center', va='bottom', fontsize=7
            )
    xlim = plot_cfg['xlim']
    ylim = plot_cfg['ylim']
    ax.set_xlim(tuple(xlim))
    ax.set_xticks(
        ticks=range(1, 7),
        labels=['CMA-SH-WARR', '试验一', '试验二', '试验三', '试验四', '试验五']
    )
    ax.set_ylim(tuple(ylim))
    ax.set_yticks(
        ticks=np.linspace(start=0, stop=1, num=6),
        labels=[
            f'{val * max_y:g}{unit}'
            for val in np.linspace(start=0, stop=1, num=6)
        ]
    )
    ax.legend()
    utils.save_figure(
        fig, pathlib.Path(output_dir) / filename,
        cfg, bbox_inches='tight', dpi=dpi
    )
    plt.close(fig)
    gc.collect()
