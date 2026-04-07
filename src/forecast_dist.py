# -*- coding: utf-8 -*-
"""Forecast distribution analysis and plotting module."""
import gc
import os
import pathlib
import typing

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_nwp_his2d(
    vis_ob: np.ndarray,
    cma_sh_warr: np.ndarray,
    output_dir: str
) -> None:
    """
    Plot NWP 2D histogram (obs vs forecast).

    Shows the joint distribution of observed vs forecast visibility,
    helping to identify systematic biases (over/under-forecast).

    Args:
        vis_ob: Observation array.
        cma_sh_warr: Forecast array.
        output_dir: Output directory.
    """
    # 1. Filter valid pairs only (both obs and forecast available)
    index = (~np.isnan(vis_ob)) & (~np.isnan(cma_sh_warr))

    # 2. Transform visibility to log scale for better visualization
    # Log transform compresses high values (0-30km range into manageable scale)
    x_l = np.log(vis_ob[index] / 1000 + 1)
    y_l = np.log(cma_sh_warr[index] / 1000 + 1)

    # 3. Build 2D histogram (30x30 bins)
    # Range 0-30km covers most meteorologically relevant visibility values
    l = np.linspace(start=0, stop=np.log(31), num=31)  # Bin edges (log scale)
    # For tick labels (linear scale)
    l0 = np.linspace(start=0, stop=30, num=31)
    his2d = np.zeros((30, 30), dtype=np.int_)

    # 4. Count samples in each 2D bin
    for i in range(30):
        for j in range(30):
            # Find obs samples in bin i
            index0 = (x_l >= l[i]) & (x_l < l[i + 1])
            # And forecast samples in bin j
            if j < 29:
                index0 &= (y_l >= l[j]) & (y_l < l[j + 1])
            else:
                # Include upper boundary for last bin to catch max values
                index0 &= (y_l >= l[j]) & (y_l <= l[j + 1])
            index0 = index0.astype(np.int_)
            his2d[i, j] = np.sum(index0)

    # 5. Plot 2D histogram as scatter with color-coded density
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    x_r = np.reshape(x_l, -1)
    y_r = np.reshape(y_l, -1)
    c_r = np.reshape(his2d, -1)
    ax.scatter(x_r, y_r, c=c_r, s=10)
    ax.set_xlim((np.log(1), np.log(31)))
    ax.set_xticks(l0, [f'{x_val:.2f}' for x_val in l])
    ax.set_ylim((np.log(1), np.log(31)))
    ax.set_yticks(l0, [f'{y_val:.2f}' for y_val in l])
    ax.set_xlabel('实况 (km)')
    ax.set_ylabel('预报 (km)')
    fig.savefig(
        str(pathlib.Path(output_dir) / 'vis_nwp_his2d.png'),
        bbox_inches='tight', dpi=800
    )
    fig.savefig(
        str(pathlib.Path(output_dir) / 'vis_nwp_his2d.pdf'),
        bbox_inches='tight', dpi=800
    )
    plt.close(fig)
    del fig, ax
    gc.collect()


def plot_grade_frequency(
    vis_ob: np.ndarray,
    preds: typing.Dict[str, np.ndarray],
    thres: tuple,
    output_dir: str
) -> None:
    """
    Plot grade frequency distribution bars.

    Args:
        vis_ob: Observation array.
        preds: Forecast dict, key is name, value is forecast array.
        thres: Visibility grade thresholds.
        output_dir: Output directory.
    """
    index = ~np.isnan(vis_ob)
    for pred in preds.values():
        index &= ~np.isnan(pred)

    df_fh = {'grade': list(), 'ob': list()}
    for name in preds:
        df_fh[name] = list()

    ob_idx = vis_ob[index]
    for i in range(7):
        right = 999999 if i == 0 else thres[i - 1]
        left = -999999 if i == 6 else thres[i]
        df_fh['grade'].append(i)
        mask = (ob_idx < right) & (ob_idx >= left)
        df_fh['ob'].append(np.sum(mask) / np.sum(index))
        for name, pred in preds.items():
            pred_idx = pred[index]
            mask = (pred_idx < right) & (pred_idx >= left)
            df_fh[name].append(np.sum(mask) / np.sum(index))

    df_fh = pd.DataFrame(df_fh)
    csv_dir = pathlib.Path(output_dir) / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    df_fh.to_csv(str(csv_dir / 'vis_fh.csv'), index=False)

    colors = {
        'ob': 'blue',
        'CMA-SH-WARR': 'green',
        'PDFM-TLE': 'red',
        'Pred3': 'cyan',
        'Pred4': 'magenta',
        'Pred5': 'yellow',
        'Pred6': 'black',
    }

    # Plot bars
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(7)
    width = 0.8 / (len(df_fh.columns) - 1)
    for i, col in enumerate(df_fh.columns[1:]):
        offset = (i - (len(df_fh.columns) - 2) / 2) * width
        ax.bar(
            x + offset,
            df_fh[col],
            width,
            label=col,
            color=colors.get(col, 'gray')
        )
    ax.set_xlabel('Grade')
    ax.set_ylabel('Frequency')
    ax.set_title('Visibility Grade Frequency')
    ax.set_xticks(x)
    ax.legend()
    fig.savefig(str(pathlib.Path(output_dir) / 'vis_fh.png'), dpi=300)
    plt.close(fig)
    gc.collect()
