# -*- coding: utf-8 -*-
"""Forecast distribution analysis and plotting module."""
import gc
import os
import pathlib
import typing

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats


def plot_nwp_his2d(
    vis_ob: np.ndarray,
    cma_sh_warr: np.ndarray,
    output_dir: str,
    n_bins: int = 100,
    max_points: int = 100000
) -> None:
    """
    Plot NWP 2D scatter colored by density (obs vs forecast).

    Density-colored scatter with diagonal line, fitted line, and metrics.
    Uses full data for statistics, samples for display if data is large.

    Args:
        vis_ob: Observation array (m).
        cma_sh_warr: Forecast array (m).
        output_dir: Output directory.
        n_bins: Number of bins for density estimation (default 100).
        max_points: Max points to display. -1 means all. Default 100000.
    """
    # 1. Filter valid pairs
    index = (~np.isnan(vis_ob)) & (~np.isnan(cma_sh_warr))
    x_all = vis_ob[index] / 1000  # Convert to km
    y_all = cma_sh_warr[index] / 1000
    n_total = len(x_all)

    # 2. Determine sample size for plotting
    if max_points < 0:
        n_plot = n_total
    else:
        n_plot = min(n_total, max_points)

    print(f'[plot_nwp_his2d] Total samples: {n_total:,}, Plot samples: {n_plot:,}')

    # 3. Compute statistics on full data
    r2 = np.corrcoef(x_all, y_all)[0, 1] ** 2
    rmse = np.sqrt(np.mean((y_all - x_all) ** 2))
    mae = np.mean(np.abs(y_all - x_all))
    bias = np.mean(y_all - x_all)
    slope, intercept, _, _, _ = stats.linregress(x_all, y_all)

    # 4. Sample for display if needed
    if n_plot < n_total:
        rng = np.random.default_rng(seed=42)
        sample_idx = rng.choice(n_total, size=n_plot, replace=False)
        x_plot = x_all[sample_idx]
        y_plot = y_all[sample_idx]
    else:
        x_plot = x_all
        y_plot = y_all

    # 5. Compute density on plot sample using 2D histogram
    hist, x_edges, y_edges = np.histogram2d(x_plot, y_plot, bins=n_bins)
    xi = np.searchsorted(x_edges, x_plot, side='right') - 1
    yi = np.searchsorted(y_edges, y_plot, side='right') - 1
    xi = np.clip(xi, 0, hist.shape[0] - 1)
    yi = np.clip(yi, 0, hist.shape[1] - 1)
    density = hist[xi, yi]

    # 6. Create plot
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    # Scatter colored by density
    scatter = ax.scatter(
        x_plot, y_plot,
        c=density,
        s=3,
        cmap='viridis',
        alpha=0.6,
        edgecolors='none'
    )
    plt.colorbar(scatter, ax=ax, label='Count')

    # Diagonal line (y=x) in red
    lim = [min(x_all.min(), y_all.min()), max(x_all.max(), y_all.max())]
    ax.plot(lim, lim, 'r-', linewidth=1.5, label='1:1 line')

    # Fitted line in black dashed
    x_fit = np.array(lim)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'k--', linewidth=1.5,
            label=f'y={slope:.3f}x+{intercept:.3f}')

    # Set limits and labels
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('Observation (km)')
    ax.set_ylabel('Forecast (km)')
    ax.set_aspect('equal')
    ax.legend(loc='upper left')

    # Add metrics text box
    textstr = (f'$R^2$={r2:.3f}\n'
               f'RMSE={rmse:.2f}km\n'
               f'MAE={mae:.2f}km\n'
               f'Bias={bias:.2f}km')
    ax.text(0.65, 0.15, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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
