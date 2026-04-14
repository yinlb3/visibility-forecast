# -*- coding: utf-8 -*-
"""
Weather-type forecast evaluation and CDF plotting module (Stage 5).

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""

import gc
import os
import pathlib
import typing

import numpy as np
from matplotlib import pyplot as plt

from src import vis_acc

VisAcc = vis_acc.VisAcc


def load_weather_type(data_dir: str, idx_mlyr: np.ndarray) -> np.ndarray:
    """
    Load weather type data and reshape to match forecast.

    Args:
        data_dir (str): Data root directory.
        idx_mlyr (np.ndarray): Index for MLYR (Middle-Lower Yangtze River).

    Returns:
        np.ndarray: Weather type array, shape (-1, 24, 502).
    """
    path = pathlib.Path(data_dir) / 'weather_type.npy'
    weather_type = np.load(str(path), mmap_mode='r')
    # val_wt: Validation period Weather Type (验证时段天气类型)
    val_wt = np.reshape(weather_type[1096:1461, ..., idx_mlyr],
                        shape=(-1, 24, 502))
    return val_wt


def calc_weather_type_metrics(
    vis_ob: np.ndarray,
    preds: typing.Tuple[np.ndarray, ...],
    val_wt: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Calc quantitative and grade metrics by weather type.

    qem shape: (4 metrics, N schemes, 3 weather types);
    cem shape: (6 grades, 4 metrics, N schemes, 3 weather types).

    Args:
        vis_ob (np.ndarray): Observation array.
        preds (tuple): Forecast array tuple, usually
            [CMA-SH-WARR, S1, S2, S3, S4, S5].
        val_wt (np.ndarray): Validation period Weather Type (验证时段天气类型: 1=precip, 2=fog, 3=haze).

    Returns:
        tuple: (qem, cem).
    """
    # 1. Init result arrays
    n_pred = len(preds)
    # qem: Quantitative Evaluation Metrics (定量指标: R, MAE, RMSE, MRE)
    qem = np.zeros((4, n_pred, 3), dtype=np.float32) + np.nan
    # cem: Categorical/Grade Evaluation Metrics (等级指标: TS, FAR, MAR, POD)
    cem = np.zeros((6, 4, n_pred, 3), dtype=np.float32) + np.nan
    # 2. Loop through 3 weather types (1=precip, 2=fog, 3=haze)
    for i in range(3):
        mask = val_wt == i + 1
        ob_masked = vis_ob[mask]
        # 3. Calc metrics for each prediction scheme
        for j, pred in enumerate(preds):
            pr_masked = pred[mask]
            acc = VisAcc(ob_masked, pr_masked)
            # 3.1 Store quantitative metrics (R, MAE, RMSE, MRE)
            qem[0, j, i] = acc.get_r()
            qem[1, j, i] = acc.get_mae()
            qem[2, j, i] = acc.get_rmse()
            qem[3, j, i] = acc.get_mre()
            # 3.2 Store grade metrics (TS, FAR, MAR, POD)
            cem[:, 0, j, i] = acc.get_ts_ge()
            cem[:, 1, j, i] = acc.get_far_ge()
            cem[:, 2, j, i] = acc.get_mar_ge()
            cem[:, 3, j, i] = acc.get_pod_ge()
    return qem, cem


def save_weather_type_metrics(
    qem: np.ndarray,
    cem: np.ndarray,
    output_dir: str
) -> None:
    """
    Save weather type metrics (qem, cem) to NPY files.

    Args:
        qem (np.ndarray): Quantitative evaluation metrics array.
        cem (np.ndarray): Categorical/Grade evaluation metrics array.
        output_dir (str): Output directory path.
    """
    csv_dir = pathlib.Path(output_dir) / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(csv_dir / 'qem.npy'), qem)
    np.save(str(csv_dir / 'cem.npy'), cem)
    print(f'[Weather Type] Saved qem and cem to {csv_dir}')


def load_weather_type_metrics(output_dir: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Load weather type metrics (qem, cem) from NPY files.

    Args:
        output_dir (str): Output directory containing 'csv' subfolder.

    Returns:
        tuple: (qem, cem) arrays.

    Raises:
        FileNotFoundError: If NPY files do not exist.
    """
    csv_dir = pathlib.Path(output_dir) / 'csv'
    qem_path = csv_dir / 'qem.npy'
    cem_path = csv_dir / 'cem.npy'
    
    if not qem_path.exists():
        raise FileNotFoundError(f'Weather type metrics not found: {qem_path}')
    if not cem_path.exists():
        raise FileNotFoundError(f'Weather type metrics not found: {cem_path}')
    
    qem = np.load(str(qem_path))
    cem = np.load(str(cem_path))
    print(f'[Weather Type] Loaded qem and cem from {csv_dir}')
    return qem, cem


def plot_vis_cdf(
    vis_ob: np.ndarray,
    cma_sh_warr: np.ndarray,
    pred_pdfm_tle0: np.ndarray,
    output_dir: str
) -> None:
    """
    Calc and plot visibility CDF.

    Args:
        vis_ob (np.ndarray): Observation array.
        cma_sh_warr (np.ndarray): CMA-SH-WARR forecast array.
        pred_pdfm_tle0 (np.ndarray): PDFM-TLE forecast array.
        output_dir (str): Output directory.
    """
    # 1. Define CDF evaluation points (vis range: -1 to 30000m)
    vis_values = np.arange(-1, 30001, 1, dtype=np.float32)
    cdf = np.zeros((3, vis_values.size), dtype=np.float32)

    # 2. Get valid data mask (exclude NaN)
    mask_ob = ~np.isnan(vis_ob)
    mask_nwp = ~np.isnan(cma_sh_warr)
    mask_pr = ~np.isnan(pred_pdfm_tle0)
    index = mask_ob & mask_nwp & mask_pr
    ob = vis_ob[index]
    nwp = cma_sh_warr[index]
    pr = pred_pdfm_tle0[index]

    # 3. Calc CDF using searchsorted (O(M log M) complexity)
    ob_sorted = np.sort(ob)
    nwp_sorted = np.sort(nwp)
    pr_sorted = np.sort(pr)
    cdf[0, :] = np.searchsorted(ob_sorted, vis_values, side='right')
    cdf[0, :] /= ob.size
    cdf[1, :] = np.searchsorted(nwp_sorted, vis_values, side='right')
    cdf[1, :] /= nwp.size
    cdf[2, :] = np.searchsorted(pr_sorted, vis_values, side='right')
    cdf[2, :] /= pr.size

    fig, ax = plt.subplots(figsize=(5, 5), dpi=800)
    ax.plot(vis_values / 1000, cdf[0, :], '-', c='black', label='实况')
    ax.plot(vis_values / 1000, cdf[1, :], '-', c='blue', label='CMA-SH3-WARR')
    ax.plot(vis_values / 1000, cdf[2, :], '-', c='red', label='PDFM-TLE')
    ax.set_xlim((-1, 31))
    ax.set_xticks((0, 10, 20, 30))
    ax.set_ylim((0, 1))
    ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    ax.set_xlabel('能见度 (km) ')
    ax.set_ylabel('累积概率')
    ax.legend()
    fig.savefig(
        fname=str(pathlib.Path(output_dir) / 'vis_cdf.png'),
        bbox_inches='tight',
        dpi=800
    )
    fig.savefig(
        fname=str(pathlib.Path(output_dir) / 'vis_cdf.pdf'),
        bbox_inches='tight',
        dpi=800
    )
    plt.close(fig)
    del fig, ax
    gc.collect()


def plot_weather_type_eval_bw(
    qem: np.ndarray,
    filename: str,
    max_y: float,
    output_dir: str
) -> None:
    """
    Plot weather-type metric comparison bars (B&W style).

    Args:
        qem: Shape (6, 3) quantitative metric array,
            6 schemes * 3 weather types;
        filename: Output filename (no extension);
        max_y: Max physical value for y-axis, used for normalization;
        output_dir: Output directory path.
    """
    os.makedirs(name=output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    # X positions for 6 schemes
    x_pos = np.linspace(start=1, stop=6, num=6)
    # Draw three groups: precip (black), fog (white hatched), haze (white)
    bars1 = ax.bar(x=x_pos - 0.2, height=qem[:, 0] / max_y, width=0.2,
                   color='black', edgecolor='black', label='降水类')
    bars2 = ax.bar(x=x_pos, height=qem[:, 1] / max_y, width=0.2,
                   color='white', edgecolor='black', hatch='///', label='雾类')
    bars3 = ax.bar(x=x_pos + 0.2, height=qem[:, 2] / max_y, width=0.2,
                   color='white', edgecolor='black', label='霾类')
    # Annotate normalized value above each bar
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            height = 0 if height < 0 else height
            ax.text(x=bar.get_x() + bar.get_width() / 2.,
                    y=height + 0.02,
                    s=f'{height:.2f}',
                    ha='center', va='bottom', fontsize=7)
    # Set axis range and tick labels
    ax.set_xlim((0, 7))
    ax.set_xticks(
        ticks=range(1, 7),
        labels=['CMA-SH-WARR', '试验一', '试验二', '试验三', '试验四', '试验五']
    )
    ax.set_ylim((0, 1))
    ax.set_yticks(
        ticks=np.linspace(start=0, stop=1, num=6),
        labels=[
            f'{val * max_y:g}'
            for val in np.linspace(start=0, stop=1, num=6)
        ]
    )
    ax.legend()
    # Save as PDF and release memory
    fig.savefig(
        fname=str(pathlib.Path(output_dir) / f'{filename}.pdf'),
        bbox_inches='tight',
        dpi=800
    )
    plt.close(fig)
    gc.collect()
