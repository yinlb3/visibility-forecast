# -*- coding: utf-8 -*-
"""
分类型预报检验与绘图模块.

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""

import os
from typing import Tuple

import gc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.vis_acc import VisAcc


def load_weather_type(data_dir: str, index_cjzxy: np.ndarray) -> np.ndarray:
    """
    加载天气类型数据并 reshape 为与预报数据一致的形状.

    Args:
        data_dir (str): 数据根目录.
        index_cjzxy (np.ndarray): 长江中下游站点筛选索引.

    Returns:
        np.ndarray: 天气类型数组, 形状为 (-1, 24, 502).
    """
    weather_type = np.load(rf'{data_dir}\weather_type.npy', mmap_mode='r')
    val_wt = np.reshape(weather_type[1096:1461, ..., index_cjzxy], shape=(-1, 24, 502))
    return val_wt


def calc_weather_type_metrics(
    vis_ob: np.ndarray,
    preds: Tuple[np.ndarray, ...],
    val_wt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    按天气类型计算定量与分级检验指标.

    qem 形状为 (4 指标, N方案, 3 天气类型);
    cem 形状为 (6 等级, 4 指标, N方案, 3 天气类型).

    Args:
        vis_ob (np.ndarray): 观测数组.
        preds (tuple): 预报数组元组, 顺序通常为
            [CMA-SH-WARR, 方案一, 方案二, 方案三, 方案四, 方案五].
        val_wt (np.ndarray): 天气类型掩码数组 (值为 1,2,3).

    Returns:
        tuple: (qem, cem).
    """
    n_pred = len(preds)
    qem = np.zeros((4, n_pred, 3), dtype=np.float32) + np.nan
    cem = np.zeros((6, 4, n_pred, 3), dtype=np.float32) + np.nan
    for i in range(3):
        mask = val_wt == i + 1
        ob_masked = vis_ob[mask]
        for j, pred in enumerate(preds):
            pr_masked = pred[mask]
            acc = VisAcc(ob_masked, pr_masked)
            qem[0, j, i] = acc.get_r()
            qem[1, j, i] = acc.get_mae()
            qem[2, j, i] = acc.get_rmse()
            qem[3, j, i] = acc.get_mre()
            cem[:, 0, j, i] = acc.get_ts2()
            cem[:, 1, j, i] = acc.get_far2()
            cem[:, 2, j, i] = acc.get_mar2()
            cem[:, 3, j, i] = acc.get_pod2()
    return qem, cem


def plot_vis_cdf(
    vis_ob: np.ndarray,
    cma_sh_warr: np.ndarray,
    pred_pdfm_tle0: np.ndarray,
    output_dir: str
) -> None:
    """
    计算并绘制能见度累积分布函数(CDF).

    Args:
        vis_ob (np.ndarray): 观测数组.
        cma_sh_warr (np.ndarray): CMA-SH-WARR 预报数组.
        pred_pdfm_tle0 (np.ndarray): PDFM-TLE 预报数组.
        output_dir (str): 输出目录.
    """
    vis_values = np.arange(-1, 30001, 1, dtype=np.float32)
    cdf = np.zeros((3, vis_values.size), dtype=np.float32)

    index = (~np.isnan(vis_ob)) & (~np.isnan(cma_sh_warr)) & (~np.isnan(pred_pdfm_tle0))
    ob = vis_ob[index]
    nwp = cma_sh_warr[index]
    pr = pred_pdfm_tle0[index]

    # 使用 searchsorted 替代循环,将复杂度从 O(N*M) 降至 O(M log M)
    ob_sorted = np.sort(ob)
    nwp_sorted = np.sort(nwp)
    pr_sorted = np.sort(pr)
    cdf[0, :] = np.searchsorted(ob_sorted, vis_values, side='right') / ob.size
    cdf[1, :] = np.searchsorted(nwp_sorted, vis_values, side='right') / nwp.size
    cdf[2, :] = np.searchsorted(pr_sorted, vis_values, side='right') / pr.size

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
    fig.savefig(fname=rf'{output_dir}\vis_cdf.png', bbox_inches='tight', dpi=800)
    fig.savefig(fname=rf'{output_dir}\vis_cdf.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()


def plot_nwp_his2d(
    vis_ob: np.ndarray,
    cma_sh_warr: np.ndarray,
    output_dir: str
) -> None:
    """
    绘制预报-实况二维频率分布图.

    Args:
        vis_ob (np.ndarray): 观测数组.
        cma_sh_warr (np.ndarray): CMA-SH-WARR 预报数组.
        output_dir (str): 输出目录.
    """
    index = ~np.isnan(vis_ob) & ~np.isnan(cma_sh_warr)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=800)
    x = vis_ob[index] / 1000
    y = cma_sh_warr[index] / 1000
    his2d = np.zeros((10, 10), dtype=np.float32) + np.nan
    l0 = np.linspace(np.log(1), np.log(31), 11)
    l = np.exp(l0) - 1
    x_l = np.zeros((10, 10), dtype=np.float32) + np.nan
    y_l = np.zeros((10, 10), dtype=np.float32) + np.nan
    for i in range(10):
        if i != 9:
            index0 = (x >= l[i]) & (x < l[i + 1])
        else:
            index0 = (x >= l[i]) & (x <= l[i + 1])
        for j in range(10):
            x_l[i, j] = (l0[i] + l0[i + 1]) / 2
            y_l[i, j] = (l0[j] + l0[j + 1]) / 2
            if j != 9:
                index0 &= (y >= l[j]) & (y < l[j + 1])
            else:
                index0 &= (y >= l[j]) & (y <= l[j + 1])
            index0 = index0.astype(np.int_)
            his2d[i, j] = np.sum(index0)
    ax.scatter(np.reshape(x_l, -1), np.reshape(y_l, -1), c=np.reshape(his2d, -1), s=10)
    ax.set_xlim((np.log(1), np.log(31)))
    ax.set_xticks(l0, [f'{x_val:.2f}' for x_val in l])
    ax.set_ylim((np.log(1), np.log(31)))
    ax.set_yticks(l0, [f'{y_val:.2f}' for y_val in l])
    ax.set_xlabel('实况 (km)')
    ax.set_ylabel('预报 (km)')
    fig.savefig(rf'{output_dir}\vis_nwp_his2d.png', bbox_inches='tight', dpi=800)
    fig.savefig(rf'{output_dir}\vis_nwp_his2d.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()


def plot_grade_frequency(
    vis_ob: np.ndarray,
    preds: dict,
    thres: tuple,
    output_dir: str
) -> None:
    """
    绘制各方案分级频率分布柱状图.

    Args:
        vis_ob (np.ndarray): 观测数组.
        preds (dict): 预报数据字典, key 为图例名称, value 为预报数组.
        thres (tuple): 能见度分级阈值.
        output_dir (str): 输出目录.
    """
    index = ~np.isnan(vis_ob)
    for pred in preds.values():
        index &= ~np.isnan(pred)

    df_fh = {'grade': list(), 'ob': list()}
    for name in preds:
        df_fh[name] = list()

    for i in range(7):
        right = 999999 if i == 0 else thres[i - 1]
        left = -999999 if i == 6 else thres[i]
        df_fh['grade'].append(i)
        df_fh['ob'].append(
            np.sum((vis_ob[index] < right) & (vis_ob[index] >= left)) / np.sum(index)
        )
        for name, pred in preds.items():
            df_fh[name].append(
                np.sum((pred[index] < right) & (pred[index] >= left)) / np.sum(index)
            )

    df_fh = pd.DataFrame(df_fh)
    df_fh.to_csv(rf'{output_dir}\vis_fh.csv', index=False)

    colors = {
        'ob': 'black',
        'CMA-SH-WARR': 'blue',
        'OTS': 'green',
        'PDF': 'yellow',
        'TL': 'orange',
        'PDFM-TLE': 'red',
    }
    offsets = {
        'ob': -0.4,
        'CMA-SH-WARR': -0.24,
        'OTS': -0.08,
        'PDF': 0.08,
        'TL': 0.24,
        'PDFM-TLE': 0.4,
    }

    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    col_order = ['ob'] + list(preds.keys())
    for col in col_order:
        ax.bar(
            x=df_fh.loc[:, 'grade'] + offsets[col],
            height=df_fh.loc[:, col],
            width=0.16,
            color=colors.get(col, 'gray'),
            label='实况' if col == 'ob' else col
        )
    ax.set_xlim((-1, 7))
    ax.set_xticks(range(7), df_fh.loc[:, 'grade'])
    ax.set_ylim((0, 1.0))
    ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
    ax.set_xlabel('低能见度等级')
    ax.set_ylabel('频率')
    ax.legend()
    fig.savefig(fname=rf'{output_dir}\fh.png', bbox_inches='tight', dpi=800)
    fig.savefig(fname=rf'{output_dir}\fh.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()


def plot_weather_type_eval_bw(qem: np.ndarray, filename: str, max_y: float) -> None:
    """
    绘制按天气类型 (降水、雾、霾) 分类的检验指标对比柱状图 (黑白风格).

    Args:
        qem: 形状为 (6, 3) 的定量检验指标数组, 6 个方案 * 3 类天气;
        filename: 输出文件名 (不含扩展名);
        max_y: y 轴实际物理量最大值, 用于归一化显示.
    """
    # 确保输出目录存在
    os.makedirs(name=r'D:\Project\vis\图', exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    # 6 个方案对应的 x 轴位置
    x_pos = np.linspace(start=1, stop=6, num=6)
    # 绘制三组并列柱子:降水类(纯黑)、雾类(白色斜线)、霾类(纯白边框)
    bars1 = ax.bar(x=x_pos - 0.2, height=qem[:, 0] / max_y, width=0.2,
                   color='black', edgecolor='black', label='降水类')
    bars2 = ax.bar(x=x_pos, height=qem[:, 1] / max_y, width=0.2,
                   color='white', edgecolor='black', hatch='///', label='雾类')
    bars3 = ax.bar(x=x_pos + 0.2, height=qem[:, 2] / max_y, width=0.2,
                   color='white', edgecolor='black', label='霾类')
    # 在每个柱子上方标注归一化后的数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            height = 0 if height < 0 else height
            ax.text(x=bar.get_x() + bar.get_width() / 2., y=height + 0.02, s=f'{height:.2f}',
                    ha='center', va='bottom', fontsize=7)
    # 设置坐标轴范围与刻度标签
    ax.set_xlim((0, 7))
    ax.set_xticks(ticks=range(1, 7), labels=['CMA-SH-WARR', '试验一', '试验二', '试验三', '试验四', '试验五'])
    ax.set_ylim((0, 1))
    ax.set_yticks(
        ticks=np.linspace(start=0, stop=1, num=6),
        labels=[f'{val * max_y:g}' for val in np.linspace(start=0, stop=1, num=6)]
    )
    ax.legend()
    # 保存为 PDF 并释放内存
    fig.savefig(fname=fr'D:\Project\vis\图\{filename}.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    gc.collect()
