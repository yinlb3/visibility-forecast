# -*- coding: utf-8 -*-
"""
MRE 与地理要素关系可视化模块 (阶段 8.4 后半段).

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""

import gc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats


def _get_north_china_mask(sta: pd.DataFrame, n_sta: int = 1183) -> np.ndarray:
    """返回华北站点掩码."""
    mask = np.zeros(n_sta, dtype=np.bool_)
    provinces = {'河南省', '山东省', '河北省', '北京市', '天津市', '山西省'}
    for i in range(n_sta):
        if sta.loc[i, 'province'] in provinces:
            mask[i] = True
    return mask


def _plot_geo_scatter(
    x_vals: np.ndarray,
    mre_before: np.ndarray,
    mre_after: np.ndarray,
    mre_improvement: np.ndarray,
    xlabel: str,
    xticks: list,
    xticklabels: list,
    xlim: tuple,
    output_dir: str,
    prefix: str
) -> None:
    """绘制某地理要素与 MRE 的散点图及改善率拟合图."""
    # MRE 散点对比
    plt.figure(figsize=(5, 6))
    plt.scatter(x=x_vals, y=mre_before, s=1, c='blue', label='CMA-SH-WARR')
    plt.scatter(x=x_vals, y=mre_after, s=1, c='red', label='PDFM-TLE')
    plt.xlim(xlim)
    plt.xticks(xticks, xticklabels)
    plt.ylim((0, 0.6))
    plt.yticks((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    plt.xlabel(xlabel)
    plt.ylabel('MRE')
    plt.legend()
    plt.savefig(rf'{output_dir}\sta_{prefix}-mre.png', bbox_inches='tight', dpi=800)
    plt.savefig(rf'{output_dir}\sta_{prefix}-mre.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    gc.collect()
    print(stats.pearsonr(x_vals, mre_before)[0])
    print(stats.pearsonr(x_vals, mre_after)[0])

    # 改善率线性拟合
    plt.figure(figsize=(5, 6))
    a, b = np.polyfit(x_vals, mre_improvement, deg=1)
    print(f'a: {a}, b: {b}')
    x_l = np.linspace(xlim[0], xlim[1], 10000)
    y_est = a * x_l + b
    t = stats.t.isf(0.05 / 2, x_vals.size - 2)
    y_err = t * np.std(y_est) * (
        1 + 1 / x_vals.size + (x_l - np.mean(x_vals)) ** 2 / np.sum((x_vals - np.mean(x_vals)) ** 2)
    ) ** 0.5
    y_ = a * x_vals + b
    y__ = t * np.std(y_est) * (
        1 + 1 / x_vals.size + (x_vals - np.mean(x_vals)) ** 2 / np.sum((x_vals - np.mean(x_vals)) ** 2)
    ) ** 0.5
    is_95 = (mre_improvement >= y_ - y__) & (mre_improvement <= y_ + y__)
    print(np.mean(is_95))
    plt.plot(x_l, y_est, c='black')
    plt.fill_between(x_l, y_est - y_err, y_est + y_err, alpha=0.1, color='blue')
    plt.scatter(x=x_vals, y=mre_improvement, s=1, c='red', label='PDFM-TLE')
    plt.xlim(xlim)
    plt.xticks(xticks, xticklabels)
    plt.ylim((-40, 80))
    plt.yticks((-40, -20, 0, 20, 40, 60, 80))
    plt.xlabel(xlabel)
    plt.ylabel('MRE改善率(%)')
    plt.savefig(rf'{output_dir}\sta_{prefix}-mre_improvement.png', bbox_inches='tight', dpi=800)
    plt.savefig(rf'{output_dir}\sta_{prefix}-mre_improvement.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    gc.collect()
    cc, p_value = stats.pearsonr(x_vals, mre_improvement)
    print(cc, p_value)


def plot_geo_mre_relations(sta: pd.DataFrame, output_dir: str) -> None:
    """
    绘制 MRE 与经度、纬度、高程的关系图及改善率.
    """
    df_sta = pd.read_csv(rf'{output_dir}\vis_sta_mre.csv', low_memory=False)
    index_in = _get_north_china_mask(sta, n_sta=1183)
    mean_in = np.mean(df_sta.loc[index_in, 'PDFM-TLE'])
    mean_out = np.mean(df_sta.loc[~index_in, 'PDFM-TLE'])
    print(f'PDFM-TLE: {mean_in}, {mean_out}')

    mre_before = np.array(df_sta.loc[:, 'CMA-SH-WARR'])
    mre_after = np.array(df_sta.loc[:, 'PDFM-TLE'])
    mre_improvement = (mre_before - mre_after) / mre_before * 100
    mean_in = np.mean(mre_improvement[index_in])
    mean_out = np.mean(mre_improvement[~index_in])
    print(f'MRE_improvement: {mean_in}, {mean_out}')
    print(np.min(mre_before), np.max(mre_before))
    print(np.min(mre_after), np.max(mre_after))
    print(np.min(mre_improvement), np.max(mre_improvement))

    lon = np.array(sta.loc[:, 'lon'])
    _plot_geo_scatter(
        lon, mre_before, mre_after, mre_improvement,
        xlabel='经度',
        xticks=[105, 110, 115, 120, 125],
        xticklabels=['105°', '110°', '115°', '120°', '125°N'],
        xlim=(105, 125),
        output_dir=output_dir,
        prefix='lon'
    )

    lat = np.array(sta.loc[:, 'lat'])
    _plot_geo_scatter(
        lat, mre_before, mre_after, mre_improvement,
        xlabel='纬度',
        xticks=[20, 25, 30, 35, 40, 45],
        xticklabels=['20°', '25°', '30°', '35°', '40', '45°N'],
        xlim=(20, 45),
        output_dir=output_dir,
        prefix='lat'
    )

    alti = np.array(sta.loc[:, 'alti'])
    _plot_geo_scatter(
        alti, mre_before, mre_after, mre_improvement,
        xlabel='高程(m)',
        xticks=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500],
        xticklabels=['0', '500', '1000', '1500', '2000', '2500', '3000', '3500'],
        xlim=(-100, 3600),
        output_dir=output_dir,
        prefix='alti'
    )
