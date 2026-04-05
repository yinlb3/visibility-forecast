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


def _get_north_china_mask(sta: pd.DataFrame) -> np.ndarray:
    """Return North China station mask."""
    n_sta = len(sta)
    mask = np.zeros(n_sta, dtype=np.bool_)
    provinces = {'河南省', '山东省', '河北省', '北京市', '天津市', '山西省'}
    sta_reset = sta.reset_index(drop=True)  # 重置索引为 0-n
    for i in range(n_sta):
        if sta_reset.loc[i, 'province'] in provinces:
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
    """Plot scatter and improvement rate fit for geo feature vs MRE."""
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
    # 过滤 nan 后计算相关系数
    valid_mask_before = np.isfinite(x_vals) & np.isfinite(mre_before)
    valid_mask_after = np.isfinite(x_vals) & np.isfinite(mre_after)
    if np.sum(valid_mask_before) >= 2:
        print(f'[_plot_geo_scatter] Corr({prefix}, mre_before): {stats.pearsonr(x_vals[valid_mask_before], mre_before[valid_mask_before])[0]}')
    else:
        print(f'[_plot_geo_scatter] Warning: insufficient valid data for pearsonr (mre_before, {prefix})')
    if np.sum(valid_mask_after) >= 2:
        print(f'[_plot_geo_scatter] Corr({prefix}, mre_after): {stats.pearsonr(x_vals[valid_mask_after], mre_after[valid_mask_after])[0]}')
    else:
        print(f'[_plot_geo_scatter] Warning: insufficient valid data for pearsonr (mre_after, {prefix})')

    # 改善率线性拟合
    plt.figure(figsize=(5, 6))
    
    # 过滤 nan 和 inf 值
    valid_mask = np.isfinite(x_vals) & np.isfinite(mre_improvement)
    x_vals_valid = x_vals[valid_mask]
    mre_improvement_valid = mre_improvement[valid_mask]
    
    n_valid = len(x_vals_valid)
    if n_valid < 2:
        print(f'[_plot_geo_scatter] Warning: insufficient valid data for polyfit ({prefix}, {n_valid} points)')
        a, b = 0, 0
        x_l = np.linspace(xlim[0], xlim[1], 10000)
        y_est = np.zeros_like(x_l)
        y_err = np.zeros_like(x_l)
        is_95 = np.zeros(len(mre_improvement), dtype=bool)
        print(f'[_plot_geo_scatter] {prefix} 95% CI: N/A')
    else:
        a, b = np.polyfit(x_vals_valid, mre_improvement_valid, deg=1)
        print(f'[_plot_geo_scatter] {prefix} fit: a={a}, b={b}')
        x_l = np.linspace(xlim[0], xlim[1], 10000)
        y_est = a * x_l + b
        t = stats.t.isf(0.05 / 2, n_valid - 2)
        x_mean = np.mean(x_vals_valid)
        x_var = np.sum((x_vals_valid - x_mean) ** 2)
        y_err = t * np.std(y_est) * (
            1 + 1 / n_valid + (x_l - x_mean) ** 2 / x_var
        ) ** 0.5
        y_ = a * x_vals + b
        y__ = t * np.std(y_est) * (
            1 + 1 / n_valid + (x_vals - x_mean) ** 2 / x_var
        ) ** 0.5
        is_95 = (mre_improvement >= y_ - y__) & (mre_improvement <= y_ + y__)
        print(f'[_plot_geo_scatter] {prefix} 95% CI ratio: {np.mean(is_95)}')
    
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
    print(f'[_plot_geo_scatter] Corr({prefix}, mre_improvement): r={cc}, p={p_value}')


def plot_geo_mre_relations(sta: pd.DataFrame, output_dir: str) -> None:
    """
    绘制 MRE 与经度、纬度、高程的关系图及改善率.
    """
    df_sta = pd.read_csv(rf'{output_dir}\csv\vis_sta_mre.csv', low_memory=False)
    
    # 检查数据一致性
    if len(sta) != len(df_sta):
        print(f'[plot_geo_mre_relations] Warning: sta ({len(sta)}) and df_sta ({len(df_sta)}) have different lengths')
        # 使用较小的长度
        min_len = min(len(sta), len(df_sta))
        sta = sta.iloc[:min_len].reset_index(drop=True)
        df_sta = df_sta.iloc[:min_len].reset_index(drop=True)
    
    index_in = _get_north_china_mask(sta)
    print(f'[DEBUG] sta length: {len(sta)}, df_sta length: {len(df_sta)}')
    print(f'[DEBUG] index_in sum: {np.sum(index_in)}, provinces in sta: {sta["province"].unique()}')
    print(f'[DEBUG] index_in dtype: {index_in.dtype}, shape: {index_in.shape}')
    mean_in = np.mean(df_sta.loc[index_in, 'PDFM-TLE'])
    mean_out = np.mean(df_sta.loc[~index_in, 'PDFM-TLE'])
    print(f'PDFM-TLE: {mean_in}, {mean_out}')

    mre_before = np.array(df_sta.loc[:, 'CMA-SH-WARR'])
    mre_after = np.array(df_sta.loc[:, 'PDFM-TLE'])
    mre_improvement = (mre_before - mre_after) / mre_before * 100
    mean_in = np.mean(mre_improvement[index_in])
    mean_out = np.mean(mre_improvement[~index_in])
    print(f'[plot_geo_mre_relations] MRE improvement (North China/Other): {mean_in}, {mean_out}')
    print(f'[plot_geo_mre_relations] MRE before range: {np.min(mre_before)} ~ {np.max(mre_before)}')
    print(f'[plot_geo_mre_relations] MRE after range: {np.min(mre_after)} ~ {np.max(mre_after)}')
    print(f'[plot_geo_mre_relations] MRE improvement range: {np.min(mre_improvement)} ~ {np.max(mre_improvement)}')

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
