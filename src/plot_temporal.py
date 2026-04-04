# -*- coding: utf-8 -*-
"""
时效特征可视化模块 (阶段 8.1-8.2).

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""

import gc
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_hour_access_heatmaps(output_dir: str) -> None:
    """
    8.1 绘制起报时次-预报时效二维热图.

    包括 CMA-SH-WARR、PDFM-TLE 的 TS4+ 热图以及改善率热图.
    """
    hour_access = np.load(rf'{output_dir}\hour_access.npy')
    gc.collect()

    # CMA-SH-WARR TS4+
    sns.heatmap(hour_access[0, :, :, 7], cmap='Reds', vmin=0, vmax=0.4, linewidths=0.3)
    plt.xticks(np.arange(24) + 0.5, [str(x) for x in range(1, 25)])
    plt.yticks(np.arange(24) + 0.5, [f'{x:02d}:00' for x in range(24)], rotation=0)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.xlabel('预报时效 (h) ')
    plt.ylabel('起报时次 (UTC) ')
    plt.savefig(rf'{output_dir}\npw_hour_ts4+.png', bbox_inches='tight', dpi=800)
    plt.savefig(rf'{output_dir}\npw_hour_ts4+.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    gc.collect()
    print(np.min(hour_access[0, :, :, 7]), np.max(hour_access[0, :, :, 7]))

    # PDFM-TLE TS4+
    sns.heatmap(hour_access[3, :, :, 7], cmap='Reds', vmin=0, vmax=0.4, linewidths=0.3)
    plt.xticks(np.arange(24) + 0.5, [str(x + 1) for x in range(24)])
    plt.yticks(np.arange(24) + 0.5, [f'{x:02d}:00' for x in range(24)], rotation=0)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.xlabel('预报时效 (h) ')
    plt.ylabel('起报时次 (UTC) ')
    plt.savefig(rf'{output_dir}\pdf-tl_hour_ts4+.png', bbox_inches='tight', dpi=800)
    plt.savefig(rf'{output_dir}\pdf-tl_hour_ts4+.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    gc.collect()
    print(np.min(hour_access[3, :, :, 7]), np.max(hour_access[3, :, :, 7]))

    # 改善率热图
    ts_before = hour_access[0, :, :, 7]
    ts_after = hour_access[3, :, :, 7]
    ts_improvement = (ts_after - ts_before) / ts_before * 100
    sns.heatmap(ts_improvement, cmap='Reds', vmin=0, vmax=150, linewidths=0.3)
    plt.xticks(np.arange(24) + 0.5, [str(x + 1) for x in range(24)])
    plt.yticks(np.arange(24) + 0.5, [f'{x:02d}:00' for x in range(24)], rotation=0)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.xlabel('预报时效 (h) ')
    plt.ylabel('起报时次 (UTC) ')
    plt.savefig(rf'{output_dir}\pdf-tl_hour_ts4+_improvement.png', bbox_inches='tight', dpi=800)
    plt.savefig(rf'{output_dir}\pdf-tl_hour_ts4+_improvement.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    gc.collect()
    print(np.min(ts_improvement), np.max(ts_improvement))
    print(np.mean(ts_improvement))
    print(np.where(ts_improvement == np.max(ts_improvement)))


def plot_ts_comparison_bars(output_dir: str) -> None:
    """
    8.2 绘制预报时效与预报时间的 TS4+ 对比柱状图.
    """
    # 预报时效
    df_vt_ts4 = pd.read_csv(filepath_or_buffer=rf'{output_dir}\vis_vt_ts4+.csv', low_memory=False)
    ts_before = np.array(df_vt_ts4.loc[:, 'CMA-SH-WARR'])
    ts_after = np.array(df_vt_ts4.loc[:, 'PDFM-TLE'])
    ts_improvement = (ts_after - ts_before) / ts_before * 100
    print(np.min(ts_improvement), np.max(ts_improvement))
    print(ts_improvement)
    print(np.max(df_vt_ts4.loc[:, 'PDFM-TLE']))

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(
        x=df_vt_ts4.loc[:, 'vt'] - 0.2,
        height=df_vt_ts4.loc[:, 'CMA-SH-WARR'],
        width=0.4,
        color='blue',
        label='CMA-SH-WARR的TS'
    )
    ax1.bar(
        x=df_vt_ts4.loc[:, 'vt'] + 0.2,
        height=df_vt_ts4.loc[:, 'PDFM-TLE'],
        width=0.4,
        color='red',
        label='PDFM-TLE的TS'
    )
    ax1.set_xlim((0, 25))
    ax1.set_xticks(range(1, 25), [str(x) for x in range(1, 25)])
    ax1.set_ylim((0, 0.4))
    ax1.set_yticks((0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4))
    ax1.set_ylabel('TS')
    ax2 = ax1.twinx()
    ax2.plot(
        df_vt_ts4.loc[:, 'vt'],
        ts_improvement,
        '-o',
        c='black',
        label='PDFM-TLE的TS改善率'
    )
    ax2.set_ylim((0, 150))
    ax2.set_yticks((0, 30, 60, 90, 120, 150))
    ax2.set_ylabel('TS改善率 (%) ')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    plt.xlabel('预报时效 (h) ')
    plt.savefig(rf'{output_dir}\vis_vt_ts4+.png', bbox_inches='tight', dpi=800)
    plt.savefig(rf'{output_dir}\vis_vt_ts4+.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    del fig, ax1, ax2
    gc.collect()

    # 预报时间
    df_fhour_ts4 = pd.read_csv(filepath_or_buffer=rf'{output_dir}\vis_fhour_ts4+.csv', low_memory=False)
    ts_before = np.array(df_fhour_ts4.loc[:, 'CMA-SH-WARR'])
    ts_after = np.array(df_fhour_ts4.loc[:, 'PDFM-TLE'])
    ts_improvement = (ts_after - ts_before) / ts_before * 100
    print(np.min(ts_improvement), np.max(ts_improvement))
    print(ts_improvement)
    print(np.max(df_fhour_ts4.loc[:, 'PDFM-TLE']))

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(
        x=df_fhour_ts4.loc[:, 'fhour'] - 0.2,
        height=df_fhour_ts4.loc[:, 'CMA-SH-WARR'],
        width=0.4,
        color='blue',
        label='CMA-SH-WARR的TS'
    )
    ax1.bar(
        x=df_fhour_ts4.loc[:, 'fhour'] + 0.2,
        height=df_fhour_ts4.loc[:, 'PDFM-TLE'],
        width=0.4,
        color='red',
        label='PDFM-TLE的TS'
    )
    ax1.set_xlim((-1, 24))
    ax1.set_xticks(range(24), [f'{x:02d}:00' for x in range(24)])
    ax1.set_ylim((0, 0.4))
    ax1.set_yticks((0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4))
    ax1.set_ylabel('TS')
    ax2 = ax1.twinx()
    ax2.plot(
        df_fhour_ts4.loc[:, 'fhour'],
        ts_improvement,
        '-o',
        c='black',
        label='PDFM-TLE的TS改善率'
    )
    ax2.set_ylim((0, 150))
    ax2.set_yticks((0, 30, 60, 90, 120, 150))
    ax2.set_ylabel('TS改善率 (%) ')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    plt.xlabel('预报时间 (UTC) ')
    plt.savefig(rf'{output_dir}\vis_fhour_ts4+.png', bbox_inches='tight', dpi=800)
    plt.savefig(rf'{output_dir}\vis_fhour_ts4+.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    del fig, ax
    gc.collect()
