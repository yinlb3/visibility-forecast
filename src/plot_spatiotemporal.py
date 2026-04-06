# -*- coding: utf-8 -*-
"""
Spatiotemporal feature visualization module (Stage 8).

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""

import gc

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from meteva import base as meb
from scipy import stats


def plot_hour_access_heatmaps(output_dir: str) -> None:
    """
    8.1 Plot init time-lead time 2D heatmap.

    Including TS4+ heatmaps for CMA-SH-WARR, PDFM-TLE and improvement rate.
    """
    # 1. Load pre-calculated hour_access data
    hour_access = np.load(rf'{output_dir}\csv\hour_access.npy')
    gc.collect()

    # 2. Plot CMA-SH-WARR TS4+ heatmap
    sns.heatmap(
        hour_access[0, :, :, 7], cmap='Reds', vmin=0, vmax=0.4, linewidths=0.3
    )
    plt.xticks(np.arange(24) + 0.5, [str(x) for x in range(1, 25)])
    yticks = [f'{x:02d}:00' for x in range(24)]
    plt.yticks(np.arange(24) + 0.5, yticks, rotation=0)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.xlabel('预报时效 (h) ')
    plt.ylabel('起报时次 (UTC) ')
    plt.savefig(
        rf'{output_dir}\npw_hour_ts4+.png', bbox_inches='tight', dpi=800
    )
    plt.savefig(
        rf'{output_dir}\npw_hour_ts4+.pdf', bbox_inches='tight', dpi=800
    )
    plt.cla()
    plt.close('all')
    gc.collect()
    ts_min = np.min(hour_access[0, :, :, 7])
    ts_max = np.max(hour_access[0, :, :, 7])
    prefix = '[plot_hour_access_heatmaps]'
    print(f'{prefix} CMA-SH-WARR TS4+ range: {ts_min:.4f} ~ {ts_max:.4f}')

    # 3. Plot PDFM-TLE TS4+ heatmap
    sns.heatmap(
        hour_access[3, :, :, 7], cmap='Reds', vmin=0, vmax=0.4, linewidths=0.3
    )
    plt.xticks(np.arange(24) + 0.5, [str(x + 1) for x in range(24)])
    yticks = [f'{x:02d}:00' for x in range(24)]
    plt.yticks(np.arange(24) + 0.5, yticks, rotation=0)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.xlabel('预报时效 (h) ')
    plt.ylabel('起报时次 (UTC) ')
    plt.savefig(
        rf'{output_dir}\pdf-tl_hour_ts4+.png', bbox_inches='tight', dpi=800
    )
    plt.savefig(
        rf'{output_dir}\pdf-tl_hour_ts4+.pdf', bbox_inches='tight', dpi=800
    )
    plt.cla()
    plt.close('all')
    gc.collect()
    ts_min = np.min(hour_access[3, :, :, 7])
    ts_max = np.max(hour_access[3, :, :, 7])
    prefix = '[plot_hour_access_heatmaps]'
    print(f'{prefix} PDFM-TLE TS4+ range: {ts_min:.4f} ~ {ts_max:.4f}')

    # 4. Calc and plot improvement rate heatmap
    ts_before = hour_access[0, :, :, 7]
    ts_after = hour_access[3, :, :, 7]
    ts_improvement = (ts_after - ts_before) / ts_before * 100
    sns.heatmap(ts_improvement, cmap='Reds', vmin=0, vmax=150, linewidths=0.3)
    plt.xticks(np.arange(24) + 0.5, [str(x + 1) for x in range(24)])
    yticks = [f'{x:02d}:00' for x in range(24)]
    plt.yticks(np.arange(24) + 0.5, yticks, rotation=0)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.xlabel('预报时效 (h) ')
    plt.ylabel('起报时次 (UTC) ')
    plt.savefig(
        rf'{output_dir}\pdf-tl_hour_ts4+_improvement.png',
        bbox_inches='tight', dpi=800
    )
    plt.savefig(
        rf'{output_dir}\pdf-tl_hour_ts4+_improvement.pdf',
        bbox_inches='tight', dpi=800
    )
    plt.cla()
    plt.close('all')
    gc.collect()
    loc = np.where(ts_improvement == np.max(ts_improvement))
    ts_min = np.min(ts_improvement)
    ts_max = np.max(ts_improvement)
    prefix = '[plot_hour_access_heatmaps]'
    print(f'{prefix} TS improvement range: {ts_min:.4f}% ~ {ts_max:.4f}%')
    print(f'{prefix} TS improvement mean: {np.mean(ts_improvement):.4f}%')
    init_h, lead_h = int(loc[0][0]), int(loc[1][0])
    loc_str = f'init_hour={init_h}, lead_hour={lead_h}'
    print(f'{prefix} TS improvement max location: {loc_str}')


def plot_ts_comparison_bars(output_dir: str) -> None:
    """
    8.2 Plot TS4+ comparison bars for lead time and forecast time.
    """
    # Forecast lead time
    csv_path = rf'{output_dir}\csv\vis_vt_ts4+.csv'
    df_vt_ts4 = pd.read_csv(filepath_or_buffer=csv_path, low_memory=False)
    ts_before = np.array(df_vt_ts4.loc[:, 'CMA-SH-WARR'])
    ts_after = np.array(df_vt_ts4.loc[:, 'PDFM-TLE'])
    ts_improvement = (ts_after - ts_before) / ts_before * 100
    ts_min = np.min(ts_improvement)
    ts_max = np.max(ts_improvement)
    prefix = '[plot_ts_comparison_bars]'
    print(f'{prefix} VT TS improve: {ts_min:.4f}% ~ {ts_max:.4f}%')
    ts_str = " ".join(f"{x:.4f}" for x in ts_improvement)
    print(f'[plot_ts_comparison_bars] VT TS improve: [{ts_str}]')
    ts_max_val = np.max(df_vt_ts4.loc[:, 'PDFM-TLE'])
    print(f'[plot_ts_comparison_bars] VT PDFM-TLE max: {ts_max_val:.4f}')

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(
        x=df_vt_ts4.loc[:, 'vt'] - 0.2,
        height=df_vt_ts4.loc[:, 'CMA-SH-WARR'],
        width=0.4,
        color='blue',
        label='CMA-SH-WARR TS'
    )
    ax1.bar(
        x=df_vt_ts4.loc[:, 'vt'] + 0.2,
        height=df_vt_ts4.loc[:, 'PDFM-TLE'],
        width=0.4,
        color='red',
        label='PDFM-TLE TS'
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
        label='PDFM-TLE TS improve'
    )
    ax2.set_ylim((0, 150))
    ax2.set_yticks((0, 30, 60, 90, 120, 150))
    ax2.set_ylabel('TS改善率(%)')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    plt.xlabel('预报时效 (h) ')
    plt.savefig(
        rf'{output_dir}\vis_vt_ts4+.png', bbox_inches='tight', dpi=800
    )
    plt.savefig(
        rf'{output_dir}\vis_vt_ts4+.pdf', bbox_inches='tight', dpi=800
    )
    plt.cla()
    plt.close('all')
    del fig, ax1, ax2
    gc.collect()

    # Forecast time
    csv_path = rf'{output_dir}\csv\vis_fhour_ts4+.csv'
    df_fhour_ts4 = pd.read_csv(filepath_or_buffer=csv_path, low_memory=False)
    ts_before = np.array(df_fhour_ts4.loc[:, 'CMA-SH-WARR'])
    ts_after = np.array(df_fhour_ts4.loc[:, 'PDFM-TLE'])
    ts_improvement = (ts_after - ts_before) / ts_before * 100
    ts_min = np.min(ts_improvement)
    ts_max = np.max(ts_improvement)
    prefix = '[plot_ts_comparison_bars]'
    print(f'{prefix} FHour TS improve: {ts_min:.4f}% ~ {ts_max:.4f}%')
    ts_str = " ".join(f"{x:.4f}" for x in ts_improvement)
    print(f'[plot_ts_comparison_bars] FHour TS improve: [{ts_str}]')
    ts_max_val = np.max(df_fhour_ts4.loc[:, 'PDFM-TLE'])
    print(f'[plot_ts_comparison_bars] FHour PDFM-TLE max: {ts_max_val:.4f}')

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
    plt.savefig(
        rf'{output_dir}\vis_fhour_ts4+.png', bbox_inches='tight', dpi=800
    )
    plt.savefig(
        rf'{output_dir}\vis_fhour_ts4+.pdf', bbox_inches='tight', dpi=800
    )
    plt.cla()
    plt.close('all')
    del fig, ax1, ax2
    gc.collect()


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation excluding NaN values."""
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return np.nan
    return stats.pearsonr(x[mask], y[mask])[0]


def plot_sta_ts4_maps(sta: pd.DataFrame, output_dir: str) -> None:
    """
    8.3 Plot station-level TS4+ and improvement rate spatial maps.

    Args:
        sta (pd.DataFrame): Station info DataFrame.
        output_dir (str): Output directory path.
    """
    csv_path = rf'{output_dir}\csv\vis_sta_ts4+.csv'
    df_sta = pd.read_csv(csv_path, low_memory=False)
    prefix = '[plot_sta_ts4_maps]'
    ts_min = np.min(df_sta.loc[:, 'CMA-SH-WARR'])
    ts_max = np.max(df_sta.loc[:, 'CMA-SH-WARR'])
    print(f'{prefix} CMA-SH-WARR TS4+ range: {ts_min:.4f} ~ {ts_max:.4f}')
    ts_min = np.min(df_sta.loc[:, 'PDFM-TLE'])
    ts_max = np.max(df_sta.loc[:, 'PDFM-TLE'])
    print(f'{prefix} PDFM-TLE TS4+ range: {ts_min:.4f} ~ {ts_max:.4f}')
    lon_arr = np.array(sta.loc[:, 'lon'])
    lat_arr = np.array(sta.loc[:, 'lat'])
    alti_arr = np.array(sta.loc[:, 'alti'])
    cma_arr = np.array(df_sta.loc[:, 'CMA-SH-WARR'])
    pdfm_arr = np.array(df_sta.loc[:, 'PDFM-TLE'])
    r_lon_cma = _safe_pearsonr(lon_arr, cma_arr)
    r_lon_pdfm = _safe_pearsonr(lon_arr, pdfm_arr)
    r_lat_cma = _safe_pearsonr(lat_arr, cma_arr)
    r_lat_pdfm = _safe_pearsonr(lat_arr, pdfm_arr)
    r_alti_cma = _safe_pearsonr(alti_arr, cma_arr)
    r_alti_pdfm = _safe_pearsonr(alti_arr, pdfm_arr)
    print(f'{prefix} Corr(lon, CMA-SH-WARR): {r_lon_cma:.4f}')
    print(f'{prefix} Corr(lon, PDFM-TLE): {r_lon_pdfm:.4f}')
    print(f'{prefix} Corr(lat, CMA-SH-WARR): {r_lat_cma:.4f}')
    print(f'{prefix} Corr(lat, PDFM-TLE): {r_lat_pdfm:.4f}')
    print(f'{prefix} Corr(alti, CMA-SH-WARR): {r_alti_cma:.4f}')
    print(f'{prefix} Corr(alti, PDFM-TLE): {r_alti_pdfm:.4f}')

    cmap, clevs = meb.def_cmap_clevs(
        meb.def_cmap_clevs(meb.cmaps.ts, vmin=0, vmax=0.7)[0],
        clevs=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    )

    sta0 = sta.loc[:, ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')]
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'CMA-SH-WARR']
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='max',
        title=[''],
        save_path=rf'{output_dir}\sta_ts4+_nwp.png',
        dpi=800
    )
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='max',
        title=[''],
        save_path=rf'{output_dir}\sta_ts4+_nwp.pdf',
        dpi=800
    )

    sta0 = sta.loc[:, ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')]
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'PDFM-TLE']
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='max',
        title=[''],
        save_path=rf'{output_dir}\sta_ts4+.png',
        dpi=800
    )
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='max',
        title=[''],
        save_path=rf'{output_dir}\sta_ts4+.pdf',
        dpi=800
    )


def plot_mre_violins(sta: pd.DataFrame, output_dir: str) -> None:
    """
    8.4 Plot MRE and improvement rate violin/box plots.

    Args:
        sta (pd.DataFrame): Station info DataFrame.
        output_dir (str): Output directory path.
    """
    csv_path = rf'{output_dir}\csv\vis_sta_mre.csv'
    df_sta = pd.read_csv(csv_path, low_memory=False)
    mre_before = np.array(df_sta.loc[:, 'CMA-SH-WARR'])
    mre_after = np.array(df_sta.loc[:, 'PDFM-TLE'])
    with np.errstate(divide='ignore', invalid='ignore'):
        mre_improvement = np.where(
            mre_before == 0, np.nan,
            (mre_before - mre_after) / mre_before * 100
        )

    plt.figure(figsize=(5, 6))
    sns.violinplot(
        data={
            'CMA-SH-WARR': df_sta.loc[:, 'CMA-SH-WARR'],
            'PDFM-TLE': df_sta.loc[:, 'PDFM-TLE'],
        },
        palette=['blue', 'red']
    )
    plt.ylabel('MRE')
    plt.savefig(
        rf'{output_dir}\boxplot_mre.png', bbox_inches='tight', dpi=800
    )
    plt.savefig(
        rf'{output_dir}\boxplot_mre.pdf', bbox_inches='tight', dpi=800
    )
    plt.cla()
    plt.close('all')
    gc.collect()
    median_before = np.nanmedian(df_sta.loc[:, 'CMA-SH-WARR'])
    median_after = np.nanmedian(df_sta.loc[:, 'PDFM-TLE'])
    median_improve = median_before - median_after
    prefix = '[plot_mre_violins]'
    median_str = f'{median_before:.4f}, {median_after:.4f}'
    print(f'{prefix} MRE median (CMA-SH-WARR, PDFM-TLE): {median_str}')
    print(f'{prefix} MRE median improvement: {median_improve:.4f}')

    plt.figure(figsize=(5, 6))
    sns.violinplot(
        data={'PDFM-TLE': mre_improvement},
        color='skyblue'
    )
    plt.ylabel('MRE改善率')
    plt.savefig(
        rf'{output_dir}\boxplot_mre_improvement.png',
        bbox_inches='tight', dpi=800
    )
    plt.savefig(
        rf'{output_dir}\boxplot_mre_improvement.pdf',
        bbox_inches='tight', dpi=800
    )
    plt.cla()
    plt.close('all')
    gc.collect()
    pos_ratio = np.mean(mre_improvement > 0) * 100
    prefix = '[plot_mre_violins]'
    print(f'{prefix} Positive improvement ratio: {pos_ratio:.2f}%')
