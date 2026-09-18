#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 3.2: Different init/lead time evaluation result plotting module.

Founded in 2026-04-14
Modified in 2026-08-06
@author: yinlb
"""

import contextlib
import gc
import io
import pathlib
import typing

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from meteva import base as meb    # type: ignore
from scipy import stats

from src import utils


mpl.use('Agg')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'SimSun']
mpl.rcParams['axes.unicode_minus'] = False


def _save_scatter_sta(
    sta0: pd.DataFrame,
    save_path: str,
    cmap: object,
    clevs: object,
    cfg: typing.Dict,
) -> None:
    """Helper: save scatter_sta to configured formats."""
    formats = cfg['draw']['plot']['output_formats']
    for fmt in formats:
        fmt_clean = fmt.lstrip('.').lower()
        path = str(pathlib.Path(save_path).with_suffix(f'.{fmt_clean}'))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                meb.tool.plot_tools.scatter_sta(
                    sta0=sta0.copy(), point_size=20,
                    map_extend=[108, 123, 24, 36], clevs=clevs, cmap=cmap,
                    extend='max', title=[''], save_path=path, dpi=800
                )
        except Exception as e:
            print(f'[_save_scatter_sta] Error saving {fmt_clean}: {e}')


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation excluding NaN values."""
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return np.nan
    return stats.pearsonr(x[mask], y[mask])[0]


def plot_hour_access_heatmaps(
    hour_access: np.ndarray,
    output_dir: str,
    cfg: typing.Dict
) -> None:
    """Plot init time-lead time 2D heatmap."""
    plot_cfg = cfg['draw']['plot']['hour_access_heatmaps']
    gc.collect()

    cmap = plot_cfg['cmap']
    linewidths = plot_cfg['linewidths']
    ts4_vmin = plot_cfg['ts4_vmin']
    ts4_vmax = plot_cfg['ts4_vmax']
    improvement_vmin = plot_cfg['improvement_vmin']
    improvement_vmax = plot_cfg['improvement_vmax']
    dpi = plot_cfg['dpi']

    # 1. Plot CMA-SH-WARR TS4+ heatmap
    # hour_access[0, :, :, 7]: scheme=0 (CMA),
    # init-hour, lead-hour, grade=7 (TS4+)
    sns.heatmap(
        hour_access[0, :, :, 7],
        cmap=cmap, vmin=ts4_vmin, vmax=ts4_vmax,
        linewidths=linewidths
    )
    plt.xticks(np.arange(24) + 0.5, [str(x) for x in range(1, 25)])
    yticks = [f'{x:02d}:00' for x in range(24)]
    plt.yticks(np.arange(24) + 0.5, yticks, rotation=0)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.xlabel('预报时效 (h) ')
    plt.ylabel('起报时次 (UTC) ')
    utils.save_figure(
        plt.gcf(), pathlib.Path(output_dir) / 'npw_hour_ts4+',
        cfg, bbox_inches='tight', dpi=dpi
    )
    plt.cla()
    plt.close('all')
    gc.collect()
    ts_min = np.min(hour_access[0, :, :, 7])
    ts_max = np.max(hour_access[0, :, :, 7])
    prefix = '[plot_hour_access_heatmaps]'
    print(f'{prefix} CMA-SH-WARR TS4+ range: {ts_min:.4f} ~ {ts_max:.4f}')

    # 2. Plot PDFM-TLE TS4+ heatmap
    # hour_access[1, :, :, 7]: scheme=1 (PDFM),
    # same dims as above
    sns.heatmap(
        hour_access[1, :, :, 7],
        cmap=cmap, vmin=ts4_vmin, vmax=ts4_vmax,
        linewidths=linewidths
    )
    plt.xticks(np.arange(24) + 0.5, [str(x + 1) for x in range(24)])
    yticks = [f'{x:02d}:00' for x in range(24)]
    plt.yticks(np.arange(24) + 0.5, yticks, rotation=0)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.xlabel('预报时效 (h) ')
    plt.ylabel('起报时次 (UTC) ')
    utils.save_figure(
        plt.gcf(), pathlib.Path(output_dir) / 'pdf-tl_hour_ts4+',
        cfg, bbox_inches='tight', dpi=dpi
    )
    plt.cla()
    plt.close('all')
    gc.collect()
    ts_min = np.min(hour_access[1, :, :, 7])
    ts_max = np.max(hour_access[1, :, :, 7])
    print(f'{prefix} PDFM-TLE TS4+ range: {ts_min:.4f} ~ {ts_max:.4f}')

    # 3. Plot TS improvement rate heatmap and locate max
    # Improvement = (PDFM - CMA) / CMA * 100,
    # guard against div-zero handled by vmin/vmax
    ts_before = hour_access[0, :, :, 7]
    ts_after = hour_access[1, :, :, 7]
    ts_improvement = (ts_after - ts_before) / ts_before * 100
    sns.heatmap(
        ts_improvement,
        cmap=cmap, vmin=improvement_vmin, vmax=improvement_vmax,
        linewidths=linewidths
    )
    plt.xticks(np.arange(24) + 0.5, [str(x + 1) for x in range(24)])
    yticks = [f'{x:02d}:00' for x in range(24)]
    plt.yticks(np.arange(24) + 0.5, yticks, rotation=0)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.xlabel('预报时效 (h) ')
    plt.ylabel('起报时次 (UTC) ')
    utils.save_figure(
        plt.gcf(), pathlib.Path(output_dir) / 'pdf-tl_hour_ts4+_improvement',
        cfg, bbox_inches='tight', dpi=dpi
    )
    plt.cla()
    plt.close('all')
    gc.collect()
    # np.where may return multiple equal-maximum cells; report the first one
    # as a representative location for the largest improvement.
    loc = np.where(ts_improvement == np.max(ts_improvement))
    ts_min = np.min(ts_improvement)
    ts_max = np.max(ts_improvement)
    print(f'{prefix} TS improvement range: {ts_min:.4f}% ~ {ts_max:.4f}%')
    print(f'{prefix} TS improvement mean: {np.mean(ts_improvement):.4f}%')
    init_h, lead_h = int(loc[0][0]), int(loc[1][0])
    loc_str = f'init_hour={init_h}, lead_hour={lead_h}'
    print(f'{prefix} TS improvement max location: {loc_str}')


def plot_ts_comparison_bars(
    df_vt_ts4: pd.DataFrame,
    df_fhour_ts4: pd.DataFrame,
    output_dir: str,
    cfg: typing.Dict
) -> None:
    """Plot TS4+ comparison bars for lead time and forecast time."""
    plot_cfg = cfg['draw']['plot']['ts_comparison_bars']
    figsize = plot_cfg['figsize']
    dpi = plot_cfg['dpi']
    bar_width = plot_cfg['bar_width']
    color_cma = plot_cfg['color_cma']
    color_pdfm = plot_cfg['color_pdfm']
    color_improvement = plot_cfg['color_improvement']
    vt_xlim = plot_cfg['vt_xlim']
    fhour_xlim = plot_cfg['fhour_xlim']
    primary_ylim = plot_cfg['primary_ylim']
    primary_yticks = plot_cfg['primary_yticks']
    twin_ylim = plot_cfg['twin_ylim']
    twin_yticks = plot_cfg['twin_yticks']
    ts_before = np.array(df_vt_ts4.loc[:, 'CMA-SH-WARR'])
    ts_after = np.array(df_vt_ts4.loc[:, 'PDFM-TLE'])
    ts_improvement = (ts_after - ts_before) / ts_before * 100
    ts_min = np.min(ts_improvement)
    ts_max = np.max(ts_improvement)
    prefix = '[plot_ts_comparison_bars]'
    print(f'{prefix} VT TS improve: {ts_min:.4f}% ~ {ts_max:.4f}%')
    ts_str = ' '.join(f'{x:.4f}' for x in ts_improvement)
    print(f'[plot_ts_comparison_bars] VT TS improve: [{ts_str}]')
    ts_max_val = np.max(df_vt_ts4.loc[:, 'PDFM-TLE'])
    print(f'[plot_ts_comparison_bars] VT PDFM-TLE max: {ts_max_val:.4f}')

    # 1. Plot lead-time (VT) TS4+ comparison bars
    # Twin-axis: left bars for TS values, right line for improvement rate
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.bar(
        x=df_vt_ts4.loc[:, 'vt'] + 1 - bar_width / 2,
        height=df_vt_ts4.loc[:, 'CMA-SH-WARR'],
        width=bar_width, color=color_cma, label='CMA-SH-WARR TS'
    )
    ax1.bar(
        x=df_vt_ts4.loc[:, 'vt'] + 1 + bar_width / 2,
        height=df_vt_ts4.loc[:, 'PDFM-TLE'],
        width=bar_width, color=color_pdfm, label='PDFM-TLE TS'
    )
    ax1.set_xlim(tuple(vt_xlim))
    ax1.set_xticks(range(1, 25), [str(x) for x in range(1, 25)])
    ax1.set_ylim(tuple(primary_ylim))
    ax1.set_yticks(tuple(primary_yticks))
    ax1.set_ylabel('TS')
    # Add secondary y-axis for improvement rate (percentage)
    ax2 = ax1.twinx()
    ax2.plot(
        df_vt_ts4.loc[:, 'vt'] + 1, ts_improvement, '-o',
        c=color_improvement, label='PDFM-TLE TS improve'
    )
    ax2.set_ylim(tuple(twin_ylim))
    ax2.set_yticks(tuple(twin_yticks))
    ax2.set_ylabel('TS改善率(%)')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    plt.xlabel('预报时效 (h) ')
    utils.save_figure(
        plt.gcf(), pathlib.Path(output_dir) / 'vis_vt_ts4+',
        cfg, bbox_inches='tight', dpi=dpi
    )
    plt.cla()
    plt.close('all')
    del fig, ax1, ax2
    gc.collect()

    ts_before = np.array(df_fhour_ts4.loc[:, 'CMA-SH-WARR'])
    ts_after = np.array(df_fhour_ts4.loc[:, 'PDFM-TLE'])
    ts_improvement = (ts_after - ts_before) / ts_before * 100
    ts_min = np.min(ts_improvement)
    ts_max = np.max(ts_improvement)
    print(f'{prefix} FHour TS improve: {ts_min:.4f}% ~ {ts_max:.4f}%')
    ts_str = ' '.join(f'{x:.4f}' for x in ts_improvement)
    print(f'[plot_ts_comparison_bars] FHour TS improve: [{ts_str}]')
    ts_max_val = np.max(df_fhour_ts4.loc[:, 'PDFM-TLE'])
    print(f'[plot_ts_comparison_bars] FHour PDFM-TLE max: {ts_max_val:.4f}')

    # 2. Plot forecast-time (FHour) TS4+ comparison bars
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.bar(
        x=df_fhour_ts4.loc[:, 'fhour'] - bar_width / 2,
        height=df_fhour_ts4.loc[:, 'CMA-SH-WARR'],
        width=bar_width, color=color_cma, label='CMA-SH-WARR的TS'
    )
    ax1.bar(
        x=df_fhour_ts4.loc[:, 'fhour'] + bar_width / 2,
        height=df_fhour_ts4.loc[:, 'PDFM-TLE'],
        width=bar_width, color=color_pdfm, label='PDFM-TLE的TS'
    )
    ax1.set_xlim(tuple(fhour_xlim))
    ax1.set_xticks(range(24), [f'{x:02d}:00' for x in range(24)])
    ax1.set_ylim(tuple(primary_ylim))
    ax1.set_yticks(tuple(primary_yticks))
    ax1.set_ylabel('TS')
    ax2 = ax1.twinx()
    ax2.plot(
        df_fhour_ts4.loc[:, 'fhour'], ts_improvement, '-o',
        c=color_improvement, label='PDFM-TLE的TS改善率'
    )
    ax2.set_ylim(tuple(twin_ylim))
    ax2.set_yticks(tuple(twin_yticks))
    ax2.set_ylabel('TS改善率 (%) ')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    plt.xlabel('预报时间 (UTC) ')
    utils.save_figure(
        plt.gcf(), pathlib.Path(output_dir) / 'vis_fhour_ts4+',
        cfg, bbox_inches='tight', dpi=dpi
    )
    plt.cla()
    plt.close('all')
    del fig, ax1, ax2
    gc.collect()


def plot_sta_ts4_maps(
    sta: pd.DataFrame,
    df_sta: pd.DataFrame,
    output_dir: str,
    cfg: typing.Dict
) -> None:
    """Plot station-level TS4+ and improvement rate spatial maps."""
    plot_cfg = cfg['draw']['plot']['sta_ts4_maps']
    prefix = '[plot_sta_ts4_maps]'
    # 1. Calc correlation between spatial coords and TS4+
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

    # 2. Build colormap and plot spatial distribution maps
    # Nested def_cmap_clevs: first get cmap object from range,
    # then attach discrete levels
    cmap_name = plot_cfg['cmap']
    cmap_obj = getattr(meb.cmaps, cmap_name)
    vmin = plot_cfg['vmin']
    vmax = plot_cfg['vmax']
    clevs = plot_cfg['clevs']
    cmap, clevs = meb.def_cmap_clevs(
        meb.def_cmap_clevs(cmap_obj, vmin=vmin, vmax=vmax)[0], clevs=clevs
    )

    sta_cols = ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')
    sta0 = sta.loc[:, sta_cols].copy()
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'CMA-SH-WARR']
    _save_scatter_sta(
        sta0, str(pathlib.Path(output_dir) / 'sta_ts4+_nwp'),
        cmap, clevs, cfg
    )

    sta0 = sta.loc[:, sta_cols].copy()
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'PDFM-TLE']
    _save_scatter_sta(
        sta0, str(pathlib.Path(output_dir) / 'sta_ts4+'), cmap, clevs, cfg
    )


def plot_mre_violins(
    sta: pd.DataFrame,
    df_sta: pd.DataFrame,
    output_dir: str,
    cfg: typing.Dict
) -> None:
    """Plot MRE and improvement rate violin/box plots."""
    plot_cfg = cfg['draw']['plot']['mre_violins']
    # 1. Plot MRE violin comparison (CMA vs PDFM)
    # Guard against zero-division when MRE before is zero
    mre_before = np.array(df_sta.loc[:, 'CMA-SH-WARR'])
    mre_after = np.array(df_sta.loc[:, 'PDFM-TLE'])
    with np.errstate(divide='ignore', invalid='ignore'):
        mre_improvement = np.where(
            mre_before == 0, np.nan, (mre_before - mre_after) / mre_before * 100
        )

    mre_figsize = plot_cfg['mre_figsize']
    mre_palette = plot_cfg['mre_palette']
    improvement_figsize = plot_cfg['improvement_figsize']
    improvement_color = plot_cfg['improvement_color']
    dpi = plot_cfg['dpi']

    plt.figure(figsize=mre_figsize)
    sns.violinplot(
        data={
            'CMA-SH-WARR': df_sta.loc[:, 'CMA-SH-WARR'],
            'PDFM-TLE': df_sta.loc[:, 'PDFM-TLE']
        },
        palette=mre_palette
    )
    plt.ylabel('MRE')
    utils.save_figure(
        plt.gcf(), pathlib.Path(output_dir) / 'boxplot_mre',
        cfg, bbox_inches='tight', dpi=dpi
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

    # 2. Plot MRE improvement rate violin
    plt.figure(figsize=improvement_figsize)
    sns.violinplot(data={'PDFM-TLE': mre_improvement}, color=improvement_color)
    plt.ylabel('MRE改善率')
    utils.save_figure(
        plt.gcf(), pathlib.Path(output_dir) / 'boxplot_mre_improvement',
        cfg, bbox_inches='tight', dpi=dpi
    )
    plt.cla()
    plt.close('all')
    gc.collect()
    pos_ratio = np.mean(mre_improvement > 0) * 100
    print(f'{prefix} Positive improvement ratio: {pos_ratio:.2f}%')
