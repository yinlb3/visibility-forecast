#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry module for visibility data visualization. Responsibilities:
1. Read meteorological station info and obs data (visibility, precip, RH);
2. Grade-based statistics (monthly, hourly, station dimensions);
3. Draw statistical charts for papers (bars, box, violin, pie, heatmap).

Founded in 2024-04-18
Modified in 2026-09-30
@author: yinlb
"""

import os
import pathlib

import arrow
import pandas as pd

from src import (
    utils,
    p1_config_data as p1,
    p2_1_dist_feature as p21,
    p2_2_event_type as p22,
    p2_3_temporal as p23,
    p2_4_spatial as p24,
    p3_1_eval_calc as p31,
    p3_2_init_lead as p32,
    p3_3_wt_ts4 as p33,
    p3_4_ablation as p34,
    p3_5_case_study as p35,
)


def _log_section(name: str, start: arrow.Arrow) -> None:
    """Print section completion with elapsed time."""
    elapsed = (arrow.now() - start).total_seconds()
    print(f'[{name}] completed in {utils.format_time(elapsed)}')


def main() -> None:
    """
    Main function: execute the visibility analysis pipeline.

    Parts:
        1. Configuration and data loading
        2. Low visibility event distribution
        3. Rapid update visibility forecast
    """
    # ==========================================
    # Part 1. Configuration and data loading
    # ==========================================
    print()
    print('=' * 50)
    print('Part 1. Configuration and data loading')
    print('=' * 50)
    part_start = arrow.now()

    # Load merged configuration (base + local override)
    cfg = utils.load_config()
    stages = cfg['draw']['stages']

    print('Loading configuration...')
    provinces = tuple(cfg['draw']['regions']['provinces'])
    output_dir = cfg['paths']['output_dir']
    cache_dir = cfg['paths']['cache_dir']
    data_dir = cfg['paths']['data_dir']
    thres = tuple(cfg['visibility']['grade_thresholds'])

    # Create the output directory up front: every stage writes below it
    os.makedirs(output_dir, exist_ok=True)

    # Either run full data prep or load from cached stage 1 results
    if stages['stage_1_data_prep']:
        sta, idx_east_china = p1.read_station(
            sta_path=str(pathlib.Path(data_dir) / 'sta2411.csv'),
            provinces=provinces
        )
        vis, pre, rhu = p1.load_obs(
            data_dir=data_dir,
            idx_east_china=idx_east_china
        )
        mlyr_provinces = tuple(cfg['draw']['regions']['mlyr_provinces'])
        sta, vis, pre, rhu, idx_mlyr = p1.filter_region(
            sta=sta, vis=vis, pre=pre, rhu=rhu,
            region_provinces=mlyr_provinces
        )
        print(f'[Data Prep] Loaded {len(sta)} MLYR stations')
        vis_grade = p1.grade_visibility(vis=vis)
        month_ind = p1.build_month_index(start_year=2020, n_hours=35064)
        p1._save_stage1_cache(
            sta, vis, pre, rhu, vis_grade, month_ind, idx_mlyr, cache_dir
        )
    else:
        print('[Cache] Loading stage 1 from cache...')
        sta, vis, pre, rhu, vis_grade, month_ind, idx_mlyr = (
            p1._load_stage1_cache(cache_dir)
        )

    # Load forecast datasets: CMA-SH-WARR + 5 PDFM-TLE experiment schemes
    print('[Data Prep] Loading forecast data...')
    vis_ob, cma_sh_warr = p1.load_forecast_data(
        data_dir=data_dir, idx_mlyr=idx_mlyr
    )
    pred_tle = p1.load_experiment_preds(data_dir=data_dir)
    pred_pdfm_tle0, pred_pdfm_tle1, pred_pdfm_tle2, pred_pdfm_tle3, \
        pred_pdfm_tle4 = pred_tle

    _log_section('Part 1', part_start)
    print()

    # ==========================================
    # Part 2. Low visibility event distribution
    # ==========================================
    print('=' * 50)
    print('Part 2. Low visibility event distribution')
    print('=' * 50)
    part_start = arrow.now()

    # 2.1 Distribution feature calculation
    print('2.1 Distribution feature calculation')
    sec_start = arrow.now()
    if stages['stage_2_1_dist_feature']:
        df_month = pd.DataFrame(p21.build_month_stats(
            vis_grade=vis_grade, pre=pre, rhu=rhu,
            month_ind=month_ind, thres=thres
        ))
        df_hour = pd.DataFrame(p21.build_hour_stats(
            vis_grade=vis_grade, pre=pre, rhu=rhu, thres=thres
        ))
        df_sta = pd.DataFrame(p21.build_sta_stats(
            sta=sta, vis=vis, vis_grade=vis_grade,
            pre=pre, rhu=rhu, thres=thres
        ))
        p21.save_obs_stats(
            df_month=df_month, df_hour=df_hour,
            df_sta=df_sta, output_dir=output_dir
        )
    else:
        print('[Cache] Loading stage 2 from cache...')
        csv_dir = str(pathlib.Path(output_dir) / 'csv')
        df_month = pd.read_csv(
            filepath_or_buffer=str(pathlib.Path(csv_dir) / 'vis_month.csv'),
            low_memory=False
        )
        df_hour = pd.read_csv(
            filepath_or_buffer=str(pathlib.Path(csv_dir) / 'vis_hour.csv'),
            low_memory=False
        )
        df_sta = pd.read_csv(
            filepath_or_buffer=str(pathlib.Path(csv_dir) / 'vis_sta.csv'),
            low_memory=False
        )
    _log_section('2.1', sec_start)
    print()

    # 2.2 Event type proportion output and plotting
    print('2.2 Event type proportion output and plotting')
    sec_start = arrow.now()
    if stages['stage_2_2_event_type']:
        p22.plot_obs_pies(
            vis_grade=vis_grade, pre=pre, rhu=rhu, thres=thres,
            output_dir=output_dir, cfg=cfg
        )
        p22.plot_obs_violin_box(
            vis=vis, pre=pre, rhu=rhu, output_dir=output_dir, cfg=cfg
        )
    _log_section('2.2', sec_start)
    print()

    # 2.3 Temporal distribution feature plotting
    print('2.3 Temporal distribution feature plotting')
    sec_start = arrow.now()
    if stages['stage_2_3_temporal']:
        p23.plot_monthly_bars(
            df_month=df_month, output_dir=output_dir, cfg=cfg
        )
        p23.plot_monthly_violins(
            vis=vis, pre=pre, rhu=rhu, month_ind=month_ind,
            output_dir=output_dir, cfg=cfg
        )
        p23.plot_hourly_bars(
            df_hour=df_hour, output_dir=output_dir, cfg=cfg
        )
        p23.plot_hourly_violins(
            vis=vis, pre=pre, rhu=rhu, output_dir=output_dir, cfg=cfg
        )
    _log_section('2.3', sec_start)
    print()

    # 2.4 Spatial distribution feature plotting
    print('2.4 Spatial distribution feature plotting')
    sec_start = arrow.now()
    if stages['stage_2_4_spatial']:
        p24.plot_sta_frequency_maps(
            sta=sta, df_sta=df_sta, output_dir=output_dir, cfg=cfg
        )
        p24.plot_sta_mean_maps(
            sta=sta, df_sta=df_sta, output_dir=output_dir, cfg=cfg
        )
    _log_section('2.4', sec_start)
    print()

    _log_section('Part 2', part_start)
    print()

    # ==========================================
    # Part 3. Rapid update visibility forecast
    # ==========================================
    print('=' * 50)
    print('Part 3. Rapid update visibility forecast')
    print('=' * 50)
    part_start = arrow.now()

    # 3.1 Evaluation result calculation (includes overall result output)
    print('3.1 Evaluation result calculation')
    sec_start = arrow.now()
    overall_strs = list()
    type_str = ''

    if stages['stage_3_1_eval_calc']:
        overall_strs.append(
            p31.format_overall_metrics(vis_ob, cma_sh_warr, 'CMA-SH-WARR')
        )
        for i, pred in enumerate(pred_tle):
            overall_strs.append(
                p31.format_overall_metrics(vis_ob, pred, f'Scheme {i+1}')
            )
        station_metrics = p31.calc_and_save_station_metrics(
            vis_ob, cma_sh_warr, pred_tle[2], output_dir=output_dir
        )

        val_wt = p31.load_weather_type(
            data_dir=data_dir, idx_mlyr=idx_mlyr
        )
        qem, cem = p31.calc_weather_type_metrics(
            vis_ob=vis_ob,
            preds=(cma_sh_warr,) + pred_tle,
            val_wt=val_wt
        )
        p31.save_weather_type_metrics(qem, cem, output_dir)

        hour_access, df_vt_ts4, df_fhour_ts4 = p31.calc_temporal_metrics(
            vis_ob=vis_ob,
            cma_sh_warr=cma_sh_warr,
            pred_pdfm_tle0=pred_pdfm_tle0,
            pred_pdfm_tle2=pred_pdfm_tle2,
            output_dir=output_dir
        )
        lve_type = p31.load_lve_type(
            data_dir=data_dir, idx_mlyr=idx_mlyr
        )
        type_dfs, type_str = p31.calc_type_metrics(
            vis_ob=vis_ob,
            cma_sh_warr=cma_sh_warr,
            pred_pdfm_tle0=pred_pdfm_tle0,
            lve_type=lve_type
        )
        p31.save_type_results(type_dfs, output_dir=output_dir)
    else:
        print('[Cache] Loading 3.1 evaluation metrics from cache...')
        station_metrics = p31.load_station_metrics(output_dir=output_dir)
        if stages['stage_3_3_wt_ts4'] or stages['stage_3_4_ablation']:
            qem, cem = p31.load_weather_type_metrics(output_dir)
        hour_access = p31.load_temporal_metrics(output_dir)
        csv_dir = pathlib.Path(output_dir) / 'csv'
        df_vt_ts4 = pd.read_csv(
            str(csv_dir / 'vis_vt_ts4+.csv'), low_memory=False
        )
        df_fhour_ts4 = pd.read_csv(
            str(csv_dir / 'vis_ft_ts4+.csv'), low_memory=False
        )

    for s in overall_strs:
        print(s)
    if type_str:
        print(type_str)
    _log_section('3.1', sec_start)
    print()

    # 3.2 Different init/lead time evaluation result plotting
    print('3.2 Different init/lead time evaluation result plotting')
    sec_start = arrow.now()
    if stages['stage_3_2_init_lead']:
        p32.plot_hour_access_heatmaps(
            hour_access, output_dir=output_dir, cfg=cfg
        )
        p32.plot_ts_comparison_bars(
            df_vt_ts4, df_fhour_ts4, output_dir=output_dir, cfg=cfg
        )
        p32.plot_sta_ts4_maps(
            sta=sta, df_sta=station_metrics['ts4+'],
            output_dir=output_dir, cfg=cfg
        )
        p32.plot_mre_violins(
            sta=sta, df_sta=station_metrics['mre'],
            output_dir=output_dir, cfg=cfg
        )
    _log_section('3.2', sec_start)
    print()

    # 3.3 Weather type evaluation
    print('3.3 Weather type evaluation')
    sec_start = arrow.now()
    if stages['stage_3_3_wt_ts4']:
        wt_ts4_str = p33.format_wt_ts4_impr(
            qem=qem, vis_ob=vis_ob, cma_sh_warr=cma_sh_warr,
            pred_pdfm_tle2=pred_pdfm_tle2
        )
        print(wt_ts4_str)
    _log_section('3.3', sec_start)
    print()

    # 3.4 Ablation experiment result output
    print('3.4 Ablation experiment result output')
    sec_start = arrow.now()
    if stages['stage_3_4_ablation']:
        p34.plot_weather_type_eval_bw(
            qem=qem[0, ...], filename='wt_cc_bw',
            max_y=0.4, output_dir=output_dir, cfg=cfg
        )
        p34.plot_weather_type_eval_bw(
            qem=qem[1, ...], filename='wt_mae_bw',
            max_y=10000, output_dir=output_dir, cfg=cfg, unit='m'
        )
        p34.plot_weather_type_eval_bw(
            qem=qem[2, ...], filename='wt_rmse_bw',
            max_y=12000, output_dir=output_dir, cfg=cfg, unit='m'
        )
        p34.plot_weather_type_eval_bw(
            qem=qem[3, ...], filename='wt_mre_bw',
            max_y=0.6, output_dir=output_dir, cfg=cfg
        )
        p34.plot_vis_cdf(
            vis_ob=vis_ob, cma_sh_warr=cma_sh_warr,
            pred_pdfm_tle0=pred_pdfm_tle0,
            output_dir=output_dir, cfg=cfg
        )
    _log_section('3.4', sec_start)
    print()

    # 3.5 Operational application case study evaluation result output
    print('3.5 Operational application case study evaluation result output')
    sec_start = arrow.now()
    if stages['stage_3_5_case_study']:
        vis_ob_2024 = p35.load_2024_obs(
            data_dir=data_dir, idx_mlyr=idx_mlyr
        )
        pred_pdfm2, pred_tle2 = p35.load_2024_preds(
            data_dir=data_dir
        )
        cma_sh_warr_2024, pred_pdfm2_2024 = p35.load_2024_eval_data(
            data_dir=data_dir, idx_mlyr=idx_mlyr
        )
        p35.print_2024_overall_metrics(
            vis_ob_2024, pred_pdfm2, pred_tle2
        )
        vis_ob_, cma_sh_warr_, pred_pdfm2_ = p35.align_forecast_times(
            vis_ob_2024, cma_sh_warr_2024, pred_pdfm2_2024
        )
        p35.analyze_case_studies(vis_ob_, cma_sh_warr_, pred_pdfm2_)
    _log_section('3.5', sec_start)
    print()

    _log_section('Part 3', part_start)


if __name__ == '__main__':
    print('Program draw.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    elapsed_str = utils.format_time(total_elapsed)
    print(f'Program draw.py finished, total time: {elapsed_str}')
