#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry module for visibility data visualization. Responsibilities:
1. Read meteorological station info and obs data (visibility, precip, RH);
2. Grade-based statistics (monthly, hourly, station dimensions);
3. Draw statistical charts for papers (bars, box, violin, pie, heatmap).

Founded in 2024-04-18
Modified in 2026-04-08
@author: yinlb
"""

import os
import pathlib

import arrow
import numpy as np
import pandas as pd

from src import (
    case_study,
    data_prep,
    data_stats,
    forecast_dist,
    forecast_prep,
    forecast_type,
    plot_obs,
    plot_spatiotemporal,
    temporal_eval,
    utils,
    vis_acc,
)


def _save_stage1_cache(sta, vis, pre, rhu, vis_grade, month_ind,
                       idx_mlyr, cache_dir: str) -> None:
    """
    Save stage 1 intermediate results for skipping.

    Args:
        sta (pd.DataFrame): Station info DataFrame.
        vis (np.ndarray): Visibility array.
        pre (np.ndarray): Precipitation array.
        rhu (np.ndarray): Relative humidity array.
        vis_grade (np.ndarray): Visibility grade array.
        month_ind (np.ndarray): Month index array.
        idx_mlyr (np.ndarray): MLYR station filter index.
        cache_dir (str): Cache directory path.
    """
    os.makedirs(cache_dir, exist_ok=True)
    sta.to_csv(str(pathlib.Path(cache_dir) / 'sta.csv'), index=False)
    np.save(str(pathlib.Path(cache_dir) / 'vis.npy'), vis)
    np.save(str(pathlib.Path(cache_dir) / 'pre.npy'), pre)
    np.save(str(pathlib.Path(cache_dir) / 'rhu.npy'), rhu)
    np.save(str(pathlib.Path(cache_dir) / 'vis_grade.npy'), vis_grade)
    np.save(str(pathlib.Path(cache_dir) / 'month_ind.npy'), month_ind)
    np.save(str(pathlib.Path(cache_dir) / 'index_mlyr.npy'), idx_mlyr)


def _load_stage1_cache(cache_dir: str):
    """
    Load stage 1 intermediate results.

    Args:
        cache_dir (str): Cache directory path.

    Returns:
        tuple: (sta, vis, pre, rhu, vis_grade, month_ind, idx_mlyr).
    """
    sta = pd.read_csv(str(pathlib.Path(cache_dir) / 'sta.csv'))
    vis = np.load(str(pathlib.Path(cache_dir) / 'vis.npy'))
    pre = np.load(str(pathlib.Path(cache_dir) / 'pre.npy'))
    rhu = np.load(str(pathlib.Path(cache_dir) / 'rhu.npy'))
    vis_grade = np.load(str(pathlib.Path(cache_dir) / 'vis_grade.npy'))
    month_ind = np.load(str(pathlib.Path(cache_dir) / 'month_ind.npy'))
    # idx_mlyr: index for Middle-Lower Yangtze River region
    idx_mlyr = np.load(str(pathlib.Path(cache_dir) / 'index_mlyr.npy'))
    return sta, vis, pre, rhu, vis_grade, month_ind, idx_mlyr


def main() -> None:
    """
    Main function: execute the 10-stage visibility analysis pipeline.

    Stages:
        1. Load configuration
        2. Data preparation
        3. Observation statistics and output
        4. Observation data visualization
        5. Forecast data loading and verification
        6. Weather-type forecast verification
        7. Forecast data distribution analysis
        8. Temporal analysis
        9. Spatiotemporal feature visualization
        10. 2024 independent case study analysis
    """
    # 1. Load configuration from config/config.yaml
    # Local overrides are merged from config.local.{platform}.yaml
    print('Loading configuration...')
    cfg = utils.load_config()
    provinces = tuple(cfg['regions']['provinces'])
    stages = cfg['stages']
    output_dir = cfg['paths']['output_dir']
    cache_dir = cfg['paths']['cache_dir']
    data_dir = cfg['paths']['data_dir']
    thres = tuple(cfg['visibility']['grade_thresholds'])

    # ==========================================
    # Stage 2. Data preparation
    # ==========================================
    if stages['data_prep']:
        # 2.1 Read China eastern station info and obs data (Type1, no output)
        sta, idx_east_china = data_prep.read_sta(
            sta_path=str(pathlib.Path(data_dir) / 'sta2411.csv'),
            provinces=provinces
        )
        vis, pre, rhu = data_prep.load_obs(
            data_dir=data_dir,
            idx_east_china=idx_east_china
        )

        # 2.2 Filter to middle-lower Yangtze region and cache (Type 3)
        # MLYR_PROVINCES: provinces in Middle-Lower Yangtze River region
        mlyr_provinces = tuple(cfg['regions']['mlyr_provinces'])
        sta, vis, pre, rhu, idx_mlyr = data_prep.filter_region(
            sta=sta,
            vis=vis,
            pre=pre,
            rhu=rhu,
            region_provinces=mlyr_provinces
        )
        print(f'[Data Prep] Loaded {len(sta)} MLYR stations')
        vis_grade = data_prep.grade_visibility(vis=vis)
        month_ind = data_prep.build_month_index(start_year=2020, n_hours=35064)
        _save_stage1_cache(
            sta, vis, pre, rhu, vis_grade, month_ind, idx_mlyr, cache_dir
        )
    else:
        # 2.3 Load MLYR data from cache (Type3)
        print('[Cache] Loading stage 1 from cache...')
        sta, vis, pre, rhu, vis_grade, month_ind, idx_mlyr = (
            _load_stage1_cache(cache_dir)
        )
        # Ensure output dir exists for subsequent stages
        os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # Stage 3. Observation statistics and output (MLYR)
    # ==========================================
    if stages['stats_calc']:
        # 3.1 Build month/hour/sta stat dicts
        df_month = pd.DataFrame(data_stats.build_month_stats(
            vis_grade=vis_grade,
            pre=pre,
            rhu=rhu,
            month_ind=month_ind,
            thres=thres
        ))
        df_hour = pd.DataFrame(data_stats.build_hour_stats(
            vis_grade=vis_grade,
            pre=pre,
            rhu=rhu,
            thres=thres
        ))
        df_sta = pd.DataFrame(data_stats.build_sta_stats(
            sta=sta,
            vis=vis,
            vis_grade=vis_grade,
            pre=pre,
            rhu=rhu,
            thres=thres
        ))

        # 3.2 Output stats to CSV
        data_stats.save_obs_stats(
            df_month=df_month,
            df_hour=df_hour,
            df_sta=df_sta,
            output_dir=output_dir
        )
    else:
        # Read pre-generated CSV
        print('[Cache] Loading stage 2 from cache...')
        csv_dir = str(pathlib.Path(output_dir) / 'csv')
        csv_path_month = str(pathlib.Path(csv_dir) / 'vis_month.csv')
        df_month = pd.read_csv(
            filepath_or_buffer=csv_path_month, low_memory=False)
        csv_path_hour = str(pathlib.Path(csv_dir) / 'vis_hour.csv')
        df_hour = pd.read_csv(
            filepath_or_buffer=csv_path_hour, low_memory=False)
        csv_path_sta = str(pathlib.Path(csv_dir) / 'vis_sta.csv')
        df_sta = pd.read_csv(filepath_or_buffer=csv_path_sta, low_memory=False)

    # ==========================================
    # Stage 4. Observation data visualization (MLYR)
    # ==========================================
    if stages['obs_viz']:
        # 4.1 Plot weather type pie charts by grade
        plot_obs.plot_obs_pies(
            vis_grade=vis_grade,
            pre=pre,
            rhu=rhu,
            thres=thres,
            output_dir=output_dir
        )

        # 4.2 Plot violin/box for vis < 500m events
        plot_obs.plot_obs_violin_box(
            vis=vis,
            pre=pre,
            rhu=rhu,
            output_dir=output_dir
        )

        # 4.3 Plot monthly prob stacked bars and box
        plot_obs.plot_monthly_bars(
            df_month=df_month,
            output_dir=output_dir
        )
        plot_obs.plot_monthly_violins(
            vis=vis,
            pre=pre,
            rhu=rhu,
            month_ind=month_ind,
            output_dir=output_dir
        )

        # 4.4 Plot hourly prob stacked bars and box
        plot_obs.plot_hourly_bars(
            df_hour=df_hour,
            output_dir=output_dir
        )
        plot_obs.plot_hourly_violins(
            vis=vis,
            pre=pre,
            rhu=rhu,
            output_dir=output_dir
        )
        # 4.5 Plot station spatial distribution
        plot_obs.plot_sta_frequency_maps(
            sta=sta,
            df_sta=df_sta,
            output_dir=output_dir
        )
        plot_obs.plot_sta_mean_maps(
            sta=sta,
            df_sta=df_sta,
            output_dir=output_dir
        )

    # ==========================================
    # Stage 5. Forecast data loading and verification
    # ==========================================
    if stages['forecast_prep']:
        # 5.1 Load forecast data (Type1: basic data prep, no output)
        # cma_sh_warr: CMA Shanghai WARR model forecast
        vis_ob, cma_sh_warr = forecast_prep.load_forecast_data(
            data_dir=data_dir,
            idx_mlyr=idx_mlyr
        )
        pred_tle = forecast_prep.load_experiment_preds(
            data_dir=data_dir
        )
        # pred_pdfm_tle: PDF matching temporal lead experiment predictions
        pred_pdfm_tle0, pred_pdfm_tle1, pred_pdfm_tle2, pred_pdfm_tle3, \
            pred_pdfm_tle4 = pred_tle

        # 5.2 Calc and output overall metrics (Type3: MLYR output)
        forecast_prep.print_overall_metrics(vis_ob, cma_sh_warr, 'CMA-SH-WARR')
        for i, pred in enumerate(pred_tle):
            forecast_prep.print_overall_metrics(vis_ob, pred, f'Scheme {i+1}')

        # 5.3 Calc and save station-level metrics (Type3)
        forecast_prep.calc_and_save_station_metrics(
            vis_ob, cma_sh_warr, pred_tle[2], output_dir=output_dir
        )
    else:
        # 5.4 Load pre-calculated forecast data and metrics from cache
        print('[Cache] Loading stage 5 from cache...')
        vis_ob, cma_sh_warr = forecast_prep.load_forecast_data(
            data_dir=data_dir,
            idx_mlyr=idx_mlyr
        )
        pred_tle = forecast_prep.load_experiment_preds(
            data_dir=data_dir
        )
        pred_pdfm_tle0, pred_pdfm_tle1, pred_pdfm_tle2, pred_pdfm_tle3, \
            pred_pdfm_tle4 = pred_tle
        # Load station metrics to verify they exist
        _ = forecast_prep.load_station_metrics(output_dir=output_dir)

    # ==========================================
    # Stage 6. Weather-type forecast verification (MLYR)
    # ==========================================
    if stages['type_eval']:
        # 6.1 Calc quantitative/grade metrics by weather type
        val_wt = forecast_type.load_weather_type(
            data_dir=data_dir,
            idx_mlyr=idx_mlyr
        )
        # qem: quantitative evaluation metrics (R, MAE, RMSE, MRE)
        # cem: categorical/grade evaluation metrics (TS, FAR, MAR, POD)
        qem, cem = forecast_type.calc_weather_type_metrics(
            vis_ob=vis_ob,
            preds=(cma_sh_warr,) + pred_tle,
            val_wt=val_wt
        )
        # Save weather type metrics for cache
        forecast_type.save_weather_type_metrics(qem, cem, output_dir)

        # 6.2 Plot weather type comparison
        forecast_type.plot_weather_type_eval_bw(
            qem=qem[0, ...], filename='wt_cc_bw',
            max_y=0.4, output_dir=output_dir
        )
        forecast_type.plot_weather_type_eval_bw(
            qem=qem[1, ...], filename='wt_mae_bw', max_y=10,
            output_dir=output_dir
        )
        forecast_type.plot_weather_type_eval_bw(
            qem=qem[2, ...], filename='wt_rmse_bw', max_y=15,
            output_dir=output_dir
        )
        forecast_type.plot_weather_type_eval_bw(
            qem=qem[3, ...], filename='wt_mre_bw', max_y=0.6,
            output_dir=output_dir
        )

        # 6.3 Calc and plot visibility CDF
        forecast_type.plot_vis_cdf(
            vis_ob=vis_ob,
            cma_sh_warr=cma_sh_warr,
            pred_pdfm_tle0=pred_pdfm_tle0,
            output_dir=output_dir
        )
    else:
        # 6.4 Load pre-calculated weather type metrics from cache
        print('[Cache] Loading stage 6 from cache...')
        qem, cem = forecast_type.load_weather_type_metrics(output_dir)

    # ==========================================
    # Stage 7. Forecast data distribution analysis (MLYR)
    # ==========================================
    if stages['dist_analysis']:
        # NOTE: plot_nwp_his2d temporarily disabled due to data quality issues
        # (CMA-SH-WARR shows weak correlation with obs, resulting in
        # scattered distribution instead of diagonal concentration)
        # forecast_dist.plot_nwp_his2d(
        #     vis_ob=vis_ob,
        #     cma_sh_warr=cma_sh_warr,
        #     output_dir=output_dir,
        #     n_bins=100,
        #     max_points=100000,
        # )

        # 7.1 Plot grade freq distribution bars
        forecast_dist.plot_grade_frequency(
            vis_ob=vis_ob,
            preds={
                'CMA-SH-WARR': pred_pdfm_tle0,
                'OTS': pred_pdfm_tle1,
                'PDF': pred_pdfm_tle2,
                'TL': pred_pdfm_tle3,
                'PDFM-TLE': pred_pdfm_tle4,
            },
            thres=thres,
            output_dir=output_dir
        )

    # ==========================================
    # Stage 8. Temporal analysis (MLYR)
    # ==========================================
    if stages['temporal']:
        # 8.1-8.3 Calc init time/lead time/forecast time metrics
        temporal_eval.calc_temporal_metrics(
            vis_ob=vis_ob,
            cma_sh_warr=cma_sh_warr,
            pred_pdfm_tle0=pred_pdfm_tle0,
            pred_pdfm_tle2=pred_pdfm_tle2,
            output_dir=output_dir
        )

        # 8.4 Calc metrics by visibility type
        v_type = temporal_eval.load_v_type(
            data_dir=data_dir,
            idx_mlyr=idx_mlyr
        )
        type_dfs = temporal_eval.calc_type_metrics(
            vis_ob=vis_ob,
            cma_sh_warr=cma_sh_warr,
            pred_pdfm_tle0=pred_pdfm_tle0,
            v_type=v_type
        )
        temporal_eval.save_type_results(type_dfs, output_dir=output_dir)
    else:
        # 8.5 Load pre-calculated temporal metrics from cache
        print('[Cache] Loading stage 8 from cache...')
        _ = temporal_eval.load_temporal_metrics(output_dir)

    # ==========================================
    # Stage 9. Spatiotemporal feature visualization (MLYR)
    # ==========================================
    if stages['spatiotemporal']:
        # 9.1 Plot init time-lead time heatmap (MLYR)
        plot_spatiotemporal.plot_hour_access_heatmaps(output_dir=output_dir)

        # 9.2 Plot lead time/forecast time TS4+ bars (MLYR)
        plot_spatiotemporal.plot_ts_comparison_bars(output_dir=output_dir)

        # 9.3 Plot station-level TS4+ spatial map (MLYR)
        plot_spatiotemporal.plot_sta_ts4_maps(sta=sta, output_dir=output_dir)

        # 9.4 Plot MRE improvement box/violin (MLYR)
        plot_spatiotemporal.plot_mre_violins(sta=sta, output_dir=output_dir)

    # ==========================================
    # Stage 10. 2024 independent case study analysis
    # ==========================================
    if stages['case_study']:
        # 10.1 Load 2024 obs and forecast data (Type1, no output)
        vis_ob_2024 = case_study.load_2024_obs(
            data_dir=data_dir, idx_mlyr=idx_mlyr
        )
        pred_pdfm2, pred_tle2 = case_study.load_2024_preds(
            data_dir=data_dir
        )
        cma_sh_warr_2024, pred_pdfm2_2024 = case_study.load_2024_eval_data(
            data_dir=data_dir, idx_mlyr=idx_mlyr
        )

        # 10.2 Output PDFM/TLE overall metrics (Type3: MLYR output)
        case_study.print_2024_overall_metrics(
            vis_ob_2024, pred_pdfm2, pred_tle2
        )

        # 10.3 Calc TS4+ for 18 cases (Type3)
        vis_ob_, cma_sh_warr_, pred_pdfm2_ = case_study.align_forecast_times(
            vis_ob_2024, cma_sh_warr_2024, pred_pdfm2_2024
        )
        case_study.analyze_case_studies(vis_ob_, cma_sh_warr_, pred_pdfm2_)


if __name__ == '__main__':
    print('Program draw.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    elapsed_str = utils.format_time(total_elapsed)
    print(f'Program draw.py finished, total time: {elapsed_str}')
