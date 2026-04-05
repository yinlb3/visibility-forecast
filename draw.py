#!user/bin.python3
# -*- coding: utf-8 -*-
"""
本模块为能见度数据可视化主入口, 职责包括:
1. 读取气象站点信息及能见度、降水、相对湿度等观测数据;
2. 对能见度进行分级统计 (月、小时、站点维度);
3. 绘制论文所需的各类统计图 (柱状图、箱线图、小提琴图、饼图、热图等).

Founded in 2024-04-18
Modified in 2026-03-17
@author: yinlb
"""

import os

import arrow
import numpy as np
import pandas as pd

from src import (
    case_study,
    data_prep,
    data_stats,
    forecast_eval,
    forecast_prep,
    plot_geo_mre,
    plot_mre_violin,
    plot_obs,
    plot_spatial,
    plot_temporal,
    temporal_eval,
)
from src.utils import format_time
from src.vis_acc import THRES, VisAcc

# 参与分析的省级行政区列表,用于初始筛选站点
PROVINCES = ('北京市', '上海市', '天津市', '安徽省', '福建省', '广东省', '江苏省', '江西省', '河北省', '河南省', '湖北省', '湖南省',
             '山东省', '山西省', '浙江省')

# Stage control flags: set to False to skip computation and read from cache
STAGES = {
    'data_prep': True,      # Stage 1: Data preparation (sta, vis, pre, rhu)
    'stats_calc': True,     # Stage 2: Statistics calculation (month/hour/sta stats)
    'obs_viz': True,        # Stage 3: Observation visualization
    'forecast_prep': True,  # Stage 4: Forecast data loading
    'type_eval': True,      # Stage 5: Weather type evaluation
    'dist_analysis': True,  # Stage 6: Distribution analysis
    'temporal': True,       # Stage 7: Temporal analysis
    'spatiotemporal': True, # Stage 8: Spatiotemporal visualization
    'case_study': True,     # Stage 9: 2024 case study
}

# Output directories
OUTPUT_DIR = r'D:\Project\vis\figures'
CACHE_DIR = r'D:\Project\vis\figures\csv'


def _save_stage1_cache(sta, vis, pre, rhu, vis_grade, month_ind, index_cjzxy) -> None:
    """Save stage 1 intermediate results for skipping."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    sta.to_csv(rf'{CACHE_DIR}\sta.csv', index=False)
    np.save(rf'{CACHE_DIR}\vis.npy', vis)
    np.save(rf'{CACHE_DIR}\pre.npy', pre)
    np.save(rf'{CACHE_DIR}\rhu.npy', rhu)
    np.save(rf'{CACHE_DIR}\vis_grade.npy', vis_grade)
    np.save(rf'{CACHE_DIR}\month_ind.npy', month_ind)
    np.save(rf'{CACHE_DIR}\index_cjzxy.npy', index_cjzxy)


def _load_stage1_cache():
    """Load stage 1 intermediate results."""
    sta = pd.read_csv(rf'{CACHE_DIR}\sta.csv')
    vis = np.load(rf'{CACHE_DIR}\vis.npy')
    pre = np.load(rf'{CACHE_DIR}\pre.npy')
    rhu = np.load(rf'{CACHE_DIR}\rhu.npy')
    vis_grade = np.load(rf'{CACHE_DIR}\vis_grade.npy')
    month_ind = np.load(rf'{CACHE_DIR}\month_ind.npy')
    index_cjzxy = np.load(rf'{CACHE_DIR}\index_cjzxy.npy')
    return sta, vis, pre, rhu, vis_grade, month_ind, index_cjzxy


def main() -> None:
    # 1. Observation data preparation
    if STAGES['data_prep']:
        # 1.1 Read station info and filter target provinces
        sta, index_zgdb = data_prep.read_sta(
            sta_path=r'D:\data\vis\sta2411.csv',
            provinces=PROVINCES
        )

        # 1.2 Load vis/precip/RH data and QC
        vis, pre, rhu = data_prep.load_obs(
            data_dir=r'D:\data\vis',
            index_zgdb=index_zgdb
        )

        # 1.3 Secondary filter for middle-lower Yangtze stations
        CJZXY_PROVINCES = (
            '湖北省', '湖南省', '江西省', '安徽省',
            '江苏省', '浙江省', '上海市'
        )
        sta, vis, pre, rhu, index_cjzxy = data_prep.filter_region(
            sta=sta,
            vis=vis,
            pre=pre,
            rhu=rhu,
            region_provinces=CJZXY_PROVINCES
        )
        print(sta)

        # 1.4 Grade visibility and build month index
        vis_grade = data_prep.grade_visibility(vis=vis)
        month_ind = data_prep.build_month_index(start_year=2020, n_hours=35064)

        # Save cache for skipping
        _save_stage1_cache(sta, vis, pre, rhu, vis_grade, month_ind, index_cjzxy)
    else:
        # Load from cache
        print('Loading stage 1 from cache...')
        sta, vis, pre, rhu, vis_grade, month_ind, index_cjzxy = _load_stage1_cache()
        # Ensure output dir exists for subsequent stages
        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Observation statistics and output
    if STAGES['stats_calc']:
        # 2.1 Build month/hour/sta stat dicts
        df_month = pd.DataFrame(data_stats.build_month_stats(
            vis_grade=vis_grade,
            pre=pre,
            rhu=rhu,
            month_ind=month_ind,
            thres=THRES
        ))
        df_hour = pd.DataFrame(data_stats.build_hour_stats(
            vis_grade=vis_grade,
            pre=pre,
            rhu=rhu,
            thres=THRES
        ))
        df_sta = pd.DataFrame(data_stats.build_sta_stats(
            sta=sta,
            vis=vis,
            vis_grade=vis_grade,
            pre=pre,
            rhu=rhu,
            thres=THRES
        ))

        # 2.2 Output stats to CSV
        data_stats.save_obs_stats(
            df_month=df_month,
            df_hour=df_hour,
            df_sta=df_sta,
            output_dir=OUTPUT_DIR
        )
    else:
        # Read pre-generated CSV
        print('Loading stage 2 from cache...')
        csv_dir = rf'{OUTPUT_DIR}\csv'
        df_month = pd.read_csv(filepath_or_buffer=rf'{csv_dir}\vis_month.csv', low_memory=False)
        df_hour = pd.read_csv(filepath_or_buffer=rf'{csv_dir}\vis_hour.csv', low_memory=False)
        df_sta = pd.read_csv(rf'{csv_dir}\vis_sta.csv', low_memory=False)

    # 3. Observation data visualization
    if STAGES['obs_viz']:
        # 3.1 Plot weather type pie charts by grade
        plot_obs.plot_obs_pies(
            vis_grade=vis_grade,
            pre=pre,
            rhu=rhu,
            thres=THRES,
            output_dir=OUTPUT_DIR
        )

        # 3.2 Plot violin/box for vis < 500m events
        plot_obs.plot_obs_violin_box(
            vis=vis,
            pre=pre,
            rhu=rhu,
            output_dir=OUTPUT_DIR
        )

        # 3.3 Plot monthly prob stacked bars and box
        plot_obs.plot_monthly_bars(
            df_month=df_month,
            output_dir=OUTPUT_DIR
        )
        plot_obs.plot_monthly_violins(
            vis=vis,
            pre=pre,
            rhu=rhu,
            month_ind=month_ind,
            output_dir=OUTPUT_DIR
        )

        # 3.4 Plot hourly prob stacked bars and box
        plot_obs.plot_hourly_bars(
            df_hour=df_hour,
            output_dir=OUTPUT_DIR
        )
        plot_obs.plot_hourly_violins(
            vis=vis,
            pre=pre,
            rhu=rhu,
            output_dir=OUTPUT_DIR
        )
        # 3.5 Plot station spatial distribution
        plot_obs.plot_sta_frequency_maps(
            sta=sta,
            df_sta=df_sta,
            output_dir=OUTPUT_DIR
        )
        plot_obs.plot_sta_mean_maps(
            sta=sta,
            df_sta=df_sta,
            output_dir=OUTPUT_DIR
        )
        plot_obs.show_cmap_legend(output_dir=OUTPUT_DIR)

    # 4. Forecast data loading and verification
    if STAGES['forecast_prep']:
        # 4.1 Load forecast verification data
        vis_ob, cma_sh_warr = forecast_prep.load_forecast_data(
            data_dir=r'D:\data\vis',
            index_cjzxy=index_cjzxy
        )
        pred_pdfm_tle0, pred_pdfm_tle1, pred_pdfm_tle2, pred_pdfm_tle3, pred_pdfm_tle4 = (
            forecast_prep.load_experiment_preds(data_dir=r'D:\data\vis')
        )

        # 4.2 Calculate overall metrics for all schemes
        forecast_prep.print_overall_metrics(vis_ob, cma_sh_warr, 'CMA-SH-WARR')
        forecast_prep.print_overall_metrics(vis_ob, pred_pdfm_tle0, 'Scheme 1')
        forecast_prep.print_overall_metrics(vis_ob, pred_pdfm_tle1, 'Scheme 2')
        forecast_prep.print_overall_metrics(vis_ob, pred_pdfm_tle2, 'Scheme 3')
        forecast_prep.print_overall_metrics(vis_ob, pred_pdfm_tle3, 'Scheme 4')
        forecast_prep.print_overall_metrics(vis_ob, pred_pdfm_tle4, 'Scheme 5')

    # 5. Weather-type forecast verification
    if STAGES['type_eval']:
        # 5.1 Calc quantitative/grade metrics by weather type
        val_wt = forecast_eval.load_weather_type(
            data_dir=r'D:\data\vis',
            index_cjzxy=index_cjzxy
        )
        qem, cem = forecast_eval.calc_weather_type_metrics(
            vis_ob=vis_ob,
            preds=(cma_sh_warr, pred_pdfm_tle0, pred_pdfm_tle1,
                   pred_pdfm_tle2, pred_pdfm_tle3, pred_pdfm_tle4),
            val_wt=val_wt
        )

        # 5.2 Plot weather type comparison
        forecast_eval.plot_weather_type_eval_bw(qem=qem[0, ...], filename='wt_cc_bw', max_y=0.4)
        forecast_eval.plot_weather_type_eval_bw(qem=qem[1, ...], filename='wt_mae_bw', max_y=10)
        forecast_eval.plot_weather_type_eval_bw(qem=qem[2, ...], filename='wt_rmse_bw', max_y=15)
        forecast_eval.plot_weather_type_eval_bw(qem=qem[3, ...], filename='wt_mre_bw', max_y=0.6)

        # 5.3 Calc and plot visibility CDF
        forecast_eval.plot_vis_cdf(
            vis_ob=vis_ob,
            cma_sh_warr=cma_sh_warr,
            pred_pdfm_tle0=pred_pdfm_tle0,
            output_dir=OUTPUT_DIR
        )

    # 6. Forecast data distribution analysis
    if STAGES['dist_analysis']:
        # 6.1 Plot 2D freq distribution (fcst vs obs)
        forecast_eval.plot_nwp_his2d(
            vis_ob=vis_ob,
            cma_sh_warr=cma_sh_warr,
            output_dir=OUTPUT_DIR
        )

        # 6.2 Plot grade freq distribution bars
        forecast_eval.plot_grade_frequency(
            vis_ob=vis_ob,
            preds={
                'CMA-SH-WARR': pred_pdfm_tle0,
                'OTS': pred_pdfm_tle1,
                'PDF': pred_pdfm_tle2,
                'TL': pred_pdfm_tle3,
                'PDFM-TLE': pred_pdfm_tle4,
            },
            thres=THRES,
            output_dir=OUTPUT_DIR
        )

    # 7. Temporal analysis
    if STAGES['temporal']:
        # 7.1-7.3 Calc init time/lead time/forecast time metrics
        temporal_eval.calc_temporal_metrics(
            vis_ob=vis_ob,
            cma_sh_warr=cma_sh_warr,
            pred_pdfm_tle0=pred_pdfm_tle0,
            pred_pdfm_tle2=pred_pdfm_tle2,
            output_dir=OUTPUT_DIR
        )

        # 7.4 Calc metrics by visibility type
        v_type = temporal_eval.load_v_type(
            data_dir=r'D:\data\vis',
            index_cjzxy=index_cjzxy
        )
        type_dfs = temporal_eval.calc_type_metrics(
            vis_ob=vis_ob,
            cma_sh_warr=cma_sh_warr,
            pred_pdfm_tle0=pred_pdfm_tle0,
            v_type=v_type
        )
        temporal_eval.save_type_results(type_dfs, output_dir=OUTPUT_DIR)

        # 7.5 Calc metrics by station
        sta_dfs = temporal_eval.calc_sta_metrics(
            vis_ob=vis_ob,
            cma_sh_warr=cma_sh_warr,
            pred_pdfm_tle2=pred_pdfm_tle2
        )
        temporal_eval.save_sta_results(sta_dfs, output_dir=OUTPUT_DIR)

    # 8. Spatiotemporal feature visualization
    if STAGES['spatiotemporal']:
        # 8.1 Plot init time-lead time heatmap (通用)
        plot_temporal.plot_hour_access_heatmaps(output_dir=OUTPUT_DIR)

        # 8.2 Plot lead time/forecast time TS4+ bars (通用)
        plot_temporal.plot_ts_comparison_bars(output_dir=OUTPUT_DIR)

        # 8.3 Plot station-level TS4+ spatial map (长江中下游)
        plot_spatial.plot_sta_ts4_maps(sta=sta, output_dir=OUTPUT_DIR)

        # # 8.4 Plot RMSE improvement spatial map (含华北内外对比，注释)
        # plot_spatial.plot_rmse_improvement_map(sta=sta, output_dir=OUTPUT_DIR)

        # # 8.5 Plot MRE vs geographic features (华北mask，注释)
        # plot_geo_mre.plot_geo_mre_relations(sta=sta, output_dir=OUTPUT_DIR)

        # 8.6 Plot MRE improvement box/violin (通用)
        plot_mre_violin.plot_mre_violins(sta=sta, output_dir=OUTPUT_DIR)

    # 9. 2024 independent sample case study (长江中下游_cjzxy数据)
    if STAGES['case_study']:
        # 9.1 Load data and compare PDFM/TLE metrics
        pred_pdfm2, pred_tle2 = case_study.load_2024_preds(data_dir=r'D:\data\vis')
        case_study.print_2024_overall_metrics(vis_ob, pred_pdfm2, pred_tle2)

        # 9.2 Calc TS4+ for 18 cases (phases and merged)
        cma_sh_warr_2024, pred_pdfm2_2024 = case_study.load_2024_eval_data(
            data_dir=r'D:\data\vis', index_cjzxy=index_cjzxy
        )
        vis_ob = np.reshape(vis_ob, shape=(-1, 24, 502))
        vis_ob[vis_ob >= 30000] = 30000
        vis_ob_, cma_sh_warr_, pred_pdfm2_ = case_study.align_forecast_times(
            vis_ob, cma_sh_warr_2024, pred_pdfm2_2024
        )
        case_study.analyze_case_studies(vis_ob_, cma_sh_warr_, pred_pdfm2_)


if __name__ == '__main__':
    print('Program draw.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(f'Program draw.py finished, total time: {format_time(total_elapsed)}')
