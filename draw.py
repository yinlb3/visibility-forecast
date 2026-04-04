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


def main() -> None:
    # 1. 观测数据准备与预处理
    # 1.1 读取站点信息并筛选目标省份
    sta, index_zgdb = data_prep.read_sta(
        sta_path=r'D:\data\vis\sta2411.csv',
        provinces=PROVINCES
    )

    # 1.2 加载能见度/降水/湿度数据并质控
    vis, pre, rhu = data_prep.load_obs(
        data_dir=r'D:\data\vis',
        index_zgdb=index_zgdb
    )

    # 1.3 二次筛选长江中下游站点
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

    # 1.4 能见度六级分级与月份索引生成
    vis_grade = data_prep.grade_visibility(vis=vis)
    month_ind = data_prep.build_month_index(start_year=2020, n_hours=35064)

    # 2. 观测数据统计与输出
    # 2.1 构建月/时/站三级统计字典
    df_month = data_stats.build_month_stats(
        vis_grade=vis_grade,
        pre=pre,
        rhu=rhu,
        month_ind=month_ind,
        thres=THRES
    )
    df_hour = data_stats.build_hour_stats(
        vis_grade=vis_grade,
        pre=pre,
        rhu=rhu,
        thres=THRES
    )
    df_sta = data_stats.build_sta_stats(
        sta=sta,
        vis=vis,
        vis_grade=vis_grade,
        pre=pre,
        rhu=rhu,
        thres=THRES
    )

    # 2.2 输出统计结果为 CSV
    data_stats.save_obs_stats(
        df_month=df_month,
        df_hour=df_hour,
        df_sta=df_sta,
        output_dir=r'D:\Project\vis\图'
    )

    # 3. 观测数据可视化
    # 3.1 绘制各级低能见度事件的天气类型占比饼图
    plot_obs.plot_obs_pies(
        vis_grade=vis_grade,
        pre=pre,
        rhu=rhu,
        thres=THRES,
        output_dir=r'D:\Project\vis\图'
    )

    # 3.2 绘制低能见度事件(vis < 500m)的小提琴图与箱线图
    plot_obs.plot_obs_violin_box(
        vis=vis,
        pre=pre,
        rhu=rhu,
        output_dir=r'D:\Project\vis\图'
    )

    # 3.3 绘制月份概率堆叠柱状图与箱线图
    df_month = pd.read_csv(filepath_or_buffer=r'D:\Project\vis\图\vis_month.csv', low_memory=False)
    plot_obs.plot_monthly_bars(
        df_month=df_month,
        output_dir=r'D:\Project\vis\图'
    )
    plot_obs.plot_monthly_violins(
        vis=vis,
        pre=pre,
        rhu=rhu,
        month_ind=month_ind,
        output_dir=r'D:\Project\vis\图'
    )

    # 3.4 绘制小时概率堆叠柱状图与箱线图
    df_hour = pd.read_csv(filepath_or_buffer=r'D:\Project\vis\图\vis_hour.csv', low_memory=False)
    plot_obs.plot_hourly_bars(
        df_hour=df_hour,
        output_dir=r'D:\Project\vis\图'
    )
    plot_obs.plot_hourly_violins(
        vis=vis,
        pre=pre,
        rhu=rhu,
        output_dir=r'D:\Project\vis\图'
    )
    # 3.5 绘制站点空间分布图
    df_sta = pd.read_csv(r'D:\Project\vis\图\vis_sta.csv', low_memory=False)
    plot_obs.plot_sta_frequency_maps(
        sta=sta,
        df_sta=df_sta,
        output_dir=r'D:\Project\vis\图'
    )
    plot_obs.plot_sta_mean_maps(
        sta=sta,
        df_sta=df_sta,
        output_dir=r'D:\Project\vis\图'
    )
    plot_obs.show_cmap_legend(output_dir=r'D:\Project\vis\图')

    # 4. 预报数据加载与整体检验
    # 4.1 加载预报检验数据
    vis_ob, cma_sh_warr = forecast_prep.load_forecast_data(
        data_dir=r'D:\data\vis',
        index_cjzxy=index_cjzxy
    )
    pred_pdfm_tle0, pred_pdfm_tle1, pred_pdfm_tle2, pred_pdfm_tle3, pred_pdfm_tle4 = (
        forecast_prep.load_experiment_preds(data_dir=r'D:\data\vis')
    )

    # 4.2 计算各方案整体检验指标
    forecast_prep.print_overall_metrics(vis_ob, cma_sh_warr, 'CMA-SH-WARR')
    forecast_prep.print_overall_metrics(vis_ob, pred_pdfm_tle0, '方案一')
    forecast_prep.print_overall_metrics(vis_ob, pred_pdfm_tle1, '方案二')
    forecast_prep.print_overall_metrics(vis_ob, pred_pdfm_tle2, '方案三')
    forecast_prep.print_overall_metrics(vis_ob, pred_pdfm_tle3, '方案四')
    forecast_prep.print_overall_metrics(vis_ob, pred_pdfm_tle4, '方案五')

    # 5. 分类型预报检验与绘图
    # 5.1 按天气类型计算定量与分级检验指标
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

    # 5.2 绘制天气类型检验对比图
    forecast_eval.plot_weather_type_eval_bw(qem=qem[0, ...], filename='wt_cc_bw', max_y=0.4)
    forecast_eval.plot_weather_type_eval_bw(qem=qem[1, ...], filename='wt_mae_bw', max_y=10)
    forecast_eval.plot_weather_type_eval_bw(qem=qem[2, ...], filename='wt_rmse_bw', max_y=15)
    forecast_eval.plot_weather_type_eval_bw(qem=qem[3, ...], filename='wt_mre_bw', max_y=0.6)

    # 5.3 计算并绘制能见度累积分布函数(CDF)
    forecast_eval.plot_vis_cdf(
        vis_ob=vis_ob,
        cma_sh_warr=cma_sh_warr,
        pred_pdfm_tle0=pred_pdfm_tle0,
        output_dir=r'D:\Project\vis\图'
    )

    # 6. 预报数据分布分析
    # 6.1 绘制预报-实况二维频率分布图
    forecast_eval.plot_nwp_his2d(
        vis_ob=vis_ob,
        cma_sh_warr=cma_sh_warr,
        output_dir=r'D:\Project\vis\图'
    )

    # 6.2 绘制各方案分级频率分布柱状图
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
        output_dir=r'D:\Project\vis\图'
    )

    # 7. 时效分析
    # 7.1-7.3 计算起报时次/预报时效/预报时间检验指标并保存
    temporal_eval.calc_temporal_metrics(
        vis_ob=vis_ob,
        cma_sh_warr=cma_sh_warr,
        pred_pdfm_tle0=pred_pdfm_tle0,
        pred_pdfm_tle2=pred_pdfm_tle2,
        output_dir=r'D:\Project\vis\图'
    )

    # 7.4 按能见度类型计算检验指标
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
    temporal_eval.save_type_results(type_dfs, output_dir=r'D:\Project\vis\图')

    # 7.5 按站点计算检验指标
    sta_dfs = temporal_eval.calc_sta_metrics(
        vis_ob=vis_ob,
        cma_sh_warr=cma_sh_warr,
        pred_pdfm_tle2=pred_pdfm_tle2,
        n_sta=502
    )
    temporal_eval.save_sta_results(sta_dfs, output_dir=r'D:\Project\vis\图')

    # 8. 时空特征与改善率可视化
    # 8.1 绘制起报时次-预报时效二维热图
    plot_temporal.plot_hour_access_heatmaps(output_dir=r'D:\Project\vis\图')

    # 8.2 绘制预报时效与预报时间的 TS4+ 对比柱状图
    plot_temporal.plot_ts_comparison_bars(output_dir=r'D:\Project\vis\图')

    # 8.3 绘制站点级 TS4+ 空间分布图
    plot_spatial.plot_sta_ts4_maps(sta=sta, output_dir=r'D:\Project\vis\图')

    # 8.4 绘制 RMSE 改善率空间分布图
    plot_spatial.plot_rmse_improvement_map(sta=sta, output_dir=r'D:\Project\vis\图')

    # 8.5 绘制 MRE 与地理要素关系图
    plot_geo_mre.plot_geo_mre_relations(sta=sta, output_dir=r'D:\Project\vis\图')

    # 8.6 绘制 MRE 改善率箱线图/小提琴图
    plot_mre_violin.plot_mre_violins(sta=sta, output_dir=r'D:\Project\vis\图')

    # 9. 2024 年独立样本个例分析
    # 9.1 加载数据并对比 PDFM/TLE 整体指标
    pred_pdfm2, pred_tle2 = case_study.load_2024_preds(data_dir=r'D:\data\vis')
    case_study.print_2024_overall_metrics(vis_ob, pred_pdfm2, pred_tle2)

    # 9.2 计算 18 组个例各阶段及合并后的 TS4+
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
