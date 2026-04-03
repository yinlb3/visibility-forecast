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
import sys
import typing

import arrow
import gc
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from meteva import base as meb
from scipy import stats


# 参与分析的省级行政区列表,用于初始筛选站点
PROVINCES = ('北京市', '上海市', '天津市', '安徽省', '福建省', '广东省', '江苏省', '江西省', '河北省', '河南省', '湖北省', '湖南省',
             '山东省', '山西省', '浙江省')
# 能见度六级分级阈值(单位:米),从好到差依次为 <10000, <2000, <1000, <500, <200, <50
THRES = (10000., 2000., 1000., 500., 200., 50.)


def plot_weather_type_eval_bw(qem: np.ndarray, filename: str, max_y: float) -> None:
    """
    绘制按天气类型 (降水、雾、霾) 分类的检验指标对比柱状图 (黑白风格).

    :param qem: 形状为 (6, 3) 的定量检验指标数组, 6 个方案 * 3 类天气;
    :param filename: 输出文件名 (不含扩展名);
    :param max_y: y 轴实际物理量最大值, 用于归一化显示.
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


class VisAcc:
    """
    能见度预报检验指标计算类.

    根据观测 (ob) 与预报 (pr) 数组, 自动计算能见度分级, 并构建混淆矩阵, 
    提供定量指标 (ME、MAE、RMSE、MRE、R) 和分级指标 (TS、ETS、HSS、TSS、BIAS、FAR、MAR、POD、OA、Kappa) 的查询方法.
    """

    def __init__(self, ob: np.ndarray, pr: np.ndarray):
        """
        初始化检验对象.

        :param ob: 观测能见度数组, 缺失值用 np.nan 表示;
        :param pr: 预报能见度数组, 缺失值用 np.nan 表示.
        """
        self.thres = THRES
        self.n_grades = len(self.thres)
        self.ob = ob
        self.pr = pr
        # 初始化观测分级矩阵:缺测为 -1,有值为 0(≥ 最大阈值),再按阈值递减赋 1~6
        self.ob_grade = np.zeros_like(self.ob, dtype=np.int_) - 1
        self.ob_grade[~np.isnan(self.ob)] = 0
        for i in range(self.n_grades):
            self.ob_grade[self.ob < self.thres[i]] = i + 1
        # 初始化预报分级矩阵,逻辑同上
        self.pr_grade = np.zeros_like(self.pr, dtype=np.int_) - 1
        self.pr_grade[~np.isnan(self.pr)] = 0
        for i in range(self.n_grades):
            self.pr_grade[self.pr < self.thres[i]] = i + 1
        # 构建 (n_grades+1) * (n_grades+1) 的混淆矩阵( hxjz )
        self.hxjz = np.zeros((self.n_grades + 1, self.n_grades + 1), dtype=np.int_)
        for i in range(self.n_grades + 1):
            for j in range(self.n_grades + 1):
                self.hxjz[i, j] = np.sum((self.ob_grade == i) & (self.pr_grade == j))
        # 有效样本总数
        self.n = np.sum(self.hxjz)

    def get_me(self) -> float:
        """计算平均误差 (Mean Error, ME)."""
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean(pr - ob))

    def get_mae(self) -> float:
        """计算平均绝对误差 (Mean Absolute Error, MAE)."""
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean(np.abs(pr - ob)))

    def get_rmse(self) -> float:
        """计算均方根误差 (Root Mean Square Error, RMSE)."""
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean((pr - ob) ** 2) ** 0.5)

    def get_mre(self) -> float:
        """计算平均相对误差 (Mean Relative Error, MRE)."""
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr)) & (self.ob + self.pr != 0)
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean(np.abs((pr - ob) / (pr + ob))))

    def get_nme(self) -> float:
        """计算归一化平均误差 (Normalized ME)."""
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean(pr - ob) / (np.max(ob) - np.min(ob)))

    def get_nmae(self) -> float:
        """计算归一化平均绝对误差 (Normalized MAE)."""
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean(np.abs(pr - ob)) / (np.max(ob) - np.min(ob)))

    def get_nrmse(self) -> float:
        """计算归一化均方根误差 (Normalized RMSE)."""
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean((pr - ob) ** 2) ** 0.5 / (np.max(ob) - np.min(ob)))

    def get_r(self) -> float:
        """计算 Pearson 相关系数 (R)."""
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(stats.pearsonr(ob, pr)[0])

    def get_me1(self) -> np.ndarray:
        """按观测分级计算各级的平均误差 (ME), 返回长度为 n_grades+1 的数组."""
        me = np.zeros(self.n_grades + 1, dtype=np.float32)
        for i in range(self.n_grades + 1):
            index = (~np.isnan(self.ob)) & (~np.isnan(self.pr)) & (self.ob_grade == i)
            ob = self.ob[index]
            pr = self.pr[index]
            me[i] = np.mean(pr - ob)
        return me

    def get_mae1(self) -> np.ndarray:
        """按观测分级计算各级的平均绝对误差 (MAE), 返回长度为 n_grades+1 的数组."""
        mae = np.zeros(self.n_grades + 1, dtype=np.float32)
        for i in range(self.n_grades + 1):
            index = (~np.isnan(self.ob)) & (~np.isnan(self.pr)) & (self.ob_grade == i)
            ob = self.ob[index]
            pr = self.pr[index]
            mae[i] = np.mean(np.abs(pr - ob))
        return mae

    def get_rmse1(self) -> np.ndarray:
        """按观测分级计算各级的均方根误差 (RMSE), 返回长度为 n_grades+1 的数组."""
        rmse = np.zeros(self.n_grades + 1, dtype=np.float32)
        for i in range(self.n_grades + 1):
            index = (~np.isnan(self.ob)) & (~np.isnan(self.pr)) & (self.ob_grade == i)
            ob = self.ob[index]
            pr = self.pr[index]
            rmse[i] = np.mean((pr - ob) ** 2) ** 0.5
        return rmse

    def get_mre1(self) -> float:
        """按观测分级计算各级的平均相对误差 (MRE), 返回长度为 n_grades+1 的数组."""
        mre = np.zeros(self.n_grades + 1, dtype=np.float32)
        for i in range(self.n_grades + 1):
            index = (~np.isnan(self.ob)) & (~np.isnan(self.pr)) & (self.ob + self.pr != 0) & (self.ob_grade == i)
            ob = self.ob[index]
            pr = self.pr[index]
            mre[i] = np.mean(np.abs((pr - ob) / (pr + ob)))
        return mre

    def get_r1(self) -> float:
        """按观测分级计算各级的 Pearson 相关系数 (R), 返回长度为 n_grades+1 的数组."""
        r = np.zeros(self.n_grades + 1, dtype=np.float32)
        for i in range(self.n_grades + 1):
            index = (~np.isnan(self.ob)) & (~np.isnan(self.pr)) & (self.ob + self.pr != 0) & (self.ob_grade == i)
            ob = self.ob[index]
            pr = self.pr[index]
            r[i] = stats.pearsonr(ob, pr)[0]
        return r

    def get_hxjz(self) -> np.ndarray:
        """返回混淆矩阵 (整数计数)."""
        return self.hxjz

    def get_hxjz2(self) -> np.ndarray:
        """返回归一化混淆矩阵, 每行除以该观测级的总站次数."""
        hxjz2 = np.zeros_like(self.hxjz, dtype=np.float32) + np.nan
        for i in range(self.n_grades + 1):
            for j in range(self.n_grades + 1):
                hxjz2[i, j] = self.hxjz[i, j] / np.sum(self.hxjz[i, :])
        return hxjz2

    def get_oa(self) -> float:
        """计算总体准确度 (Overall Accuracy, OA), 即对角线之和占总样本比例."""
        return np.sum(np.diag(self.hxjz)) / self.n

    def get_kappa(self) -> float:
        """计算 Kappa 系数, 衡量分级一致性."""
        a = np.sum(self.hxjz, axis=1)
        a = a.astype(np.float32)
        b = np.sum(self.hxjz, axis=0)
        b = b.astype(np.float32)
        pe = np.sum(a * b) / self.n / self.n
        return (self.get_oa() - pe) / (1 - pe)

    def get_ts(self) -> np.ndarray:
        """计算各级的威胁评分 (Threat Score, TS), 返回长度为 n_grades+1 的数组."""
        ts = np.zeros(self.n_grades + 1, dtype=np.float32)
        for i in range(self.n_grades + 1):
            na = self.hxjz[i, i]
            nb = np.sum(self.hxjz[:, i]) - na
            nc = np.sum(self.hxjz[i, :]) - na
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ts2(self) -> np.ndarray:
        """计算合并等级的威胁评分 (TS), 返回长度为 n_grades 的数组.

        对第 i 个阈值, 将等级 i+1 及以上视为正例, 其余为负例, 构造 2 * 2 联表:
        na = 观测正且预报正, nb = 观测负且预报正, nc = 观测正且预报负.
        """
        ts = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(self.hxjz[i + 1:, i + 1:])
            nb = np.sum(self.hxjz[:i + 1, i + 1:])
            nc = np.sum(self.hxjz[i + 1:, :i + 1])
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ets2(self) -> np.ndarray:
        """计算合并等级的公平威胁评分 (Equitable Threat Score, ETS)."""
        ets = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(self.hxjz[i + 1:, i + 1:])
            nb = np.sum(self.hxjz[:i + 1, i + 1:])
            nc = np.sum(self.hxjz[i + 1:, :i + 1])
            nd = np.sum(self.hxjz[:i + 1, :i + 1])
            r = (na + nb) / (na + nb + nc + nd) * (na + nc)
            ets[i] = (na - r) / (na + nb + nc - r) if na + nb + nc != 0 else np.nan
        return ets

    def get_hss2(self) -> np.ndarray:
        """计算合并等级的 Heidke 技巧评分 (Heidke Skill Score, HSS)."""
        hss = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(self.hxjz[i + 1:, i + 1:])
            nb = np.sum(self.hxjz[:i + 1, i + 1:])
            nc = np.sum(self.hxjz[i + 1:, :i + 1])
            nd = np.sum(self.hxjz[:i + 1, :i + 1])
            hss[i] = 2 * (na * nd - nb * nc) / ((na + nc) * (nc + nd) + (na + nb) * (nb + nd))
        return hss

    def get_tss2(self) -> np.ndarray:
        """计算合并等级的 True Skill Statistic (TSS, 即 Pierce's Skill Score)."""
        tss = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(self.hxjz[i + 1:, i + 1:])
            nb = np.sum(self.hxjz[:i + 1, i + 1:])
            nc = np.sum(self.hxjz[i + 1:, :i + 1])
            nd = np.sum(self.hxjz[:i + 1, :i + 1])
            tss[i] = (na * nd - nb * nc) / (na + nc) / (nb + nd)
        return tss

    def get_bias2(self) -> np.ndarray:
        """计算合并等级的频率偏差 (Frequency BIAS)."""
        bias = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(self.hxjz[i + 1:, i + 1:])
            nb = np.sum(self.hxjz[:i + 1, i + 1:])
            nc = np.sum(self.hxjz[i + 1:, :i + 1])
            bias[i] = (na + nb) / (na + nc)
        return bias

    def get_far2(self) -> np.ndarray:
        """计算合并等级的空报率 (False Alarm Ratio, FAR)."""
        far = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(self.hxjz[i + 1:, i + 1:])
            nb = np.sum(self.hxjz[:i + 1, i + 1:])
            far[i] = nb / (na + nb)
        return far

    def get_mar2(self) -> np.ndarray:
        """计算合并等级的漏报率 (Miss Alarm Ratio, MAR)."""
        mar = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(self.hxjz[i + 1:, i + 1:])
            nc = np.sum(self.hxjz[i + 1:, :i + 1])
            mar[i] = nc / (na + nc)
        return mar

    def get_pod2(self) -> np.ndarray:
        """计算合并等级的命中率 (Probability of Detection, POD)."""
        pod = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(self.hxjz[i + 1:, i + 1:])
            nc = np.sum(self.hxjz[i + 1:, :i + 1])
            pod[i] = na / (na + nc)
        return pod


def format_time(second: float, is_abbreviation: bool = False) -> str:
    """
    将秒数转换为人类可读的时间字符串.

    :param second: 秒数, 浮点数;
    :param is_abbreviation: 是否使用缩写格式 (如 '12.5m'), 默认 False;
    :return: 格式化后的时间字符串, 例如 '43.5 seconds' 或 '12.5m';
    :raise ValueError: 当 second 为负数时抛出异常.
    """
    if second < 0:
        raise ValueError('输入参数 \'second\' 不能为负数.')
    elif is_abbreviation:
        if second <= 60:
            time_str = str(second) + 's'
        elif second <= 3600:
            time_str = str(second / 60) + 'm'
        else:
            time_str = str(second / 3600) + 'h'
    else:
        if second <= 1:
            time_str = str(second) + ' second'
        elif second <= 60:
            time_str = str(second) + ' seconds'
        elif second <= 3600:
            time_str = str(second / 60) + 'minutes'
        else:
            time_str = str(second / 3600) + 'hours'

    return time_str


def main() -> None:
    # ------------------------------------------------------------------
    # 1. 读取站点信息并筛选目标省份
    # ------------------------------------------------------------------
    sta = pd.read_csv(filepath_or_buffer=r'D:\data\vis\sta2411.csv', low_memory=False)
    sta = sta.sort_values(by=['id'])
    sta.reset_index(drop=True, inplace=True)
    # 按 PROVINCES 列表筛选初始站点
    index_zgdb = None
    for province in PROVINCES:
        if index_zgdb is None:
            index_zgdb = sta.loc[:, 'province'] == province
        else:
            index_zgdb |= sta.loc[:, 'province'] == province
    sta = sta.loc[index_zgdb]
    sta.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # 2. 加载 2020-2023 年能见度、降水、相对湿度数据并进行质控
    # ------------------------------------------------------------------
    vis = np.load(r'D:\data\vis\vis20-23.npy')[:, index_zgdb]
    vis[vis >= 999990] = np.nan  # 缺测值替换
    vis[vis >= 30000] = 30000    # 能见度上限截断
    pre = np.load(r'D:\data\vis\pre20-23.npy')[:, index_zgdb]
    pre[pre >= 200] = np.nan
    pre[pre >= 30000] = 30000
    rhu = np.load(r'D:\data\vis\rhu20-23.npy')[:, index_zgdb]
    rhu[rhu >= 999990] = np.nan
    rhu[rhu > 100] = 100
    rhu[rhu < 0] = 0

    # 进一步筛选长江中下游 7 省市站点(湖北省、湖南省、江西省、安徽省、江苏省、浙江省、上海市)
    index_cjzxy = np.zeros(1183, dtype=np.bool_)
    for i in range(1183):
        if sta.loc[i, 'province'] in ('湖北省', '湖南省', '江西省', '安徽省', '江苏省', '浙江省', '上海市'):
            index_cjzxy[i] = True
    sta = sta.loc[index_cjzxy]
    sta.reset_index(drop=True, inplace=True)
    sta = sta.astype({'data0': float})
    print(sta)
    vis = vis[:, index_cjzxy]
    pre = pre[:, index_cjzxy]
    rhu = rhu[:, index_cjzxy]

    # ------------------------------------------------------------------
    # 3. 能见度六级分级与月份索引生成
    # ------------------------------------------------------------------
    vis_grade = np.zeros_like(vis, dtype=np.int_) - 1
    vis_grade[~np.isnan(vis)] = 0
    for i, t in enumerate(THRES):
        vis_grade[vis < t] = i + 1
    # 2020 年起逐小时对应的月份编号(35064 = 4 年 * 8766 小时,含闰年)
    month_ind = np.zeros(shape=35064, dtype=np.int_)
    for i in range(35064):
        month_ind[i] = arrow.get('2020').shift(hours=i).datetime.month

    # ------------------------------------------------------------------
    # 4. 构建月、时、站三级统计字典
    #    分别统计总体、降水类、雾类、霾类的各级低能见度出现频率
    # ------------------------------------------------------------------
    df_month = {'month': list()}
    df_hour = {'hour': list()}
    df_sta = {'sta_id': sta.loc[:, 'id'], 'lon': sta.loc[:, 'lon'], 'lat': sta.loc[:, 'lat'],
              'alti': sta.loc[:, 'alti'], 'n': list(), 'mean': list(), 'lvpe': list(), 'lvfe': list(), 'lvhe': list()}
    for i in range(len(THRES)):
        # a: 降水类(pre > 0),b: 雾类(pre == 0 且 rhu >= 80),c: 霾类(pre == 0 且 rhu < 80)
        a = np.copy(vis_grade)
        a[pre == 0] = -1
        b = np.copy(vis_grade)
        b[(pre > 0) | (rhu < 80)] = -1
        c = np.copy(vis_grade)
        c[(pre < 0) | (rhu >= 80)] = -1
        # 按月统计
        for j in range(12):
            if i == 0:
                df_month['month'].append(j + 1)
            if j == 0:
                df_month[str(i + 1)] = list()
                df_month[str(i + 1) + 'pre'] = list()
                df_month[str(i + 1) + 'fog'] = list()
                df_month[str(i + 1) + 'haze'] = list()
            df_month[str(i + 1)].append(np.sum(vis_grade[month_ind == j + 1, :] >= i + 1)
                                        / np.sum(vis_grade[month_ind == j + 1, :] >= 0))
            df_month[str(i + 1) + 'pre'].append(np.sum(a[month_ind == j + 1, :] >= i + 1)
                                                / np.sum(vis_grade[month_ind == j + 1, :] >= 0))
            df_month[str(i + 1) + 'fog'].append(np.sum(b[month_ind == j + 1, :] >= i + 1)
                                                / np.sum(vis_grade[month_ind == j + 1, :] >= 0))
            df_month[str(i + 1) + 'haze'].append(np.sum(c[month_ind == j + 1, :] >= i + 1)
                                                 / np.sum(vis_grade[month_ind == j + 1, :] >= 0))
        # 按小时统计(UTC)
        for j in range(24):
            if i == 0:
                df_hour['hour'].append(j)
            if j == 0:
                df_hour[str(i + 1)] = list()
                df_hour[str(i + 1) + 'pre'] = list()
                df_hour[str(i + 1) + 'fog'] = list()
                df_hour[str(i + 1) + 'haze'] = list()
            df_hour[str(i + 1)].append(np.sum(vis_grade[j::24, :] >= i + 1) / np.sum(vis_grade[j::24, :] >= 0))
            df_hour[str(i + 1) + 'pre'].append(np.sum(a[j::24, :] >= i + 1) / np.sum(vis_grade[j::24, :] >= 0))
            df_hour[str(i + 1) + 'fog'].append(np.sum(b[j::24, :] >= i + 1) / np.sum(vis_grade[j::24, :] >= 0))
            df_hour[str(i + 1) + 'haze'].append(np.sum(c[j::24, :] >= i + 1) / np.sum(vis_grade[j::24, :] >= 0))
        # 按站点统计,并计算各站平均能见度及不同天气类型下的平均能见度
        for j in range(len(sta)):
            if j == 0:
                df_sta[str(i + 1)] = list()
                df_sta[str(i + 1) + 'pre'] = list()
                df_sta[str(i + 1) + 'fog'] = list()
                df_sta[str(i + 1) + 'haze'] = list()
            df_sta[str(i + 1)].append(np.sum(vis_grade[:, j] >= i + 1) / np.sum(vis_grade[:, j] >= 0))
            df_sta[str(i + 1) + 'pre'].append(np.sum(a[:, j] >= i + 1) / np.sum(vis_grade[:, j] >= 0))
            df_sta[str(i + 1) + 'fog'].append(np.sum(b[:, j] >= i + 1) / np.sum(vis_grade[:, j] >= 0))
            df_sta[str(i + 1) + 'haze'].append(np.sum(c[:, j] >= i + 1) / np.sum(vis_grade[:, j] >= 0))
            if i == 0:
                df_sta['n'].append(np.sum(vis_grade[:, j] >= 0))
                df_sta['mean'].append(np.nanmean(vis[:, j]))
                index_e = pre[:, j] > 0
                df_sta['lvpe'].append(np.nanmean(vis[:, j][index_e]))
                index_e = (pre[:, j] == 0) & (rhu[:, j] >= 80)
                df_sta['lvfe'].append(np.nanmean(vis[:, j][index_e]))
                index_e = (pre[:, j] == 0) & (rhu[:, j] < 80)
                df_sta['lvhe'].append(np.nanmean(vis[:, j][index_e]))

    # ------------------------------------------------------------------
    # 5. 将统计结果输出为 CSV
    # ------------------------------------------------------------------
    df_month = pd.DataFrame(df_month)
    df_month.to_csv(path_or_buf=r'D:\Project\vis\图\vis_month.csv', index=False)
    df_hour = pd.DataFrame(df_hour)
    df_hour.to_csv(path_or_buf=r'D:\Project\vis\图\vis_hour.csv', index=False)
    df_sta = pd.DataFrame(df_sta)
    df_sta.to_csv(r'D:\Project\vis\图\vis_sta.csv', index=False)

    # ------------------------------------------------------------------
    # 6. 绘制各级低能见度事件的天气类型占比饼图
    # ------------------------------------------------------------------
    for i in range(len(THRES)):
        index = (vis_grade == i + 1) & ~np.isnan(pre) & ~np.isnan(rhu)
        a = np.sum((pre > 0) & index)
        b = np.sum((pre == 0) & (rhu >= 80) & index)
        c = np.sum((pre == 0) & (rhu < 80) & index)
        abc = np.array([a, b, c])
        print(abc / np.sum(abc) * 100)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=800)
        ax.pie(x=(a, b, c), labels=('降水', '雾', '霾'), autopct='%.2f%%', startangle=90)
        fig.savefig(fname=rf'D:\Project\vis\图\pie_{i + 1}.png', bbox_inches='tight', dpi=800)
        fig.savefig(fname=rf'D:\Project\vis\图\pie_{i + 1}.pdf', bbox_inches='tight', dpi=800)
        plt.close(fig)
        del fig, ax
        gc.collect()

    # ------------------------------------------------------------------
    # 7. 绘制低能见度事件(vis < 500m)的小提琴图与箱线图
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5), dpi=800)
    index0 = vis < 500
    index1 = (vis < 500) & (pre > 0)
    index2 = (vis < 500) & (pre == 0) & (rhu >= 80)
    index3 = (vis < 500) & (pre == 0) & (rhu < 80)
    sns.violinplot(
        data={
            '整体': vis[index0] / 1000,
            '降水类': vis[index1] / 1000,
            '雾类': vis[index2] / 1000,
            '霾类': vis[index3] / 1000
        },
        color='skyblue'
    )
    ax.set_xlabel('低能见度事件类型')
    ax.set_ylabel('能见度 (km) ')
    plt.savefig(r'D:\Project\vis\图\violinplot_ob.png', bbox_inches='tight', dpi=800)
    plt.savefig(r'D:\Project\vis\图\violinplot_ob.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()

    fig, ax = plt.subplots(figsize=(5, 5), dpi=800)
    sns.boxplot(
        data={
            '整体': vis[index0] / 1000,
            '降水类': vis[index1] / 1000,
            '雾类': vis[index2] / 1000,
            '霾类': vis[index3] / 1000
        },
        color='black',
        width=0.1,
        showfliers=False,
        medianprops={'color': 'white'}
    )
    ax.set_xlabel('低能见度事件类型')
    ax.set_ylabel('能见度 (km) ')
    plt.savefig(r'D:\Project\vis\图\boxplot_ob.png', bbox_inches='tight', dpi=800)
    plt.savefig(r'D:\Project\vis\图\boxplot_ob.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()

    # ------------------------------------------------------------------
    # 8. 绘制月份概率堆叠柱状图(彩色版与黑白版)
    # ------------------------------------------------------------------
    df_month = pd.read_csv(filepath_or_buffer=r'D:\Project\vis\图\vis_month.csv', low_memory=False)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    ax.bar(
        x=np.linspace(start=1, stop=12, num=12),
        height=df_month.loc[:, '1pre'],
        width=0.4,
        color=(31 / 255, 119 / 255, 180 / 255),
        label='降水类'
    )
    ax.bar(
        x=np.linspace(start=1, stop=12, num=12),
        height=df_month.loc[:, '1fog'],
        bottom=df_month.loc[:, '1pre'],
        width=0.4,
        color=(255 / 255, 127 / 255, 14 / 255),
        label='雾类'
    )
    ax.bar(
        x=np.linspace(start=1, stop=12, num=12),
        height=df_month.loc[:, '1haze'],
        bottom=df_month.loc[:, '1pre'] + df_month.loc[:, '1fog'],
        width=0.4,
        color=(44 / 255, 160 / 255, 44 / 255),
        label='霾类'
    )
    ax.set_xlim((0, 13))
    ax.set_xticks(range(1, 13), [f'{x}月' for x in range(1, 13)])
    ax.set_ylim((0, 1.0))
    ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
    ax.set_xlabel('月份')
    ax.set_ylabel('概率')
    ax.legend()
    fig.savefig(fname=rf'D:\Project\vis\图\month_1+.png', bbox_inches='tight', dpi=800)
    fig.savefig(fname=rf'D:\Project\vis\图\month_1+.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()

    # 黑白版月份堆叠柱状图
    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    ax.bar(
        x=np.linspace(start=1, stop=12, num=12),
        height=df_month.loc[:, '1pre'],
        width=0.4,
        color='black',
        edgecolor='black',
        label='降水类'
    )
    ax.bar(
        x=np.linspace(start=1, stop=12, num=12),
        height=df_month.loc[:, '1fog'],
        bottom=df_month.loc[:, '1pre'],
        width=0.4,
        color='white',
        edgecolor='black',
        hatch='///',
        label='雾类'
    )
    ax.bar(
        x=np.linspace(start=1, stop=12, num=12),
        height=df_month.loc[:, '1haze'],
        bottom=df_month.loc[:, '1pre'] + df_month.loc[:, '1fog'],
        width=0.4,
        color='white',
        edgecolor='black',
        label='霾类'
    )
    ax.set_xlim((0, 13))
    ax.set_xticks(range(1, 13), [f'{x}月' for x in range(1, 13)])
    ax.set_ylim((0, 1.0))
    ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
    ax.set_xlabel('月份')
    ax.set_ylabel('概率')
    ax.legend()
    fig.savefig(fname=rf'D:\Project\vis\图\month_1+_bw.png', bbox_inches='tight', dpi=800)
    fig.savefig(fname=rf'D:\Project\vis\图\month_1+_bw.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()
    # 输出低能见度事件概率最大与最小的月份
    lve = np.array(df_month.loc[:, '1haze'] + df_month.loc[:, '1pre'] + df_month.loc[:, '1fog'])
    print(np.argmax(lve) + 1, np.max(lve))
    print(np.argmin(lve) + 1, np.min(lve))
    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    d = dict()
    for i in range(12):
        index = vis < 10000
        vis_ob = vis[month_ind == i + 1, :]
        index = index[month_ind == i + 1, :]
        d[f'{i + 1}月'] = vis_ob[index] / 1000
    sns.violinplot(data=d, color='skyblue')
    ax.set_xlabel('月份')
    ax.set_ylabel('能见度(km)')
    fig.savefig(r'D:\Project\vis\图\boxplot_ob_month.png', bbox_inches='tight', dpi=800)
    fig.savefig(r'D:\Project\vis\图\boxplot_ob_month.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()
    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    d = dict()
    for i in range(12):
        index = (vis < 10000) & (pre > 0)
        vis_ob = vis[month_ind == i + 1, :]
        index = index[month_ind == i + 1, :]
        d[f'{i + 1}月'] = vis_ob[index] / 1000
    sns.violinplot(data=d, color='skyblue')
    ax.set_xlabel('月份')
    ax.set_ylabel('能见度(km)')
    fig.savefig(r'D:\Project\vis\图\boxplot_ob_month_pre.png', bbox_inches='tight', dpi=800)
    fig.savefig(r'D:\Project\vis\图\boxplot_ob_month_pre.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()
    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    d = dict()
    for i in range(12):
        index = (vis < 10000) & (pre == 0) & (rhu >= 80)
        vis_ob = vis[month_ind == i + 1, :]
        index = index[month_ind == i + 1, :]
        d[f'{i + 1}月'] = vis_ob[index] / 1000
    sns.violinplot(data=d, color='skyblue')
    ax.set_xlabel('月份')
    ax.set_ylabel('能见度(km)')
    fig.savefig(r'D:\Project\vis\图\boxplot_ob_month_fog.png', bbox_inches='tight', dpi=800)
    fig.savefig(r'D:\Project\vis\图\boxplot_ob_month_fog.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()
    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    d = dict()
    for i in range(12):
        index = (vis < 10000) & (pre == 0) & (rhu < 80)
        vis_ob = vis[month_ind == i + 1, :]
        index = index[month_ind == i + 1, :]
        d[f'{i + 1}月'] = vis_ob[index] / 1000
    sns.violinplot(data=d, color='skyblue')
    ax.set_xlabel('月份')
    ax.set_ylabel('能见度(km)')
    fig.savefig(r'D:\Project\vis\图\boxplot_ob_month_haze.png', bbox_inches='tight', dpi=800)
    fig.savefig(r'D:\Project\vis\图\boxplot_ob_month_haze.pdf', bbox_inches='tight', dpi=800)
    plt.close('all')
    del fig, ax
    gc.collect()

    # ------------------------------------------------------------------
    # 9. 绘制小时概率堆叠柱状图(彩色版与黑白版)
    # ------------------------------------------------------------------
    df_hour = pd.read_csv(filepath_or_buffer=r'D:\Project\vis\图\vis_hour.csv', low_memory=False)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    ax.bar(
        x=np.linspace(start=0, stop=23, num=24),
        height=df_hour.loc[:, '1pre'],
        width=0.4,
        color=(31 / 255, 119 / 255, 180 / 255),
        label='降水类'
    )
    ax.bar(
        x=np.linspace(start=0, stop=23, num=24),
        height=df_hour.loc[:, '1fog'],
        bottom=df_hour.loc[:, '1pre'],
        width=0.4,
        color=(255 / 255, 127 / 255, 14 / 255),
        label='雾类'
    )
    ax.bar(
        x=np.linspace(start=0, stop=23, num=24),
        height=df_hour.loc[:, '1haze'],
        bottom=df_hour.loc[:, '1pre'] + df_hour.loc[:, '1fog'],
        width=0.4,
        color=(44 / 255, 160 / 255, 44 / 255),
        label='霾类'
    )
    ax.set_xlim((-1, 24))
    ax.set_xticks(range(24), [f'{x:02d}:00' for x in range(24)])
    ax.set_ylim((0, 1.0))
    ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
    ax.set_xlabel('时间 (UTC) ')
    ax.set_ylabel('概率')
    ax.legend()
    fig.savefig(fname=rf'D:\Project\vis\图\hour_1+.png', bbox_inches='tight', dpi=800)
    fig.savefig(fname=rf'D:\Project\vis\图\hour_1+.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()

    # 黑白版小时堆叠柱状图
    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    ax.bar(
        x=np.linspace(start=0, stop=23, num=24),
        height=df_hour.loc[:, '1pre'],
        width=0.4,
        color='black',
        edgecolor='black',
        label='降水类'
    )
    ax.bar(
        x=np.linspace(start=0, stop=23, num=24),
        height=df_hour.loc[:, '1fog'],
        bottom=df_hour.loc[:, '1pre'],
        width=0.4,
        color='white',
        edgecolor='black',
        hatch='///',
        label='雾类'
    )
    ax.bar(
        x=np.linspace(start=0, stop=23, num=24),
        height=df_hour.loc[:, '1haze'],
        bottom=df_hour.loc[:, '1pre'] + df_hour.loc[:, '1fog'],
        width=0.4,
        color='white',
        edgecolor='black',
        label='霾类'
    )
    ax.set_xlim((-1, 24))
    ax.set_xticks(range(24), [f'{x:02d}:00' for x in range(24)])
    ax.set_ylim((0, 1.0))
    ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
    ax.set_xlabel('时间 (UTC) ')
    ax.set_ylabel('概率')
    ax.legend()
    fig.savefig(fname=rf'D:\Project\vis\图\hour_1+_bw.png', bbox_inches='tight', dpi=800)
    fig.savefig(fname=rf'D:\Project\vis\图\hour_1+_bw.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()
    # 输出低能见度事件概率最大与最小的小时
    lve = np.array(df_hour.loc[:, '1haze'] + df_hour.loc[:, '1pre'] + df_hour.loc[:, '1fog'])
    print(np.argmax(lve), np.max(lve))
    print(np.argmin(lve), np.min(lve))
    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    d = dict()
    for i in range(24):
        index = vis < 10000
        vis_ob = vis[i::24, :]
        index = index[i::24, :]
        d[f'{i}:00'] = vis_ob[index] / 1000
    sns.violinplot(data=d, color='skyblue')
    ax.set_xlabel('小时(UTC)')
    ax.set_ylabel('能见度(km)')
    fig.savefig(r'D:\Project\vis\图\boxplot_ob_hour.png', bbox_inches='tight', dpi=800)
    fig.savefig(r'D:\Project\vis\图\boxplot_ob_hour.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()
    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    d = dict()
    for i in range(24):
        index = (vis < 10000) & (pre > 0)
        vis_ob = vis[i::24, :]
        index = index[i::24, :]
        d[f'{i}:00'] = vis_ob[index] / 1000
    sns.violinplot(data=d, color='skyblue')
    ax.set_xlabel('小时(UTC)')
    ax.set_ylabel('能见度(km)')
    fig.savefig(r'D:\Project\vis\图\boxplot_ob_hour_pre.png', bbox_inches='tight', dpi=800)
    fig.savefig(r'D:\Project\vis\图\boxplot_ob_hour_pre.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()
    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    d = dict()
    for i in range(24):
        index = (vis < 10000) & (pre == 0) & (rhu >= 80)
        vis_ob = vis[i::24, :]
        index = index[i::24, :]
        d[f'{i}:00'] = vis_ob[index] / 1000
    sns.violinplot(data=d, color='skyblue')
    ax.set_xlabel('小时(UTC)')
    ax.set_ylabel('能见度(km)')
    fig.savefig(r'D:\Project\vis\图\boxplot_ob_hour_fog.png', bbox_inches='tight', dpi=800)
    fig.savefig(r'D:\Project\vis\图\boxplot_ob_hour_fog.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()
    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    d = dict()
    for i in range(24):
        index = (vis < 10000) & (pre == 0) & (rhu < 80)
        vis_ob = vis[i::24, :]
        index = index[i::24, :]
        d[f'{i}:00'] = vis_ob[index] / 1000
    sns.violinplot(data=d, color='skyblue')
    ax.set_xlabel('小时(UTC)')
    ax.set_ylabel('能见度(km)')
    fig.savefig(r'D:\Project\vis\图\boxplot_ob_hour_haze.png', bbox_inches='tight', dpi=800)
    fig.savefig(r'D:\Project\vis\图\boxplot_ob_hour_haze.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()
    
    df_sta = pd.read_csv(r'D:\Project\vis\图\vis_sta.csv', low_memory=False)
    cmap, clevs = meb.def_cmap_clevs(
        meb.def_cmap_clevs(meb.cmaps.ts, vmin=0, vmax=0.9)[0],
        clevs=[0, 0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72, 0.81, 0.9]
    )
    sta0 = sta.loc[:, ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')]
    sta0.loc[:, 'data0'] = df_sta.loc[:, '1']
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='max',
        title=[''],
        save_path=r'D:\Project\vis\图\sta_1+.png',
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
        save_path=r'D:\Project\vis\图\sta_1+.pdf',
        dpi=800
    )
    index_in = np.zeros(502, dtype=np.bool_)
    for i in range(502):
        if sta.loc[i, 'province'] in ['湖南省']:
            index_in[i] = True
    mean_in = np.mean(df_sta.loc[index_in, '1'])
    mean_out = np.mean(df_sta.loc[~index_in, '1'])
    print(f'LVE: {mean_in}, {mean_out}')
    cmap, clevs = meb.def_cmap_clevs(
        meb.def_cmap_clevs(meb.cmaps.ts, vmin=0, vmax=0.2)[0],
        clevs=[0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    )
    sta0 = sta.loc[:, ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')]
    sta0.loc[:, 'data0'] = df_sta.loc[:, '1pre']
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='max',
        title=[''],
        save_path=r'D:\Project\vis\图\sta_1+_pre.png',
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
        save_path=r'D:\Project\vis\图\sta_1+_pre.pdf',
        dpi=800
    )
    index_in = np.zeros(502, dtype=np.bool_)
    for i in range(502):
        if sta.loc[i, 'province'] in ['湖南省', '江西省', '浙江省']:
            index_in[i] = True
    mean_in = np.mean(df_sta.loc[index_in, '1pre'])
    mean_out = np.mean(df_sta.loc[~index_in, '1pre'])
    print(f'LVPE: {mean_in}, {mean_out}')
    cmap, clevs = meb.def_cmap_clevs(
        meb.def_cmap_clevs(meb.cmaps.ts, vmin=0, vmax=40)[0],
        clevs=[0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4]
    )
    sta0 = sta.loc[:, ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')]
    sta0.loc[:, 'data0'] = df_sta.loc[:, '1fog']
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='max',
        title=[''],
        save_path=r'D:\Project\vis\图\sta_1+_fog.png',
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
        save_path=r'D:\Project\vis\图\sta_1+_fog.pdf',
        dpi=800
    )
    index_in = np.zeros(502, dtype=np.bool_)
    for i in range(502):
        if sta.loc[i, 'province'] in ['湖南省', '江苏省']:
            index_in[i] = True
    mean_in = np.mean(df_sta.loc[index_in, '1fog'])
    mean_out = np.mean(df_sta.loc[~index_in, '1fog'])
    print(f'LVFE: {mean_in}, {mean_out}')
    cmap, clevs = meb.def_cmap_clevs(
        meb.def_cmap_clevs(meb.cmaps.ts, vmin=0, vmax=40)[0],
        clevs=[0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4]
    )
    sta0 = sta.loc[:, ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')]
    sta0.loc[:, 'data0'] = df_sta.loc[:, '1haze']
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='max',
        title=[''],
        save_path=r'D:\Project\vis\图\sta_1+_haze.png',
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
        save_path=r'D:\Project\vis\图\sta_1+_haze.pdf',
        dpi=800
    )
    index_in = np.zeros(502, dtype=np.bool_)
    for i in range(502):
        if sta.loc[i, 'province'] in ['湖南省']:
            index_in[i] = True
    mean_in = np.mean(df_sta.loc[index_in, '1haze'])
    mean_out = np.mean(df_sta.loc[~index_in, '1haze'])
    print(f'LVHE: {mean_in}, {mean_out}')
    
    cmap, clevs = meb.def_cmap_clevs(
        meb.def_cmap_clevs(meb.cmaps.vis, vmin=0, vmax=30)[0],
        clevs=[0, 1, 2, 4, 7, 10, 15, 20, 25, 30]
    )
    sta0 = sta.loc[:, ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')]
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'mean'] / 1000
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='max',
        title=[''],
        save_path=r'D:\Project\vis\图\sta_ob_mean.png',
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
        save_path=r'D:\Project\vis\图\sta_ob_mean.pdf',
        dpi=800
    )
    index_in = np.zeros(502, dtype=np.bool_)
    for i in range(502):
        if sta.loc[i, 'province'] in ['湖南省', '江苏省', '浙江省']:
            index_in[i] = True
    mean_in = np.mean(df_sta.loc[index_in, 'mean'])
    mean_out = np.mean(df_sta.loc[~index_in, 'mean'])
    print(f'VIS: {mean_in}, {mean_out}')
    sta0 = sta.loc[:, ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')]
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'lvpe'] / 1000
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='max',
        title=[''],
        save_path=r'D:\Project\vis\图\sta_ob_lvpe_mean.png',
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
        save_path=r'D:\Project\vis\图\sta_ob_lvpe_mean.pdf',
        dpi=800
    )
    sta0 = sta.loc[:, ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')]
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'lvfe'] / 1000
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='max',
        title=[''],
        save_path=r'D:\Project\vis\图\sta_ob_lvfe_mean.png',
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
        save_path=r'D:\Project\vis\图\sta_ob_lvfe_mean.pdf',
        dpi=800
    )
    sta0 = sta.loc[:, ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')]
    sta0.loc[:, 'data0'] = df_sta.loc[:, 'lvhe'] / 1000
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='max',
        title=[''],
        save_path=r'D:\Project\vis\图\sta_ob_lvhe_mean.png',
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
        save_path=r'D:\Project\vis\图\sta_ob_lvhe_mean.pdf',
        dpi=800
    )
    cmap, clevs = meb.def_cmap_clevs(meb.cmaps.ts)
    meb.tool.color_tools.show_cmap_clev(cmap, clevs, save_path=r'D:\Project\vis\图\000.png')
    meb.tool.color_tools.show_cmap_clev(cmap, clevs, save_path=r'D:\Project\vis\图\000.pdf')

    # ------------------------------------------------------------------
    # 10. 加载 2020-2023 年预报检验数据(最后 365 天)
    #     包括观测、CMA-SH-WARR 模式及 5 组 PDFM-TLE 试验结果
    # ------------------------------------------------------------------
    vis_ob = np.load(r'D:\data\vis\vis1183_ob.npy')[-365:, :, 1:, index_cjzxy]
    vis_ob = np.reshape(vis_ob, (-1, 24, 502))
    vis_ob[vis_ob >= 999990] = np.nan
    vis_ob[vis_ob >= 30000] = 30000
    cma_sh_warr = np.load(r'D:\data\vis\vis1183_pr.npy')[-365:, :, 1:, index_cjzxy]
    cma_sh_warr = np.reshape(cma_sh_warr, (-1, 24, 502))
    cma_sh_warr[cma_sh_warr >= 30000] = 30000
    pred_pdfm_tle0 = np.load(r'D:\data\vis\vis_gjz_pdfm_tle0_cjzxy.npy')
    pred_pdfm_tle0 = np.reshape(pred_pdfm_tle0, (-1, 24, 502))
    pred_pdfm_tle0[pred_pdfm_tle0 >= 30000] = 30000
    pred_pdfm_tle1 = np.load(r'D:\data\vis\vis_gjz_pdfm_tle1_cjzxy.npy')
    pred_pdfm_tle1 = np.reshape(pred_pdfm_tle1, (-1, 24, 502))
    pred_pdfm_tle1[pred_pdfm_tle1 >= 30000] = 30000
    pred_pdfm_tle2 = np.load(r'D:\data\vis\vis_gjz_pdfm_tle2_cjzxy.npy')
    pred_pdfm_tle2 = np.reshape(pred_pdfm_tle2, (-1, 24, 502))
    pred_pdfm_tle2[pred_pdfm_tle2 >= 30000] = 30000
    pred_pdfm_tle3 = np.load(r'D:\data\vis\vis_gjz_pdfm_tle3_cjzxy.npy')
    pred_pdfm_tle3 = np.reshape(pred_pdfm_tle3, (-1, 24, 502))
    pred_pdfm_tle3[pred_pdfm_tle3 >= 30000] = 30000
    pred_pdfm_tle4 = np.load(r'D:\data\vis\vis_gjz_pdfm_tle4_cjzxy.npy')
    pred_pdfm_tle4 = np.reshape(pred_pdfm_tle4, (-1, 24, 502))
    pred_pdfm_tle4[pred_pdfm_tle4 >= 30000] = 30000
    acc = VisAcc(vis_ob, cma_sh_warr)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_ets2())
    print(acc.get_hss2())
    print(acc.get_tss2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print(acc.get_pod2())
    print('方案一')
    acc = VisAcc(vis_ob, pred_pdfm_tle0)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_ets2())
    print(acc.get_hss2())
    print(acc.get_tss2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print(acc.get_pod2())
    print('方案二')
    acc = VisAcc(vis_ob, pred_pdfm_tle1)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_ets2())
    print(acc.get_hss2())
    print(acc.get_tss2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print(acc.get_pod2())
    print('方案三')
    acc = VisAcc(vis_ob, pred_pdfm_tle2)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_ets2())
    print(acc.get_hss2())
    print(acc.get_tss2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print(acc.get_pod2())
    print('方案四')
    acc = VisAcc(vis_ob, pred_pdfm_tle3)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_ets2())
    print(acc.get_hss2())
    print(acc.get_tss2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print(acc.get_pod2())
    print('方案五')
    acc = VisAcc(vis_ob, pred_pdfm_tle4)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_ets2())
    print(acc.get_hss2())
    print(acc.get_tss2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print(acc.get_pod2())

    # ------------------------------------------------------------------
    # 11. 按天气类型(降水、雾、霾)计算各方案的定量与分级检验指标并绘图
    # ------------------------------------------------------------------
    weather_type = np.load(r'D:\data\vis\weather_type.npy', mmap_mode='r')
    val_wt = np.reshape(weather_type[1096:1461, ..., index_cjzxy], shape=(-1, 24, 502))
    # qem: (4 指标, 6 方案, 3 天气类型);cem: (6 等级, 4 指标, 6 方案, 3 天气类型)
    qem = np.zeros(shape=(4, 6, 3), dtype=np.float32) + np.nan
    cem = np.zeros(shape=(6, 4, 6, 3), dtype=np.float32) + np.nan
    for i in range(3):
        acc = VisAcc(vis_ob[val_wt == i + 1], cma_sh_warr[val_wt == i + 1])
        qem[0, 0, i] = acc.get_r()
        qem[1, 0, i] = acc.get_mae()
        qem[2, 0, i] = acc.get_rmse()
        qem[3, 0, i] = acc.get_mre()
        cem[:, 0, 0, i] = acc.get_ts2()
        cem[:, 1, 0, i] = acc.get_far2()
        cem[:, 2, 0, i] = acc.get_mar2()
        cem[:, 3, 0, i] = acc.get_pod2()
        acc = VisAcc(vis_ob[val_wt == i + 1], pred_pdfm_tle0[val_wt == i + 1])
        qem[0, 1, i] = acc.get_r()
        qem[1, 1, i] = acc.get_mae()
        qem[2, 1, i] = acc.get_rmse()
        qem[3, 1, i] = acc.get_mre()
        cem[:, 0, 1, i] = acc.get_ts2()
        cem[:, 1, 1, i] = acc.get_far2()
        cem[:, 2, 1, i] = acc.get_mar2()
        cem[:, 3, 1, i] = acc.get_pod2()
        acc = VisAcc(vis_ob[val_wt == i + 1], pred_pdfm_tle1[val_wt == i + 1])
        qem[0, 2, i] = acc.get_r()
        qem[1, 2, i] = acc.get_mae()
        qem[2, 2, i] = acc.get_rmse()
        qem[3, 2, i] = acc.get_mre()
        cem[:, 0, 2, i] = acc.get_ts2()
        cem[:, 1, 2, i] = acc.get_far2()
        cem[:, 2, 2, i] = acc.get_mar2()
        cem[:, 3, 2, i] = acc.get_pod2()
        acc = VisAcc(vis_ob[val_wt == i + 1], pred_pdfm_tle2[val_wt == i + 1])
        qem[0, 3, i] = acc.get_r()
        qem[1, 3, i] = acc.get_mae()
        qem[2, 3, i] = acc.get_rmse()
        qem[3, 3, i] = acc.get_mre()
        cem[:, 0, 3, i] = acc.get_ts2()
        cem[:, 1, 3, i] = acc.get_far2()
        cem[:, 2, 3, i] = acc.get_mar2()
        cem[:, 3, 3, i] = acc.get_pod2()
        acc = VisAcc(vis_ob[val_wt == i + 1], pred_pdfm_tle3[val_wt == i + 1])
        qem[0, 4, i] = acc.get_r()
        qem[1, 4, i] = acc.get_mae()
        qem[2, 4, i] = acc.get_rmse()
        qem[3, 4, i] = acc.get_mre()
        cem[:, 0, 4, i] = acc.get_ts2()
        cem[:, 1, 4, i] = acc.get_far2()
        cem[:, 2, 4, i] = acc.get_mar2()
        cem[:, 3, 4, i] = acc.get_pod2()
        acc = VisAcc(vis_ob[val_wt == i + 1], pred_pdfm_tle4[val_wt == i + 1])
        qem[0, 5, i] = acc.get_r()
        qem[1, 5, i] = acc.get_mae()
        qem[2, 5, i] = acc.get_rmse()
        qem[3, 5, i] = acc.get_mre()
        cem[:, 0, 5, i] = acc.get_ts2()
        cem[:, 1, 5, i] = acc.get_far2()
        cem[:, 2, 5, i] = acc.get_mar2()
        cem[:, 3, 5, i] = acc.get_pod2()
    # 绘制天气类型检验对比图(相关系数、MAE、RMSE、MRE)
    plot_weather_type_eval_bw(qem=qem[0, ...], filename='wt_cc_bw', max_y=0.4)
    plot_weather_type_eval_bw(qem=qem[1, ...], filename='wt_mae_bw', max_y=10)
    plot_weather_type_eval_bw(qem=qem[2, ...], filename='wt_rmse_bw', max_y=15)
    plot_weather_type_eval_bw(qem=qem[3, ...], filename='wt_mre_bw', max_y=0.6)

    vis_values = list(range(-1, 100, 1)) + list(range(100, 1000, 10)) + list(range(1000, 30001, 100))
    vis_values = list(range(-1, 30001, 1))
    vis_values = np.array(vis_values)
    cdf = np.zeros((3, vis_values.size), dtype=np.float32) + np.nan
    index = (~np.isnan(vis_ob)) & (~np.isnan(cma_sh_warr)) & (~np.isnan(pred_pdfm_tle0))
    ob = vis_ob[index]
    nwp = cma_sh_warr[index]
    pr = pred_pdfm_tle0[index]
    for i in range(vis_values.size):
        cdf[0, i] = np.mean(ob <= vis_values[i])
        cdf[1, i] = np.mean(nwp <= vis_values[i])
        cdf[2, i] = np.mean(pr <= vis_values[i])
    fig, ax = plt.subplots(figsize=(5, 5), dpi=800)
    ax.plot(
        vis_values / 1000,
        cdf[0, :],
        '-',
        c='black',
        label='实况'
    )
    ax.plot(
        vis_values / 1000,
        cdf[1, :],
        '-',
        c='blue',
        label='CMA-SH3-WARR'
    )
    ax.plot(
        vis_values / 1000,
        cdf[2, :],
        '-',
        c='red',
        label='PDFM-TLE'
    )
    ax.set_xlim((-1, 31))
    ax.set_xticks((0, 10, 20, 30))
    ax.set_ylim((0, 1))
    ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    ax.set_xlabel('能见度/km')
    ax.set_ylabel('累积概率')
    ax.legend()
    fig.savefig(fname=rf'D:\Project\vis\图\vis_cdf.png', bbox_inches='tight', dpi=800)
    fig.savefig(fname=rf'D:\Project\vis\图\vis_cdf.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()

    index = ~np.isnan(vis_ob) & ~np.isnan(cma_sh_warr) & ~np.isnan(pred_pdfm_tle0)
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
    ax.set_xticks(l0, [f'{x:.2f}' for x in l])
    ax.set_ylim((np.log(1), np.log(31)))
    ax.set_yticks(l0, [f'{x:.2f}' for x in l])
    ax.set_xlabel('实况 (km)')
    ax.set_ylabel('预报 (km)')
    fig.savefig(r'D:\Project\vis\图\vis_nwp_his2d.png', bbox_inches='tight', dpi=800)
    fig.savefig(r'D:\Project\vis\图\vis_nwp_his2d.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()

    index = ~np.isnan(vis_ob)
    index &= ~np.isnan(pred_pdfm_tle0)
    index &= ~np.isnan(pred_pdfm_tle1)
    index &= ~np.isnan(pred_pdfm_tle2)
    index &= ~np.isnan(pred_pdfm_tle3)
    index &= ~np.isnan(pred_pdfm_tle4)
    df_fh = {'grade': list(), 'ob': list(), 'CMA-SH-WARR': list(), 'OTS': list(),
             'PDF': list(), 'TL': list(), 'PDFM-TLE': list()}
    for i in range(7):
        right = 999999 if i == 0 else THRES[i - 1]
        left = -999999 if i == 6 else THRES[i]
        df_fh['grade'].append(i)
        df_fh['ob'].append(np.sum((vis_ob[index] < right) & (vis_ob[index] >= left)) / np.sum(index))
        df_fh['CMA-SH-WARR'].append(np.sum((pred_pdfm_tle0[index] < right) & (pred_pdfm_tle0[index] >= left)) / np.sum(index))
        df_fh['OTS'].append(np.sum((pred_pdfm_tle1[index] < right) & (pred_pdfm_tle1[index] >= left)) / np.sum(index))
        df_fh['PDF'].append(np.sum((pred_pdfm_tle2[index] < right) & (pred_pdfm_tle2[index] >= left)) / np.sum(index))
        df_fh['TL'].append(np.sum((pred_pdfm_tle3[index] < right) & (pred_pdfm_tle3[index] >= left)) / np.sum(index))
        df_fh['PDFM-TLE'].append(np.sum((pred_pdfm_tle4[index] < right) & (pred_pdfm_tle4[index] >= left)) / np.sum(index))
    df_fh = pd.DataFrame(df_fh)
    df_fh.to_csv(r'D:\Project\vis\图\vis_fh.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=800)
    ax.bar(
        x=df_fh.loc[:, 'grade'] - 0.4,
        height=df_fh.loc[:, 'ob'],
        width=0.16,
        color='black',
        label='实况'
    )
    ax.bar(
        x=df_fh.loc[:, 'grade'] - 0.24,
        height=df_fh.loc[:, 'CMA-SH-WARR'],
        width=0.16,
        color='blue',
        label='CMA-SH-WARR'
    )
    ax.bar(
        x=df_fh.loc[:, 'grade'] - 0.08,
        height=df_fh.loc[:, 'OTS'],
        width=0.16,
        color='green',
        label='OTS'
    )
    ax.bar(
        x=df_fh.loc[:, 'grade'] + 0.08,
        height=df_fh.loc[:, 'PDF'],
        width=0.16,
        color='yellow',
        label='PDF'
    )
    ax.bar(
        x=df_fh.loc[:, 'grade'] + 0.24,
        height=df_fh.loc[:, 'TL'],
        width=0.16,
        color='orange',
        label='TL'
    )
    ax.bar(
        x=df_fh.loc[:, 'grade'] + 0.4,
        height=df_fh.loc[:, 'PDFM-TLE'],
        width=0.16,
        color='red',
        label='PDFM-TLE'
    )
    ax.set_xlim((-1, 7))
    ax.set_xticks(range(7), df_fh.loc[:, 'grade'])
    ax.set_ylim((0, 1.0))
    ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
    ax.set_xlabel('低能见度等级')
    ax.set_ylabel('频率')
    ax.legend()
    fig.savefig(fname=rf'D:\Project\vis\图\fh.png', bbox_inches='tight', dpi=800)
    fig.savefig(fname=rf'D:\Project\vis\图\fh.pdf', bbox_inches='tight', dpi=800)
    plt.close(fig)
    del fig, ax
    gc.collect()

    # ------------------------------------------------------------------
    # 12. 时效分析:计算预报时效(fhour)与起报时次(vt)对应的 TS 评分及四维检验数组
    # ------------------------------------------------------------------
    # fhour_ind[i, j] 表示第 i 个起报日、第 j 个预报时效对应的实际预报时间(UTC)
    fhour_ind = np.zeros((8760, 24), dtype=np.int_)
    for i in range(8760):
        for j in range(24):
            fhour_ind[i, j] = (i + j) % 24
    df_vt_corr = {'vt': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_vt_mae = {'vt': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_vt_rmse = {'vt': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_vt_mre = {'vt': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_vt_ts1 = {'vt': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_vt_ts2 = {'vt': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_vt_ts3 = {'vt': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_vt_ts4 = {'vt': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_vt_ts5 = {'vt': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_vt_ts6 = {'vt': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_shour_corr = {'shour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_shour_mae = {'shour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_shour_rmse = {'shour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_shour_mre = {'shour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_shour_ts1 = {'shour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_shour_ts2 = {'shour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_shour_ts3 = {'shour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_shour_ts4 = {'shour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_shour_ts5 = {'shour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_shour_ts6 = {'shour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_fhour_corr = {'fhour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_fhour_mae = {'fhour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_fhour_rmse = {'fhour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_fhour_mre = {'fhour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_fhour_ts1 = {'fhour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_fhour_ts2 = {'fhour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_fhour_ts3 = {'fhour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_fhour_ts4 = {'fhour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_fhour_ts5 = {'fhour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_fhour_ts6 = {'fhour': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    # hour_access: (4 方法, 24 起报时次, 24 预报时效, 10 指标)
    # 指标顺序:R, MAE, RMSE, MRE, TS1~TS6
    hour_access = np.zeros(shape=(4, 24, 24, 10), dtype=np.float32) + np.nan
    for i in range(24):
        df_vt_corr['vt'].append(i + 1)
        df_vt_mae['vt'].append(i + 1)
        df_vt_rmse['vt'].append(i + 1)
        df_vt_mre['vt'].append(i + 1)
        df_vt_ts1['vt'].append(i + 1)
        df_vt_ts2['vt'].append(i + 1)
        df_vt_ts3['vt'].append(i + 1)
        df_vt_ts4['vt'].append(i + 1)
        df_vt_ts5['vt'].append(i + 1)
        df_vt_ts6['vt'].append(i + 1)
        df_shour_corr['shour'].append(i)
        df_shour_mae['shour'].append(i)
        df_shour_rmse['shour'].append(i)
        df_shour_mre['shour'].append(i)
        df_shour_ts1['shour'].append(i)
        df_shour_ts2['shour'].append(i)
        df_shour_ts3['shour'].append(i)
        df_shour_ts4['shour'].append(i)
        df_shour_ts5['shour'].append(i)
        df_shour_ts6['shour'].append(i)
        df_fhour_corr['fhour'].append(i)
        df_fhour_mae['fhour'].append(i)
        df_fhour_rmse['fhour'].append(i)
        df_fhour_mre['fhour'].append(i)
        df_fhour_ts1['fhour'].append(i)
        df_fhour_ts2['fhour'].append(i)
        df_fhour_ts3['fhour'].append(i)
        df_fhour_ts4['fhour'].append(i)
        df_fhour_ts5['fhour'].append(i)
        df_fhour_ts6['fhour'].append(i)
        acc_nwp = VisAcc(vis_ob[:, i, :], cma_sh_warr[:, i, :])
        df_vt_corr['CMA-SH-WARR'].append(acc_nwp.get_r())
        df_vt_mae['CMA-SH-WARR'].append(acc_nwp.get_mae())
        df_vt_rmse['CMA-SH-WARR'].append(acc_nwp.get_rmse())
        df_vt_mre['CMA-SH-WARR'].append(acc_nwp.get_mre())
        ts = acc_nwp.get_ts2()
        df_vt_ts1['CMA-SH-WARR'].append(ts[0])
        df_vt_ts2['CMA-SH-WARR'].append(ts[1])
        df_vt_ts3['CMA-SH-WARR'].append(ts[2])
        df_vt_ts4['CMA-SH-WARR'].append(ts[3])
        df_vt_ts5['CMA-SH-WARR'].append(ts[4])
        df_vt_ts6['CMA-SH-WARR'].append(ts[5])
        acc = VisAcc(vis_ob[:, i, :], pred_pdfm_tle2[:, i, :])
        df_vt_corr['PDFM-TLE'].append(acc.get_r())
        df_vt_mae['PDFM-TLE'].append(acc.get_mae())
        df_vt_rmse['PDFM-TLE'].append(acc.get_rmse())
        df_vt_mre['PDFM-TLE'].append(acc.get_mre())
        ts = acc.get_ts2()
        df_vt_ts1['PDFM-TLE'].append(ts[0])
        df_vt_ts2['PDFM-TLE'].append(ts[1])
        df_vt_ts3['PDFM-TLE'].append(ts[2])
        df_vt_ts4['PDFM-TLE'].append(ts[3])
        df_vt_ts5['PDFM-TLE'].append(ts[4])
        df_vt_ts6['PDFM-TLE'].append(ts[5])
        acc_nwp = VisAcc(vis_ob[i::24, :, :], cma_sh_warr[i::24, :, :])
        df_shour_corr['CMA-SH-WARR'].append(acc_nwp.get_r())
        df_shour_mae['CMA-SH-WARR'].append(acc_nwp.get_mae())
        df_shour_rmse['CMA-SH-WARR'].append(acc_nwp.get_rmse())
        df_shour_mre['CMA-SH-WARR'].append(acc_nwp.get_mre())
        ts = acc_nwp.get_ts2()
        df_shour_ts1['CMA-SH-WARR'].append(ts[0])
        df_shour_ts2['CMA-SH-WARR'].append(ts[1])
        df_shour_ts3['CMA-SH-WARR'].append(ts[2])
        df_shour_ts4['CMA-SH-WARR'].append(ts[3])
        df_shour_ts5['CMA-SH-WARR'].append(ts[4])
        df_shour_ts6['CMA-SH-WARR'].append(ts[5])
        acc = VisAcc(vis_ob[i::24, :, :], pred_pdfm_tle0[i::24, :, :])
        df_shour_corr['PDFM-TLE'].append(acc.get_r())
        df_shour_mae['PDFM-TLE'].append(acc.get_mae())
        df_shour_rmse['PDFM-TLE'].append(acc.get_rmse())
        df_shour_mre['PDFM-TLE'].append(acc.get_mre())
        ts = acc.get_ts2()
        df_shour_ts1['PDFM-TLE'].append(ts[0])
        df_shour_ts2['PDFM-TLE'].append(ts[1])
        df_shour_ts3['PDFM-TLE'].append(ts[2])
        df_shour_ts4['PDFM-TLE'].append(ts[3])
        df_shour_ts5['PDFM-TLE'].append(ts[4])
        df_shour_ts6['PDFM-TLE'].append(ts[5])
        acc_nwp = VisAcc(vis_ob[fhour_ind == i], cma_sh_warr[fhour_ind == i])
        df_fhour_corr['CMA-SH-WARR'].append(acc_nwp.get_r())
        df_fhour_mae['CMA-SH-WARR'].append(acc_nwp.get_mae())
        df_fhour_rmse['CMA-SH-WARR'].append(acc_nwp.get_rmse())
        df_fhour_mre['CMA-SH-WARR'].append(acc_nwp.get_mre())
        ts = acc_nwp.get_ts2()
        df_fhour_ts1['CMA-SH-WARR'].append(ts[0])
        df_fhour_ts2['CMA-SH-WARR'].append(ts[1])
        df_fhour_ts3['CMA-SH-WARR'].append(ts[2])
        df_fhour_ts4['CMA-SH-WARR'].append(ts[3])
        df_fhour_ts5['CMA-SH-WARR'].append(ts[4])
        df_fhour_ts6['CMA-SH-WARR'].append(ts[5])
        acc = VisAcc(vis_ob[fhour_ind == i, :], pred_pdfm_tle2[fhour_ind == i, :])
        df_fhour_corr['PDFM-TLE'].append(acc.get_r())
        df_fhour_mae['PDFM-TLE'].append(acc.get_mae())
        df_fhour_rmse['PDFM-TLE'].append(acc.get_rmse())
        df_fhour_mre['PDFM-TLE'].append(acc.get_mre())
        ts = acc.get_ts2()
        df_fhour_ts1['PDFM-TLE'].append(ts[0])
        df_fhour_ts2['PDFM-TLE'].append(ts[1])
        df_fhour_ts3['PDFM-TLE'].append(ts[2])
        df_fhour_ts4['PDFM-TLE'].append(ts[3])
        df_fhour_ts5['PDFM-TLE'].append(ts[4])
        df_fhour_ts6['PDFM-TLE'].append(ts[5])
        for j in range(24):
            acc_nwp = VisAcc(vis_ob[i::24, j, :], cma_sh_warr[i::24, j, :])
            hour_access[0, i, j, 0] = acc_nwp.get_r()
            hour_access[0, i, j, 1] = acc_nwp.get_mae()
            hour_access[0, i, j, 2] = acc_nwp.get_rmse()
            hour_access[0, i, j, 3] = acc_nwp.get_mre()
            hour_access[0, i, j, 4:] = acc_nwp.get_ts2()
            acc = VisAcc(vis_ob[i::24, j, :], pred_pdfm_tle2[i::24, j, :])
            hour_access[1, i, j, 0] = acc.get_r()
            hour_access[1, i, j, 1] = acc.get_mae()
            hour_access[1, i, j, 2] = acc.get_rmse()
            hour_access[1, i, j, 3] = acc.get_mre()
            hour_access[1, i, j, 4:] = acc.get_ts2()
            acc_nwp = VisAcc(vis_ob[i::24, j, :], pred_pdfm_tle2[i::24, j, :])
            hour_access[2, i, j, 0] = acc_nwp.get_r()
            hour_access[2, i, j, 1] = acc_nwp.get_mae()
            hour_access[2, i, j, 2] = acc_nwp.get_rmse()
            hour_access[2, i, j, 3] = acc_nwp.get_mre()
            hour_access[2, i, j, 4:] = acc_nwp.get_ts2()
            acc = VisAcc(vis_ob[i::24, j, :], pred_pdfm_tle2[i::24, j, :])
            hour_access[3, i, j, 0] = acc.get_r()
            hour_access[3, i, j, 1] = acc.get_mae()
            hour_access[3, i, j, 2] = acc.get_rmse()
            hour_access[3, i, j, 3] = acc.get_mre()
            hour_access[3, i, j, 4:] = acc.get_ts2()
    # 将时效 TS 评分结果输出为 CSV,并保存四维检验数组
    df_vt_ts4 = pd.DataFrame(df_vt_ts4)
    df_vt_ts4.to_csv(path_or_buf=r'D:\Project\vis\图\vis_vt_ts4+.csv', index=False)
    df_fhour_ts4 = pd.DataFrame(df_fhour_ts4)
    df_fhour_ts4.to_csv(path_or_buf=r'D:\Project\vis\图\vis_fhour_ts4+.csv', index=False)
    np.save(r'D:\Project\vis\图\hour_access.npy', hour_access)
    v_type = np.load(r'D:\data\vis\v_type.npy')
    v_type = v_type[-365:, :, :, index_zgdb]
    v_type = np.reshape(v_type, (-1, 24, 1183))
    df_type_corr = {'type': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_type_mae = {'type': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_type_rmse = {'type': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_type_mre = {'type': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_type_ts1 = {'type': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_type_ts2 = {'type': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_type_ts3 = {'type': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_type_ts4 = {'type': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_type_ts5 = {'type': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_type_ts6 = {'type': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    for i in range(3):
        df_type_corr['type'].append(i)
        df_type_mae['type'].append(i)
        df_type_rmse['type'].append(i)
        df_type_mre['type'].append(i)
        df_type_ts1['type'].append(i)
        df_type_ts2['type'].append(i)
        df_type_ts3['type'].append(i)
        df_type_ts4['type'].append(i)
        df_type_ts5['type'].append(i)
        df_type_ts6['type'].append(i)
        acc_nwp = VisAcc(vis_ob[v_type == i], cma_sh_warr[v_type == i])
        df_type_corr['CMA-SH-WARR'].append(acc_nwp.get_r())
        df_type_mae['CMA-SH-WARR'].append(acc_nwp.get_mae())
        df_type_rmse['CMA-SH-WARR'].append(acc_nwp.get_rmse())
        df_type_mre['CMA-SH-WARR'].append(acc_nwp.get_mre())
        ts = acc_nwp.get_ts2()
        df_type_ts1['CMA-SH-WARR'].append(ts[0])
        df_type_ts2['CMA-SH-WARR'].append(ts[1])
        df_type_ts3['CMA-SH-WARR'].append(ts[2])
        df_type_ts4['CMA-SH-WARR'].append(ts[3])
        df_type_ts5['CMA-SH-WARR'].append(ts[4])
        df_type_ts6['CMA-SH-WARR'].append(ts[5])
        acc = VisAcc(vis_ob[v_type == i], pred_pdfm_tle0[v_type == i])
        df_type_corr['PDFM-TLE'].append(acc.get_r())
        df_type_mae['PDFM-TLE'].append(acc.get_mae())
        df_type_rmse['PDFM-TLE'].append(acc.get_rmse())
        df_type_mre['PDFM-TLE'].append(acc.get_mre())
        ts = acc.get_ts2()
        df_type_ts1['PDFM-TLE'].append(ts[0])
        df_type_ts2['PDFM-TLE'].append(ts[1])
        df_type_ts3['PDFM-TLE'].append(ts[2])
        df_type_ts4['PDFM-TLE'].append(ts[3])
        df_type_ts5['PDFM-TLE'].append(ts[4])
        df_type_ts6['PDFM-TLE'].append(ts[5])
    print(df_type_corr['CMA-SH-WARR'])
    print(df_type_corr['PDFM-TLE'])
    print(df_type_mae['CMA-SH-WARR'])
    print(df_type_mae['PDFM-TLE'])
    print(df_type_rmse['CMA-SH-WARR'])
    print(df_type_rmse['PDFM-TLE'])
    print(df_type_mre['CMA-SH-WARR'])
    print(df_type_mre['PDFM-TLE'])
    print(df_type_ts1['CMA-SH-WARR'])
    print(df_type_ts1['PDFM-TLE'])
    print(df_type_ts2['CMA-SH-WARR'])
    print(df_type_ts2['PDFM-TLE'])
    print(df_type_ts3['CMA-SH-WARR'])
    print(df_type_ts3['PDFM-TLE'])
    print(df_type_ts4['CMA-SH-WARR'])
    print(df_type_ts4['PDFM-TLE'])
    print(df_type_ts5['CMA-SH-WARR'])
    print(df_type_ts5['PDFM-TLE'])
    print(df_type_ts6['CMA-SH-WARR'])
    print(df_type_ts6['PDFM-TLE'])
    df_type_corr = pd.DataFrame(df_type_corr)
    df_type_mae = pd.DataFrame(df_type_mae)
    df_type_rmse = pd.DataFrame(df_type_rmse)
    df_type_mre = pd.DataFrame(df_type_mre)
    df_type_ts1 = pd.DataFrame(df_type_ts1)
    df_type_ts2 = pd.DataFrame(df_type_ts2)
    df_type_ts3 = pd.DataFrame(df_type_ts3)
    df_type_ts4 = pd.DataFrame(df_type_ts4)
    df_type_ts5 = pd.DataFrame(df_type_ts5)
    df_type_ts6 = pd.DataFrame(df_type_ts6)
    df_type_corr.to_csv(r'D:\Project\vis\图\vis_type_corr.csv', index=False)
    df_type_mae.to_csv(r'D:\Project\vis\图\vis_type_mae.csv', index=False)
    df_type_rmse.to_csv(r'D:\Project\vis\图\vis_type_rmse.csv', index=False)
    df_type_mre.to_csv(r'D:\Project\vis\图\vis_type_mre.csv', index=False)
    df_type_ts1.to_csv(r'D:\Project\vis\图\vis_type_ts1+.csv', index=False)
    df_type_ts2.to_csv(r'D:\Project\vis\图\vis_type_ts2+.csv', index=False)
    df_type_ts3.to_csv(r'D:\Project\vis\图\vis_type_ts3+.csv', index=False)
    df_type_ts4.to_csv(r'D:\Project\vis\图\vis_type_ts4+.csv', index=False)
    df_type_ts5.to_csv(r'D:\Project\vis\图\vis_type_ts5+.csv', index=False)
    df_type_ts6.to_csv(r'D:\Project\vis\图\vis_type_ts6+.csv', index=False)
    df_sta_corr = {'sta': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_sta_mae = {'sta': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    # ------------------------------------------------------------------
    # 14. 站点级检验:计算各站的 RMSE、MRE、TS4+ 并输出 CSV
    # ------------------------------------------------------------------
    df_sta_rmse = {'sta': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_sta_mre = {'sta': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    df_sta_ts4 = {'sta': list(), 'CMA-SH-WARR': list(), 'PDFM-TLE': list()}
    for i in range(502):
        df_sta_corr['sta'].append(i)
        df_sta_mae['sta'].append(i)
        df_sta_rmse['sta'].append(i)
        df_sta_mre['sta'].append(i)
        df_sta_ts1['sta'].append(i)
        df_sta_ts2['sta'].append(i)
        df_sta_ts3['sta'].append(i)
        df_sta_ts4['sta'].append(i)
        df_sta_ts5['sta'].append(i)
        df_sta_ts6['sta'].append(i)
        acc_nwp = VisAcc(vis_ob[:, :, i], cma_sh_warr[:, :, i])
        df_sta_corr['CMA-SH-WARR'].append(acc_nwp.get_r())
        df_sta_mae['CMA-SH-WARR'].append(acc_nwp.get_mae())
        df_sta_rmse['CMA-SH-WARR'].append(acc_nwp.get_rmse())
        df_sta_mre['CMA-SH-WARR'].append(acc_nwp.get_mre())
        ts = acc_nwp.get_ts2()
        df_sta_ts1['CMA-SH-WARR'].append(ts[0])
        df_sta_ts2['CMA-SH-WARR'].append(ts[1])
        df_sta_ts3['CMA-SH-WARR'].append(ts[2])
        df_sta_ts4['CMA-SH-WARR'].append(ts[3])
        df_sta_ts5['CMA-SH-WARR'].append(ts[4])
        df_sta_ts6['CMA-SH-WARR'].append(ts[5])
        acc = VisAcc(vis_ob[:, :, i], pred_pdfm_tle2[:, :, i])
        df_sta_corr['PDFM-TLE'].append(acc.get_r())
        df_sta_mae['PDFM-TLE'].append(acc.get_mae())
        df_sta_rmse['PDFM-TLE'].append(acc.get_rmse())
        df_sta_mre['PDFM-TLE'].append(acc.get_mre())
        ts = acc.get_ts2()
        df_sta_ts1['PDFM-TLE'].append(ts[0])
        df_sta_ts2['PDFM-TLE'].append(ts[1])
        df_sta_ts3['PDFM-TLE'].append(ts[2])
        df_sta_ts4['PDFM-TLE'].append(ts[3])
        df_sta_ts5['PDFM-TLE'].append(ts[4])
        df_sta_ts6['PDFM-TLE'].append(ts[5])
    df_sta_corr = pd.DataFrame(df_sta_corr)
    df_sta_mae = pd.DataFrame(df_sta_mae)
    df_sta_rmse = pd.DataFrame(df_sta_rmse)
    df_sta_mre = pd.DataFrame(df_sta_mre)
    df_sta_ts1 = pd.DataFrame(df_sta_ts1)
    df_sta_ts2 = pd.DataFrame(df_sta_ts2)
    df_sta_ts3 = pd.DataFrame(df_sta_ts3)
    df_sta_ts4 = pd.DataFrame(df_sta_ts4)
    df_sta_ts5 = pd.DataFrame(df_sta_ts5)
    df_sta_ts6 = pd.DataFrame(df_sta_ts6)
    df_sta_corr.to_csv(r'D:\Project\vis\图\vis_sta_corr.csv', index=False)
    df_sta_mae.to_csv(r'D:\Project\vis\图\vis_sta_mae.csv', index=False)
    df_sta_rmse.to_csv(path_or_buf=r'D:\Project\vis\图\vis_sta_rmse.csv', index=False)
    df_sta_mre.to_csv(path_or_buf=r'D:\Project\vis\图\vis_sta_mre.csv', index=False)
    df_sta_ts1.to_csv(r'D:\Project\vis\图\vis_sta_ts1+.csv', index=False)
    df_sta_ts2.to_csv(r'D:\Project\vis\图\vis_sta_ts2+.csv', index=False)
    df_sta_ts3.to_csv(r'D:\Project\vis\图\vis_sta_ts3+.csv', index=False)
    df_sta_ts4.to_csv(path_or_buf=r'D:\Project\vis\图\vis_sta_ts4+.csv', index=False)
    df_sta_ts5.to_csv(r'D:\Project\vis\图\vis_sta_ts5+.csv', index=False)
    df_sta_ts6.to_csv(r'D:\Project\vis\图\vis_sta_ts6+.csv', index=False)

    # ------------------------------------------------------------------
    # 13. 绘制起报时次-预报时效二维热图(TS4+)及改善率热图
    # ------------------------------------------------------------------
    hour_access = np.load(r'D:\Project\vis\图\hour_access.npy')
    sns.heatmap(hour_access[0, :, :, 7], cmap='Reds', vmin=0, vmax=0.4, linewidths=0.3)
    plt.xticks(np.arange(24) + 0.5, [str(x) for x in range(1, 25)])
    plt.yticks(np.arange(24) + 0.5, [f'{x:02d}:00' for x in range(24)], rotation=0)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.xlabel('预报时效 (h) ')
    plt.ylabel('起报时次 (UTC) ')
    plt.savefig(r'D:\Project\vis\图\npw_hour_ts4+.png', bbox_inches='tight')
    plt.savefig(r'D:\Project\vis\图\npw_hour_ts4+.pdf', bbox_inches='tight')
    plt.cla()
    plt.close('all')
    del fig, ax
    gc.collect()
    sns.heatmap(hour_access[0, :, :, 7], cmap='Reds', vmin=0, vmax=0.4, linewidths=0.3)
    plt.xticks(np.arange(24) + 0.5, [str(x) for x in range(1, 25)])
    plt.yticks(np.arange(24) + 0.5, [f'{x:02d}:00' for x in range(24)], rotation=0)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.xlabel('预报时效 (h) ')
    plt.ylabel('起报时次 (UTC) ')
    plt.savefig(r'D:\Project\vis\图\npw_hour_ts4+.png', bbox_inches='tight', dpi=800)
    plt.savefig(r'D:\Project\vis\图\npw_hour_ts4+.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    del fig, ax
    gc.collect()
    print(np.min(hour_access[0, :, :, 7]), np.max(hour_access[0, :, :, 7]))
    sns.heatmap(hour_access[3, :, :, 7], cmap='Reds', vmin=0, vmax=0.4, linewidths=0.3)
    plt.xticks(np.arange(24) + 0.5, [str(x + 1) for x in range(24)])
    plt.yticks(np.arange(24) + 0.5, [f'{x:02d}:00' for x in range(24)], rotation=0)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.xlabel('预报时效 (h) ')
    plt.ylabel('起报时次 (UTC) ')
    plt.savefig(r'D:\Project\vis\图\pdf-tl_hour_ts4+.png', bbox_inches='tight')
    plt.savefig(r'D:\Project\vis\图\pdf-tl_hour_ts4+.pdf', bbox_inches='tight')
    plt.cla()
    plt.close('all')
    del fig, ax
    gc.collect()
    sns.heatmap(hour_access[3, :, :, 7], cmap='Reds', vmin=0, vmax=0.4, linewidths=0.3)
    plt.xticks(np.arange(24) + 0.5, [str(x + 1) for x in range(24)])
    plt.yticks(np.arange(24) + 0.5, [f'{x:02d}:00' for x in range(24)], rotation=0)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.xlabel('预报时效 (h) ')
    plt.ylabel('起报时次 (UTC) ')
    plt.savefig(r'D:\Project\vis\图\pdf-tl_hour_ts4+.png', bbox_inches='tight', dpi=800)
    plt.savefig(r'D:\Project\vis\图\pdf-tl_hour_ts4+.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    del fig, ax
    gc.collect()
    print(np.min(hour_access[3, :, :, 7]), np.max(hour_access[3, :, :, 7]))
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
    plt.savefig(r'D:\Project\vis\图\pdf-tl_hour_ts4+_improvement.png', bbox_inches='tight', dpi=800)
    plt.savefig(r'D:\Project\vis\图\pdf-tl_hour_ts4+_improvement.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    del fig, ax
    gc.collect()
    sns.heatmap(ts_improvement, cmap='Reds', vmin=0, vmax=150, linewidths=0.3)
    plt.xticks(np.arange(24) + 0.5, [str(x + 1) for x in range(24)])
    plt.yticks(np.arange(24) + 0.5, [f'{x:02d}:00' for x in range(24)], rotation=0)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.xlabel('预报时效 (h) ')
    plt.ylabel('起报时次 (UTC) ')
    plt.savefig(r'D:\Project\vis\图\pdf-tl_hour_ts4+_improvement.png', bbox_inches='tight', dpi=800)
    plt.savefig(r'D:\Project\vis\图\pdf-tl_hour_ts4+_improvement.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    del fig, ax
    gc.collect()
    print(np.min(ts_improvement), np.max(ts_improvement))
    print(np.mean(ts_improvement))
    print(np.where(ts_improvement == np.max(ts_improvement)))

    # ------------------------------------------------------------------
    # 15. 绘制预报时效(vt)与预报时间(fhour)的 TS4+ 对比柱状图
    # ------------------------------------------------------------------
    df_vt_ts4 = pd.read_csv(filepath_or_buffer=r'D:\Project\vis\图\vis_vt_ts4+.csv', low_memory=False)
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
    plt.savefig(rf'D:\Project\vis\图\vis_vt_ts4+.png', bbox_inches='tight', dpi=800)
    plt.savefig(rf'D:\Project\vis\图\vis_vt_ts4+.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    del fig, ax
    gc.collect()

    df_fhour_ts4 = pd.read_csv(filepath_or_buffer=r'D:\Project\vis\图\vis_fhour_ts4+.csv', low_memory=False)
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
    plt.savefig(rf'D:\Project\vis\图\vis_fhour_ts4+.png', bbox_inches='tight', dpi=800)
    plt.savefig(rf'D:\Project\vis\图\vis_fhour_ts4+.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    del fig, ax
    gc.collect()

    df_sta = pd.read_csv(r'D:\Project\vis\图\vis_sta_ts4+.csv', low_memory=False)
    print(np.min(df_sta.loc[:, 'CMA-SH-WARR']), np.max(df_sta.loc[:, 'CMA-SH-WARR']))
    print(np.min(df_sta.loc[:, 'PDFM-TLE']), np.max(df_sta.loc[:, 'PDFM-TLE']))
    print(stats.pearsonr(sta.loc[:, 'lon'], df_sta.loc[:, 'CMA-SH-WARR'])[0])
    print(stats.pearsonr(sta.loc[:, 'lon'], df_sta.loc[:, 'PDFM-TLE'])[0])
    print(stats.pearsonr(sta.loc[:, 'lat'], df_sta.loc[:, 'CMA-SH-WARR'])[0])
    print(stats.pearsonr(sta.loc[:, 'lat'], df_sta.loc[:, 'PDFM-TLE'])[0])
    print(stats.pearsonr(sta.loc[:, 'alti'], df_sta.loc[:, 'CMA-SH-WARR'])[0])
    print(stats.pearsonr(sta.loc[:, 'alti'], df_sta.loc[:, 'PDFM-TLE'])[0])
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
        save_path=r'D:\Project\vis\图\sta_ts4+_nwp.png',
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
        save_path=r'D:\Project\vis\图\sta_ts4+_nwp.pdf',
        dpi=800
    )
    index_in = np.zeros(502, dtype=np.bool_)
    for i in range(502):
        if sta.loc[i, 'province'] in ['河南省', '山东省', '河北省', '北京市', '天津市', '山西省']:
            index_in[i] = True
    mean_in = np.mean(df_sta.loc[index_in, 'CMA-SH-WARR'])
    mean_out = np.mean(df_sta.loc[~index_in, 'CMA-SH-WARR'])
    print(f'CMA-SH-WARR: {mean_in}, {mean_out}')
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
        save_path=r'D:\Project\vis\图\sta_ts4+.png',
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
        save_path=r'D:\Project\vis\图\sta_ts4+.pdf',
        dpi=800
    )
    df_sta = pd.read_csv(r'D:\Project\vis\图\vis_sta_rmse.csv', low_memory=False)
    index_in = np.zeros(502, dtype=np.bool_)
    for i in range(502):
        if sta.loc[i, 'province'] in ['河南省', '山东省', '河北省', '北京市', '天津市', '山西省']:
            index_in[i] = True
    mean_in = np.mean(df_sta.loc[index_in, 'PDFM-TLE'])
    mean_out = np.mean(df_sta.loc[~index_in, 'PDFM-TLE'])
    print(f'PDFM-TLE: {mean_in}, {mean_out}')
    rmse_before = np.array(df_sta.loc[:, 'CMA-SH-WARR'])
    rmse_after = np.array(df_sta.loc[:, 'PDFM-TLE'])
    rmse_improvement = (rmse_before - rmse_after) / rmse_before * 100
    print(np.min(rmse_improvement), np.max(rmse_improvement))
    cmap, clevs = meb.def_cmap_clevs(
        meb.def_cmap_clevs(meb.cmaps.me, vmin=-40, vmax=60)[0],
        clevs=[-40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
    )
    sta0 = sta.loc[:, ('level', 'time', 'dtime', 'id', 'lat', 'lon', 'data0')]
    sta0.loc[:, 'data0'] = rmse_improvement
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='both',
        title=[''],
        save_path=r'D:\Project\vis\图\sta_mre_improvement.png',
        dpi=800
    )
    meb.tool.plot_tools.scatter_sta(
        sta0=sta0.copy(),
        point_size=20,
        map_extend=[108, 123, 24, 36],
        clevs=clevs,
        cmap=cmap,
        extend='both',
        title=[''],
        save_path=r'D:\Project\vis\图\sta_mre_improvement.pdf',
        dpi=800
    )
    print(np.mean(rmse_improvement))

    df_sta = pd.read_csv(r'D:\Project\vis\图\vis_sta_mre.csv', low_memory=False)
    index_in = np.zeros(1183, dtype=np.bool_)
    for i in range(1183):
        if sta.loc[i, 'province'] in ['河南省', '山东省', '河北省', '北京市', '天津市', '山西省']:
            index_in[i] = True
    mean_in = np.mean(df_sta.loc[index_in, 'PDFM-TLE'])
    mean_out = np.mean(df_sta.loc[~index_in, 'PDFM-TLE'])
    print(f'PDFM-TLE: {mean_in}, {mean_out}')
    mre_before = np.array(df_sta.loc[:, 'CMA-SH-WARR'])
    mre_after = np.array(df_sta.loc[:, 'PDFM-TLE'])
    mre_improvement = (mre_before - mre_after) / mre_before * 100
    mean_in = np.mean(mre_improvement[index_in])
    mean_out = np.mean(mre_improvement[~index_in])
    print(f'MRE_improvement: {mean_in}, {mean_out}')

    print(np.min(df_sta.loc[:, 'CMA-SH-WARR']), np.max(df_sta.loc[:, 'CMA-SH-WARR']))
    print(np.min(df_sta.loc[:, 'PDFM-TLE']), np.max(df_sta.loc[:, 'PDFM-TLE']))
    print(np.min(mre_improvement), np.max(mre_improvement))
    plt.figure(figsize=(5, 6))
    plt.scatter(
        x=sta.loc[:, 'lon'],
        y=df_sta.loc[:, 'CMA-SH-WARR'],
        s=1,
        c='blue',
        label='CMA-SH-WARR'
    )
    plt.scatter(
        x=sta.loc[:, 'lon'],
        y=df_sta.loc[:, 'PDFM-TLE'],
        s=1,
        c='red',
        label='PDFM-TLE'
    )
    plt.xlim((105, 125))
    plt.xticks([105, 110, 115, 120, 125], ['105°', '110°', '115°', '120°', '125°N'])
    plt.ylim((0, 0.6))
    plt.yticks((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    plt.xlabel('经度')
    plt.ylabel('MRE')
    plt.legend()
    plt.savefig(rf'D:\Project\vis\图\sta_lon-mre.png', bbox_inches='tight', dpi=800)
    plt.savefig(rf'D:\Project\vis\图\sta_lon-mre.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    del fig, ax
    gc.collect()
    print(stats.pearsonr(sta.loc[:, 'lon'], df_sta.loc[:, 'CMA-SH-WARR'])[0])
    print(stats.pearsonr(sta.loc[:, 'lon'], df_sta.loc[:, 'PDFM-TLE'])[0])
    lon = np.array(sta.loc[:, 'lon'])
    plt.figure(figsize=(5, 6))
    a, b = np.polyfit(lon, mre_improvement, deg=1)
    print(f'a: {a}, b: {b}')
    x_l = np.linspace(105, 125, 10000)
    y_est = a * x_l + b
    t = stats.t.isf(0.05 / 2, lon.size - 2)
    y_err = t * np.std(y_est) * (1 + 1 / lon.size + (x_l - np.mean(lon)) ** 2 / np.sum((lon - np.mean(lon)) ** 2)) ** 0.5
    y_ = a * lon + b
    y__ = t * np.std(y_est) * (1 + 1 / lon.size + (lon - np.mean(lon)) ** 2 / np.sum((lon - np.mean(lon)) ** 2)) ** 0.5
    is_95 = (mre_improvement >= y_ - y__) & (mre_improvement <= y_ + y__)
    print(np.mean(is_95))
    plt.plot(x_l, y_est, c='black')
    plt.fill_between(x_l, y_est - y_err, y_est + y_err, alpha=0.1, color='blue')
    plt.scatter(
        x=lon,
        y=mre_improvement,
        s=1,
        c='red',
        label='PDFM-TLE'
    )
    plt.xlim((105, 125))
    plt.xticks([105, 110, 115, 120, 125], ['105°', '110°', '115°', '120°', '125°N'])
    plt.ylim((-40, 80))
    plt.yticks((-40, -20, 0, 20, 40, 60, 80))
    plt.xlabel('经度')
    plt.ylabel('MRE改善率(%)')
    plt.savefig(r'D:\Project\vis\图\sta_lon-mre_improvement.png', bbox_inches='tight', dpi=800)
    plt.savefig(r'D:\Project\vis\图\sta_lon-mre_improvement.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    del fig, ax
    gc.collect()
    cc, p_value = stats.pearsonr(sta.loc[:, 'lon'], mre_improvement)
    print(cc, p_value)
    plt.figure(figsize=(5, 6))
    plt.scatter(
        x=sta.loc[:, 'lat'],
        y=df_sta.loc[:, 'CMA-SH-WARR'],
        s=1,
        c='blue',
        label='CMA-SH-WARR'
    )
    plt.scatter(
        x=sta.loc[:, 'lat'],
        y=df_sta.loc[:, 'PDFM-TLE'],
        s=1,
        c='red',
        label='PDFM-TLE'
    )
    plt.xlim((20, 45))
    plt.xticks([20, 25, 30, 35, 40, 45], ['20°', '25°', '30°', '35°', '40', '45°N'])
    plt.ylim((0, 0.6))
    plt.yticks((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    plt.xlabel('纬度')
    plt.ylabel('MRE')
    plt.legend()
    plt.savefig(rf'D:\Project\vis\图\sta_lat-mre.png', bbox_inches='tight', dpi=800)
    plt.savefig(rf'D:\Project\vis\图\sta_lat-mre.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    del fig, ax
    gc.collect()
    print(stats.pearsonr(sta.loc[:, 'lat'], df_sta.loc[:, 'CMA-SH-WARR'])[0])
    print(stats.pearsonr(sta.loc[:, 'lat'], df_sta.loc[:, 'PDFM-TLE'])[0])
    lat = np.array(sta.loc[:, 'lat'])
    plt.figure(figsize=(5, 6))
    a, b = np.polyfit(lat, mre_improvement, deg=1)
    print(f'a: {a}, b: {b}')
    x_l = np.linspace(20, 45, 10000)
    y_est = a * x_l + b
    t = stats.t.isf(0.05 / 2, lat.size - 2)
    y_err = t * np.std(y_est) * (1 + 1 / lat.size + (x_l - np.mean(lat)) ** 2 / np.sum((lat - np.mean(lat)) ** 2)) ** 0.5
    y_ = a * lat + b
    y__ = t * np.std(y_est) * (1 + 1 / lat.size + (lat - np.mean(lat)) ** 2 / np.sum((lat - np.mean(lat)) ** 2)) ** 0.5
    is_95 = (mre_improvement >= y_ - y__) & (mre_improvement <= y_ + y__)
    print(np.mean(is_95))
    plt.plot(x_l, y_est, c='black')
    plt.fill_between(x_l, y_est - y_err, y_est + y_err, alpha=0.1, color='blue')
    plt.scatter(
        x=lat,
        y=mre_improvement,
        s=1,
        c='red',
        label='PDFM-TLE'
    )
    plt.xlim((20, 45))
    plt.xticks([20, 25, 30, 35, 40, 45], ['20°', '25°', '30°', '35°', '40', '45°N'])
    plt.ylim((-40, 80))
    plt.yticks((-40, -20, 0, 20, 40, 60, 80))
    plt.xlabel('纬度')
    plt.ylabel('MRE改善率(%)')
    plt.savefig(r'D:\Project\vis\图\sta_lat-mre_improvement.png', bbox_inches='tight', dpi=800)
    plt.savefig(r'D:\Project\vis\图\sta_lat-mre_improvement.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    del fig, ax
    gc.collect()
    cc, p_value = stats.pearsonr(sta.loc[:, 'lat'], mre_improvement)
    print(cc, p_value)
    plt.figure(figsize=(5, 6))
    plt.scatter(
        x=sta.loc[:, 'alti'],
        y=df_sta.loc[:, 'CMA-SH-WARR'],
        s=1,
        c='blue',
        label='CMA-SH-WARR'
    )
    plt.scatter(
        x=sta.loc[:, 'alti'],
        y=df_sta.loc[:, 'PDFM-TLE'],
        s=1,
        c='red',
        label='PDFM-TLE'
    )
    plt.xlim((-100, 3600))
    plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500],
               ['0', '500', '1000', '1500', '2000', '2500', '3000', '3500'])
    plt.ylim((0, 0.6))
    plt.yticks((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    plt.xlabel('纬度')
    plt.ylabel('MRE')
    plt.legend()
    plt.savefig(rf'D:\Project\vis\图\sta_alti-mre.png', bbox_inches='tight', dpi=800)
    plt.savefig(rf'D:\Project\vis\图\sta_alti-mre.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    del fig, ax
    gc.collect()
    print(stats.pearsonr(sta.loc[:, 'alti'], df_sta.loc[:, 'CMA-SH-WARR'])[0])
    print(stats.pearsonr(sta.loc[:, 'alti'], df_sta.loc[:, 'PDFM-TLE'])[0])
    alti = np.array(sta.loc[:, 'alti'])
    plt.figure(figsize=(5, 6))
    a, b = np.polyfit(alti, mre_improvement, deg=1)
    print(f'a: {a}, b: {b}')
    x_l = np.linspace(-100, 3600, 10000)
    y_est = a * x_l + b
    t = stats.t.isf(0.05 / 2, alti.size - 2)
    y_err = t * np.std(y_est) * (1 + 1 / alti.size + (x_l - np.mean(alti)) ** 2 / np.sum((alti - np.mean(alti)) ** 2)) ** 0.5
    y_ = a * alti + b
    y__ = t * np.std(y_est) * (1 + 1 / alti.size + (alti - np.mean(alti)) ** 2 / np.sum((alti - np.mean(alti)) ** 2)) ** 0.5
    is_95 = (mre_improvement >= y_ - y__) & (mre_improvement <= y_ + y__)
    print(np.mean(is_95))
    plt.plot(x_l, y_est, c='black')
    plt.fill_between(x_l, y_est - y_err, y_est + y_err, alpha=0.1, color='blue')
    plt.scatter(
        x=alti,
        y=mre_improvement,
        s=1,
        c='red',
        label='PDFM-TLE'
    )
    plt.xlim((-100, 3600))
    plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500],
               ['0', '500', '1000', '1500', '2000', '2500', '3000', '3500'])
    plt.ylim((-40, 80))
    plt.yticks((-40, -20, 0, 20, 40, 60, 80))
    plt.xlabel('高程(m)')
    plt.ylabel('MRE改善率(%)')
    plt.savefig(r'D:\Project\vis\图\sta_alti-mre_improvement.png', bbox_inches='tight', dpi=800)
    plt.savefig(r'D:\Project\vis\图\sta_alti-mre_improvement.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    del fig, ax
    gc.collect()
    cc, p_value = stats.pearsonr(sta.loc[:, 'alti'], mre_improvement)
    print(cc, p_value)
    
    plt.figure(figsize=(5, 6))
    sns.violinplot(
        data={
            'CMA-SH-WARR': df_sta.loc[:, 'CMA-SH-WARR'],
            'PDFM-TLE': df_sta.loc[:, 'PDFM-TLE'],
        },
        palette=['blue', 'red']
    )
    plt.ylabel('MRE')
    plt.savefig(r'D:\Project\vis\图\boxplot_mre.png', bbox_inches='tight', dpi=800)
    plt.savefig(r'D:\Project\vis\图\boxplot_mre.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    del fig, ax
    gc.collect()
    print(np.median(df_sta.loc[:, 'CMA-SH-WARR']), np.median(df_sta.loc[:, 'PDFM-TLE']))
    print(np.median(df_sta.loc[:, 'CMA-SH-WARR']) - np.median(df_sta.loc[:, 'PDFM-TLE']))
    
    plt.figure(figsize=(5, 6))
    sns.violinplot(
        data={'PDFM-TLE': mre_improvement},
        color='skyblue'
    )
    plt.ylabel('MRE改善率')
    plt.savefig(r'D:\Project\vis\图\boxplot_mre_improvement.png', bbox_inches='tight', dpi=800)
    plt.savefig(r'D:\Project\vis\图\boxplot_mre_improvement.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    del fig, ax
    gc.collect()
    print(np.mean(mre_improvement > 0))

    # ------------------------------------------------------------------
    # 16. 2024 年独立样本个例分析:对选定低能见度过程计算 TS4+
    # ------------------------------------------------------------------
    # 先对比 PDFM 与 TLE 两种方案在 2020-2023 年上的整体检验指标
    pred_pdfm2 = np.load(r'D:\data\vis\vis_gjz_pdfm2_cjzxy.npy')
    pred_pdfm2 = np.reshape(pred_pdfm2, shape=(-1, 24, 502))
    pred_pdfm2[pred_pdfm2 >= 30000] = 30000
    pred_tle2 = np.load(r'D:\data\vis\vis_gjz_tle2_cjzxy.npy')
    pred_tle2 = np.reshape(pred_tle2, shape=(-1, 24, 502))
    pred_tle2[pred_tle2 >= 30000] = 30000
    acc = VisAcc(vis_ob, pred_pdfm2)
    print('PDFM')
    print(acc.get_r(), acc.get_mae(), acc.get_rmse(), acc.get_mre())
    print(acc.get_ts2(), acc.get_far2(), acc.get_mar2())
    acc = VisAcc(vis_ob, pred_tle2)
    print('TLE')
    print(acc.get_r(), acc.get_mae(), acc.get_rmse(), acc.get_mre())
    print(acc.get_ts2(), acc.get_far2(), acc.get_mar2())

    # 加载 2024 年观测与预报数据,并进行时效对齐偏移
    vis_ob = np.load(r'D:\data\vis\vis1183_ob_2024.npy')[:, 1:, index_cjzxy]
    vis_ob = np.reshape(vis_ob, shape=(-1, 24, 502))
    vis_ob[vis_ob >= 30000] = 30000
    cma_sh_warr = np.load(r'D:\data\vis\vis_gjz_2024.npy')
    cma_sh_warr = np.reshape(cma_sh_warr[:, 1:, index_cjzxy], shape=(-1, 24, 502))
    cma_sh_warr[cma_sh_warr >= 30000] = 30000
    pred_pdfm2 = np.load(r'D:\data\vis\vis_gjz_pdfm2_cjzxy_2024.npy')
    pred_pdfm2 = np.reshape(pred_pdfm2, shape=(-1, 24, 502))
    pred_pdfm2[pred_pdfm2 >= 30000] = 30000
    # 对 24 个起报时次分别做时间偏移,使预报时效与实况时间对齐
    vis_ob_ = np.zeros_like(vis_ob) + np.nan
    pred_pdfm2_ = np.zeros_like(pred_pdfm2) + np.nan
    cma_sh_warr_ = np.zeros_like(cma_sh_warr) + np.nan
    for i in range(24):
        vis_ob_[i + 1:, i, :] = vis_ob[: 8783 - i, i, :]
        pred_pdfm2_[i + 1:, i, :] = pred_pdfm2[: 8783 - i, i, :]
        cma_sh_warr_[i + 1:, i, :] = cma_sh_warr[: 8783 - i, i, :]
    # 18 组低能见度个例,每组包含 [过程开始, 过程结束, 峰值日期]
    date_list = [
        ['2024-01-02', '2024-01-13', '2024-01-03'],
        ['2024-01-29', '2024-02-01', '2024-01-30'],
        ['2024-02-08', '2024-02-11', '2024-02-10'],
        ['2024-03-02', '2024-03-08', '2024-03-04'],
        ['2024-03-11', '2024-03-18', '2024-03-14'],
        ['2024-03-23', '2024-03-28', '2024-03-27'],
        ['2024-03-31', '2024-04-02', '2024-04-01'],
        ['2024-04-06', '2024-04-08', '2024-04-08'],
        ['2024-04-11', '2024-04-29', '2024-04-14'],
        ['2024-05-02', '2024-05-06', '2024-05-05'],
        ['2024-06-09', '2024-06-12', '2024-06-10'],
        ['2024-06-18', '2024-06-21', '2024-06-20'],
        ['2024-06-24', '2024-07-01', '2024-06-27'],
        ['2024-07-11', '2024-07-13', '2024-07-11'],
        ['2024-10-14', '2024-10-18', '2024-10-14'],
        ['2024-10-27', '2024-10-29', '2024-10-28'],
        ['2024-11-10', '2024-11-12', '2024-11-10'],
        ['2024-12-04', '2024-12-07', '2024-12-05']
    ]
    ind = np.zeros(shape=(8784, 4), dtype=np.bool_)
    for d in date_list:
        i = round((arrow.get(d[0]) - arrow.get('2024')).total_seconds() / 3600)
        j = round((arrow.get(d[1]) - arrow.get('2024')).total_seconds() / 3600)
        k = round((arrow.get(d[2]) - arrow.get('2024')).total_seconds() / 3600)
        # 输出各阶段(全程、开始、结束、峰值)的 CMA-SH-WARR 与 PDFM-TLE 的 TS4+
        print(
            VisAcc(vis_ob_[i: j + 24, :, :], cma_sh_warr_[i: j + 24, :, :]).get_ts2()[3],
            VisAcc(vis_ob_[i: i + 24, :, :], cma_sh_warr_[i: i + 24, :, :]).get_ts2()[3],
            VisAcc(vis_ob_[j: j + 24, :, :], cma_sh_warr_[j: j + 24, :, :]).get_ts2()[3],
            VisAcc(vis_ob_[k: k + 24, :, :], cma_sh_warr_[k: k + 24, :, :]).get_ts2()[3],
            VisAcc(vis_ob_[i: j + 24, :, :], pred_pdfm2_[i: j + 24, :, :]).get_ts2()[3],
            VisAcc(vis_ob_[i: i + 24, :, :], pred_pdfm2_[i: i + 24, :, :]).get_ts2()[3],
            VisAcc(vis_ob_[j: j + 24, :, :], pred_pdfm2_[j: j + 24, :, :]).get_ts2()[3],
            VisAcc(vis_ob_[k: k + 24, :, :], pred_pdfm2_[k: k + 24, :, :]).get_ts2()[3]
        )
        ind[i: j + 24, 0] = True
        ind[i: i + 24, 1] = True
        ind[j: j + 24, 2] = True
        ind[k: k + 24, 3] = True
    # 输出全部个例合并后的 TS4+
    print(
        VisAcc(vis_ob_[ind[:, 0], :, :], cma_sh_warr_[ind[:, 0], :, :]).get_ts2()[3],
        VisAcc(vis_ob_[ind[:, 1], :, :], cma_sh_warr_[ind[:, 1], :, :]).get_ts2()[3],
        VisAcc(vis_ob_[ind[:, 2], :, :], cma_sh_warr_[ind[:, 2], :, :]).get_ts2()[3],
        VisAcc(vis_ob_[ind[:, 3], :, :], cma_sh_warr_[ind[:, 3], :, :]).get_ts2()[3],
        VisAcc(vis_ob_[ind[:, 0], :, :], pred_pdfm2_[ind[:, 0], :, :]).get_ts2()[3],
        VisAcc(vis_ob_[ind[:, 1], :, :], pred_pdfm2_[ind[:, 1], :, :]).get_ts2()[3],
        VisAcc(vis_ob_[ind[:, 2], :, :], pred_pdfm2_[ind[:, 2], :, :]).get_ts2()[3],
        VisAcc(vis_ob_[ind[:, 3], :, :], pred_pdfm2_[ind[:, 3], :, :]).get_ts2()[3]
    )


if __name__ == '__main__':
    print('Program draw.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(f'Program draw.py finished, total time: {format_time(total_elapsed)}')
