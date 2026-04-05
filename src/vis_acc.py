# -*- coding: utf-8 -*-
"""
能见度预报检验指标计算模块.

Founded in 2024-04-18
Modified in 2026-04-04
@author: yinlb
"""

import copy

import numpy as np
from scipy import stats


# Six-grade visibility thresholds (m): <10000, <2000, <1000, <500, <200, <50
THRES = (10000., 2000., 1000., 500., 200., 50.)


class VisAcc:
    """
    能见度预报检验指标计算类.

    Auto-grade visibility from obs/fcst arrays and build confusion matrix,
    providing quantitative metrics (ME, MAE, RMSE, MRE, R) and grade metrics (TS, ETS, HSS, TSS,
    BIAS, FAR, MAR, POD, OA, Kappa) query methods.
    """

    def __init__(self, ob: np.ndarray, pr: np.ndarray) -> None:
        """
        初始化检验对象.

        Args:
            ob (np.ndarray): Obs visibility array, np.nan for missing.
            pr (np.ndarray): Forecast visibility array, np.nan for missing.
        """
        self._thres = THRES
        self._n_grades = len(self._thres)
        self._ob = ob
        self._pr = pr

        # Init obs grade matrix: -1=missing, 0=above max threshold, 1-6 by thresholds
        self._ob_grade = np.zeros_like(self._ob, dtype=np.int_) - 1
        self._ob_grade[~np.isnan(self._ob)] = 0
        for i in range(self._n_grades):
            self._ob_grade[self._ob < self._thres[i]] = i + 1

        # Init forecast grade matrix, same logic
        self._pr_grade = np.zeros_like(self._pr, dtype=np.int_) - 1
        self._pr_grade[~np.isnan(self._pr)] = 0
        for i in range(self._n_grades):
            self._pr_grade[self._pr < self._thres[i]] = i + 1

        # Build (n_grades+1)*(n_grades+1) confusion matrix
        self._hxjz = np.zeros((self._n_grades + 1, self._n_grades + 1), dtype=np.int_)
        for i in range(self._n_grades + 1):
            for j in range(self._n_grades + 1):
                self._hxjz[i, j] = np.sum((self._ob_grade == i) & (self._pr_grade == j))

        # Total valid samples
        self._n = np.sum(self._hxjz)

    def copy(self) -> 'VisAcc':
        """返回深拷贝对象."""
        return copy.deepcopy(self)

    def get_me(self) -> float:
        """计算平均误差(Mean Error, ME)."""
        index = (~np.isnan(self._ob)) & (~np.isnan(self._pr))
        ob = self._ob[index]
        pr = self._pr[index]
        if len(ob) == 0:
            return np.nan
        return float(np.mean(pr - ob))

    def get_mae(self) -> float:
        """计算平均绝对误差(Mean Absolute Error, MAE)."""
        index = (~np.isnan(self._ob)) & (~np.isnan(self._pr))
        ob = self._ob[index]
        pr = self._pr[index]
        if len(ob) == 0:
            return np.nan
        return float(np.mean(np.abs(pr - ob)))

    def get_rmse(self) -> float:
        """计算均方根误差(Root Mean Square Error, RMSE)."""
        index = (~np.isnan(self._ob)) & (~np.isnan(self._pr))
        ob = self._ob[index]
        pr = self._pr[index]
        if len(ob) == 0:
            return np.nan
        return float(np.mean((pr - ob) ** 2) ** 0.5)

    def get_mre(self) -> float:
        """计算平均相对误差(Mean Relative Error, MRE)."""
        # 严格过滤：排除nan、排除分母过小（避免除零或数值爆炸）
        index = (~np.isnan(self._ob)) & (~np.isnan(self._pr)) & (self._ob + self._pr > 1e-6)
        ob = self._ob[index]
        pr = self._pr[index]
        if len(ob) == 0:
            return np.nan  # 无有效数据时明确返回nan
        return float(np.mean(np.abs((pr - ob) / (pr + ob))))

    def get_nme(self) -> float:
        """计算归一化平均误差(Normalized ME)."""
        index = (~np.isnan(self._ob)) & (~np.isnan(self._pr))
        ob = self._ob[index]
        pr = self._pr[index]
        if len(ob) == 0:
            return np.nan
        denom = np.max(ob) - np.min(ob)
        if denom < 1e-6:
            return np.nan  # 避免除零
        return float(np.mean(pr - ob) / denom)

    def get_nmae(self) -> float:
        """计算归一化平均绝对误差(Normalized MAE)."""
        index = (~np.isnan(self._ob)) & (~np.isnan(self._pr))
        ob = self._ob[index]
        pr = self._pr[index]
        if len(ob) == 0:
            return np.nan
        denom = np.max(ob) - np.min(ob)
        if denom < 1e-6:
            return np.nan  # 避免除零
        return float(np.mean(np.abs(pr - ob)) / denom)

    def get_nrmse(self) -> float:
        """计算归一化均方根误差(Normalized RMSE)."""
        index = (~np.isnan(self._ob)) & (~np.isnan(self._pr))
        ob = self._ob[index]
        pr = self._pr[index]
        if len(ob) == 0:
            return np.nan
        denom = np.max(ob) - np.min(ob)
        if denom < 1e-6:
            return np.nan  # 避免除零
        return float(np.mean((pr - ob) ** 2) ** 0.5 / denom)

    def get_r(self) -> float:
        """计算Pearson相关系数(R)."""
        index = (~np.isnan(self._ob)) & (~np.isnan(self._pr))
        ob = self._ob[index]
        pr = self._pr[index]
        if np.sum(index) > 2:
            r = float(stats.pearsonr(ob, pr)[0])
        else:
            r = np.nan
        return r

    def get_me1(self) -> np.ndarray:
        """按观测分级计算各级的平均误差(ME),返回长度为n_grades+1的数组."""
        me = np.zeros(self._n_grades + 1, dtype=np.float32) + np.nan
        for i in range(self._n_grades + 1):
            index = (~np.isnan(self._ob)) & (~np.isnan(self._pr)) & (self._ob_grade == i)
            ob = self._ob[index]
            pr = self._pr[index]
            if len(ob) > 0:
                me[i] = np.mean(pr - ob)
        return me

    def get_mae1(self) -> np.ndarray:
        """按观测分级计算各级的平均绝对误差(MAE),返回长度为n_grades+1的数组."""
        mae = np.zeros(self._n_grades + 1, dtype=np.float32) + np.nan
        for i in range(self._n_grades + 1):
            index = (~np.isnan(self._ob)) & (~np.isnan(self._pr)) & (self._ob_grade == i)
            ob = self._ob[index]
            pr = self._pr[index]
            if len(ob) > 0:
                mae[i] = np.mean(np.abs(pr - ob))
        return mae

    def get_rmse1(self) -> np.ndarray:
        """按观测分级计算各级的均方根误差(RMSE),返回长度为n_grades+1的数组."""
        rmse = np.zeros(self._n_grades + 1, dtype=np.float32) + np.nan
        for i in range(self._n_grades + 1):
            index = (~np.isnan(self._ob)) & (~np.isnan(self._pr)) & (self._ob_grade == i)
            ob = self._ob[index]
            pr = self._pr[index]
            if len(ob) > 0:
                rmse[i] = np.mean((pr - ob) ** 2) ** 0.5
        return rmse

    def get_mre1(self) -> np.ndarray:
        """按观测分级计算各级的平均相对误差(MRE),返回长度为n_grades+1的数组."""
        mre = np.zeros(self._n_grades + 1, dtype=np.float32) + np.nan
        for i in range(self._n_grades + 1):
            index = (
                (~np.isnan(self._ob))
                & (~np.isnan(self._pr))
                & (self._ob + self._pr > 1e-6)  # 严格过滤，避免除零
                & (self._ob_grade == i)
            )
            ob = self._ob[index]
            pr = self._pr[index]
            if len(ob) > 0:
                mre[i] = np.mean(np.abs((pr - ob) / (pr + ob)))
        return mre

    def get_r1(self) -> np.ndarray:
        """按观测分级计算各级的Pearson相关系数(R),返回长度为n_grades+1的数组."""
        r = np.zeros(self._n_grades + 1, dtype=np.float32) + np.nan
        for i in range(self._n_grades + 1):
            index = (
                (~np.isnan(self._ob))
                & (~np.isnan(self._pr))
                & (self._ob_grade == i)
            )
            ob = self._ob[index]
            pr = self._pr[index]
            # pearsonr需要至少2个样本
            if len(ob) > 2:
                r[i] = stats.pearsonr(ob, pr)[0]
        return r

    def get_hxjz(self) -> np.ndarray:
        """返回混淆矩阵(整数计数)."""
        return self._hxjz

    def get_hxjz2(self) -> np.ndarray:
        """返回归一化混淆矩阵,每行除以该观测级的总站次数."""
        hxjz2 = np.zeros_like(self._hxjz, dtype=np.float32) + np.nan
        for i in range(self._n_grades + 1):
            row_sum = np.sum(self._hxjz[i, :])
            if row_sum > 0:
                for j in range(self._n_grades + 1):
                    hxjz2[i, j] = self._hxjz[i, j] / row_sum
        return hxjz2

    def get_oa(self) -> float:
        """计算总体准确度(Overall Accuracy, OA),即对角线之和占总样本比例."""
        if self._n == 0:
            return np.nan
        return np.sum(np.diag(self._hxjz)) / self._n

    def get_kappa(self) -> float:
        """计算Kappa系数,衡量分级一致性."""
        if self._n == 0:
            return np.nan
        a = np.sum(self._hxjz, axis=1).astype(np.float32)
        b = np.sum(self._hxjz, axis=0).astype(np.float32)
        pe = np.sum(a * b) / self._n / self._n
        if abs(1 - pe) < 1e-6:
            return np.nan  # 避免除零
        return (self.get_oa() - pe) / (1 - pe)

    def get_ts(self) -> np.ndarray:
        """计算各级的威胁评分(Threat Score, TS),返回长度为n_grades+1的数组."""
        ts = np.zeros(self._n_grades + 1, dtype=np.float32)
        for i in range(self._n_grades + 1):
            na = self._hxjz[i, i]
            nb = np.sum(self._hxjz[:, i]) - na
            nc = np.sum(self._hxjz[i, :]) - na
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ts2(self) -> np.ndarray:
        """计算合并等级的威胁评分(TS),返回长度为n_grades的数组.

        对第i个阈值,将等级i+1及以上视为正例,其余为负例,构造2*2联表:
        na = 观测正且预报正, nb = 观测负且预报正, nc = 观测正且预报负.
        """
        ts = np.zeros(self._n_grades, dtype=np.float32)
        for i in range(self._n_grades):
            na = np.sum(self._hxjz[i + 1:, i + 1:])
            nb = np.sum(self._hxjz[:i + 1, i + 1:])
            nc = np.sum(self._hxjz[i + 1:, :i + 1])
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ets2(self) -> np.ndarray:
        """计算合并等级的公平威胁评分(Equitable Threat Score, ETS)."""
        ets = np.zeros(self._n_grades, dtype=np.float32)
        for i in range(self._n_grades):
            na = np.sum(self._hxjz[i + 1:, i + 1:])
            nb = np.sum(self._hxjz[:i + 1, i + 1:])
            nc = np.sum(self._hxjz[i + 1:, :i + 1])
            nd = np.sum(self._hxjz[:i + 1, :i + 1])
            r = (na + nb) / (na + nb + nc + nd) * (na + nc)
            ets[i] = (na - r) / (na + nb + nc - r) if na + nb + nc != 0 else np.nan
        return ets

    def get_hss2(self) -> np.ndarray:
        """计算合并等级的Heidke技巧评分(Heidke Skill Score, HSS)."""
        hss = np.zeros(self._n_grades, dtype=np.float32) + np.nan
        for i in range(self._n_grades):
            na = np.sum(self._hxjz[i + 1:, i + 1:])
            nb = np.sum(self._hxjz[:i + 1, i + 1:])
            nc = np.sum(self._hxjz[i + 1:, :i + 1])
            nd = np.sum(self._hxjz[:i + 1, :i + 1])
            denom = (na + nc) * (nc + nd) + (na + nb) * (nb + nd)
            if abs(denom) > 1e-6:
                hss[i] = 2 * (na * nd - nb * nc) / denom
        return hss

    def get_tss2(self) -> np.ndarray:
        """计算合并等级的True Skill Statistic(TSS,即Pierce's Skill Score)."""
        tss = np.zeros(self._n_grades, dtype=np.float32) + np.nan
        for i in range(self._n_grades):
            na = np.sum(self._hxjz[i + 1:, i + 1:])
            nb = np.sum(self._hxjz[:i + 1, i + 1:])
            nc = np.sum(self._hxjz[i + 1:, :i + 1])
            nd = np.sum(self._hxjz[:i + 1, :i + 1])
            denom1 = na + nc
            denom2 = nb + nd
            if abs(denom1) > 1e-6 and abs(denom2) > 1e-6:
                tss[i] = (na * nd - nb * nc) / denom1 / denom2
        return tss

    def get_bias2(self) -> np.ndarray:
        """计算合并等级的频率偏差(Frequency BIAS)."""
        bias = np.zeros(self._n_grades, dtype=np.float32) + np.nan
        for i in range(self._n_grades):
            na = np.sum(self._hxjz[i + 1:, i + 1:])
            nb = np.sum(self._hxjz[:i + 1, i + 1:])
            nc = np.sum(self._hxjz[i + 1:, :i + 1])
            denom = na + nc
            if abs(denom) > 1e-6:
                bias[i] = (na + nb) / denom
        return bias

    def get_far2(self) -> np.ndarray:
        """计算合并等级的空报率(False Alarm Ratio, FAR)."""
        far = np.zeros(self._n_grades, dtype=np.float32) + np.nan
        for i in range(self._n_grades):
            na = np.sum(self._hxjz[i + 1:, i + 1:])
            nb = np.sum(self._hxjz[:i + 1, i + 1:])
            denom = na + nb
            if abs(denom) > 1e-6:
                far[i] = nb / denom
        return far

    def get_mar2(self) -> np.ndarray:
        """计算合并等级的漏报率(Miss Alarm Ratio, MAR)."""
        mar = np.zeros(self._n_grades, dtype=np.float32) + np.nan
        for i in range(self._n_grades):
            na = np.sum(self._hxjz[i + 1:, i + 1:])
            nc = np.sum(self._hxjz[i + 1:, :i + 1])
            denom = na + nc
            if abs(denom) > 1e-6:
                mar[i] = nc / denom
        return mar

    def get_pod2(self) -> np.ndarray:
        """计算合并等级的命中率(Probability of Detection, POD)."""
        pod = np.zeros(self._n_grades, dtype=np.float32) + np.nan
        for i in range(self._n_grades):
            na = np.sum(self._hxjz[i + 1:, i + 1:])
            nc = np.sum(self._hxjz[i + 1:, :i + 1])
            denom = na + nc
            if abs(denom) > 1e-6:
                pod[i] = na / denom
        return pod
