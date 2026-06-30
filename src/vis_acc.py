# -*- coding: utf-8 -*-
"""
Visibility forecast verification metrics calculation module.

Founded in 2024-04-18
Modified in 2026-05-09
@author: yinlb
"""

import copy
import typing

import numpy as np
from scipy import stats

# Import load_config at module level for THRES initialization
from src import utils

# Visibility grade thresholds loaded from config
# Default: [10000., 2000., 1000., 500., 200., 50.] (meters)
# Grade 0: >=10000m (good), Grade 1-5: moderate to poor,
# Grade 6: <50m (dense fog)
THRES: typing.Tuple[float, ...] = tuple(
    utils.load_config()['visibility']['grade_thresholds']
)


class VisAcc:
    """
    Visibility forecast verification metrics calculator class.

    Auto-grade visibility from obs/fcst arrays and build confusion matrix,
    providing metrics (ME, MAE, RMSE, MRE, R, TS, ETS, HSS, BIAS, FAR, etc).

    Attributes:
        _thres: Grade thresholds tuple.
        _n_grades: Number of grades.
        _ob: Observation array.
        _pr: Forecast array.
        _ob_grade: Observation grade array.
        _pr_grade: Forecast grade array.
        _conf_mat: Confusion matrix (n_grades+1)^2.
        _n: Total valid samples.
    """

    def __init__(self, ob: np.ndarray, pr: np.ndarray) -> None:
        """
        Init verification object.

        Args:
            ob (np.ndarray): Obs visibility array, np.nan for missing.
            pr (np.ndarray): Forecast visibility array, np.nan for missing.
        """
        self._thres = THRES
        self._n_grades = len(self._thres)
        self._ob = ob
        self._pr = pr

        # Init obs grade: -1=missing, 0=above max, 1-6 by thresholds
        # Note: Lower grade number = better visibility (Grade 0 is best)
        self._ob_grade = np.zeros_like(self._ob, dtype=np.int_) - 1
        self._ob_grade[~np.isnan(self._ob)] = 0
        for i in range(self._n_grades):
            # Assign grade i+1 if visibility below threshold i
            # This creates an ordered grade system: lower vis = higher grade
            self._ob_grade[self._ob < self._thres[i]] = i + 1

        # Init forecast grade matrix, same logic
        self._pr_grade = np.zeros_like(self._pr, dtype=np.int_) - 1
        self._pr_grade[~np.isnan(self._pr)] = 0
        for i in range(self._n_grades):
            self._pr_grade[self._pr < self._thres[i]] = i + 1

        # Build (n_grades+1)^2 confusion matrix using vectorized bincount
        n_classes = self._n_grades + 1
        valid = (self._ob_grade >= 0) & (self._pr_grade >= 0)
        idx = self._ob_grade[valid] * n_classes + self._pr_grade[valid]
        counts = np.bincount(idx, minlength=n_classes * n_classes)
        self._conf_mat = counts.reshape(n_classes, n_classes).astype(np.int_)

        # Total valid samples
        self._n = np.sum(self._conf_mat)

    def copy(self) -> 'VisAcc':
        """
        Return deep copy of object.

        Returns:
            VisAcc: Deep copy of this instance.
        """
        return copy.deepcopy(self)

    def get_me(self) -> float:
        """
        Calc Mean Error (ME).

        Returns:
            float: Mean error value.
        """
        # 1. Filter valid data (exclude NaN)
        index = (~np.isnan(self._ob)) & (~np.isnan(self._pr))
        ob = self._ob[index]
        pr = self._pr[index]
        # 2. Handle empty case
        if len(ob) == 0:
            return np.nan
        # 3. Calc mean difference (forecast - obs)
        return float(np.mean(pr - ob))

    def get_mae(self) -> float:
        """
        Calc Mean Absolute Error (MAE).

        Returns:
            float: Mean absolute error value.
        """
        # 1. Filter valid data
        index = (~np.isnan(self._ob)) & (~np.isnan(self._pr))
        ob = self._ob[index]
        pr = self._pr[index]
        # 2. Handle empty case
        if len(ob) == 0:
            return np.nan
        # 3. Calc mean absolute difference
        return float(np.mean(np.abs(pr - ob)))

    def get_rmse(self) -> float:
        """
        Calc Root Mean Square Error (RMSE).

        Returns:
            float: Root mean square error value.
        """
        # 1. Filter valid data
        index = (~np.isnan(self._ob)) & (~np.isnan(self._pr))
        ob = self._ob[index]
        pr = self._pr[index]
        # 2. Handle empty case
        if len(ob) == 0:
            return np.nan
        # 3. Calc root mean square difference
        return float(np.mean((pr - ob) ** 2) ** 0.5)

    def get_mre(self) -> float:
        """
        Calc Mean Relative Error (MRE).

        Returns:
            float: Mean relative error value.
        """
        # 1. Strict filter: exclude nan and small denominator (avoid div zero)
        valid = (~np.isnan(self._ob)) & (~np.isnan(self._pr))
        index = valid & (self._ob + self._pr > 1e-6)
        ob = self._ob[index]
        pr = self._pr[index]
        # 2. Handle empty case
        if len(ob) == 0:
            return np.nan
        # 3. Calc mean relative error using |pr-ob|/(pr+ob)
        return float(np.mean(np.abs((pr - ob) / (pr + ob))))

    def get_nme(self) -> float:
        """
        Calc Normalized ME.

        Normalizes by the obs range (max - min) to make metric comparable
        across different visibility regimes.

        Returns:
            float: Normalized mean error value.
        """
        index = (~np.isnan(self._ob)) & (~np.isnan(self._pr))
        ob = self._ob[index]
        pr = self._pr[index]
        if len(ob) == 0:
            return np.nan
        denom = np.max(ob) - np.min(ob)
        if denom < 1e-6:
            return np.nan  # Avoid div zero
        return float(np.mean(pr - ob) / denom)

    def get_nmae(self) -> float:
        """
        Calc Normalized MAE.

        Normalizes by the obs range (max - min) for cross-regime comparability.

        Returns:
            float: Normalized mean absolute error value.
        """
        index = (~np.isnan(self._ob)) & (~np.isnan(self._pr))
        ob = self._ob[index]
        pr = self._pr[index]
        if len(ob) == 0:
            return np.nan
        denom = np.max(ob) - np.min(ob)
        if denom < 1e-6:
            return np.nan  # Avoid div zero
        return float(np.mean(np.abs(pr - ob)) / denom)

    def get_nrmse(self) -> float:
        """
        Calc Normalized RMSE.

        Normalizes by the obs range (max - min) for cross-regime comparability.

        Returns:
            float: Normalized root mean square error value.
        """
        index = (~np.isnan(self._ob)) & (~np.isnan(self._pr))
        ob = self._ob[index]
        pr = self._pr[index]
        if len(ob) == 0:
            return np.nan
        denom = np.max(ob) - np.min(ob)
        if denom < 1e-6:
            return np.nan  # Avoid div zero
        return float(np.mean((pr - ob) ** 2) ** 0.5 / denom)

    def get_r(self) -> float:
        """
        Calc Pearson correlation coefficient (R).

        Returns:
            float: Pearson correlation coefficient.
        """
        index = (~np.isnan(self._ob)) & (~np.isnan(self._pr))
        ob = self._ob[index]
        pr = self._pr[index]
        if np.sum(index) > 2:
            corr = float(stats.pearsonr(ob, pr)[0])
        else:
            corr = np.nan
        return corr

    def get_me_grade(self) -> np.ndarray:
        """
        Calc ME by obs grade, return array of length n_grades+1.

        Iterates over each observed grade (0 to n_grades) and computes
        the mean error for samples where obs grade equals that grade.

        Returns:
            np.ndarray: ME array by obs grade.
        """
        me = np.zeros(self._n_grades + 1, dtype=np.float32) + np.nan
        for i in range(self._n_grades + 1):
            valid = (~np.isnan(self._ob)) & (~np.isnan(self._pr))
            index = valid & (self._ob_grade == i)
            ob = self._ob[index]
            pr = self._pr[index]
            if len(ob) > 0:
                me[i] = np.mean(pr - ob)
        return me

    def get_mae_grade(self) -> np.ndarray:
        """
        Calc MAE by obs grade, return array of length n_grades+1.

        Same per-grade filtering pattern as get_me_grade.

        Returns:
            np.ndarray: MAE array by obs grade.
        """
        mae = np.zeros(self._n_grades + 1, dtype=np.float32) + np.nan
        for i in range(self._n_grades + 1):
            valid = (~np.isnan(self._ob)) & (~np.isnan(self._pr))
            index = valid & (self._ob_grade == i)
            ob = self._ob[index]
            pr = self._pr[index]
            if len(ob) > 0:
                mae[i] = np.mean(np.abs(pr - ob))
        return mae

    def get_rmse_grade(self) -> np.ndarray:
        """
        Calc RMSE by obs grade, return array of length n_grades+1.

        Same per-grade filtering pattern as get_me_grade.

        Returns:
            np.ndarray: RMSE array by obs grade.
        """
        rmse = np.zeros(self._n_grades + 1, dtype=np.float32) + np.nan
        for i in range(self._n_grades + 1):
            valid = (~np.isnan(self._ob)) & (~np.isnan(self._pr))
            index = valid & (self._ob_grade == i)
            ob = self._ob[index]
            pr = self._pr[index]
            if len(ob) > 0:
                rmse[i] = np.mean((pr - ob) ** 2) ** 0.5
        return rmse

    def get_mre_grade(self) -> np.ndarray:
        """
        Calc MRE by obs grade, return array of length n_grades+1.

        Same per-grade filtering pattern as get_me_grade,
        with additional denominator guard (ob + pr > 1e-6).

        Returns:
            np.ndarray: MRE array by obs grade.
        """
        mre = np.zeros(self._n_grades + 1, dtype=np.float32) + np.nan
        for i in range(self._n_grades + 1):
            index = (
                (~np.isnan(self._ob))
                & (~np.isnan(self._pr))
                & (self._ob + self._pr > 1e-6)  # Strict filter, avoid div zero
                & (self._ob_grade == i)
            )
            ob = self._ob[index]
            pr = self._pr[index]
            if len(ob) > 0:
                mre[i] = np.mean(np.abs((pr - ob) / (pr + ob)))
        return mre

    def get_r_grade(self) -> np.ndarray:
        """
        Calc Pearson R by obs grade, return array of length n_grades+1.

        Same per-grade filtering pattern as get_me_grade.
        Requires at least 3 samples for pearsonr.

        Returns:
            np.ndarray: R array by obs grade.
        """
        corr_arr = np.zeros(self._n_grades + 1, dtype=np.float32) + np.nan
        for i in range(self._n_grades + 1):
            index = (
                (~np.isnan(self._ob))
                & (~np.isnan(self._pr))
                & (self._ob_grade == i)
            )
            ob = self._ob[index]
            pr = self._pr[index]
            # pearsonr needs at least 2 samples
            if len(ob) > 2:
                r[i] = stats.pearsonr(ob, pr)[0]
        return r

    def get_conf_mat(self) -> np.ndarray:
        """
        Return confusion matrix (int counts).

        Returns:
            np.ndarray: Confusion matrix array (7x7, grade 0-6).
        """
        return self._conf_mat

    def get_conf_mat_norm(self) -> np.ndarray:
        """
        Return normalized confusion matrix, each row divided by obs total.

        Returns:
            np.ndarray: Normalized confusion matrix array (row sum = 1).
        """
        # conf_mat_norm: Normalized confusion matrix
        conf_mat_norm = np.zeros_like(self._conf_mat, dtype=np.float32) + np.nan
        for i in range(self._n_grades + 1):
            row_sum = np.sum(self._conf_mat[i, :])
            if row_sum > 0:
                for j in range(self._n_grades + 1):
                    conf_mat_norm[i, j] = self._conf_mat[i, j] / row_sum
        return conf_mat_norm

    def get_oa(self) -> float:
        """
        Calc Overall Accuracy (OA), ratio of diagonal sum to total samples.

        OA = (correct forecasts) / (total samples)
        Correct means forecast grade exactly matches observed grade.

        Returns:
            float: Overall accuracy value, or NaN if no valid samples.
        """
        # Guard against empty data (no valid obs/forecast pairs)
        if self._n == 0:
            return np.nan
        # Diagonal elements represent correct grade forecasts
        return np.sum(np.diag(self._conf_mat)) / self._n

    def get_kappa(self) -> float:
        """
        Calc Kappa coefficient, measuring grade consistency.

        Kappa accounts for chance agreement, more robust than OA alone.
        Formula: (OA - Pe) / (1 - Pe), where Pe is expected agreement
        by chance.

        Returns:
            float: Kappa coefficient value, or NaN if undefined.
        """
        # Guard against empty data
        if self._n == 0:
            return np.nan
        # Calculate marginal frequencies (row and column sums)
        row_sum = np.sum(self._conf_mat, axis=1).astype(np.float32)
        col_sum = np.sum(self._conf_mat, axis=0).astype(np.float32)
        # Expected agreement by chance (Pe)
        pe = np.sum(row_sum * col_sum) / self._n / self._n
        # Guard against degenerate case (all forecasts same)
        if abs(1 - pe) < 1e-6:
            return np.nan
        return (self.get_oa() - pe) / (1 - pe)

    def get_ts(self) -> np.ndarray:
        """
        Calc Threat Score (TS) for each grade.

        TS = hits / (hits + misses + false alarms)
        Also known as Critical Success Index (CSI).

        Returns:
            np.ndarray: TS array by grade.
        """
        ts = np.zeros(self._n_grades + 1, dtype=np.float32)
        for i in range(self._n_grades + 1):
            # na: correct forecasts for grade i (hits)
            na = self._conf_mat[i, i]
            # nb: false alarms (forecast grade i, obs not i)
            nb = np.sum(self._conf_mat[:, i]) - na
            # nc: misses (obs grade i, forecast not i)
            nc = np.sum(self._conf_mat[i, :]) - na
            # Avoid division by zero
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ts_ge(self) -> np.ndarray:
        """
        Calc merged-grade Threat Score (TS), return array of length n_grades.

        For i-th threshold, grade i+1+ as positive, else negative,
        build 2x2 table: na = obs+ fcst+, nb = obs- fcst+, nc = obs+ fcst-.

        Returns:
            np.ndarray: Merged-grade TS array.
        """
        ts = np.zeros(self._n_grades, dtype=np.float32)
        for i in range(self._n_grades):
            na = np.sum(self._conf_mat[i + 1:, i + 1:])
            nb = np.sum(self._conf_mat[:i + 1, i + 1:])
            nc = np.sum(self._conf_mat[i + 1:, :i + 1])
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ets_ge(self) -> np.ndarray:
        """
        Calc merged-grade Equitable Threat Score (ETS).

        Returns:
            np.ndarray: Merged-grade ETS array.
        """
        ets = np.zeros(self._n_grades, dtype=np.float32)
        for i in range(self._n_grades):
            na = np.sum(self._conf_mat[i + 1:, i + 1:])
            nb = np.sum(self._conf_mat[:i + 1, i + 1:])
            nc = np.sum(self._conf_mat[i + 1:, :i + 1])
            nd = np.sum(self._conf_mat[:i + 1, :i + 1])
            exp_agree = (na + nb) / (na + nb + nc + nd) * (na + nc)
            denom = na + nb + nc - exp_agree
            ets[i] = (na - r) / denom if na + nb + nc != 0 else np.nan
        return ets

    def get_hss_ge(self) -> np.ndarray:
        """
        Calc merged-grade Heidke Skill Score (HSS).

        Returns:
            np.ndarray: Merged-grade HSS array.
        """
        hss = np.zeros(self._n_grades, dtype=np.float32) + np.nan
        for i in range(self._n_grades):
            na = np.sum(self._conf_mat[i + 1:, i + 1:])
            nb = np.sum(self._conf_mat[:i + 1, i + 1:])
            nc = np.sum(self._conf_mat[i + 1:, :i + 1])
            nd = np.sum(self._conf_mat[:i + 1, :i + 1])
            denom = (na + nc) * (nc + nd) + (na + nb) * (nb + nd)
            if abs(denom) > 1e-6:
                hss[i] = 2 * (na * nd - nb * nc) / denom
        return hss

    def get_tss_ge(self) -> np.ndarray:
        """
        Calc merged-grade True Skill Statistic (TSS, Pierce's Skill Score).

        Returns:
            np.ndarray: Merged-grade TSS array.
        """
        tss = np.zeros(self._n_grades, dtype=np.float32) + np.nan
        for i in range(self._n_grades):
            na = np.sum(self._conf_mat[i + 1:, i + 1:])
            nb = np.sum(self._conf_mat[:i + 1, i + 1:])
            nc = np.sum(self._conf_mat[i + 1:, :i + 1])
            nd = np.sum(self._conf_mat[:i + 1, :i + 1])
            denom1 = na + nc
            denom2 = nb + nd
            if abs(denom1) > 1e-6 and abs(denom2) > 1e-6:
                tss[i] = (na * nd - nb * nc) / denom1 / denom2
        return tss

    def get_bias_ge(self) -> np.ndarray:
        """
        Calc merged-grade Frequency BIAS.

        Returns:
            np.ndarray: Merged-grade BIAS array.
        """
        bias = np.zeros(self._n_grades, dtype=np.float32) + np.nan
        for i in range(self._n_grades):
            na = np.sum(self._conf_mat[i + 1:, i + 1:])
            nb = np.sum(self._conf_mat[:i + 1, i + 1:])
            nc = np.sum(self._conf_mat[i + 1:, :i + 1])
            denom = na + nc
            if abs(denom) > 1e-6:
                bias[i] = (na + nb) / denom
        return bias

    def get_far_ge(self) -> np.ndarray:
        """
        Calc merged-grade False Alarm Ratio (FAR).

        Returns:
            np.ndarray: Merged-grade FAR array.
        """
        far = np.zeros(self._n_grades, dtype=np.float32) + np.nan
        for i in range(self._n_grades):
            na = np.sum(self._conf_mat[i + 1:, i + 1:])
            nb = np.sum(self._conf_mat[:i + 1, i + 1:])
            denom = na + nb
            if abs(denom) > 1e-6:
                far[i] = nb / denom
        return far

    def get_mar_ge(self) -> np.ndarray:
        """
        Calc merged-grade Miss Alarm Ratio (MAR).

        Returns:
            np.ndarray: Merged-grade MAR array.
        """
        mar = np.zeros(self._n_grades, dtype=np.float32) + np.nan
        for i in range(self._n_grades):
            na = np.sum(self._conf_mat[i + 1:, i + 1:])
            nc = np.sum(self._conf_mat[i + 1:, :i + 1])
            denom = na + nc
            if abs(denom) > 1e-6:
                mar[i] = nc / denom
        return mar

    def get_pod_ge(self) -> np.ndarray:
        """
        Calc merged-grade Probability of Detection (POD).

        Returns:
            np.ndarray: Merged-grade POD array.
        """
        pod = np.zeros(self._n_grades, dtype=np.float32) + np.nan
        for i in range(self._n_grades):
            na = np.sum(self._conf_mat[i + 1:, i + 1:])
            nc = np.sum(self._conf_mat[i + 1:, :i + 1])
            denom = na + nc
            if abs(denom) > 1e-6:
                pod[i] = na / denom
        return pod
