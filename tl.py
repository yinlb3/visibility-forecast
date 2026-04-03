#!user/bin.python3

"""
Founded in 2024-07-10
Modified in 2024-07-24
@author: yinlb
"""
import os
import sys
import typing

import arrow
import numpy as np


THRES = (10000., 2000., 1000., 500., 200., 50.)


class VisAcc:
    def __init__(self, ob: np.ndarray, pr: np.ndarray):
        self.thres = THRES
        self.n_grades = len(self.thres)
        self.ob = ob
        self.pr = pr
        self.ob_grade = np.zeros_like(self.ob, dtype=np.int_) - 1
        self.ob_grade[~np.isnan(self.ob)] = 0
        for i in range(self.n_grades):
            self.ob_grade[self.ob < self.thres[i]] = i + 1
        self.pr_grade = np.zeros_like(self.pr, dtype=np.int_) - 1
        self.pr_grade[~np.isnan(self.pr)] = 0
        for i in range(self.n_grades):
            self.pr_grade[self.pr < self.thres[i]] = i + 1
        self.hxjz = np.zeros((self.n_grades + 1, self.n_grades + 1), dtype=np.int_)
        for i in range(self.n_grades + 1):
            for j in range(self.n_grades + 1):
                self.hxjz[i, j] = np.sum((self.ob_grade == i) & (self.pr_grade == j))
        self.n = np.sum(self.hxjz)

    def get_me(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean(pr - ob))

    def get_mae(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean(np.abs(pr - ob)))

    def get_rmse(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean((pr - ob) ** 2) ** 0.5)

    def get_mre(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean(np.abs((pr - ob) / (pr + ob))))

    def get_r(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.corrcoef(ob, pr)[0, 1])

    def get_hxjz(self) -> np.ndarray:
        return self.hxjz

    def get_oa(self) -> float:
        return np.sum(np.diag(self.hxjz)) / self.n

    def get_kappa(self) -> float:
        a = np.sum(self.hxjz, axis=1)
        a = a.astype(np.float32)
        b = np.sum(self.hxjz, axis=0)
        b = b.astype(np.float32)
        pe = np.sum(a * b) / self.n / self.n
        return (self.get_oa() - pe) / (1 - pe)

    def get_ts(self) -> np.ndarray:
        ts = np.zeros(self.n_grades + 1, dtype=np.float32)
        for i in range(self.n_grades + 1):
            na = self.hxjz[i, i]
            nb = np.sum(self.hxjz[:, i]) - na
            nc = np.sum(self.hxjz[i, :]) - na
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ts2(self) -> np.ndarray:
        ts = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(self.hxjz[i + 1:, i + 1:])
            nb = np.sum(self.hxjz[:i + 1, i + 1:])
            nc = np.sum(self.hxjz[i + 1:, :i + 1])
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ets2(self) -> np.ndarray:
        ets = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(self.hxjz[i + 1:, i + 1:])
            nb = np.sum(self.hxjz[:i + 1, i + 1:])
            nc = np.sum(self.hxjz[i + 1:, :i + 1])
            nd = np.sum(self.hxjz[:i + 1, :i + 1])
            r = (na + nb) / (na + nb + nc + nd) * (na + nc)
            ets[i] = (na - r) / (na + nb + nc - r) if na + nb + nc != 0 else np.nan
        return ets

    def get_bias2(self) -> np.ndarray:
        bias = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(self.hxjz[i + 1:, i + 1:])
            nb = np.sum(self.hxjz[:i + 1, i + 1:])
            nc = np.sum(self.hxjz[i + 1:, :i + 1])
            bias[i] = (na + nb) / (na + nc)
        return bias

    def get_far2(self) -> np.ndarray:
        far = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(self.hxjz[i + 1:, i + 1:])
            nb = np.sum(self.hxjz[:i + 1, i + 1:])
            far[i] = nb / (na + nb)
        return far

    def get_mar2(self) -> np.ndarray:
        mar = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(self.hxjz[i + 1:, i + 1:])
            nc = np.sum(self.hxjz[i + 1:, :i + 1])
            mar[i] = nc / (na + nc)
        return mar

    def get_thres_pdf(self) -> typing.List[float]:
        thres = list()
        pr = list(self.pr[~np.isnan(self.pr)])
        pr.sort()
        for i in range(self.n_grades):
            y0 = np.sum(self.ob < self.thres[i]) / np.sum(~np.isnan(self.ob))
            j = round(y0 * (len(pr) - 1))
            thres.append(float(pr[j]))
        return thres

    # def get_pdf_model(self) -> PDF:
    #     model = PDF()
    #     model.fit(self.ob, self.pr)
    #     return model


def format_time(second: float, is_abbreviation: bool = False) -> str:
    r"""Format time.

    :param second: A float number representing the number of seconds.
    :param is_abbreviation: A boolean variable representing whether processing to abbreviation.
        The default value is False.
    :return: A sequence of strings representing the time. For example: '43.5 seconds'
    :raise ValueError: The value of input parameter 'second' is wrong.
    """
    if second < 0:
        raise ValueError('The input parameter \'second\' cannot be negative.')
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
    ob = np.load(r'D:\data\vis\vis1183_ob.npy')[:, :, 1:, :]
    ob[ob >= 999990] = np.nan
    pr = np.load(r'D:\data\vis\vis1183_pr.npy')[:, :, 1:, :]
    index = np.zeros(1183, dtype=np.bool_)
    for i in range(1183):
        if np.sum(~np.isnan(ob[:, :, :, i]) & ~np.isnan(pr[:, :, :, i])) > 0:
            index[i] = True
    index_sta08 = np.load(r'D:\Project\vis\index_sta08.npy')
    print(np.sum(index_sta08))
    val_ob = ob[-365:, :, :, index_sta08]
    val_ob = np.reshape(val_ob, (-1, 24, 1085))
    val_pr = pr[-365:, :, :, index_sta08]
    val_pr = np.reshape(val_pr, (-1, 24, 1085))
    del ob, pr
    print('****** nwp ******')
    acc = VisAcc(val_ob, val_pr)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())

    pred_pr_tle0 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    pred_pr_tle1 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    pred_pr_tle2 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    pred_pr_tle3 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    pred_pr_tle4 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    # pred_pr_tle5 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    for i in range(48, 6960):
        w0 = np.zeros(24) + np.nan
        w1 = np.zeros(24) + np.nan
        w2 = np.zeros(24) + np.nan
        w3 = np.zeros(24) + np.nan
        w4 = np.zeros(24) + np.nan
        # w5 = np.zeros(24) + np.nan
        for j in range(24):
            val_pr_ = val_pr[i - 48: i - 24, j, :]
            val_ob_ = val_ob[i - 48: i - 24, j, :]
            w0[j] = 1
            w1[j] = np.corrcoef(val_pr_, val_ob_)[0, 1]
            w2[j] = 1 + 1 / np.mean(np.abs(val_pr_ - val_ob_))
            w3[j] = 1 + 1 / (np.mean((val_pr_ - val_ob_) ** 2) ** 0.5)
            index = val_pr_ + val_ob_ != 0
            w4[j] = 1 + 1 / np.mean(np.abs(val_pr_ - val_ob_)[index] / (val_pr_ + val_ob_)[index])
            # w5[j] = 1 - np.mean(np.abs(val_pr_ - val_ob_)[index] / (val_pr_ + val_ob_)[index])
        for j in range(24):
            vis0 = 0
            vis1 = 0
            vis2 = 0
            vis3 = 0
            vis4 = 0
    #         vis5 = 0
            for k in range(j, 24):
                vis0 += w0[k] * val_pr[i + j - k, k, :]
                vis1 += w1[k] * val_pr[i + j - k, k, :]
                vis2 += w2[k] * val_pr[i + j - k, k, :]
                vis3 += w3[k] * val_pr[i + j - k, k, :]
                vis4 += w4[k] * val_pr[i + j - k, k, :]
    #             vis5 += w5[k] * val_pr[i + j - k, k, :]
            pred_pr_tle0[i, j, :] = vis0 / np.sum(w0[j:])
            pred_pr_tle1[i, j, :] = vis1 / np.sum(w1[j:])
            pred_pr_tle2[i, j, :] = vis2 / np.sum(w2[j:])
            pred_pr_tle3[i, j, :] = vis3 / np.sum(w3[j:])
            pred_pr_tle4[i, j, :] = vis4 / np.sum(w4[j:])
    #         pred_pr_tle5[i, j, :] = vis5 / np.sum(w5[j:])
    np.save(r'D:\data\vis\vis1085_shr_pred_tle0.npy', pred_pr_tle0)
    np.save(r'D:\data\vis\vis1085_shr_pred_tle1.npy', pred_pr_tle1)
    np.save(r'D:\data\vis\vis1085_shr_pred_tle2.npy', pred_pr_tle2)
    np.save(r'D:\data\vis\vis1085_shr_pred_tle3.npy', pred_pr_tle3)
    np.save(r'D:\data\vis\vis1085_shr_pred_tle4.npy', pred_pr_tle4)
    # np.save(r'D:\data\vis\vis1085_shr_pred_tle5.npy', pred_pr_tle5)
    pred_pr_tle0 = np.load(r'D:\data\vis\vis1085_shr_pred_tle0.npy')
    pred_pr_tle1 = np.load(r'D:\data\vis\vis1085_shr_pred_tle1.npy')
    pred_pr_tle2 = np.load(r'D:\data\vis\vis1085_shr_pred_tle2.npy')
    pred_pr_tle3 = np.load(r'D:\data\vis\vis1085_shr_pred_tle3.npy')
    pred_pr_tle4 = np.load(r'D:\data\vis\vis1085_shr_pred_tle4.npy')
    pred_pr_tle5 = np.load(r'D:\data\vis\vis1085_shr_pred_tle5.npy')
    print('****** tle0 ******')
    pred_pr_tle0[pred_pr_tle0 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_tle0)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** tle1 ******')
    pred_pr_tle1[pred_pr_tle1 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_tle1)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** tle2 ******')
    pred_pr_tle2[pred_pr_tle2 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_tle2)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** tle3 ******')
    pred_pr_tle3[pred_pr_tle3 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_tle3)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** tle4 ******')
    pred_pr_tle4[pred_pr_tle4 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_tle4)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** tle5 ******')
    pred_pr_tle5[pred_pr_tle5 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_tle5)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())

    pred_pdfm = np.load(r'D:\data\vis\vis1085_shr_pred2.npy')
    pred_pdfm = np.reshape(pred_pdfm, (-1, 24, 1085))
    print('****** pdfm ******')
    acc = VisAcc(val_ob, pred_pdfm)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())

    pred_pr_pdfm_tle0 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    pred_pr_pdfm_tle1 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    pred_pr_pdfm_tle2 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    pred_pr_pdfm_tle3 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    pred_pr_pdfm_tle4 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    # pred_pr_pdfm_tle5 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    for i in range(48, 6960):
        w0 = np.zeros(24) + np.nan
        w1 = np.zeros(24) + np.nan
        w2 = np.zeros(24) + np.nan
        w3 = np.zeros(24) + np.nan
        w4 = np.zeros(24) + np.nan
    #     w5 = np.zeros(24) + np.nan
        for j in range(24):
            val_pr_ = val_pr[i - 48: i - 24, j, :]
            val_ob_ = val_ob[i - 48: i - 24, j, :]
            w0[j] = 1
            w1[j] = np.corrcoef(val_pr_, val_ob_)[0, 1]
            w2[j] = 1 + 1 / np.mean(np.abs(val_pr_ - val_ob_))
            w3[j] = 1 + 1 / (np.mean((val_pr_ - val_ob_) ** 2) ** 0.5)
            index = val_pr_ + val_ob_ != 0
            w4[j] = 1 + 1 / np.mean(np.abs(val_pr_ - val_ob_)[index] / (val_pr_ + val_ob_)[index])
    #         w5[j] = 1 - np.mean(np.abs(val_pr_ - val_ob_)[index] / (val_pr_ + val_ob_)[index])
        for j in range(24):
            vis0 = 0
            vis1 = 0
            vis2 = 0
            vis3 = 0
            vis4 = 0
    #         vis5 = 0
            for k in range(j, 24):
                vis0 += w0[k] * pred_pdfm[i + j - k, k, :]
                vis1 += w1[k] * pred_pdfm[i + j - k, k, :]
                vis2 += w2[k] * pred_pdfm[i + j - k, k, :]
                vis3 += w3[k] * pred_pdfm[i + j - k, k, :]
                vis4 += w4[k] * pred_pdfm[i + j - k, k, :]
    #             vis5 += w5[k] * pred_pdfm[i + j - k, k, :]
            pred_pr_pdfm_tle0[i, j, :] = vis0 / np.sum(w0[j:])
            pred_pr_pdfm_tle1[i, j, :] = vis1 / np.sum(w1[j:])
            pred_pr_pdfm_tle2[i, j, :] = vis2 / np.sum(w2[j:])
            pred_pr_pdfm_tle3[i, j, :] = vis3 / np.sum(w3[j:])
            pred_pr_pdfm_tle4[i, j, :] = vis4 / np.sum(w4[j:])
    #         pred_pr_pdfm_tle5[i, j, :] = vis5 / np.sum(w5[j:])
    np.save(r'D:\data\vis\vis1085_shr_pred_pdfm_tle0.npy', pred_pr_pdfm_tle0)
    np.save(r'D:\data\vis\vis1085_shr_pred_pdfm_tle1.npy', pred_pr_pdfm_tle1)
    np.save(r'D:\data\vis\vis1085_shr_pred_pdfm_tle2.npy', pred_pr_pdfm_tle2)
    np.save(r'D:\data\vis\vis1085_shr_pred_pdfm_tle3.npy', pred_pr_pdfm_tle3)
    np.save(r'D:\data\vis\vis1085_shr_pred_pdfm_tle4.npy', pred_pr_pdfm_tle4)
    # np.save(r'D:\data\vis\vis1085_shr_pred_pdfm_tle5.npy', pred_pr_pdfm_tle5)
    pred_pr_pdfm_tle0 = np.load(r'D:\data\vis\vis1085_shr_pred_pdfm_tle0.npy')
    pred_pr_pdfm_tle1 = np.load(r'D:\data\vis\vis1085_shr_pred_pdfm_tle1.npy')
    pred_pr_pdfm_tle2 = np.load(r'D:\data\vis\vis1085_shr_pred_pdfm_tle2.npy')
    pred_pr_pdfm_tle3 = np.load(r'D:\data\vis\vis1085_shr_pred_pdfm_tle3.npy')
    pred_pr_pdfm_tle4 = np.load(r'D:\data\vis\vis1085_shr_pred_pdfm_tle4.npy')
    pred_pr_pdfm_tle5 = np.load(r'D:\data\vis\vis1085_shr_pred_pdfm_tle5.npy')
    print('****** pdfm-tle0 ******')
    pred_pr_pdfm_tle0[pred_pr_pdfm_tle0 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_pdfm_tle0)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** pdfm-tle1 ******')
    pred_pr_pdfm_tle1[pred_pr_pdfm_tle1 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_pdfm_tle1)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** pdfm-tle2 ******')
    pred_pr_pdfm_tle2[pred_pr_pdfm_tle2 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_pdfm_tle2)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** pdfm-tle3 ******')
    pred_pr_pdfm_tle3[pred_pr_pdfm_tle3 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_pdfm_tle3)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** pdfm-tle4 ******')
    pred_pr_pdfm_tle4[pred_pr_pdfm_tle4 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_pdfm_tle4)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** pdfm-tle5 ******')
    pred_pr_pdfm_tle5[pred_pr_pdfm_tle5 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_pdfm_tle5)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())#!user/bin.python3

"""
Founded in 2024-07-10
Modified in 2024-07-24
@author: yinlb
"""
import os
import sys
import typing

import arrow
import numpy as np


THRES = (10000., 2000., 1000., 500., 200., 50.)


class VisAcc:
    def __init__(self, ob: np.ndarray, pr: np.ndarray):
        self.thres = THRES
        self.n_grades = len(self.thres)
        self.ob = ob
        self.pr = pr
        self.ob_grade = np.zeros_like(self.ob, dtype=np.int_) - 1
        self.ob_grade[~np.isnan(self.ob)] = 0
        for i in range(self.n_grades):
            self.ob_grade[self.ob < self.thres[i]] = i + 1
        self.pr_grade = np.zeros_like(self.pr, dtype=np.int_) - 1
        self.pr_grade[~np.isnan(self.pr)] = 0
        for i in range(self.n_grades):
            self.pr_grade[self.pr < self.thres[i]] = i + 1
        self.hxjz = np.zeros((self.n_grades + 1, self.n_grades + 1), dtype=np.int_)
        for i in range(self.n_grades + 1):
            for j in range(self.n_grades + 1):
                self.hxjz[i, j] = np.sum((self.ob_grade == i) & (self.pr_grade == j))
        self.n = np.sum(self.hxjz)

    def get_me(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean(pr - ob))

    def get_mae(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean(np.abs(pr - ob)))

    def get_rmse(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean((pr - ob) ** 2) ** 0.5)

    def get_mre(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean(np.abs((pr - ob) / (pr + ob))))

    def get_r(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.corrcoef(ob, pr)[0, 1])

    def get_hxjz(self) -> np.ndarray:
        return self.hxjz

    def get_oa(self) -> float:
        return np.sum(np.diag(self.hxjz)) / self.n

    def get_kappa(self) -> float:
        a = np.sum(self.hxjz, axis=1)
        a = a.astype(np.float32)
        b = np.sum(self.hxjz, axis=0)
        b = b.astype(np.float32)
        pe = np.sum(a * b) / self.n / self.n
        return (self.get_oa() - pe) / (1 - pe)

    def get_ts(self) -> np.ndarray:
        ts = np.zeros(self.n_grades + 1, dtype=np.float32)
        for i in range(self.n_grades + 1):
            na = self.hxjz[i, i]
            nb = np.sum(self.hxjz[:, i]) - na
            nc = np.sum(self.hxjz[i, :]) - na
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ts2(self) -> np.ndarray:
        ts = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(self.hxjz[i + 1:, i + 1:])
            nb = np.sum(self.hxjz[:i + 1, i + 1:])
            nc = np.sum(self.hxjz[i + 1:, :i + 1])
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ets2(self) -> np.ndarray:
        ets = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(self.hxjz[i + 1:, i + 1:])
            nb = np.sum(self.hxjz[:i + 1, i + 1:])
            nc = np.sum(self.hxjz[i + 1:, :i + 1])
            nd = np.sum(self.hxjz[:i + 1, :i + 1])
            r = (na + nb) / (na + nb + nc + nd) * (na + nc)
            ets[i] = (na - r) / (na + nb + nc - r) if na + nb + nc != 0 else np.nan
        return ets

    def get_bias2(self) -> np.ndarray:
        bias = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(self.hxjz[i + 1:, i + 1:])
            nb = np.sum(self.hxjz[:i + 1, i + 1:])
            nc = np.sum(self.hxjz[i + 1:, :i + 1])
            bias[i] = (na + nb) / (na + nc)
        return bias

    def get_far2(self) -> np.ndarray:
        far = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(self.hxjz[i + 1:, i + 1:])
            nb = np.sum(self.hxjz[:i + 1, i + 1:])
            far[i] = nb / (na + nb)
        return far

    def get_mar2(self) -> np.ndarray:
        mar = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(self.hxjz[i + 1:, i + 1:])
            nc = np.sum(self.hxjz[i + 1:, :i + 1])
            mar[i] = nc / (na + nc)
        return mar

    def get_thres_pdf(self) -> typing.List[float]:
        thres = list()
        pr = list(self.pr[~np.isnan(self.pr)])
        pr.sort()
        for i in range(self.n_grades):
            y0 = np.sum(self.ob < self.thres[i]) / np.sum(~np.isnan(self.ob))
            j = round(y0 * (len(pr) - 1))
            thres.append(float(pr[j]))
        return thres

    # def get_pdf_model(self) -> PDF:
    #     model = PDF()
    #     model.fit(self.ob, self.pr)
    #     return model


def format_time(second: float, is_abbreviation: bool = False) -> str:
    r"""Format time.

    :param second: A float number representing the number of seconds.
    :param is_abbreviation: A boolean variable representing whether processing to abbreviation.
        The default value is False.
    :return: A sequence of strings representing the time. For example: '43.5 seconds'
    :raise ValueError: The value of input parameter 'second' is wrong.
    """
    if second < 0:
        raise ValueError('The input parameter \'second\' cannot be negative.')
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
    ob = np.load(r'D:\data\vis\vis1183_ob.npy')[:, :, 1:, :]
    ob[ob >= 999990] = np.nan
    pr = np.load(r'D:\data\vis\vis1183_pr.npy')[:, :, 1:, :]
    index = np.zeros(1183, dtype=np.bool_)
    for i in range(1183):
        if np.sum(~np.isnan(ob[:, :, :, i]) & ~np.isnan(pr[:, :, :, i])) > 0:
            index[i] = True
    index_sta08 = np.load(r'D:\Project\vis\index_sta08.npy')
    print(np.sum(index_sta08))
    val_ob = ob[-365:, :, :, index_sta08]
    val_ob = np.reshape(val_ob, (-1, 24, 1085))
    val_pr = pr[-365:, :, :, index_sta08]
    val_pr = np.reshape(val_pr, (-1, 24, 1085))
    del ob, pr
    print('****** nwp ******')
    acc = VisAcc(val_ob, val_pr)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())

    pred_pr_tle0 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    pred_pr_tle1 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    pred_pr_tle2 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    pred_pr_tle3 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    pred_pr_tle4 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    # pred_pr_tle5 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    for i in range(48, 6960):
        w0 = np.zeros(24) + np.nan
        w1 = np.zeros(24) + np.nan
        w2 = np.zeros(24) + np.nan
        w3 = np.zeros(24) + np.nan
        w4 = np.zeros(24) + np.nan
        # w5 = np.zeros(24) + np.nan
        for j in range(24):
            val_pr_ = val_pr[i - 48: i - 24, j, :]
            val_ob_ = val_ob[i - 48: i - 24, j, :]
            w0[j] = 1
            w1[j] = np.corrcoef(val_pr_, val_ob_)[0, 1]
            w2[j] = 1 + 1 / np.mean(np.abs(val_pr_ - val_ob_))
            w3[j] = 1 + 1 / (np.mean((val_pr_ - val_ob_) ** 2) ** 0.5)
            index = val_pr_ + val_ob_ != 0
            w4[j] = 1 + 1 / np.mean(np.abs(val_pr_ - val_ob_)[index] / (val_pr_ + val_ob_)[index])
            # w5[j] = 1 - np.mean(np.abs(val_pr_ - val_ob_)[index] / (val_pr_ + val_ob_)[index])
        for j in range(24):
            vis0 = 0
            vis1 = 0
            vis2 = 0
            vis3 = 0
            vis4 = 0
    #         vis5 = 0
            for k in range(j, 24):
                vis0 += w0[k] * val_pr[i + j - k, k, :]
                vis1 += w1[k] * val_pr[i + j - k, k, :]
                vis2 += w2[k] * val_pr[i + j - k, k, :]
                vis3 += w3[k] * val_pr[i + j - k, k, :]
                vis4 += w4[k] * val_pr[i + j - k, k, :]
    #             vis5 += w5[k] * val_pr[i + j - k, k, :]
            pred_pr_tle0[i, j, :] = vis0 / np.sum(w0[j:])
            pred_pr_tle1[i, j, :] = vis1 / np.sum(w1[j:])
            pred_pr_tle2[i, j, :] = vis2 / np.sum(w2[j:])
            pred_pr_tle3[i, j, :] = vis3 / np.sum(w3[j:])
            pred_pr_tle4[i, j, :] = vis4 / np.sum(w4[j:])
    #         pred_pr_tle5[i, j, :] = vis5 / np.sum(w5[j:])
    np.save(r'D:\data\vis\vis1085_shr_pred_tle0.npy', pred_pr_tle0)
    np.save(r'D:\data\vis\vis1085_shr_pred_tle1.npy', pred_pr_tle1)
    np.save(r'D:\data\vis\vis1085_shr_pred_tle2.npy', pred_pr_tle2)
    np.save(r'D:\data\vis\vis1085_shr_pred_tle3.npy', pred_pr_tle3)
    np.save(r'D:\data\vis\vis1085_shr_pred_tle4.npy', pred_pr_tle4)
    # np.save(r'D:\data\vis\vis1085_shr_pred_tle5.npy', pred_pr_tle5)
    pred_pr_tle0 = np.load(r'D:\data\vis\vis1085_shr_pred_tle0.npy')
    pred_pr_tle1 = np.load(r'D:\data\vis\vis1085_shr_pred_tle1.npy')
    pred_pr_tle2 = np.load(r'D:\data\vis\vis1085_shr_pred_tle2.npy')
    pred_pr_tle3 = np.load(r'D:\data\vis\vis1085_shr_pred_tle3.npy')
    pred_pr_tle4 = np.load(r'D:\data\vis\vis1085_shr_pred_tle4.npy')
    pred_pr_tle5 = np.load(r'D:\data\vis\vis1085_shr_pred_tle5.npy')
    print('****** tle0 ******')
    pred_pr_tle0[pred_pr_tle0 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_tle0)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** tle1 ******')
    pred_pr_tle1[pred_pr_tle1 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_tle1)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** tle2 ******')
    pred_pr_tle2[pred_pr_tle2 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_tle2)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** tle3 ******')
    pred_pr_tle3[pred_pr_tle3 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_tle3)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** tle4 ******')
    pred_pr_tle4[pred_pr_tle4 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_tle4)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** tle5 ******')
    pred_pr_tle5[pred_pr_tle5 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_tle5)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())

    pred_pdfm = np.load(r'D:\data\vis\vis1085_shr_pred2.npy')
    pred_pdfm = np.reshape(pred_pdfm, (-1, 24, 1085))
    print('****** pdfm ******')
    acc = VisAcc(val_ob, pred_pdfm)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())

    pred_pr_pdfm_tle0 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    pred_pr_pdfm_tle1 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    pred_pr_pdfm_tle2 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    pred_pr_pdfm_tle3 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    pred_pr_pdfm_tle4 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    # pred_pr_pdfm_tle5 = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    for i in range(48, 6960):
        w0 = np.zeros(24) + np.nan
        w1 = np.zeros(24) + np.nan
        w2 = np.zeros(24) + np.nan
        w3 = np.zeros(24) + np.nan
        w4 = np.zeros(24) + np.nan
    #     w5 = np.zeros(24) + np.nan
        for j in range(24):
            val_pr_ = val_pr[i - 48: i - 24, j, :]
            val_ob_ = val_ob[i - 48: i - 24, j, :]
            w0[j] = 1
            w1[j] = np.corrcoef(val_pr_, val_ob_)[0, 1]
            w2[j] = 1 + 1 / np.mean(np.abs(val_pr_ - val_ob_))
            w3[j] = 1 + 1 / (np.mean((val_pr_ - val_ob_) ** 2) ** 0.5)
            index = val_pr_ + val_ob_ != 0
            w4[j] = 1 + 1 / np.mean(np.abs(val_pr_ - val_ob_)[index] / (val_pr_ + val_ob_)[index])
    #         w5[j] = 1 - np.mean(np.abs(val_pr_ - val_ob_)[index] / (val_pr_ + val_ob_)[index])
        for j in range(24):
            vis0 = 0
            vis1 = 0
            vis2 = 0
            vis3 = 0
            vis4 = 0
    #         vis5 = 0
            for k in range(j, 24):
                vis0 += w0[k] * pred_pdfm[i + j - k, k, :]
                vis1 += w1[k] * pred_pdfm[i + j - k, k, :]
                vis2 += w2[k] * pred_pdfm[i + j - k, k, :]
                vis3 += w3[k] * pred_pdfm[i + j - k, k, :]
                vis4 += w4[k] * pred_pdfm[i + j - k, k, :]
    #             vis5 += w5[k] * pred_pdfm[i + j - k, k, :]
            pred_pr_pdfm_tle0[i, j, :] = vis0 / np.sum(w0[j:])
            pred_pr_pdfm_tle1[i, j, :] = vis1 / np.sum(w1[j:])
            pred_pr_pdfm_tle2[i, j, :] = vis2 / np.sum(w2[j:])
            pred_pr_pdfm_tle3[i, j, :] = vis3 / np.sum(w3[j:])
            pred_pr_pdfm_tle4[i, j, :] = vis4 / np.sum(w4[j:])
    #         pred_pr_pdfm_tle5[i, j, :] = vis5 / np.sum(w5[j:])
    np.save(r'D:\data\vis\vis1085_shr_pred_pdfm_tle0.npy', pred_pr_pdfm_tle0)
    np.save(r'D:\data\vis\vis1085_shr_pred_pdfm_tle1.npy', pred_pr_pdfm_tle1)
    np.save(r'D:\data\vis\vis1085_shr_pred_pdfm_tle2.npy', pred_pr_pdfm_tle2)
    np.save(r'D:\data\vis\vis1085_shr_pred_pdfm_tle3.npy', pred_pr_pdfm_tle3)
    np.save(r'D:\data\vis\vis1085_shr_pred_pdfm_tle4.npy', pred_pr_pdfm_tle4)
    # np.save(r'D:\data\vis\vis1085_shr_pred_pdfm_tle5.npy', pred_pr_pdfm_tle5)
    pred_pr_pdfm_tle0 = np.load(r'D:\data\vis\vis1085_shr_pred_pdfm_tle0.npy')
    pred_pr_pdfm_tle1 = np.load(r'D:\data\vis\vis1085_shr_pred_pdfm_tle1.npy')
    pred_pr_pdfm_tle2 = np.load(r'D:\data\vis\vis1085_shr_pred_pdfm_tle2.npy')
    pred_pr_pdfm_tle3 = np.load(r'D:\data\vis\vis1085_shr_pred_pdfm_tle3.npy')
    pred_pr_pdfm_tle4 = np.load(r'D:\data\vis\vis1085_shr_pred_pdfm_tle4.npy')
    pred_pr_pdfm_tle5 = np.load(r'D:\data\vis\vis1085_shr_pred_pdfm_tle5.npy')
    print('****** pdfm-tle0 ******')
    pred_pr_pdfm_tle0[pred_pr_pdfm_tle0 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_pdfm_tle0)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** pdfm-tle1 ******')
    pred_pr_pdfm_tle1[pred_pr_pdfm_tle1 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_pdfm_tle1)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** pdfm-tle2 ******')
    pred_pr_pdfm_tle2[pred_pr_pdfm_tle2 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_pdfm_tle2)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** pdfm-tle3 ******')
    pred_pr_pdfm_tle3[pred_pr_pdfm_tle3 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_pdfm_tle3)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** pdfm-tle4 ******')
    pred_pr_pdfm_tle4[pred_pr_pdfm_tle4 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_pdfm_tle4)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('****** pdfm-tle5 ******')
    pred_pr_pdfm_tle5[pred_pr_pdfm_tle5 >= 30000] = 30000
    acc = VisAcc(val_ob, pred_pr_pdfm_tle5)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())


if __name__ == '__main__':
    print('Program tl.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(f'Program tl.py finished, total time: {format_time(total_elapsed)}')

    print(acc.get_mar2())


# if __name__ == '__main__':
#     print('Program tl.py started')
#     total_start = arrow.now()
#
#     main()
#
#     total_elapsed = (arrow.now() - total_start).total_seconds()
#     print(f'Program tl.py finished, total time: {format_time(total_elapsed)}')
