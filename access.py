#!user/bin.python3

"""
Founded in 2024-03-24
Modified in 2024-11-07
@author: yinlb
"""
import joblib
import os
import sys
import typing

import arrow
import numpy as np
import pandas as pd


THRES = (10000., 2000., 1000., 500., 200., 50.)
# [0.3402387  0.11927235 0.12334341 0.12732257 0.13927859 0.11669873]
# [0.50168276 0.1368752  0.12147611 0.12757798 0.14081563 0.12272818]
# [0.5015232  0.1368301  0.11941028 0.12470423 0.1369358  0.12310486] 24
# [0.5183403  0.17651607 0.16779487 0.17834826 0.24180914 0.38097325] 97


class PDF:
    def __init__(self):
        self.c = None

    def fit(self, ob: np.ndarray, pr: np.ndarray):
        ob = ob[~np.isnan(ob)][::10000]
        pr = pr[~np.isnan(pr)][::10000]
        a = np.unique(ob)
        a = np.sort(a)
        self.c = np.zeros((len(a), 2), dtype=np.float32)
        self.c[:, 0] = a
        pr = np.sort(pr)
        for i, a0 in enumerate(a):
            p0 = np.mean(ob <= a0)
            j = round(p0 * (len(pr) - 1))
            self.c[i, 1] = pr[j]

    def predict(self, pr0: np.ndarray) -> typing.Optional[np.ndarray]:
        if self.c is None:
            return None
        shape = pr0.shape
        pr0 = np.reshape(pr0, -1)
        pr = np.zeros_like(pr0) + np.nan
        for i in range(pr0.size):
            value = pr0[i]
            if np.isnan(value):
                continue
            if value < self.c[0, 1]:
                pr[i] = self.c[0, 0]
                continue
            if value > self.c[-1, 1]:
                pr[i] = self.c[-1, 0]
                continue
            left = 0
            right = self.c.shape[0] - 1
            while right - left > 1:
                mid = round((left + right) / 2)
                if value <= self.c[mid, 1]:
                    right = mid
                else:
                    left = mid
            k = (self.c[right, 0] - self.c[left, 0])
            k /= (self.c[right, 1] - self.c[left, 1])
            pr[i] = self.c[left, 0] + k * (value - self.c[left, 1])
        pr = np.reshape(pr, shape)
        return pr


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
        shape = (self.n_grades + 1, self.n_grades + 1)
        self.hxjz = np.zeros(shape, dtype=np.int_)
        for i in range(self.n_grades + 1):
            for j in range(self.n_grades + 1):
                mask = (self.ob_grade == i) & (self.pr_grade == j)
                self.hxjz[i, j] = np.sum(mask)
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
            denom = na + nb + nc - r
            ets[i] = (na - r) / denom if na + nb + nc != 0 else np.nan
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

    def get_pdf_model(self) -> PDF:
        model = PDF()
        model.fit(self.ob, self.pr)
        return model


def format_time(second: float, is_abbreviation: bool = False) -> str:
    r"""Format time.

    :param second: A float number representing the number of seconds.
    :param is_abbreviation: Whether to use abbreviation format.
        The default value is False.
    :return: Formatted time string, e.g., '43.5 seconds'.
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
    # 1. Load station info and build region filters
    sta = pd.read_csv(r'D:\data\vis\sta2411.csv', low_memory=False)
    index_zgdb = np.zeros(2411, dtype=np.bool_)
    index_cjzxy = np.zeros(1183, dtype=np.bool_)
    # 1.1 Filter stations by eastern China provinces
    for i in range(2411):
        provs = ('北京市', '上海市', '天津市', '安徽省', '福建省', '广东省',
                 '江苏省', '江西省', '河北省', '河南省', '湖北省', '湖南省',
                 '山东省', '山西省', '浙江省')
        if sta.loc[i, 'province'] in provs:
            index_zgdb[i] = True
    sta = sta.loc[index_zgdb]
    sta.reset_index(drop=True, inplace=True)
    for i in range(1183):
        provs2 = ('湖北省', '湖南省', '江西省', '安徽省', '江苏省', '浙江省', '上海市')
        if sta.loc[i, 'province'] in provs2:
            index_cjzxy[i] = True
    # 2. Load visibility observation and forecast data
    ob = np.load(r'D:\data\vis\vis1183_ob.npy')[:, :, 1:, :]
    ob[ob >= 999990] = np.nan  # Missing value marker
    pr = np.load(r'D:\data\vis\vis_gjz.npy')[:1461, :, 1:, :]
    # 2.1 Split into train/validation sets (last 365 days as validation)
    train_ob = ob[:-365, :, :, :]
    train_pr = pr[:-365, :, :, :]
    val_ob = ob[-365:, :, :, :]
    val_pr = pr[-365:, :, :, :]
    val_ob_cjzxy = val_ob[..., index_cjzxy]
    val_pr_cjzxy = val_pr[..., index_cjzxy]
    np.save(r'D:\data\vis\vis_ob_cjzxy.npy', val_ob_cjzxy)
    np.save(r'D:\data\vis\vis_pr_cjzxy.npy', val_pr_cjzxy)
    del ob, pr

    train_ob_cjzxy = train_ob[..., index_cjzxy]
    train_pr_cjzxy = train_pr[..., index_cjzxy]
    # 3. Scheme 1: Overall PDF matching
    print('Scheme 1')
    time_arrow = arrow.now()
    acc = VisAcc(train_ob, train_pr)
    model = acc.get_pdf_model()
    print((arrow.now() - time_arrow).total_seconds() / 60)
    joblib.dump(model, r'D:\data\vis\pdfm0.dat')
    pred_pdfm = model.predict(val_pr)
    pred_pdfm = np.reshape(pred_pdfm, (8760, 24, 1183))
    pred_pdfm_tle = np.zeros_like(pred_pdfm, dtype=np.float32) + np.nan
    for i in range(8760):
        for j in range(24):
            for k in range(1183):
                n = 0
                vis = 0
                for ii in range(24):
                    idx = i + j - ii
                    if 0 <= idx < 8760 and ~np.isnan(pred_pdfm[idx, ii, k]):
                        n += 1
                        vis += pred_pdfm[idx, ii, k]
                if n > 0:
                    pred_pdfm_tle[i, j, k] = vis / n
        print(i)
    pred_pdfm_tle = np.reshape(pred_pdfm_tle, (365, 24, 24, 1183))
    np.save(r'D:\data\vis\vis_gjz_pred0.npy', pred_pdfm_tle)
    time_arrow = arrow.now()
    acc = VisAcc(train_ob_cjzxy, train_pr_cjzxy)
    model = acc.get_pdf_model()
    print((arrow.now() - time_arrow).total_seconds() / 60)
    joblib.dump(model, r'D:\data\vis\pdfm0_cjzxy.dat')
    pred_pdfm = model.predict(val_pr_cjzxy)
    pred_pdfm = np.reshape(pred_pdfm, (8760, 24, 502))
    pred_pdfm_tle = np.zeros_like(pred_pdfm, dtype=np.float32) + np.nan
    for i in range(8760):
        for j in range(24):
            for k in range(502):
                n = 0
                vis = 0
                for ii in range(24):
                    idx = i + j - ii
                    if 0 <= idx < 8760 and ~np.isnan(pred_pdfm[idx, ii, k]):
                        n += 1
                        vis += pred_pdfm[idx, ii, k]
                if n > 0:
                    pred_pdfm_tle[i, j, k] = vis / n
        print(i)
    pred_pdfm_tle = np.reshape(pred_pdfm_tle, (365, 24, 24, 502))
    np.save(r'D:\data\vis\vis_gjz_pred0_cjzxy.npy', pred_pdfm_tle)
    del acc, pred_pdfm, model
    # 4. Scheme 2: Hour-specific PDF matching
    print('Scheme 2')
    pred_pdfm = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    time_arrow = arrow.now()
    models = list()
    for i in range(24):  # Build model for each init hour
        acc = VisAcc(train_ob[:, :, i, :], train_pr[:, :, i, :])
        models.append(acc.get_pdf_model())
    print((arrow.now() - time_arrow).total_seconds() / 60)
    joblib.dump(models, r'D:\data\vis\pdfm1.dat')
    for i in range(24):
        pred_pdfm[:, :, i, :] = models[i].predict(val_pr[:, :, i, :])
    pred_pdfm = np.reshape(pred_pdfm, (8760, 24, 1183))
    pred_pdfm_tle = np.zeros_like(pred_pdfm, dtype=np.float32) + np.nan
    for i in range(8760):
        for j in range(24):
            for k in range(1183):
                n = 0
                vis = 0
                for ii in range(24):
                    idx = i + j - ii
                    if 0 <= idx < 8760 and ~np.isnan(pred_pdfm[idx, ii, k]):
                        n += 1
                        vis += pred_pdfm[idx, ii, k]
                if n > 0:
                    pred_pdfm_tle[i, j, k] = vis / n
        print(i)
    pred_pdfm_tle = np.reshape(pred_pdfm_tle, (365, 24, 24, 1183))
    np.save(r'D:\data\vis\vis_gjz_pred1.npy', pred_pdfm_tle)
    pred_pdfm = np.zeros_like(val_pr_cjzxy, dtype=np.float32) + np.nan
    time_arrow = arrow.now()
    models = list()
    for i in range(24):
        acc = VisAcc(train_ob_cjzxy[:, :, i, :], train_pr_cjzxy[:, :, i, :])
        models.append(acc.get_pdf_model())
    print((arrow.now() - time_arrow).total_seconds() / 60)
    joblib.dump(models, r'D:\data\vis\pdfm1_cjzxy.dat')
    for i in range(24):
        pred_pdfm[:, :, i, :] = models[i].predict(val_pr_cjzxy[:, :, i, :])
    pred_pdfm = np.reshape(pred_pdfm, (8760, 24, 502))
    pred_pdfm_tle = np.zeros_like(pred_pdfm, dtype=np.float32) + np.nan
    for i in range(8760):
        for j in range(24):
            for k in range(502):
                n = 0
                vis = 0
                for ii in range(24):
                    idx = i + j - ii
                    if 0 <= idx < 8760 and ~np.isnan(pred_pdfm[idx, ii, k]):
                        n += 1
                        vis += pred_pdfm[idx, ii, k]
                if n > 0:
                    pred_pdfm_tle[i, j, k] = vis / n
        print(i)
    pred_pdfm_tle = np.reshape(pred_pdfm_tle, (365, 24, 24, 502))
    np.save(r'D:\data\vis\vis_gjz_pred1_cjzxy.npy', pred_pdfm_tle)
    del acc, pred_pdfm, models
    # 5. Scheme 3: Station-specific PDF matching
    print('Scheme 3')
    pred_pdfm = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    time_arrow = arrow.now()
    models = list()
    for i in range(1183):  # Build model for each station
        acc = VisAcc(train_ob[:, :, :, i], train_pr[:, :, :, i])
        models.append(acc.get_pdf_model())
    print((arrow.now() - time_arrow).total_seconds() / 60)
    joblib.dump(models, r'D:\data\vis\pdfm2.dat')
    for i in range(1183):
        pred_pdfm[:, :, :, i] = models[i].predict(val_pr[:, :, :, i])
    pred_pdfm = np.reshape(pred_pdfm, (8760, 24, 1183))
    pred_pdfm_tle = np.zeros_like(pred_pdfm, dtype=np.float32) + np.nan
    for i in range(8760):
        for j in range(24):
            for k in range(1183):
                n = 0
                vis = 0
                for ii in range(24):
                    idx = i + j - ii
                    if 0 <= idx < 8760 and ~np.isnan(pred_pdfm[idx, ii, k]):
                        n += 1
                        vis += pred_pdfm[idx, ii, k]
                if n > 0:
                    pred_pdfm_tle[i, j, k] = vis / n
        print(i)
    pred_pdfm = np.reshape(pred_pdfm, (365, 24, 24, 1183))
    pred_pdfm_tle = np.reshape(pred_pdfm_tle, (365, 24, 24, 1183))
    np.save(r'D:\data\vis\vis_gjz_pdfm.npy', pred_pdfm)
    np.save(r'D:\data\vis\vis_gjz_pred2.npy', pred_pdfm_tle)
    val_pr = np.reshape(val_pr, (8760, 24, 1183))
    pred_tle = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    for i in range(8760):
        for j in range(24):
            for k in range(1183):
                n = 0
                vis = 0
                for ii in range(24):
                    idx = i + j - ii
                    if 0 <= idx < 8760 and ~np.isnan(val_pr[idx, ii, k]):
                        n += 1
                        vis += val_pr[idx, ii, k]
                if n > 0:
                    pred_tle[i, j, k] = vis / n
        print(i)
    val_pr = np.reshape(val_pr, (365, 24, 24, 1183))
    pred_tle = np.reshape(pred_tle, (365, 24, 24, 1183))
    np.save(r'D:\data\vis\vis_gjz_tle.npy', pred_tle)
    pred_pdfm = np.zeros_like(val_pr_cjzxy, dtype=np.float32) + np.nan
    time_arrow = arrow.now()
    models = list()
    for i in range(502):
        acc = VisAcc(train_ob_cjzxy[:, :, :, i], train_pr_cjzxy[:, :, :, i])
        models.append(acc.get_pdf_model())
    print((arrow.now() - time_arrow).total_seconds() / 60)
    joblib.dump(models, r'D:\data\vis\pdfm2_cjzxy.dat')
    for i in range(502):
        pred_pdfm[:, :, :, i] = models[i].predict(val_pr_cjzxy[:, :, :, i])
    pred_pdfm = np.reshape(pred_pdfm, (8760, 24, 502))
    pred_pdfm_tle = np.zeros_like(pred_pdfm, dtype=np.float32) + np.nan
    for i in range(8760):
        for j in range(24):
            for k in range(502):
                n = 0
                vis = 0
                for ii in range(24):
                    idx = i + j - ii
                    if 0 <= idx < 8760 and ~np.isnan(pred_pdfm[idx, ii, k]):
                        n += 1
                        vis += pred_pdfm[idx, ii, k]
                if n > 0:
                    pred_pdfm_tle[i, j, k] = vis / n
    pred_pdfm = np.reshape(pred_pdfm, (365, 24, 24, 502))
    pred_pdfm_tle = np.reshape(pred_pdfm_tle, (365, 24, 24, 502))
    np.save(r'D:\data\vis\vis_gjz_pdfm_cjzxy.npy', pred_pdfm)
    np.save(r'D:\data\vis\vis_gjz_pred2_cjzxy.npy', pred_pdfm_tle)
    val_pr = np.reshape(val_pr, (8760, 24, 502))
    pred_tle = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    for i in range(8760):
        for j in range(24):
            for k in range(502):
                n = 0
                vis = 0
                for ii in range(24):
                    idx = i + j - ii
                    if 0 <= idx < 8760 and ~np.isnan(val_pr[idx, ii, k]):
                        n += 1
                        vis += val_pr[idx, ii, k]
                if n > 0:
                    pred_tle[i, j, k] = vis / n
    val_pr = np.reshape(val_pr, (365, 24, 24, 502))
    pred_tle = np.reshape(pred_tle, (365, 24, 24, 502))
    np.save(r'D:\data\vis\vis_gjz_tle_cjzxy.npy', pred_tle)
    del acc, pred_pdfm, models

    acc = VisAcc(val_ob, val_pr)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    # pred_pr = np.load(r'D:\data\vis\vis_gjz_pred0.npy')
    # acc = VisAcc(val_ob, pred_pr)
    # print(acc.get_r())
    # print(acc.get_mae())
    # print(acc.get_rmse())
    # print(acc.get_mre())
    # print(acc.get_ts2())
    # print(acc.get_far2())
    # print(acc.get_mar2())
    # pred_pr = np.load(r'D:\data\vis\vis_gjz_pred1.npy')
    # acc = VisAcc(val_ob, pred_pr)
    # print(acc.get_r())
    # print(acc.get_mae())
    # print(acc.get_rmse())
    # print(acc.get_mre())
    # print(acc.get_ts2())
    # print(acc.get_far2())
    # print(acc.get_mar2())
    pred_pr = np.load(r'D:\data\vis\vis_gjz_pred2.npy')
    acc = VisAcc(val_ob, pred_pr)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    pred_pr = np.load(r'D:\data\vis\vis_gjz_pdfm.npy')
    acc = VisAcc(val_ob, pred_pr)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    pred_pr = np.load(r'D:\data\vis\vis_gjz_tle.npy')
    acc = VisAcc(val_ob, pred_pr)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    print('*' * 20)
    acc = VisAcc(val_ob_cjzxy, val_pr_cjzxy)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    # pred_pr = np.load(r'D:\data\vis\vis_gjz_pred0_cjzxy.npy')
    # acc = VisAcc(val_ob_cjzxy, pred_pr)
    # print(acc.get_r())
    # print(acc.get_mae())
    # print(acc.get_rmse())
    # print(acc.get_mre())
    # print(acc.get_ts2())
    # print(acc.get_far2())
    # print(acc.get_mar2())
    # pred_pr = np.load(r'D:\data\vis\vis_gjz_pred1_cjzxy.npy')
    # acc = VisAcc(val_ob_cjzxy, pred_pr)
    # print(acc.get_r())
    # print(acc.get_mae())
    # print(acc.get_rmse())
    # print(acc.get_mre())
    # print(acc.get_ts2())
    # print(acc.get_far2())
    # print(acc.get_mar2())
    pred_pr = np.load(r'D:\data\vis\vis_gjz_pred2_cjzxy.npy')
    acc = VisAcc(val_ob_cjzxy, pred_pr)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    pred_pr = np.load(r'D:\data\vis\vis_gjz_pdfm_cjzxy.npy')
    acc = VisAcc(val_ob_cjzxy, pred_pr)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())
    pred_pr = np.load(r'D:\data\vis\vis_gjz_tle_cjzxy.npy')
    acc = VisAcc(val_ob_cjzxy, pred_pr)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts2())
    print(acc.get_far2())
    print(acc.get_mar2())


if __name__ == '__main__':
    print('Program access.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    elapsed_str = format_time(total_elapsed)
    print(f'Program access.py finished, total time: {elapsed_str}')
