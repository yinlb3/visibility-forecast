#!user/bin.python3

"""
Founded in 2024-07-13
Modified in 2024-07-24
@author: yinlb
"""
import os
import sys
import typing

import arrow
import numpy as np


THRES = (10000., 2000., 1000., 500., 200., 50.)


class OTS:
    def __init__(self):
        self.t = list()

    def fit(self, ob: np.ndarray, pr: np.ndarray):
        t0 = list(range(100)) + list(range(100, 1000, 10)) + list(range(1000, 60001, 100))
        t0.reverse()
        t0 = np.array(t0, dtype=np.float32)
        thres = list(THRES)
        thres.reverse()
        for i in range(6):
            ts = np.zeros_like(t0, dtype=np.float32) - 1
            for j in range(t0.size):
                if i > 0 and t0[j] <= self.t[-1]:
                    continue
                na = np.sum((pr < t0[j]) & (ob < thres[i]))
                nb = np.sum((pr < t0[j]) & (ob >= thres[i]))
                nc = np.sum((pr >= t0[j]) & (ob < thres[i]))
                ts[j] = na / (na + nb + nc) if na + nb + nc > 0 else 0
            self.t.append(float(t0[np.argmax(ts)]))
        print(self.t)

    def predict(self, pr: np.ndarray) -> np.ndarray:
        pred_pr = np.zeros_like(pr, dtype=np.float32) + np.nan
        index = pr >= self.t[0]
        pred_pr[index] = (pr[index] - self.t[0]) / (60000 - self.t[0]) * 50000 + 10000
        index = (pr >= self.t[1]) & (pr < self.t[0])
        pred_pr[index] = (pr[index] - self.t[1]) / (self.t[0] - self.t[1]) * 8000 + 2000
        index = (pr >= self.t[2]) & (pr < self.t[1])
        pred_pr[index] = (pr[index] - self.t[2]) / (self.t[1] - self.t[2]) * 1000 + 1000
        index = (pr >= self.t[3]) & (pr < self.t[2])
        pred_pr[index] = (pr[index] - self.t[3]) / (self.t[2] - self.t[3]) * 500 + 500
        index = (pr >= self.t[4]) & (pr < self.t[3])
        pred_pr[index] = (pr[index] - self.t[4]) / (self.t[3] - self.t[4]) * 300 + 200
        index = (pr >= self.t[5]) & (pr < self.t[4])
        pred_pr[index] = (pr[index] - self.t[5]) / (self.t[4] - self.t[5]) * 150 + 50
        index = pr < self.t[5]
        pred_pr[index] = pr[index] / self.t[5] * 50

        return pred_pr


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
        raise ValueError('The input parameter "second" cannot be negative.')
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
    print(np.sum(index))
    ob = ob[:, :, :, index]
    pr = pr[:, :, :, index]
    train_ob = ob[:-365, :, :, :]
    train_pr = pr[:-365, :, :, :]
    val_ob = ob[-365:, :, :, :]
    val_pr = pr[-365:, :, :, :]
    del ob, pr
    # # 原始
    # na = np.sum((val_pr < 1000) & (val_ob < 1000))
    # nb = np.sum((val_pr < 1000) & (val_ob >= 1000))
    # nc = np.sum((val_pr >= 1000) & (val_ob < 1000))
    # print(na / (na + nb + nc))
    # # 整体
    # pred_pr = np.zeros_like(val_pr)
    # ts = np.zeros(200, dtype=np.float32) + np.nan
    # for t in range(200):
    #     na = np.sum((train_pr < 10 * t + 10) & (train_ob < 1000))
    #     nb = np.sum((train_pr < 10 * t + 10) & (train_ob >= 1000))
    #     nc = np.sum((train_pr >= 10 * t + 10) & (train_ob < 1000))
    #     ts[t] = na / (na + nb + nc)
    #     print(t)
    # t = 10 * np.argmax(ts) + 10
    # pred_pr = val_pr / 1000 * t
    # na = np.sum((pred_pr < 1000) & (val_ob < 1000))
    # nb = np.sum((pred_pr < 1000) & (val_ob >= 1000))
    # nc = np.sum((pred_pr >= 1000) & (val_ob < 1000))
    # print(na / (na + nb + nc))
    # # 分时次
    # pred_pr = np.zeros_like(val_pr)
    # for i in range(24):
    #     ts = np.zeros(200, dtype=np.float32) + np.nan
    #     for t in range(200):
    #         na = np.sum((train_pr[:, :, i, :] < 10 * t + 10) & (train_ob[:, :, i, :] < 1000))
    #         nb = np.sum((train_pr[:, :, i, :] < 10 * t + 10) & (train_ob[:, :, i, :] >= 1000))
    #         nc = np.sum((train_pr[:, :, i, :] >= 10 * t + 10) & (train_ob[:, :, i, :] < 1000))
    #         ts[t] = na / (na + nb + nc)
    #     t = 10 * np.argmax(ts) + 10
    #     pred_pr[:, :, i, :] = val_pr[:, :, i, :] / 1000 * t
    #     print(i)
    # na = np.sum((pred_pr < 1000) & (val_ob < 1000))
    # nb = np.sum((pred_pr < 1000) & (val_ob >= 1000))
    # nc = np.sum((pred_pr >= 1000) & (val_ob < 1000))
    # print(na / (na + nb + nc))
    # # 分站点
    # pred_pr = np.zeros_like(val_pr)
    # for i in range(1139):
    #     ts = np.zeros(200, dtype=np.float32) + np.nan
    #     for t in range(200):
    #         na = np.sum((train_pr[:, :, :, i] < 10 * t + 10) & (train_ob[:, :, :, i] < 1000))
    #         nb = np.sum((train_pr[:, :, :, i] < 10 * t + 10) & (train_ob[:, :, :, i] >= 1000))
    #         nc = np.sum((train_pr[:, :, :, i] >= 10 * t + 10) & (train_ob[:, :, :, i] < 1000))
    #         ts[t] = na / (na + nb + nc)
    #     t = 10 * np.argmax(ts) + 10
    #     pred_pr[:, :, :, i] = val_pr[:, :, :, i] / 1000 * t
    #     print(i)
    # na = np.sum((pred_pr < 1000) & (val_ob < 1000))
    # nb = np.sum((pred_pr < 1000) & (val_ob >= 1000))
    # nc = np.sum((pred_pr >= 1000) & (val_ob < 1000))
    # print(na / (na + nb + nc))
    # # 分时次分站点
    # pred_pr = np.zeros_like(val_pr)
    # for i in range(24):
    #     for j in range(1139):
    #         ts = np.zeros(200, dtype=np.float32) + np.nan
    #         for t in range(200):
    #             na = np.sum((train_pr[:, :, i, j] < 10 * t + 10) & (train_ob[:, :, i, j] < 1000))
    #             nb = np.sum((train_pr[:, :, i, j] < 10 * t + 10) & (train_ob[:, :, i, j] >= 1000))
    #             nc = np.sum((train_pr[:, :, i, j] >= 10 * t + 10) & (train_ob[:, :, i, j] < 1000))
    #             ts[t] = na / (na + nb + nc)
    #         t = 10 * np.argmax(ts) + 10
    #         pred_pr[:, :, i, j] = val_pr[:, :, i, j] / 1000 * t
    #     print(i)
    # na = np.sum((pred_pr < 1000) & (val_ob < 1000))
    # nb = np.sum((pred_pr < 1000) & (val_ob >= 1000))
    # nc = np.sum((pred_pr >= 1000) & (val_ob < 1000))
    # print(na / (na + nb + nc))

    # t0 = list(range(100)) + list(range(100, 1000, 10)) + list(range(1000, 60001, 100))
    # t0 = np.array(t0, dtype=np.float32)
    # for i in range(6):
    #     ts = np.zeros_like(t0, dtype=np.float32) + np.nan
    #     for j in range(t0.size):
    #         na = np.sum((train_pr < t0[j]) & (train_ob < THRES[i]))
    #         nb = np.sum((train_pr < t0[j]) & (train_ob >= THRES[i]))
    #         nc = np.sum((train_pr >= t0[j]) & (train_ob < THRES[i]))
    #         ts[j] = na / (na + nb + nc) if na + nb + nc > 0 else 0
    #     print(t0[np.argmax(ts)])
    # 30500.0
    # 8900.0
    # 1300.0
    # 270.0
    # 160.0
    # 84.0

    pred_pr = np.zeros_like(val_pr, dtype=np.float32) + np.nan
    index0 = val_pr >= 30500
    pred_pr[index0] = (val_pr[index0] - 30500) / (60000 - 30500) * 50000 + 10000
    index0 = (val_pr >= 8900) & (val_pr < 30500)
    pred_pr[index0] = (val_pr[index0] - 8900) / (30500 - 8900) * 8000 + 2000
    index0 = (val_pr >= 1300) & (val_pr < 8900)
    pred_pr[index0] = (val_pr[index0] - 1300) / (8900 - 1300) * 1000 + 1000
    index0 = (val_pr >= 270) & (val_pr < 1300)
    pred_pr[index0] = (val_pr[index0] - 270) / (1300 - 270) * 500 + 500
    index0 = (val_pr >= 160) & (val_pr < 270)
    pred_pr[index0] = (val_pr[index0] - 160) / (270 - 160) * 300 + 200
    index0 = (val_pr >= 84) & (val_pr < 160)
    pred_pr[index0] = (val_pr[index0] - 84) / (160 - 84) * 150 + 50
    index0 = val_pr < 84
    pred_pr[index0] = val_pr[index0] / 84 * 50
    index_sta08 = np.load(r'D:\Project\vis\index_sta08.npy')
    pred_pr = np.reshape(pred_pr, (-1, 24, 1139))
    pred_pr = pred_pr[:, :, index_sta08[index]]
    np.save(r'D:\data\vis\vis1085_shr_ots0.npy', pred_pr)
    val_ob = np.reshape(val_ob, (-1, 24, 1139))
    val_ob = val_ob[:, :, index_sta08[index]]
    acc = VisAcc(val_ob, pred_pr)
    print(acc.get_r())
    print(acc.get_me())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    # print(acc.get_hxjz())
    # print(acc.get_oa())
    # print(acc.get_kappa())
    # print(acc.get_ts())
    print(acc.get_ts2())
    print(acc.get_ets2())
    print(acc.get_bias2())
    print(acc.get_far2())
    print(acc.get_mar2())

    # for i in range(24):
    #     print(np.nanmax(train_ob[:, :, 0, :]), np.nanmax(train_pr[:, :, 0, :]))
    # model = OTS()
    # model.fit(train_ob[:, :, 0, :], train_pr[:, :, 0, :])


if __name__ == '__main__':
    print('The program "ots.py" is beginning.')
    start = arrow.now()

    main()

    end = arrow.now()
    running_time = (end - start).total_seconds()

    print('The program "ots.py" runs out in {:s}.'.format(format_time(running_time)))
