#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimal Threshold Selection (OTS) for visibility forecast calibration.

Founded in 2024-07-13
Modified in 2026-08-06
@author: yinlb
"""
import numpy as np

from src import vis_acc


class OTS:
    """Optimal Threshold Selection for visibility forecast calibration."""

    def __init__(self, max_visibility: float = 60000.0) -> None:
        """Initialize empty threshold list.

        Args:
            max_visibility: Maximum visibility value used for threshold search
                and upper-range mapping (meters).
        """
        self.t = list()
        self._max_visibility = float(max_visibility)

    def fit(self, ob: np.ndarray, pr: np.ndarray) -> None:
        """Fit optimal thresholds that maximize TS for each visibility grade.

        Args:
            ob: Observation array.
            pr: Forecast array.
        """
        thres = list(vis_acc.THRES)
        thres.reverse()

        # Candidate thresholds from max_visibility down to 0m
        t0 = list(range(100))
        t0 += list(range(100, 1000, 10))
        t0 += list(range(1000, int(self._max_visibility) + 1, 100))
        t0.reverse()
        t0 = np.array(t0, dtype=np.float32)

        for i in range(len(thres)):
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
        """Map raw forecast to calibrated visibility using fit thresholds.

        Args:
            pr: Raw forecast array.

        Returns:
            Calibrated forecast array.
        """
        pred_pr = np.zeros_like(pr, dtype=np.float32) + np.nan
        t = self.t

        index = pr >= t[0]
        pr_i = pr[index]
        pred_pr[index] = (
            (pr_i - t[0]) / (self._max_visibility - t[0]) * 50000 + 10000
        )

        index = (pr >= t[1]) & (pr < t[0])
        pr_i = pr[index]
        pred_pr[index] = (pr_i - t[1]) / (t[0] - t[1]) * 8000 + 2000

        index = (pr >= t[2]) & (pr < t[1])
        pr_i = pr[index]
        pred_pr[index] = (pr_i - t[2]) / (t[1] - t[2]) * 1000 + 1000

        index = (pr >= t[3]) & (pr < t[2])
        pr_i = pr[index]
        pred_pr[index] = (pr_i - t[3]) / (t[2] - t[3]) * 500 + 500

        index = (pr >= t[4]) & (pr < t[3])
        pr_i = pr[index]
        pred_pr[index] = (pr_i - t[4]) / (t[3] - t[4]) * 300 + 200

        index = (pr >= t[5]) & (pr < t[4])
        pr_i = pr[index]
        pred_pr[index] = (pr_i - t[5]) / (t[4] - t[5]) * 150 + 50

        index = pr < t[5]
        pred_pr[index] = pr[index] / t[5] * 50

        return pred_pr
