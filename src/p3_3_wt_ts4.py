#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 3.3: Weather type evaluation output module.

Founded in 2026-05-19
Modified in 2026-06-30
@author: yinlb
"""

import typing

import numpy as np

from src import vis_acc

VisAcc = vis_acc.VisAcc


def format_wt_ts4_impr(
    qem: np.ndarray,
    vis_ob: np.ndarray,
    cma_sh_warr: np.ndarray,
    pred_pdfm_tle2: np.ndarray
) -> str:
    """
    Format weather-type quantitative metrics table.

    Metrics: CC, MAE (km), RMSE (km), MRE.
    Weather types: Precip, Fog, Haze, Overall.

    Args:
        qem (np.ndarray): Quantitative metrics array from
            calc_weather_type_metrics, shape (4, n_schemes, 3).
        vis_ob (np.ndarray): Observed visibility array.
        cma_sh_warr (np.ndarray): CMA-SH-WARR forecast array.
        pred_pdfm_tle2 (np.ndarray): PDFM-TLE forecast array (scheme 3).

    Returns:
        str: Tab-separated table string.
    """
    # Extract values for three weather types from qem
    # qem shape: (4 metrics, n_schemes, 3 types)
    cc_cma = qem[0, 0, :]
    mae_cma = qem[1, 0, :] / 1000.0
    rmse_cma = qem[2, 0, :] / 1000.0
    mre_cma = qem[3, 0, :]
    cc_pdfm = qem[0, 3, :]
    mae_pdfm = qem[1, 3, :] / 1000.0
    rmse_pdfm = qem[2, 3, :] / 1000.0
    mre_pdfm = qem[3, 3, :]

    # Overall: all valid triple-matched samples
    mask = ~np.isnan(vis_ob) & ~np.isnan(cma_sh_warr) & \
        ~np.isnan(pred_pdfm_tle2)
    acc_cma = VisAcc(vis_ob[mask], cma_sh_warr[mask])
    acc_pdfm = VisAcc(vis_ob[mask], pred_pdfm_tle2[mask])
    cc_cma_all = acc_cma.get_r()
    mae_cma_all = acc_cma.get_mae() / 1000.0
    rmse_cma_all = acc_cma.get_rmse() / 1000.0
    mre_cma_all = acc_cma.get_mre()
    cc_pdfm_all = acc_pdfm.get_r()
    mae_pdfm_all = acc_pdfm.get_mae() / 1000.0
    rmse_pdfm_all = acc_pdfm.get_rmse() / 1000.0
    mre_pdfm_all = acc_pdfm.get_mre()

    # Build table rows
    wt_names = ('Precip', 'Fog', 'Haze', 'Overall')
    lines = list()
    # Header row 1: weather type names
    header1 = '\t'.join([''] + list(wt_names))
    lines.append(header1)
    # Header row 2: CMA / PDFM for each type
    header2 = '\t'.join([''] + ['CMA / PDFM'] * 4)
    lines.append(header2)
    # Data rows
    lines.append(
        '\t'.join(['CC'] +
                  [f'{cc_cma[i]:.4f} / {cc_pdfm[i]:.4f}' for i in range(3)] +
                  [f'{cc_cma_all:.4f} / {cc_pdfm_all:.4f}'])
    )
    rmse_vals = [f'{rmse_cma[i]:.2f} / {rmse_pdfm[i]:.2f}' for i in range(3)]
    rmse_vals.append(f'{rmse_cma_all:.2f} / {rmse_pdfm_all:.2f}')
    lines.append('\t'.join(['RMSE (km)'] + rmse_vals))
    lines.append(
        '\t'.join(['MAE (km)'] +
                  [f'{mae_cma[i]:.2f} / {mae_pdfm[i]:.2f}' for i in range(3)] +
                  [f'{mae_cma_all:.2f} / {mae_pdfm_all:.2f}'])
    )
    lines.append(
        '\t'.join(['MRE'] +
                  [f'{mre_cma[i]:.4f} / {mre_pdfm[i]:.4f}' for i in range(3)] +
                  [f'{mre_cma_all:.4f} / {mre_pdfm_all:.4f}'])
    )
    return '\n'.join(lines)
