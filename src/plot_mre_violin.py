# -*- coding: utf-8 -*-
"""
MRE distribution violin/box plot module (Stage 8.5).

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""

import gc
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_mre_violins(sta: pd.DataFrame, output_dir: str) -> None:
    """
    绘制 MRE 与改善率的小提琴图/箱线图.
    """
    df_sta = pd.read_csv(rf'{output_dir}\csv\vis_sta_mre.csv', low_memory=False)
    mre_before = np.array(df_sta.loc[:, 'CMA-SH-WARR'])
    mre_after = np.array(df_sta.loc[:, 'PDFM-TLE'])
    mre_improvement = (mre_before - mre_after) / mre_before * 100

    plt.figure(figsize=(5, 6))
    sns.violinplot(
        data={
            'CMA-SH-WARR': df_sta.loc[:, 'CMA-SH-WARR'],
            'PDFM-TLE': df_sta.loc[:, 'PDFM-TLE'],
        },
        palette=['blue', 'red']
    )
    plt.ylabel('MRE')
    plt.savefig(rf'{output_dir}\boxplot_mre.png', bbox_inches='tight', dpi=800)
    plt.savefig(rf'{output_dir}\boxplot_mre.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    gc.collect()
    print(f'[plot_mre_violins] MRE median (CMA-SH-WARR, PDFM-TLE): {np.median(df_sta.loc[:, "CMA-SH-WARR"])}, {np.median(df_sta.loc[:, "PDFM-TLE"])}')
    print(f'[plot_mre_violins] MRE median improvement: {np.median(df_sta.loc[:, "CMA-SH-WARR"]) - np.median(df_sta.loc[:, "PDFM-TLE"])}')

    plt.figure(figsize=(5, 6))
    sns.violinplot(
        data={'PDFM-TLE': mre_improvement},
        color='skyblue'
    )
    plt.ylabel('MRE改善率')
    plt.savefig(rf'{output_dir}\boxplot_mre_improvement.png', bbox_inches='tight', dpi=800)
    plt.savefig(rf'{output_dir}\boxplot_mre_improvement.pdf', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close('all')
    gc.collect()
    print(f'[plot_mre_violins] Positive improvement ratio: {np.mean(mre_improvement > 0)}')
