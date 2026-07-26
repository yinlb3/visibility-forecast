#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Founded in 2024-07-13
Modified in 2026-07-21
@author: yinlb
"""
import pathlib

import arrow
import numpy as np

from src import ots_model, utils, vis_acc


# Load OTS configuration at import time
_OTS_CFG = utils.CFG['ots']
_OTS_INPUT = _OTS_CFG['input_files']
_OTS_OUTPUT = _OTS_CFG['output_files']
_OTS_PARAMS = _OTS_CFG['params']

_MAX_VISIBILITY = float(_OTS_PARAMS['max_visibility'])
_OUTPUT_LEAD_HOURS = int(_OTS_PARAMS['output_lead_hours'])
_MAPPING_THRESHOLDS = list(_OTS_PARAMS['mapping_thresholds'])


OTS = ots_model.OTS

def _print_metrics(acc: vis_acc.VisAcc) -> None:
    """Print standard verification metrics from a VisAcc object.

    Args:
        acc: VisAcc instance with obs and forecast data.
    """
    print(acc.get_r())
    print(acc.get_me())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_ts_ge())
    print(acc.get_ets_ge())
    print(acc.get_bias_ge())
    print(acc.get_far_ge())
    print(acc.get_mar_ge())


def main() -> None:
    """Run OTS calibration and verification."""
    # 1. Load observation and forecast data
    data_dir = pathlib.Path(utils.CFG['paths']['data_dir'])
    ob = np.load(str(data_dir / _OTS_INPUT['observation_npy']))[:, :, 1:, :]
    pr = np.load(str(data_dir / _OTS_INPUT['forecast_npy']))[:, :, 1:, :]

    # 2. Filter stations and split train/validation
    train_ob, train_pr, val_ob, val_pr, valid_stations = utils.load_vis_data(ob, pr)
    del ob, pr

    # 3. Apply pre-calibrated OTS thresholds using the OTS class
    ots = OTS(max_visibility=_MAX_VISIBILITY)
    ots.t = list(_MAPPING_THRESHOLDS)
    pred_pr = ots.predict(val_pr)

    # 4. Reshape to filtered station count and subset to 1085 stations
    index_sta08 = np.load(str(data_dir / _OTS_INPUT['station_index_npy']))
    subset_mask = index_sta08[valid_stations]
    n_valid_stations = int(np.sum(valid_stations))

    pred_pr = np.reshape(pred_pr, (-1, _OUTPUT_LEAD_HOURS, n_valid_stations))
    pred_pr = pred_pr[:, :, subset_mask]
    np.save(str(data_dir / _OTS_OUTPUT['calibrated_forecast']), pred_pr)

    val_ob = np.reshape(val_ob, (-1, _OUTPUT_LEAD_HOURS, n_valid_stations))
    val_ob = val_ob[:, :, subset_mask]

    # 5. Verify calibrated forecast
    acc = vis_acc.VisAcc(val_ob, pred_pr)
    _print_metrics(acc)


if __name__ == '__main__':
    print('Program ots.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(f'Program ots.py finished, total time: {utils.format_time(total_elapsed)}')
