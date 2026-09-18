# -*- coding: utf-8 -*-
"""Time-lagged ensemble (TLE) utilities for grid forecasts.

Provides equal-weight TLE averaging on PDFM-corrected grid forecasts.
No model parameters are trained; the algorithm is fixed.

Founded in 2026-08-06
Modified in 2026-08-06
@author: yinlb
"""

import typing

import arrow
import numpy as np


def _shift_init_time(init_time: str, hours: int) -> str:
    """Shift a YYYYMMDDHH string by a signed number of hours.

    Args:
        init_time: YYYYMMDDHH string.
        hours: Hours to add (positive) or subtract (negative).

    Returns:
        Shifted YYYYMMDDHH string.
    """
    return (
        arrow.get(init_time, 'YYYYMMDDHH')
        .shift(hours=hours)
        .format('YYYYMMDDHH')
    )


def _build_lead_index(leads: typing.List[int]) -> typing.Dict[int, int]:
    """Map lead hour value to array index.

    Args:
        leads: Ordered list of lead hours.

    Returns:
        Mapping from lead value to index.
    """
    return {lead: idx for idx, lead in enumerate(leads)}


def apply_equal_tle(
    corrected_grids: typing.Dict[str, np.ndarray],
    leads: typing.List[int],
    max_lookback: int = 24,
    fallback_to_pdfm: bool = True,
) -> typing.Dict[str, np.ndarray]:
    """Apply equal-weight time-lagged ensemble to PDFM-corrected grids.

    For each target init_time m and lead n, the TLE value is the average of
    available PDFM-corrected forecasts along the anti-diagonal:

        VIS_TLE(m, n) = mean( VIS(m-i, n+i) )

    where i ranges from 0 to min(max_lookback, 24-n) and both the lagged
    init_time and lead exist in the input. Grid cells are processed
    independently; NaNs propagate only within their own sample.

    Args:
        corrected_grids: Mapping init_time (YYYYMMDDHH) -> corrected grid
            array of shape (n_lead, nlat, nlon).
        leads: Ordered list of lead hours matching the first axis of each
            grid array.
        max_lookback: Maximum hours to look back for lagged init times.
        fallback_to_pdfm: When no anti-diagonal sample is available, keep the
            original PDFM-corrected value.

    Returns:
        Mapping init_time -> TLE grid array with the same shape as input.

    Raises:
        ValueError: If no corrected grids are provided or lead list is empty.
    """
    if not corrected_grids:
        raise ValueError('corrected_grids must not be empty')
    if not leads:
        raise ValueError('leads must not be empty')

    lead_index = _build_lead_index(leads)
    n_leads = len(leads)
    reference_shape = next(iter(corrected_grids.values())).shape

    tle_grids: typing.Dict[str, np.ndarray] = {}

    for init_time, target_grid in corrected_grids.items():
        if target_grid.shape != reference_shape:
            raise ValueError(
                f'Shape mismatch for {init_time}: {target_grid.shape} != '
                f'{reference_shape}'
            )

        tle_grid = target_grid.copy()

        for lead_idx, lead in enumerate(leads):
            max_i = min(max_lookback, 24 - lead)
            samples = []

            for i in range(max_i + 1):
                lag_init = _shift_init_time(init_time, -i)
                lag_lead = lead + i
                lag_grid = corrected_grids.get(lag_init)
                if lag_grid is None:
                    continue
                lag_lead_idx = lead_index.get(lag_lead)
                if lag_lead_idx is None:
                    continue
                samples.append(lag_grid[lag_lead_idx])

            if not samples:
                continue

            stacked = np.stack(samples, axis=0)
            with np.errstate(invalid='ignore'):
                mean = np.nanmean(stacked, axis=0)

            if fallback_to_pdfm:
                pdfm_mask = np.isnan(mean)
                mean[pdfm_mask] = target_grid[lead_idx][pdfm_mask]

            tle_grid[lead_idx] = mean

        tle_grids[init_time] = tle_grid

    return tle_grids
