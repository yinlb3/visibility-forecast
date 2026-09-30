# -*- coding: utf-8 -*-
"""
Post-processing correction for visibility forecasts.

Two correction stages live here:

- PDFM (probability density function matching) maps the raw forecast
  distribution onto the observed one through a quantile mapping curve.
  See the PDFM class and load_pdfm_file.
- TLE (time-lagged ensemble) averages PDFM-corrected grids from earlier
  initialization times along the anti-diagonal of equal valid time.
  See tle_model.apply_equal_tle.

Model files serialized before the class was renamed from PDF to PDFM
still load through load_pdfm_file, which maps the old class name and the
old public curve attribute back onto the current ones.

Founded in 2024-03-24
Modified in 2026-09-30
@author: yinlb, space-bunny
"""
import copy
import pathlib
import sys
import typing

import arrow
import joblib
import numpy as np

from src import logger, utils


class PDFM:
    """
    PDFM (probability density function matching) model for visibility
    forecast correction.

    Stores a quantile mapping curve fitted from paired observation and
    forecast samples. predict() applies the curve to a raw forecast
    field through linear interpolation.
    """

    def __init__(self) -> None:
        """Initialize empty matching curve."""
        self._c = None

    @classmethod
    def load(
        cls: typing.Type['PDFM'],
        path: typing.Union[str, pathlib.Path],
    ) -> 'PDFM':
        """Load a fitted PDFM model from a joblib-serialized file.

        Args:
            path: Path to the .dat file saved by joblib.dump().

        Returns:
            PDFM instance with fitted matching curve.

        Raises:
            TypeError: If the loaded object is not a PDFM instance.
        """
        model = load_pdfm_file(path)
        if not isinstance(model, cls):
            raise TypeError(
                f'Expected {cls.__name__} instance, got {type(model).__name__}'
            )
        return model

    def fit(self, ob: np.ndarray, pr: np.ndarray) -> None:
        """Fit the quantile mapping curve from obs and forecast samples.

        Args:
            ob: Observation array.
            pr: Forecast array.
        """
        # Subsample both arrays before building the curve: the quantile
        # mapping only needs the distribution shape, and the full arrays are
        # far too large to hold every sample.
        ob = ob[~np.isnan(ob)][::10000]
        pr = pr[~np.isnan(pr)][::10000]
        uniq_vis = np.sort(np.unique(ob))
        self._c = np.zeros((len(uniq_vis), 2), dtype=np.float32)
        self._c[:, 0] = uniq_vis
        pr = np.sort(pr)
        # For each observed value, take the forecast at the same quantile;
        # that quantile pair is the quantile mapping curve.
        for i, obs_val in enumerate(uniq_vis):
            p0 = np.mean(ob <= obs_val)
            j = round(p0 * (len(pr) - 1))
            self._c[i, 1] = pr[j]

    def copy(self) -> 'PDFM':
        """
        Return a deep copy of the instance.

        Returns:
            PDFM: A deep copy holding the same matching curve.
        """
        return copy.deepcopy(self)

    def predict(self, pr0: np.ndarray) -> typing.Optional[np.ndarray]:
        """Apply PDFM correction to a forecast array using np.interp.

        Args:
            pr0: Raw forecast array.

        Returns:
            Corrected forecast array, or None if model is not fitted.
        """
        if self._c is None:
            return None
        shape = pr0.shape
        pr0 = pr0.flatten().astype(np.float32)
        out = np.zeros_like(pr0) + np.nan
        valid = ~np.isnan(pr0)
        out[valid] = np.interp(pr0[valid], self._c[:, 1], self._c[:, 0])
        return out.reshape(shape)


def _migrate_legacy_model(
    model: typing.Union[PDFM, typing.List[PDFM]]
) -> typing.Union[PDFM, typing.List[PDFM]]:
    """Migrate a pre-rename model instance after unpickling.

    Model files saved before the curve became private store it under the
    public name ``c``. joblib restores instance __dict__ verbatim, so the
    attribute is renamed here to keep predict() working.

    Args:
        model: Object freshly restored by joblib.

    Returns:
        The same object with the legacy attribute migrated.
    """
    if hasattr(model, 'c') and not hasattr(model, '_c'):
        model._c = model.c
        del model.c
    if isinstance(model, list):
        for item in model:
            _migrate_legacy_model(item)
    return model


def load_pdfm_file(
    path: typing.Union[str, pathlib.Path]
) -> typing.Union[PDFM, typing.List[PDFM]]:
    """Load a joblib file written by an older layout of this project.

    Model files saved before the class moved into src/pdf_model.py
    reference ``__main__.PDF`` in their pickles, and files saved before the
    rename reference the module-level name ``PDF``. Both aliases are
    installed before unpickling so those artifacts keep working, and the
    restored instance is migrated by _migrate_legacy_model.

    Args:
        path: Path to the joblib-serialized file.

    Returns:
        The unpickled PDFM or list[PDFM] object.
    """
    # Pickle resolves the class by name, so the historical names must
    # exist before the file is read. Aliasing them here avoids rewriting
    # the on-disk model artifacts.
    setattr(sys.modules[__name__], 'PDF', PDFM)
    main_mod = sys.modules.get('__main__')
    if main_mod is not None and not hasattr(main_mod, 'PDF'):
        main_mod.PDF = PDFM
    return _migrate_legacy_model(joblib.load(str(path)))


# ==================== Time-Lagged Ensemble (TLE) ====================


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
    """Apply equal-weight TLE to PDFM-corrected grids.

    For each target init_time m and lead n, the TLE value is the average of
    available PDFM-corrected forecasts along the anti-diagonal:

        VIS_TLE(m, n) = mean( VIS(m-i, n+i) )

    where i ranges from 0 to min(max_lookback, 24-n) and both the lagged
    init_time and lead exist in the input. Grid cells are processed
    independently; NaNs propagate only within their own sample.
    No model parameters are trained; the algorithm is fixed.

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
    reference_shape = next(iter(corrected_grids.values())).shape

    tle_grids: typing.Dict[str, np.ndarray] = {}

    for init_time, target_grid in corrected_grids.items():
        # All init times must share one grid, otherwise the anti-diagonal
        # samples cannot be compared cell by cell.
        if target_grid.shape != reference_shape:
            raise ValueError(
                f'Shape mismatch for {init_time}: {target_grid.shape} != '
                f'{reference_shape}'
            )

        tle_grid = target_grid.copy()

        for lead_idx, lead in enumerate(leads):
            # Samples share a valid time only along the anti-diagonal, and a
            # lead beyond 24h has no earlier init time to lag onto.
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
            # All-NaN cells emit a RuntimeWarning under errstate; the mask
            # below decides whether the PDFM value is restored.
            with np.errstate(invalid='ignore'):
                mean = np.nanmean(stacked, axis=0)

            if fallback_to_pdfm:
                pdfm_mask = np.isnan(mean)
                mean[pdfm_mask] = target_grid[lead_idx][pdfm_mask]

            tle_grid[lead_idx] = mean

        tle_grids[init_time] = tle_grid

    return tle_grids


# ==================== PDFM Cache ====================
# One .npy file per initialization time, holding the PDFM-corrected grid
# array of shape (n_lead, nlat, nlon). The cache lets a later run reuse
# earlier corrected grids instead of recomputing them from GRIB.
# An empty pdfm_cache_dir or the unfilled placeholder disables it, in
# which case every function below is a no-op.


def pdfm_cache_dir(cfg: typing.Dict) -> typing.Optional[pathlib.Path]:
    """Resolve the configured PDFM cache directory.

    Args:
        cfg: Merged configuration dictionary.

    Returns:
        pathlib.Path: Cache directory, or None when the cache is disabled.
    """
    cache_dir_cfg = cfg['operational']['paths']['pdfm_cache_dir']
    if not cache_dir_cfg or cache_dir_cfg == '<PDFM_CACHE_DIR>':
        return None
    return pathlib.Path(cache_dir_cfg)


def save_pdfm_cache(
    init_time: str,
    corrected: np.ndarray,
    cfg: typing.Dict,
) -> None:
    """Save PDFM-corrected grid to cache for future TLE runs.

    Args:
        init_time: YYYYMMDDHH string.
        corrected: Corrected grid array (n_lead, nlat, nlon).
        cfg: Merged configuration dictionary.
    """
    cache_dir = pdfm_cache_dir(cfg)
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'{init_time}.npy'
    np.save(str(cache_path), corrected)
    logger.info(f'Saved PDFM cache: {cache_path}')


def load_pdfm_cache(
    init_time: str,
    max_lookback: int,
    cfg: typing.Dict,
) -> typing.Dict[str, np.ndarray]:
    """Load cached PDFM-corrected grids from previous init times.

    Loads one .npy file per lagged init_time if it exists. Missing hours are
    silently skipped; TLE fallback will use PDFM values for those.

    Args:
        init_time: Target YYYYMMDDHH string.
        max_lookback: Maximum hours to look back.
        cfg: Merged configuration dictionary.

    Returns:
        Mapping lagged_init_time -> cached grid array.
    """
    cache_dir = pdfm_cache_dir(cfg)
    if cache_dir is None:
        return {}
    cached: typing.Dict[str, np.ndarray] = {}
    for i in range(1, max_lookback + 1):
        lag_init = utils.shift_init_time(init_time, -i)
        cache_path = cache_dir / f'{lag_init}.npy'
        if cache_path.exists():
            cached[lag_init] = np.load(str(cache_path))
            logger.info(f'Loaded PDFM cache: {cache_path}')
    return cached


def cleanup_pdfm_cache(
    latest_init_time: str,
    cfg: typing.Dict,
) -> None:
    """Remove PDFM cache files older than cache_keep_hours.

    Args:
        latest_init_time: Newest YYYYMMDDHH string processed in this run.
        cfg: Merged configuration dictionary.
    """
    keep_hours = int(
        cfg['operational']['inference']['tle']['cache_keep_hours']
    )
    cache_dir = pdfm_cache_dir(cfg)
    if cache_dir is None or not cache_dir.exists():
        return
    latest_dt = arrow.get(latest_init_time, 'YYYYMMDDHH')
    cutoff = latest_dt.shift(hours=-keep_hours)
    removed = 0
    for cache_path in cache_dir.glob('*.npy'):
        try:
            file_init = arrow.get(cache_path.stem, 'YYYYMMDDHH')
            if file_init < cutoff:
                cache_path.unlink()
                removed += 1
        except (ValueError, OSError):
            # Names that are not YYYYMMDDHH are unrelated leftovers.
            continue
    if removed:
        logger.info(f'Removed {removed} old PDFM cache files')