# -*- coding: utf-8 -*-
"""GRIB2 forecast loading and station-to-grid interpolation utilities.

Reads visibility forecast fields from GRIB2 files and interpolates them
onto a regular lat-lon grid using meteva station-to-grid IDW.

Founded in 2026-07-16
Modified in 2026-07-26
@author: yinlb
"""

import pathlib
import random
import shutil
import typing

import arrow
import numpy as np
import pandas as pd

try:
    import pygrib
except ImportError as e:
    raise ImportError(f'pygrib not installed: {e}')

from scipy.interpolate import griddata

try:
    from meteva import base as meb
except ImportError as e:
    raise ImportError(f'meteva not installed: {e}')


def _read_grib_messages(
    file_path: str,
    vis_idx: int,
    lat_idx: int,
    lon_idx: int
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read vis/lat/lon from GRIB2 file using configured message indices.

    Args:
        file_path: Path to the GRIB2 file.
        vis_idx: 0-based pygrib index of the visibility message.
        lat_idx: 0-based pygrib index of the latitude message.
        lon_idx: 0-based pygrib index of the longitude message.

    Returns:
        Tuple of (vis_values, lat_values, lon_values) as 2D numpy arrays.
    """
    with pygrib.open(file_path) as grb:
        msgs = list(grb)
        return (
            msgs[vis_idx].values,
            msgs[lat_idx].values,
            msgs[lon_idx].values,
        )


def _compute_extent(
    lat: np.ndarray,
    lon: np.ndarray,
    buffer_deg: float
) -> typing.Tuple[float, float, float, float]:
    """Compute target lat-lon extent inside source data boundary.

    Args:
        lat: 2D latitude array.
        lon: 2D longitude array.
        buffer_deg: Buffer in degrees to shrink inside the boundary.

    Returns:
        Tuple of (lon_min, lon_max, lat_min, lat_max).
    """
    lon_min = float(lon.min()) + buffer_deg
    lon_max = float(lon.max()) - buffer_deg
    lat_min = float(lat.min()) + buffer_deg
    lat_max = float(lat.max()) - buffer_deg
    return lon_min, lon_max, lat_min, lat_max


def _build_grid(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    resolution: float,
) -> typing.Any:
    """Build a meteva lat-lon grid with the given extent and resolution.

    Args:
        lon_min: Western boundary in degrees.
        lon_max: Eastern boundary in degrees.
        lat_min: Southern boundary in degrees.
        lat_max: Northern boundary in degrees.
        resolution: Grid resolution in degrees.

    Returns:
        meteva grid object (max bounds are exclusive).
    """
    return meb.grid(
        [lon_min, lon_max - resolution, resolution],
        [lat_min, lat_max - resolution, resolution]
    )


def set_forecast_time(
    grd: typing.Any,
    init_time: str,
    lead: int
) -> typing.Any:
    """Return a copy of grd with time/dtime set to init_time and lead.

    The MICAPS4 writer derives the header init time and forecast lead
    from the grid time/dtime coordinates, so they must be set here.

    Args:
        grd: meteva grid_data object.
        init_time: YYYYMMDDHH string (Beijing time).
        lead: Forecast lead hour.

    Returns:
        grid_data: A copy of grd with updated time/dtime coordinates.
    """
    init_dt = np.datetime64(arrow.get(init_time, 'YYYYMMDDHH').datetime)
    return grd.assign_coords(time=[init_dt], dtime=[lead])


def _interpolate_to_grid(sta, grid, interp_cfg: typing.Dict):
    """Apply configured station-to-grid interpolation method.

    Args:
        sta: meteva station data object.
        grid: Target meteva grid.
        interp_cfg: Interpolation configuration dictionary.

    Returns:
        meteva grid_data object with interpolated values.

    Raises:
        ValueError: If the interpolation method is unsupported.
    """
    method = str(interp_cfg['method']).lower()
    near_num = int(interp_cfg['near_num'])
    effect_r = float(interp_cfg['effect_radius_m'])

    if method == 'idw':
        return meb.interp_sg_idw(
            sta, grid=grid, nearNum=near_num, effectR=effect_r
        )
    elif method == 'idw_delta':
        return meb.interp_sg_idw_delta(
            sta, grid=grid, halfR=effect_r, nearNum=near_num
        )
    elif method == 'cressman':
        r_list = interp_cfg['cressman_r_list']
        return meb.interp_sg_cressman(
            sta, grid=grid, r_list=r_list, nearNum=near_num
        )
    else:
        raise ValueError(f'Unsupported interpolation method: {method}')


def interp_station_to_grid(
    values: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    interp_cfg: typing.Dict,
):
    """Interpolate station-like vis/lon/lat data to a regular lat-lon grid.

    Supports one-step IDW to the target resolution or two-step IDW-to-coarse
    followed by bilinear interpolation to the target resolution.

    Args:
        values: 2D visibility array.
        lat: 2D latitude array.
        lon: 2D longitude array.
        interp_cfg: Interpolation configuration dictionary from
            operational.yaml.

    Returns:
        meteva grid_data object containing the interpolated grid.
    """
    fine_res = float(interp_cfg['resolution'])
    buffer_deg = float(interp_cfg['buffer_deg'])

    lon_min, lon_max, lat_min, lat_max = _compute_extent(lat, lon, buffer_deg)

    n = values.size
    df = pd.DataFrame({
        'level': np.zeros(n, dtype=np.int32),
        'time': [pd.Timestamp.now()] * n,
        'dtime': np.zeros(n, dtype=np.int32),
        'id': np.arange(n, dtype=np.int32),
        'lon': lon.flatten(),
        'lat': lat.flatten(),
        'vis': values.flatten(),
    }).dropna(subset=['vis'])

    sta = meb.sta_data(df)
    method = str(interp_cfg['method']).lower()

    if method == 'linear':
        target_extent = interp_cfg['target_extent']
        lon_min = float(target_extent['lon_min'])
        lon_max = float(target_extent['lon_max'])
        lat_min = float(target_extent['lat_min'])
        lat_max = float(target_extent['lat_max'])
        grid_fine = _build_grid(lon_min, lon_max, lat_min, lat_max, fine_res)

        points = np.column_stack([lon.flatten(), lat.flatten()])
        vals = values.flatten()
        valid = ~np.isnan(vals)
        points = points[valid]
        vals = vals[valid]

        grid_x, grid_y = np.mgrid[
            lon_min:lon_max - fine_res + fine_res * 0.5:fine_res,
            lat_min:lat_max - fine_res + fine_res * 0.5:fine_res
        ]
        grid_z = griddata(points, vals, (grid_x, grid_y), method='linear')
        # griddata returns (n_lon, n_lat); transpose to (n_lat, n_lon)
        return meb.grid_data(grid_fine, grid_z.T)

    if not bool(interp_cfg['two_step']):
        grid_fine = _build_grid(lon_min, lon_max, lat_min, lat_max, fine_res)
        return _interpolate_to_grid(sta, grid_fine, interp_cfg)

    # Two-step: IDW to coarse grid, then bilinear to fine grid.
    # Expand the coarse domain by one coarse cell so the fine grid is fully
    # contained inside the coarse grid and linear interpolation does not fail.
    coarse_res = float(interp_cfg['coarse_resolution'])
    grid_coarse = _build_grid(
        lon_min - coarse_res,
        lon_max + coarse_res,
        lat_min - coarse_res,
        lat_max + coarse_res,
        coarse_res,
    )
    grd_coarse = _interpolate_to_grid(sta, grid_coarse, interp_cfg)

    grid_fine = _build_grid(lon_min, lon_max, lat_min, lat_max, fine_res)
    return meb.interp_gg_linear(grd_coarse, grid_fine)


def _resolve_forecast_path(
    init_time: str,
    cfg: typing.Dict,
    lead: int
) -> pathlib.Path:
    """Build forecast GRIB file path from configuration template.

    Args:
        init_time: YYYYMMDDHH string.
        cfg: Merged configuration dictionary.
        lead: Forecast lead hour.

    Returns:
        Resolved pathlib.Path to the forecast GRIB file.
    """
    forecast_cfg = cfg['operational']['forecast']
    template = forecast_cfg['filename_template']

    raw_dir = pathlib.Path(forecast_cfg['raw_data_dir'])
    filename = template.format(
        init_time=init_time,
        init_date=init_time[:8],
        init_year=init_time[:4],
        lead=lead,
    )
    return raw_dir / filename


def load_forecast_for_init(
    init_time: str,
    cfg: typing.Dict,
    lead: int
):
    """Load and interpolate forecast GRIB for one lead hour.

    If a temp_dir is configured, the GRIB file is first copied to that local
    directory, read from there, and then deleted. This avoids needing write
    permission on read-only network shares.

    Args:
        init_time: YYYYMMDDHH string.
        cfg: Merged configuration dictionary.
        lead: Forecast lead hour.

    Returns:
        Tuple of (interpolated_grid_data, resolved_grib_path); grid_data
        is None when the GRIB file does not exist.
    """
    file_path = _resolve_forecast_path(init_time, cfg, lead)
    if not file_path.exists():
        # Return None instead of raising so the caller can decide whether
        # to skip or generate fallback data without copying/interpolating.
        return None, file_path

    temp_dir_cfg = cfg['operational']['paths']['temp_dir']
    use_temp = bool(temp_dir_cfg and temp_dir_cfg != '<TEMP_DIR>')

    if use_temp:
        temp_dir = pathlib.Path(temp_dir_cfg)
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / file_path.name
        shutil.copy2(str(file_path), str(temp_path))
    else:
        temp_path = file_path

    try:
        msg_cfg = cfg['operational']['grib_message_indices']
        interp_cfg = cfg['operational']['interpolation']

        values, lat, lon = _read_grib_messages(
            str(temp_path),
            int(msg_cfg['visibility']),
            int(msg_cfg['latitude']),
            int(msg_cfg['longitude']),
        )

        grid = interp_station_to_grid(values, lat, lon, interp_cfg)
        grid = set_forecast_time(grid, init_time, lead)
    finally:
        if use_temp:
            try:
                temp_path.unlink()
            except OSError:
                pass

    return grid, file_path


def forecast_file_exists(init_time: str, cfg: typing.Dict, lead: int) -> bool:
    """Check whether the GRIB file for (init_time, lead) exists.

    Treats network/storage access errors as "not available" so the caller
    can skip the task instead of crashing.

    Args:
        init_time: YYYYMMDDHH string.
        cfg: Merged configuration dictionary.
        lead: Forecast lead hour.

    Returns:
        True if the file exists, False otherwise (including I/O errors).
    """
    try:
        return _resolve_forecast_path(init_time, cfg, lead).exists()
    except OSError:
        return False


def generate_fallback_grid(
    cfg: typing.Dict,
    init_time: str,
    lead: int
) -> typing.Any:
    """Create a synthetic grid by noising a random existing m4 file.

    Picks a random MICAPS4 file from the configured sample directory, or
    from all existing products under ops/products, and multiplies its values
    by Gaussian noise N(mean, std).

    Args:
        cfg: Merged configuration dictionary.
        init_time: YYYYMMDDHH string (Beijing time).
        lead: Forecast lead hour.

    Returns:
        meteva grid_data object containing the synthetic fallback field.

    Raises:
        FileNotFoundError: If no sample m4 file can be found.
    """
    fallback_cfg = cfg['operational']['synthetic']['fallback']
    sample_dir_cfg = fallback_cfg['sample_dir']
    noise_mean = float(fallback_cfg['noise_mean'])
    noise_std = float(fallback_cfg['noise_std'])

    def _is_lead_file(path: pathlib.Path) -> bool:
        """Check if filename ends with '.NNN', e.g. 2026070100.012."""
        if not path.is_file():
            return False
        suffix = path.suffix
        return len(suffix) == 4 and suffix[1:].isdigit()

    if sample_dir_cfg and sample_dir_cfg != '<SAMPLE_DIR>':
        search_root = pathlib.Path(sample_dir_cfg)
        candidates = [p for p in search_root.iterdir() if _is_lead_file(p)]
    else:
        # Default: scan all lead files under ops/products
        product_root = pathlib.Path(
            cfg['operational']['paths']['product_dir']
        )
        candidates = [p for p in product_root.rglob('*') if _is_lead_file(p)]

    if not candidates:
        raise FileNotFoundError(
            'No sample m4 files found for fallback data generation. '
            'Please set fallback.sample_dir or generate some products first.'
        )

    sample_path = random.choice(candidates)
    sample_grd = meb.read_griddata_from_micaps4(str(sample_path))
    sample_values = np.squeeze(sample_grd.values)

    noise = np.random.normal(
        loc=noise_mean,
        scale=noise_std,
        size=sample_values.shape,
    )
    fallback_values = sample_values * noise

    # Rebuild grid from sample coordinates so the fallback grid is consistent
    lons = sample_grd.coords['lon'].values
    lats = sample_grd.coords['lat'].values
    lon_step = float(lons[1] - lons[0])
    lat_step = float(lats[1] - lats[0])
    grid = meb.grid(
        [float(lons[0]), float(lons[-1]), lon_step],
        [float(lats[0]), float(lats[-1]), lat_step],
    )
    grd = meb.grid_data(grid, fallback_values)
    return set_forecast_time(grd, init_time, lead)


def load_forecast_task(
    init_time: str,
    cfg: typing.Dict,
    lead: int
) -> typing.Tuple[str, int, typing.Any, pathlib.Path]:
    """Load and interpolate one (init_time, lead) pair for parallel execution.

    Args:
        init_time: YYYYMMDDHH string.
        cfg: Merged configuration dictionary.
        lead: Forecast lead hour.

    Returns:
        Tuple of (init_time, lead, interpolated_grid_data, resolved_grib_path).
    """
    grid, file_path = load_forecast_for_init(init_time, cfg, lead)
    return init_time, lead, grid, file_path
