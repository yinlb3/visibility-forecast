# -*- coding: utf-8 -*-
"""Nearest-station map build/load utilities for operational inference.

The map is a one-time artifact: each grid cell stores the id of its
nearest station. Build it with build_near_map.py whenever the grid spec
or station table changes; inference.py only loads and validates it.

Founded in 2026-07-26
Modified in 2026-07-26
@author: yinlb
"""

import json
import pathlib
import typing

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src import logger, p1_config_data as p1


def load_stations(cfg: typing.Dict) -> pd.DataFrame:
    """Load station table filtered to the configured region.

    Reads the configured station CSV and filters by region provinces
    with the same sort/filter semantics as p1.read_sta.

    Args:
        cfg: Merged configuration dictionary.

    Returns:
        pd.DataFrame: Station table including id/lon/lat columns.
    """
    region = cfg['operational']['runtime']['region']
    region_cfg = cfg['draw']['regions']
    if region == 'mlyr':
        provinces = tuple(region_cfg['mlyr_provinces'])
    else:
        provinces = tuple(region_cfg['provinces'])

    sta_path = cfg['operational']['inference']['station_csv']
    sta, _ = p1.read_sta(sta_path=sta_path, provinces=provinces)
    sta['id'] = sta['id'].astype(int)
    return sta


def grid_spec_from_config(cfg: typing.Dict) -> typing.Dict:
    """Build grid spec from the interpolation target extent config.

    Args:
        cfg: Merged configuration dictionary.

    Returns:
        Grid spec dict (slon/dlon/nlon/slat/dlat/nlat).
    """
    interp_cfg = cfg['operational']['interpolation']
    extent = interp_cfg['target_extent']
    res = float(interp_cfg['resolution'])
    lon_min = float(extent['lon_min'])
    lon_max = float(extent['lon_max'])
    lat_min = float(extent['lat_min'])
    lat_max = float(extent['lat_max'])
    return {
        'slon': lon_min,
        'dlon': res,
        'nlon': int(round((lon_max - lon_min) / res)),
        'slat': lat_min,
        'dlat': res,
        'nlat': int(round((lat_max - lat_min) / res)),
    }


def grid_spec_from_meta(meta_path: str) -> typing.Dict:
    """Read grid spec from a preprocess meta.json file.

    Args:
        meta_path: Path to the intermediate meta.json.

    Returns:
        Grid spec dict (slon/dlon/nlon/slat/dlat/nlat).

    Raises:
        FileNotFoundError: If the meta.json file does not exist.
        KeyError: If meta.json lacks the grid spec.
    """
    path = pathlib.Path(meta_path)
    if not path.exists():
        raise FileNotFoundError(f'meta.json not found: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    if 'grid' not in meta:
        raise KeyError(f'meta.json has no grid spec: {path}')
    return meta['grid']


def build_near_id_map(
    grid: typing.Dict,
    sta: pd.DataFrame
) -> np.ndarray:
    """Build a grid-cell map of nearest station ids via cKDTree.

    Args:
        grid: Grid spec dict (slon/dlon/nlon/slat/dlat/nlat).
        sta: Station table with id/lon/lat columns.

    Returns:
        Array of shape (nlat, nlon) with nearest station id per cell.
    """
    lons = grid['slon'] + np.arange(grid['nlon']) * grid['dlon']
    lats = grid['slat'] + np.arange(grid['nlat']) * grid['dlat']
    lon2d, lat2d = np.meshgrid(lons, lats)
    sta_xy = np.column_stack([sta['lon'].values, sta['lat'].values])
    _, idx = cKDTree(sta_xy).query(
        np.column_stack([lon2d.ravel(), lat2d.ravel()])
    )
    return sta['id'].values[idx].reshape(grid['nlat'], grid['nlon'])


def save_near_map(
    map_path: str,
    near_id: np.ndarray,
    grid: typing.Dict,
    sta: pd.DataFrame
) -> None:
    """Save nearest-station map with embedded spec and station ids.

    Args:
        map_path: Destination .npz file path.
        near_id: Array (nlat, nlon) of nearest station ids.
        grid: Grid spec used to build the map.
        sta: Station table used to build the map.
    """
    path = pathlib.Path(map_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(path),
        near_id=near_id,
        sta_ids=sta['id'].values,
        slon=grid['slon'], dlon=grid['dlon'], nlon=grid['nlon'],
        slat=grid['slat'], dlat=grid['dlat'], nlat=grid['nlat'],
    )
    logger.info(f'Saved nearest-station map: {path}')


def load_near_map(
    map_path: str,
    grid: typing.Dict,
    sta: pd.DataFrame
) -> np.ndarray:
    """Load and validate the precomputed nearest-station map.

    Args:
        map_path: Path to the .npz artifact.
        grid: Grid spec from the current intermediate meta.json.
        sta: Current station table.

    Returns:
        Array of shape (nlat, nlon) with nearest station ids.

    Raises:
        FileNotFoundError: If the artifact is missing.
        ValueError: If the embedded spec/stations do not match.
    """
    path = pathlib.Path(map_path)
    if not path.exists():
        raise FileNotFoundError(
            f'Nearest-station map not found: {path}. '
            f'Please run build_near_map.py first.'
        )
    bundle = np.load(str(path))
    same_grid = all(
        np.isclose(
            float(bundle[key]), float(grid[key]), rtol=1e-9, atol=1e-9
        )
        for key in ('slon', 'dlon', 'slat', 'dlat')
    )
    same_grid = same_grid and int(bundle['nlon']) == int(grid['nlon'])
    same_grid = same_grid and int(bundle['nlat']) == int(grid['nlat'])
    same_sta = np.array_equal(
        np.sort(bundle['sta_ids']), np.sort(sta['id'].values)
    )
    if not (same_grid and same_sta):
        raise ValueError(
            f'Nearest-station map is stale or mismatched: {path}. '
            f'Please re-run build_near_map.py.'
        )
    logger.info(f'Loaded nearest-station map: {path}')
    return bundle['near_id']
