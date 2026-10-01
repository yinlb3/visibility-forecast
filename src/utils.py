# -*- coding: utf-8 -*-
"""General utility functions.

Founded in 2026-04-04
Modified in 2026-10-01
@author: yinlb, space-bunny
"""

import contextlib
import ctypes
import io
import os
import platform
import pathlib
import random
import typing

import arrow
import matplotlib as mpl
import numba
import numpy as np
import yaml

from matplotlib import figure
from matplotlib import font_manager as fm


mpl.use('Agg')


# ==================== Plot Style Utilities ====================


def find_chinese_font(
    fonts_cfg: typing.Optional[typing.Dict] = None
) -> typing.Optional[str]:
    """
    Register the first available Chinese font file of the configuration.

    Args:
        fonts_cfg: 'fonts' section of the config; it holds a 'chinese' list
            of font file paths. None or an empty list means no candidate.

    Returns:
        Family name of the registered font, or None when no file is usable.
    """
    for font_path in (fonts_cfg or {}).get('chinese', []):
        path = pathlib.Path(font_path)
        if not path.is_file():
            continue
        try:
            fm.fontManager.addfont(str(path))
            font_name = fm.FontProperties(fname=str(path)).get_name()
        except Exception as e:
            print(f'[find_chinese_font] Skipped {path}: {e}')
            continue
        print(f'[find_chinese_font] Chinese font: {font_name} ({path})')
        return font_name
    return None


def setup_plot_style(
    fonts_cfg: typing.Optional[typing.Dict] = None,
    chinese_first: bool = False
) -> None:
    """
    Apply the project-wide matplotlib fonts and tick settings.

    Single source of truth for rcParams, so plotting modules only need to
    import src.utils. matplotlib >= 3.11 no longer falls back inside the
    generic 'serif' alias list, hence the Chinese family name must be put
    into font.family explicitly.

    Args:
        fonts_cfg: 'fonts' section of the config, optional. A usable font
            file found there replaces the built-in SimSun.
        chinese_first: When True the Chinese font also renders latin text,
            otherwise Times New Roman keeps the latin characters.
    """
    mpl.rcParams['font.serif'] = ['Times New Roman', 'SimSun']
    mpl.rcParams['axes.unicode_minus'] = False
    chinese_name = find_chinese_font(fonts_cfg) or 'SimSun'
    # The Chinese family goes into font.family instead of relying on the
    # generic 'serif' alias: matplotlib >= 3.11 does not fall back inside
    # that alias list, and third-party libraries (e.g. meteva) overwrite
    # font.serif when they are imported.
    if chinese_first:
        mpl.rcParams['font.family'] = [chinese_name]
    else:
        mpl.rcParams['font.family'] = ['Times New Roman', chinese_name]


# ==================== Figure Save Utility ====================


def save_figure(
    fig: figure.Figure,
    base_path: pathlib.Path,
    cfg: typing.Dict,
    **save_kwargs
) -> None:
    """
    Save figure to multiple formats specified in config.

    Reads output_formats from cfg['draw']['plot']['output_formats'].
    Tries each format in order, logs error and continues if one fails.

    Args:
        fig: Matplotlib figure to save.
        base_path: Base file path without extension
            (e.g., Path('output/fig1')).
        cfg: Configuration dictionary containing plot settings.
        **save_kwargs: Additional kwargs passed to fig.savefig().

    Example:
        >>> save_figure(
        ...     fig, Path('figures/chart'), cfg, dpi=300, bbox_inches='tight'
        ... )
        [save_figure] Saved: figures/chart.png
        [save_figure] Saved: figures/chart.pdf
    """
    # Get formats from config
    formats = cfg['draw']['plot']['output_formats']

    # Ensure base_path is Path object
    base = pathlib.Path(base_path)

    # Create the parent directory so callers need not prepare it in advance
    base.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        # Clean format string (remove leading dot if present)
        fmt_clean = fmt.lstrip('.').lower()
        filepath = base.with_suffix(f'.{fmt_clean}')

        try:
            fig.savefig(str(filepath), format=fmt_clean, **save_kwargs)
        except Exception as e:
            print(f'[save_figure] Error saving {fmt_clean}: {e}')
            continue

        # Vector formats such as EPS fail silently far too often, so the
        # result is checked instead of trusting savefig alone.
        if not filepath.is_file() or filepath.stat().st_size == 0:
            print(f'[save_figure] EMPTY {fmt_clean}: {filepath}')
            continue
        size_kb = filepath.stat().st_size / 1024
        print(f'[save_figure] Saved: {filepath} ({size_kb:.0f} KB)')


def save_station_scatter(
    sta0: typing.Any,
    base_path: str,
    cmap: object,
    clevs: object,
    cfg: typing.Dict,
) -> None:
    """
    Save a station scatter map through meteva in every configured format.

    meteva prints a lot while plotting, so its output is redirected here
    instead of in each caller. Vector formats such as EPS can come out
    empty without raising, so the result is checked afterwards.

    Args:
        sta0: Station DataFrame that holds a 'data0' column to be mapped.
        base_path: Base file path without extension.
        cmap: Matplotlib colormap object.
        clevs: Discrete colour levels.
        cfg: Configuration dictionary containing plot settings.
    """
    from meteva import base as meb    # type: ignore

    formats = cfg['draw']['plot']['output_formats']
    for fmt in formats:
        fmt_clean = fmt.lstrip('.').lower()
        path = pathlib.Path(base_path).with_suffix(f'.{fmt_clean}')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                meb.tool.plot_tools.scatter_sta(
                    sta0=sta0.copy(), point_size=20,
                    map_extend=[108, 123, 24, 36], clevs=clevs, cmap=cmap,
                    extend='max', title=[''], save_path=str(path), dpi=800
                )
        except Exception as e:
            print(f'[save_station_scatter] Error saving {fmt_clean}: {e}')
            continue
        if not path.is_file() or path.stat().st_size == 0:
            print(f'[save_station_scatter] EMPTY {fmt_clean}: {path}')
            continue
        size_kb = path.stat().st_size / 1024
        print(f'[save_station_scatter] Saved: {path} ({size_kb:.0f} KB)')


def format_time(second: float, is_abbreviation: bool = False) -> str:
    """
    Convert seconds to human-readable time string.

    Args:
        second: Seconds, float;
        is_abbreviation: Use abbreviation (e.g., '12.5m'), default False;

    Returns:
        Formatted time string, e.g., '43.5 seconds' or '12.5m';

    Raises:
        ValueError: When second is negative.
    """
    if second < 0:
        raise ValueError('Parameter \'second\' cannot be negative.')
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


def compute_fake_timestamp(init_time: str, fake_cfg: typing.Dict) -> float:
    """Compute filesystem timestamp for an m4 file.

    If fake_timestamp is enabled, returns init_time + offset_hours + a
    weighted random seconds offset. Otherwise returns the current time.

    Args:
        init_time: YYYYMMDDHH string.
        fake_cfg: fake_timestamp configuration dictionary.

    Returns:
        Unix timestamp.
    """
    if not fake_cfg['enabled']:
        return arrow.now().timestamp()

    # Parse init_time as Asia/Shanghai (CST, UTC+8), which is the local time
    # implied by the GRIB filenames.
    init_dt = arrow.get(init_time, 'YYYYMMDDHH').replace(
        tzinfo='Asia/Shanghai'
    )
    base_time = init_dt.shift(
        hours=int(fake_cfg['offset_hours'])
    ).floor('hour')

    sec_min = int(fake_cfg['random_seconds_min'])
    sec_max = int(fake_cfg['random_seconds_max'])
    # 04:00 boundary in seconds from the hour start
    split_seconds = 240
    weight = float(fake_cfg['first_half_weight'])

    if random.random() < weight:
        seconds = random.randint(sec_min, split_seconds)
    else:
        seconds = random.randint(split_seconds, sec_max)

    return base_time.shift(seconds=seconds).timestamp()


def utc_to_beijing(init_time: str) -> str:
    """Convert a YYYYMMDDHH string from UTC to Beijing time (CST, UTC+8).

    Args:
        init_time: YYYYMMDDHH string in UTC.

    Returns:
        YYYYMMDDHH string in Asia/Shanghai (Beijing) time.
    """
    init_dt = arrow.get(init_time, 'YYYYMMDDHH').replace(tzinfo='UTC')
    bj_dt = init_dt.to('Asia/Shanghai')
    return bj_dt.format('YYYYMMDDHH')


def shift_init_time(init_time: str, hours: int) -> str:
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


def mask_missing(
    arr: np.ndarray,
    missing_value: float = 999990.0,
) -> np.ndarray:
    """Replace missing-value markers with np.nan.

    Returns a copy so the caller's array is not mutated.

    Args:
        arr: Input array.
        missing_value: Values >= this threshold are treated as missing
            (np.nan).

    Returns:
        Array with missing values replaced by np.nan.
    """
    arr = arr.copy()
    arr[arr >= missing_value] = np.nan
    return arr


# ==================== Visibility Data Loader ====================


def load_vis_data(
    ob: np.ndarray,
    pr: np.ndarray,
    n_val_days: int = 365,
    missing_value: float = 999990.0,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mask missing values, filter stations, split train/validation.

    Processes already-loaded observation and forecast numpy arrays:
    replaces missing-value markers with np.nan, keeps only stations with
    at least one valid obs/forecast pair, and splits the first (time) axis
    into training and validation sets.

    Args:
        ob: Observation array with missing-value markers.
        pr: Forecast array with missing-value markers.
        n_val_days: Number of trailing days to use as validation set.
        missing_value: Values >= this threshold are treated as missing
            (np.nan).

    Returns:
        Tuple of (train_ob, train_pr, val_ob, val_pr, valid_stations).
    """
    # 1. Work on copies to avoid mutating caller's arrays
    ob = ob.copy()
    pr = pr.copy()

    # 2. Mask missing values
    ob[ob >= missing_value] = np.nan
    pr[pr >= missing_value] = np.nan

    # 3. Filter stations with at least one valid obs/forecast pair
    n_stations = ob.shape[-1]
    valid_stations = np.zeros(n_stations, dtype=np.bool_)
    for i in range(n_stations):
        if np.sum(~np.isnan(ob[..., i]) & ~np.isnan(pr[..., i])) > 0:
            valid_stations[i] = True
    print(f'Valid stations: {np.sum(valid_stations)}')

    ob = ob[..., valid_stations]
    pr = pr[..., valid_stations]

    # 4. Split into train/validation along the first (time) axis
    train_ob = ob[:-n_val_days, ...]
    train_pr = pr[:-n_val_days, ...]
    val_ob = ob[-n_val_days:, ...]
    val_pr = pr[-n_val_days:, ...]

    return train_ob, train_pr, val_ob, val_pr, valid_stations


# ==================== Temporal Lead Ensemble Utility ====================


@numba.njit
def _apply_tle_equal_weight_numba(
    arr: np.ndarray,
    n_hours: int,
    out: np.ndarray,
) -> None:
    """Numba-compiled equal-weight TLE averaging for access.py.

    For each target time step i and lead time j, averages
    arr[i + j - ii, ii, :] over ii in [0, n_hours - 1], matching the
    original access.py TLE loop.
    NaN values are skipped so that sparse forecast arrays still produce output.

    Args:
        arr: Forecast array of shape (n_times, n_hours, n_stations).
        n_hours: Number of lead times to average over.
        out: Pre-allocated output array of the same shape as arr.

    Returns:
        None. Results are written into out in place.
    """
    n_times = arr.shape[0]
    n_stations = arr.shape[2]
    for i in range(n_times):
        for j in range(n_hours):
            for sta_idx in range(n_stations):
                n_valid = 0
                vis_sum = np.float32(0.0)
                for ii in range(n_hours):
                    idx = i + j - ii
                    if idx < 0 or idx >= n_times:
                        continue
                    value = arr[idx, ii, sta_idx]
                    if np.isnan(value):
                        continue
                    n_valid += 1
                    vis_sum += value
                if n_valid > 0:
                    out[i, j, sta_idx] = vis_sum / np.float32(n_valid)


def apply_tle_equal_weight(
    arr: np.ndarray,
    n_hours: int = 24,
) -> np.ndarray:
    """Apply equal-weight temporal lead ensemble averaging for access.py.

    For each target time step i and lead time j, averages
    arr[i + j - ii, ii, :] over ii in [0, n_hours - 1]. NaN values are
    skipped.

    Args:
        arr: Input array shaped (n_times, n_hours, n_stations).
        n_hours: Number of lead/init hours.

    Returns:
        Array of same shape as arr with TLE-averaged values.
    """
    out = np.zeros_like(arr) + np.nan
    _apply_tle_equal_weight_numba(arr, n_hours, out)
    return out


# ==================== System Memory Utility ====================


def get_available_memory_gb() -> float:
    """Return available physical memory in GB using only the standard library.

    Returns:
        Available physical memory in gigabytes. Returns 4.0 as a conservative
        fallback if the platform cannot be determined.
    """
    system = platform.system().lower()
    try:
        if system == 'windows':
            # ctypes structure mirroring the Win32 MEMORYSTATUSEX layout.
            # It carries no state and no behaviour, so the usual copy()
            # method does not apply.
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullAvailPhys / (1024 ** 3)
        elif system == 'linux':
            avail_pages = os.sysconf('SC_AVPHYS_PAGES')
            page_size = os.sysconf('SC_PAGE_SIZE')
            return avail_pages * page_size / (1024 ** 3)
    except Exception:
        pass
    return 4.0


# ==================== Configuration Loader ====================


def _deep_merge(
    base: typing.Dict,
    override: typing.Dict
) -> typing.Dict:
    """
    Recursively merge two dictionaries.

    Override values take precedence. Nested dicts are merged,
    non-dict values are replaced.

    Args:
        base: Base configuration dict.
        override: Override configuration dict.

    Returns:
        Merged configuration dict.
    """
    # Recursively merge: override takes precedence, nested dicts merged deeply
    result: typing.Dict = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: pathlib.Path) -> typing.Dict:
    """
    Load YAML file if exists.

    Tries UTF-8 first, then common Chinese encodings, and finally latin-1
    with replacement as a last resort. This prevents crashes when a config
    file has been edited with a non-UTF-8 editor.

    Args:
        path: pathlib.Path to YAML file.

    Returns:
        Loaded dict or empty dict if file not found or unreadable.
    """
    if not path.exists():
        return {}
    raw = path.read_bytes()
    for enc in ('utf-8', 'gbk', 'gb18030', 'latin-1'):
        try:
            text = raw.decode(enc)
            return yaml.safe_load(text) or {}
        except (UnicodeDecodeError, yaml.YAMLError):
            continue
    print(f'Warning: could not decode YAML {path}, using empty config')
    return {}


def load_config(operational: bool = False) -> typing.Dict:
    """
    Load configuration with platform-specific local override.

    Loads config/config.yaml first, then merges with
    config/config.local.{platform}.yaml if it exists.
    Local config overrides base config values.

    When operational is True, additionally merges config/operational.yaml
    on top, allowing business-specific settings to override research defaults.

    Args:
        operational: If True, also load and merge config/operational.yaml.

    Returns:
        Merged configuration dictionary.
    """
    # Get config directory (project_root/config/)
    config_dir = pathlib.Path(__file__).parent.parent / 'config'

    # Load base configuration
    base_config = _load_yaml(config_dir / 'config.yaml')

    # Determine platform-specific local config filename
    system = platform.system().lower()
    if system == 'windows':
        local_file = 'config.local.windows.yaml'
    elif system == 'linux':
        local_file = 'config.local.linux.yaml'
    else:
        local_file = 'config.local.yaml'

    # Load and merge operational configuration if requested
    # Operational defaults are applied before local overrides so that
    # platform-specific local configs can override operational placeholders.
    cfg = base_config
    if operational:
        op_config = _load_yaml(config_dir / 'operational.yaml')
        if op_config:
            cfg = _deep_merge(cfg, op_config)

    # Load and merge local configuration if exists
    local_config = _load_yaml(config_dir / local_file)
    if local_config:
        cfg = _deep_merge(cfg, local_config)

    return cfg


def parse_time_args(argv: typing.List[str]) -> typing.Tuple[str, ...]:
    """
    Parse and validate 0/1/2 YYYYMMDDHH command-line arguments.

    Args:
        argv: sys.argv list.

    Returns:
        Tuple of validated time strings.

    Raises:
        SystemExit: If argument count or format is invalid.
    """
    usage = 'Usage: python <script>.py [YYYYMMDDHH] [YYYYMMDDHH]'
    if len(argv) < 2:
        return tuple()
    if len(argv) > 3:
        print(usage)
        raise SystemExit(1)

    for arg in argv[1:]:
        if len(arg) != 10:
            print(f'Invalid time format: {arg}. Expected YYYYMMDDHH.')
            raise SystemExit(1)
        try:
            arrow.get(arg, 'YYYYMMDDHH')
        except arrow.ParserError:
            print(f'Invalid datetime: {arg}')
            raise SystemExit(1)

    return tuple(argv[1:])


def resolve_init_times(
    args: typing.Tuple[str, ...],
    cfg: typing.Dict
) -> typing.List[str]:
    """
    Resolve initialization time list from command-line arguments.

    0 args: real-time mode, pick latest available init hour.
    1 arg:  single init time.
    2 args: backfill mode, generate times from start to end.

    Args:
        args: Parsed command-line arguments.
        cfg: Merged configuration dictionary.

    Returns:
        List of YYYYMMDDHH strings.
    """
    runtime = cfg['operational']['runtime']
    init_hours = sorted(runtime['init_hours'])
    step = runtime['backfill_step_hours']

    if len(args) == 0:
        # Default to 12 hours before current time, floored to the hour.
        now = arrow.now()
        init_time = now.shift(hours=-12).floor('hour')
        return [init_time.format('YYYYMMDDHH')]

    if len(args) == 1:
        return [args[0]]

    start = arrow.get(args[0], 'YYYYMMDDHH')
    end = arrow.get(args[1], 'YYYYMMDDHH')
    if end < start:
        print('End time must be equal to or after start time.')
        raise SystemExit(1)

    times = list()
    current = start
    while current <= end:
        if current.hour in init_hours:
            times.append(current.format('YYYYMMDDHH'))
        current = current.shift(hours=step)
    return times


def _validate_config(cfg: typing.Dict) -> None:
    """
    Validate required configuration keys exist.

    Raises informative KeyError if any required key is missing.

    Args:
        cfg: Loaded configuration dictionary.

    Raises:
        KeyError: If required configuration key is missing.
    """
    required_keys = [
        'paths.data_dir',
        'paths.output_dir',
        'paths.cache_dir',
        'paths.wind_data_dir',
        'draw.stages',
        'draw.plot.output_formats',
        # Plot keys used by src/p3_2_init_lead.py: without them the stage
        # fails only after the long 3.1 calculation has already finished.
        'draw.plot.hour_access_heatmaps.improvement_vmin',
        'draw.plot.hour_access_heatmaps.improvement_vmax',
        'draw.plot.ts_comparison_bars.color_improvement',
        'draw.plot.ts_comparison_bars.ft_xlim',
        'draw.plot.mre_violins.improvement_figsize',
        'draw.plot.mre_violins.improvement_color',
        'draw.regions.provinces',
        'draw.regions.mlyr_provinces',
        'tl.input_files.observation_npy',
        'tl.input_files.forecast_npy',
        'tl.input_files.pdfm_input_npy',
        'tl.input_files.station_index_npy',
        'tl.input_files.pre_existing_tle',
        'tl.input_files.existing_pdfm_tle',
        'tl.output_files.tle_template',
        'tl.output_files.pdfm_tle_template',
        'tl.params.n_hours',
        'tl.params.n_raw_stations',
        'tl.params.n_subset_stations',
        'tl.params.val_days',
        'tl.params.tle_start_idx',
        'tl.params.tle_end_idx',
        'tl.params.tle_window_hours',
        'tl.params.visibility_cap',
        'access.input_files.station_csv',
        'access.input_files.observation_npy',
        'access.input_files.forecast_npy',
        'access.params.n_raw_stations',
        'access.params.n_east_sta',
        'access.params.n_mlyr_stations',
        'access.params.val_days',
        'access.params.n_hours',
        'access.params.visibility_cap',
        'access.params.save_pdfm_models',
        'access.params.run_scheme_0',
        'access.params.run_scheme_1',
        'access.params.run_scheme_2',
        'access.params.run_region',
        'access.output_files.pdfm_models.pdfm0',
        'access.output_files.pdfm_models.pdfm0_mlyr',
        'access.output_files.pdfm_models.pdfm1',
        'access.output_files.pdfm_models.pdfm1_mlyr',
        'access.output_files.pdfm_models.pdfm2',
        'access.output_files.pdfm_models.pdfm2_mlyr',
        'access.output_files.corrected_forecasts.pred0',
        'access.output_files.corrected_forecasts.pred0_mlyr',
        'access.output_files.corrected_forecasts.pred1',
        'access.output_files.corrected_forecasts.pred1_mlyr',
        'access.output_files.corrected_forecasts.pred2',
        'access.output_files.corrected_forecasts.pdfm',
        'access.output_files.corrected_forecasts.tle',
        'access.output_files.corrected_forecasts.pred2_mlyr',
        'access.output_files.corrected_forecasts.pdfm_mlyr',
        'access.output_files.corrected_forecasts.tle_mlyr',
        'huanghua.input_files.base_dir',
        'huanghua.input_files.filename_templates',
        'huanghua.output_files.meteogram_npy',
        'huanghua.params.start_year',
        'huanghua.params.end_year',
        'huanghua.params.n_total_days',
        'huanghua.params.n_hours',
        'huanghua.params.n_variables',
        'huanghua.params.sheet_names',
        'huanghua.params.continuous_sheets',
    ]

    # Traverse dotted key paths (e.g., 'paths.data_dir') and verify existence
    for key_path in required_keys:
        parts = key_path.split('.')
        current = cfg
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                raise KeyError(
                    f'Missing required config: \'{key_path}\'. '
                    f'Please check config/config.yaml and '
                    f'config.local.{platform.system().lower()}.yaml'
                )
            current = current[part]


# Load and validate configuration at module import time
_CFG: typing.Dict = load_config()
_validate_config(_CFG)

# Global config instance (validated)
CFG: typing.Dict = _CFG

# Apply the shared plot style once, so importing this module is enough for
# every plotting module.
setup_plot_style()
