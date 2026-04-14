# -*- coding: utf-8 -*-
"""General utility functions.

Founded in 2026-04-04
Modified in 2026-04-07
@author: yinlb
"""

import platform
import pathlib
import typing

from matplotlib import figure
import yaml


# ==================== Figure Save Utility ====================


def save_figure(
    fig: figure.Figure,
    base_path: pathlib.Path,
    cfg: typing.Dict[str, typing.Any],
    **save_kwargs
) -> None:
    """
    Save figure to multiple formats specified in config.

    Reads output_formats from cfg['plot']['output_formats'].
    Tries each format in order, logs error and continues if one fails.

    Args:
        fig: Matplotlib figure to save.
        base_path: Base file path without extension (e.g., Path('output/fig1')).
        cfg: Configuration dictionary containing plot settings.
        **save_kwargs: Additional kwargs passed to fig.savefig().

    Example:
        >>> save_figure(fig, Path('figures/chart'), cfg, dpi=300, bbox_inches='tight')
        [save_figure] Saved: figures/chart.png
        [save_figure] Saved: figures/chart.pdf
    """
    # Get formats from config, default to png only
    formats = cfg.get('plot', {}).get('output_formats', ['png'])
    if not formats:
        formats = ['png']

    # Ensure base_path is Path object
    base = pathlib.Path(base_path)

    for fmt in formats:
        # Clean format string (remove leading dot if present)
        fmt_clean = fmt.lstrip('.').lower()
        filepath = base.with_suffix(f'.{fmt_clean}')

        try:
            fig.savefig(str(filepath), format=fmt_clean, **save_kwargs)
            print(f'[save_figure] Saved: {filepath}')
        except Exception as e:
            print(f'[save_figure] Error saving {fmt_clean}: {e}')
            continue


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


# ==================== Configuration Loader ====================


def _deep_merge(
    base: typing.Dict[str, typing.Any],
    override: typing.Dict[str, typing.Any]
) -> typing.Dict[str, typing.Any]:
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
    result: typing.Dict[str, typing.Any] = base.copy()
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


def _load_yaml(path: pathlib.Path) -> typing.Dict[str, typing.Any]:
    """
    Load YAML file if exists.

    Args:
        path: pathlib.Path to YAML file.

    Returns:
        Loaded dict or empty dict if file not found.
    """
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def load_config() -> typing.Dict[str, typing.Any]:
    """
    Load configuration with platform-specific local override.

    Loads config/config.yaml first, then merges with
    config/config.local.{platform}.yaml if it exists.
    Local config overrides base config values.

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

    # Load and merge local configuration if exists
    local_config = _load_yaml(config_dir / local_file)
    if local_config:
        return _deep_merge(base_config, local_config)

    return base_config


def _validate_config(cfg: typing.Dict[str, typing.Any]) -> None:
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
        'stages',
        'regions.provinces',
        'regions.mlyr_provinces',
    ]

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


# Load and validate configuration
_CFG: typing.Dict[str, typing.Any] = load_config()
_validate_config(_CFG)

# Global config instance (validated)
CFG: typing.Dict[str, typing.Any] = _CFG
