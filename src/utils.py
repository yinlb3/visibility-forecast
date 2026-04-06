# -*- coding: utf-8 -*-
"""General utility functions.

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""


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
        raise ValueError("Parameter 'second' cannot be negative.")
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
