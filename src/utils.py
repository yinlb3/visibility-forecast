# -*- coding: utf-8 -*-
"""通用工具函数模块.

Founded in 2026-04-04
Modified in 2026-04-04
@author: yinlb
"""


def format_time(second: float, is_abbreviation: bool = False) -> str:
    """
    将秒数转换为人类可读的时间字符串.

    Args:
        second: 秒数, 浮点数;
        is_abbreviation: 是否使用缩写格式 (如 '12.5m'), 默认 False;

    Returns:
        格式化后的时间字符串, 例如 '43.5 seconds' 或 '12.5m';

    Raises:
        ValueError: 当 second 为负数时抛出异常.
    """
    if second < 0:
        raise ValueError('输入参数 \'second\' 不能为负数.')
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
