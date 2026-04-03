#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对角线颜色网格图绘制程序
用于展示斜线方向的颜色分布规律
修复了中文字体和内存问题
"""

import arrow
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fm

# ==================== 全局配置参数 ====================

GRID_SIZE = 24  # 网格大小 24x24
FIG_WIDTH = 12  # 图片宽度（英寸）
FIG_HEIGHT = 14  # 图片高度（英寸）
COLOR_MAP_NAME = 'rainbow'  # 颜色映射名称
OUTPUT_DPI = 800
CHINESE_FONT_PATH = None  # 中文字体路径（自动查找）


# ==================== 字体配置函数 ====================


def setup_chinese_font():
    """
    配置中文字体支持

    自动查找系统可用的中文字体
    解决 DejaVu Sans 不支持中文的问题
    """
    global CHINESE_FONT_PATH

    # 常见中文字体路径（Windows）
    possible_fonts = [
        r'C:\Windows\Fonts\msyh.ttc',  # 微软雅黑
        r'C:\Windows\Fonts\msyhbd.ttc',  # 微软雅黑粗体
        r'C:\Windows\Fonts\simhei.ttf',  # SimHei
        r'C:\Windows\Fonts\STSONG.TTC',  # 宋体
        r'C:\Windows\Fonts\ARIALUNI.TTF',  # Arial Unicode MS
    ]

    # 尝试找到可用的中文字体
    for font_path in possible_fonts:
        try:
            # 测试字体是否可用
            prop = fm.FontProperties(fname=font_path)
            text = u'测试中文'
            # 尝试渲染测试文本
            fig_test = plt.figure()
            ax_test = fig_test.add_subplot(111)
            ax_test.text(0.5, 0.5, text, fontproperties=prop)
            plt.close(fig_test)
            CHINESE_FONT_PATH = font_path
            print(f'找到并验证可用的中文字体: {font_path}')
            break
        except Exception as e:
            continue

    if CHINESE_FONT_PATH is None:
        print('警告: 未找到可用的中文字体，将使用英文标题')


def set_matplotlib_params():
    """
    设置 matplotlib 参数

    包括字体、后端等配置
    避免字体渲染问题
    """
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    # 如果找到中文字体，则注册并使用
    if CHINESE_FONT_PATH:
        # 注册字体
        font_prop = fm.FontProperties(fname=CHINESE_FONT_PATH)
        # 设置全局字体
        plt.rcParams['font.family'] = font_prop.get_name()
        print(f'已设置字体为: {font_prop.get_name()}')


# ==================== 核心功能函数 ====================


def create_diagonal_matrix(size):
    """
    创建对角线索引矩阵

    每条从左下到右上的斜线具有相同的索引值
    用于后续映射颜色

    Args:
        size: 网格大小

    Returns:
        diagonal_idx: 对角线索引矩阵
    """
    diagonal_idx = np.zeros((size, size), dtype=int)

    for i in range(size):
        for j in range(size):
            # 计算对角线索引 (左下到右上方向)
            diagonal_idx[i, j] = i + j

    return diagonal_idx


def generate_color_map(num_colors):
    """
    生成指定数量的颜色列表

    使用 matplotlib 内置颜色映射
    可根据需要自定义颜色序列

    Args:
        num_colors: 需要的颜色数量

    Returns:
        colors: 颜色列表
    """
    base_cmap = plt.get_cmap(COLOR_MAP_NAME)
    colors = [base_cmap(i / max(1, num_colors - 1)) for i in range(num_colors)]

    return colors


def draw_grid(ax, matrix, colors, size):
    """
    绘制彩色网格

    每个格子根据对角线索引填充对应颜色
    添加黑色网格线便于区分

    Args:
        ax: matplotlib 轴对象
        matrix: 对角线索引矩阵
        colors: 颜色列表
        size: 网格大小
    """
    # 分批绘制，减少内存占用
    batch_size = 8  # 每次绘制8行

    for start_row in range(0, size, batch_size):
        end_row = min(start_row + batch_size, size)

        for i in range(start_row, end_row):
            for j in range(size):
                color_idx = matrix[i, j] % len(colors)
                # 注意：矩阵行索引与绘图 y 轴方向相反
                rect = Rectangle(
                    (j, size - 1 - i), 1, 1,
                    facecolor=colors[color_idx],
                    edgecolor='black',
                    linewidth=0.3  # 减小线宽节省内存
                )
                ax.add_patch(rect)

        # 强制垃圾回收
        import gc
        gc.collect()

    # 设置坐标轴范围
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_color_bar(ax, colors, size):
    """
    绘制底部颜色条

    显示每种颜色对应的索引
    与上方网格颜色一一对应

    Args:
        ax: matplotlib 轴对象
        colors: 颜色列表
        size: 网格大小
    """
    num_colors = min(len(colors), size)  # 限制颜色数量

    # 分批绘制颜色条
    for i in range(num_colors):
        rect = Rectangle(
            (i, 0), 1, 1,
            facecolor=colors[i],
            edgecolor='black',
            linewidth=0.3
        )
        ax.add_patch(rect)

    # 设置坐标轴
    ax.set_xlim(0, num_colors)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def add_number_labels(ax, size, position='top'):
    """
    添加数字标签

    在网格顶部或底部添加 1-24 的数字
    便于定位每个格子的位置

    Args:
        ax: matplotlib 轴对象
        size: 网格大小
        position: 标签位置 ('top' 或 'bottom')
    """
    # 减少标签密度以节省内存
    label_step = max(1, size // 12)  # 每隔几个位置显示一个标签

    for i in range(0, size, label_step):
        x_pos = i + 0.5
        if position == 'top':
            y_pos = size + 0.5
            va = 'bottom'
        else:
            y_pos = -0.5
            va = 'top'

        # 使用较小的字体
        ax.text(
            x_pos, y_pos, str(i + 1),
            ha='center', va=va,
            fontsize=6,  # 减小字体大小
            color='black'
        )


def add_arrow_annotation(ax, size, target_col):
    """
    添加箭头标注

    指示网格与颜色条的对应关系

    Args:
        ax: matplotlib 轴对象
        size: 网格大小
        target_col: 目标列位置
    """
    ax.annotate(
        '',
        xy=(target_col, 1.2),
        xytext=(target_col, 2.5),
        arrowprops=dict(
            arrowstyle='->',
            color='darkblue',
            linewidth=1.5
        )
    )


# ==================== 主程序入口 ====================


def main():
    """
    主函数：执行绘图流程

    1. 设置中文字体
    2. 创建对角线矩阵
    3. 生成颜色映射
    4. 绘制网格和颜色条
    5. 添加标注和标签
    6. 保存输出文件
    """
    # 设置字体和参数
    setup_chinese_font()
    set_matplotlib_params()

    # 创建图形和子图
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # 定义三个子图区域：顶部标签、网格、颜色条
    ax_grid = fig.add_axes([0.1, 0.15, 0.8, 0.75])
    ax_colorbar = fig.add_axes([0.1, 0.05, 0.8, 0.08])
    ax_labels = fig.add_axes([0.1, 0.15, 0.8, 0.75])

    # 生成对角线矩阵和颜色
    diagonal_matrix = create_diagonal_matrix(GRID_SIZE)
    color_list = generate_color_map(GRID_SIZE * 2 - 1)

    # 绘制主要元素
    draw_grid(ax_grid, diagonal_matrix, color_list, GRID_SIZE)
    draw_color_bar(ax_colorbar, color_list, GRID_SIZE * 2 - 1)

    # 添加数字标签
    add_number_labels(ax_labels, GRID_SIZE, position='top')
    add_number_labels(ax_labels, GRID_SIZE, position='bottom')

    # 添加箭头标注（示例：指向第 12-13 列）
    add_arrow_annotation(ax_labels, GRID_SIZE, target_col=12.5)

    # 根据是否找到中文字体设置标题
    if CHINESE_FONT_PATH:
        # 使用字体属性直接指定字体
        font_prop = fm.FontProperties(fname=CHINESE_FONT_PATH, size=14)
        fig.suptitle('对角线颜色分布示意图', fontproperties=font_prop, y=0.98)
    else:
        fig.suptitle('Diagonal Color Distribution Schematic', fontsize=14, y=0.98)

    # 保存文件（使用较低的DPI以节省内存）
    output_path_pdf = r'D:\Project\vis\图\diagonal_grid_fixed.pdf'
    plt.savefig(output_path_pdf, dpi=OUTPUT_DPI, bbox_inches='tight',
                format='pdf', facecolor='white', edgecolor='none')
    print(f'PDF图形已保存为 {output_path_pdf}')

    output_path_eps = r'D:\Project\vis\图\diagonal_grid_fixed.eps'
    plt.savefig(output_path_eps, dpi=OUTPUT_DPI, bbox_inches='tight',
                format='eps', facecolor='white', edgecolor='none')
    print(f'EPS图形已保存为 {output_path_eps}')

    # # 显示图形
    # plt.show()

    # 清理内存
    plt.close(fig)


def format_time(second: float, is_abbreviation: bool = False) -> str:
    if second < 0:
        raise ValueError('The input parameter \'second\' cannot be negative.')
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


# ==================== 程序执行 ====================

if __name__ == '__main__':
    print('Program schematic.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(f'Program schematic.py finished, total time: {format_time(total_elapsed)}')
