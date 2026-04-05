#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagonal color grid plotting program
Display color distribution along diagonal direction
Fixed Chinese font and memory issues
"""

import arrow
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fm

# ==================== Global Configuration ====================

GRID_SIZE = 24  # 网格大小 24x24
FIG_WIDTH = 12  # 图片宽度（英寸）
FIG_HEIGHT = 14  # 图片高度（英寸）
COLOR_MAP_NAME = 'rainbow'  # 颜色映射名称
OUTPUT_DPI = 800
CHINESE_FONT_PATH = None  # 中文字体路径（自动查找）


# ==================== Font Configuration ====================


def setup_chinese_font():
    """
    配置中文字体支持

    自动查找系统可用的中文字体
    解决 DejaVu Sans 不支持中文的问题
    """
    global CHINESE_FONT_PATH

    # Common Chinese font paths (Windows)
    possible_fonts = [
        r'C:\Windows\Fonts\msyh.ttc',  # Microsoft YaHei
        r'C:\Windows\Fonts\msyhbd.ttc',  # Microsoft YaHei Bold
        r'C:\Windows\Fonts\simhei.ttf',  # SimHei
        r'C:\Windows\Fonts\STSONG.TTC',  # SimSun
        r'C:\Windows\Fonts\ARIALUNI.TTF',  # Arial Unicode MS
    ]

    # Try to find available Chinese font
    for font_path in possible_fonts:
        try:
            # Test if font is available
            prop = fm.FontProperties(fname=font_path)
            text = u'Test Chinese'
            # Try to render test text
            fig_test = plt.figure()
            ax_test = fig_test.add_subplot(111)
            ax_test.text(0.5, 0.5, text, fontproperties=prop)
            plt.close(fig_test)
            CHINESE_FONT_PATH = font_path
            print(f'Found and verified Chinese font: {font_path}')
            break
        except Exception as e:
            continue

    if CHINESE_FONT_PATH is None:
        print('Warning: No Chinese font found, using English title')


def set_matplotlib_params():
    """
    设置 matplotlib 参数

    包括字体、后端等配置
    避免字体渲染问题
    """
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    # If Chinese font found, register and use
    if CHINESE_FONT_PATH:
        # Register font
        font_prop = fm.FontProperties(fname=CHINESE_FONT_PATH)
        # Set global font
        plt.rcParams['font.family'] = font_prop.get_name()
        print(f'Font set to: {font_prop.get_name()}')


# ==================== Core Functions ====================


def create_diagonal_matrix(size):
    """
    创建对角线索引矩阵

    每条从左下到右上的斜线具有相同的索引值
    用于后续映射颜色

    Args:
        size: Grid size

    Returns:
        diagonal_idx: Diagonal index matrix
    """
    diagonal_idx = np.zeros((size, size), dtype=int)

    for i in range(size):
        for j in range(size):
            # Calc diagonal index (bottom-left to top-right)
            diagonal_idx[i, j] = i + j

    return diagonal_idx


def generate_color_map(num_colors):
    """
    生成指定数量的颜色列表

    使用 matplotlib 内置颜色映射
    可根据需要自定义颜色序列

    Args:
        num_colors: Number of colors needed

    Returns:
        colors: Color list
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
        ax: matplotlib axis object
        matrix: Diagonal index matrix
        colors: Color list
        size: Grid size
    """
    # Batch draw to reduce memory
    batch_size = 8  # Draw 8 rows per batch

    for start_row in range(0, size, batch_size):
        end_row = min(start_row + batch_size, size)

        for i in range(start_row, end_row):
            for j in range(size):
                color_idx = matrix[i, j] % len(colors)
                # Note: Matrix row index opposite to y-axis
                rect = Rectangle(
                    (j, size - 1 - i), 1, 1,
                    facecolor=colors[color_idx],
                    edgecolor='black',
                    linewidth=0.3  # Thinner lines save memory
                )
                ax.add_patch(rect)

        # Force garbage collection
        import gc
        gc.collect()

    # Set axis range
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
        ax: matplotlib axis object
        colors: Color list
        size: Grid size
    """
    num_colors = min(len(colors), size)  # Limit color count

    # Batch draw color bar
    for i in range(num_colors):
        rect = Rectangle(
            (i, 0), 1, 1,
            facecolor=colors[i],
            edgecolor='black',
            linewidth=0.3
        )
        ax.add_patch(rect)

    # Set axis
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
        ax: matplotlib axis object
        size: Grid size
        position: Label position ('top' or 'bottom')
    """
    # Reduce label density to save memory
    label_step = max(1, size // 12)  # Show label every N positions

    for i in range(0, size, label_step):
        x_pos = i + 0.5
        if position == 'top':
            y_pos = size + 0.5
            va = 'bottom'
        else:
            y_pos = -0.5
            va = 'top'

        # Use smaller font
        ax.text(
            x_pos, y_pos, str(i + 1),
            ha='center', va=va,
            fontsize=6,  # Smaller font
            color='black'
        )


def add_arrow_annotation(ax, size, target_col):
    """
    添加箭头标注

    指示网格与颜色条的对应关系

    Args:
        ax: matplotlib axis object
        size: Grid size
        target_col: Target column position
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


# ==================== Main Entry ====================


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
    # Set font and parameters
    setup_chinese_font()
    set_matplotlib_params()

    # Create figure and subplots
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Define three subplot areas: top label, grid, color bar
    ax_grid = fig.add_axes([0.1, 0.15, 0.8, 0.75])
    ax_colorbar = fig.add_axes([0.1, 0.05, 0.8, 0.08])
    ax_labels = fig.add_axes([0.1, 0.15, 0.8, 0.75])

    # Generate diagonal matrix and colors
    diagonal_matrix = create_diagonal_matrix(GRID_SIZE)
    color_list = generate_color_map(GRID_SIZE * 2 - 1)

    # Draw main elements
    draw_grid(ax_grid, diagonal_matrix, color_list, GRID_SIZE)
    draw_color_bar(ax_colorbar, color_list, GRID_SIZE * 2 - 1)

    # Add number labels
    add_number_labels(ax_labels, GRID_SIZE, position='top')
    add_number_labels(ax_labels, GRID_SIZE, position='bottom')

    # Add arrow annotation (example: point to col 12-13)
    add_arrow_annotation(ax_labels, GRID_SIZE, target_col=12.5)

    # Set title based on whether Chinese font found
    if CHINESE_FONT_PATH:
        # Use font property to specify font
        font_prop = fm.FontProperties(fname=CHINESE_FONT_PATH, size=14)
        fig.suptitle('Diagonal Color Distribution', fontproperties=font_prop, y=0.98)
    else:
        fig.suptitle('Diagonal Color Distribution Schematic', fontsize=14, y=0.98)

    # Save files (use lower DPI to save memory)
    output_path_pdf = r'D:\Project\vis\图\diagonal_grid_fixed.pdf'
    plt.savefig(output_path_pdf, dpi=OUTPUT_DPI, bbox_inches='tight',
                format='pdf', facecolor='white', edgecolor='none')
    print(f'PDF figure saved to {output_path_pdf}')

    output_path_eps = r'D:\Project\vis\图\diagonal_grid_fixed.eps'
    plt.savefig(output_path_eps, dpi=OUTPUT_DPI, bbox_inches='tight',
                format='eps', facecolor='white', edgecolor='none')
    print(f'EPS figure saved to {output_path_eps}')

    # # Show figure
    # plt.show()

    # Clean up memory
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


# ==================== Execution ====================

if __name__ == '__main__':
    print('Program schematic.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(f'Program schematic.py finished, total time: {format_time(total_elapsed)}')
