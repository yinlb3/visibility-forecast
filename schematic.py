#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagonal color grid plotting program
Display color distribution along diagonal
Fixed Chinese font and memory issues

Founded in 2026-04-04
Modified in 2026-04-15
@author: yinlb
"""

import pathlib
import typing

import arrow
import numpy as np
from matplotlib import font_manager as fm
from matplotlib import patches
from matplotlib import pyplot as plt

from src import utils


def _setup_chinese_font(fonts_cfg: typing.Dict) -> typing.Optional[str]:
    """
    Configure Chinese font support.

    Auto-detect system Chinese fonts
    to fix DejaVu Sans missing CJK glyphs.

    Args:
        fonts_cfg: Font configuration dict with 'chinese' key.

    Returns:
        Path to available Chinese font, or None if not found.
    """
    possible_fonts = fonts_cfg['chinese']

    # Try to find available Chinese font
    for font_path in possible_fonts:
        try:
            # Test if font is available
            prop = fm.FontProperties(fname=font_path)
            text = 'Test Chinese'
            # Try to render test text
            fig_test = plt.figure()
            ax_test = fig_test.add_subplot(111)
            ax_test.text(0.5, 0.5, text, fontproperties=prop)
            plt.close(fig_test)
            print(f'Found font: {font_path}')
            return font_path
        except Exception:
            continue

    print('Warning: No Chinese font found, using English title')
    return None


def _set_matplotlib_params(chinese_font_path: typing.Optional[str]) -> None:
    """
    Set matplotlib parameters.

    Including fonts, backend, etc.
    Avoid font rendering issues.

    Args:
        chinese_font_path: Path to Chinese font, or None.
    """
    plt.rcParams['axes.unicode_minus'] = False  # Correct minus sign display

    # If Chinese font found, register and use
    if chinese_font_path:
        # Register font
        font_prop = fm.FontProperties(fname=chinese_font_path)
        # Set global font
        plt.rcParams['font.family'] = font_prop.get_name()
        print(f'Font set to: {font_prop.get_name()}')


# ==================== Core Functions ====================


def _create_diagonal_matrix(size: int) -> np.ndarray:
    """
    Create diagonal index matrix.

    Each diagonal line (bottom-left to top-right) has same index value.
    Used for subsequent color mapping.

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


def _generate_color_map(
    num_colors: int, color_map_name: str
) -> typing.List[typing.Tuple[float, ...]]:
    """
    Generate color list of specified count.

    Use built-in matplotlib colormap.
    Can customize color sequence as needed.

    Args:
        num_colors: Number of colors needed
        color_map_name: Name of matplotlib colormap

    Returns:
        colors: Color list
    """
    # Uniformly sample colors from the specified colormap
    base_cmap = plt.get_cmap(color_map_name)
    colors = [base_cmap(i / max(1, num_colors - 1)) for i in range(num_colors)]

    return colors


def _draw_grid(ax, matrix: np.ndarray, colors: typing.List, size: int) -> None:
    """
    Draw colored grid.

    Each cell filled with corresponding color by diagonal index.
    Add black grid lines for distinction.

    Args:
        ax: matplotlib axis object
        matrix: Diagonal index matrix
        colors: Color list
        size: Grid size
    """
    # Batch draw cells to reduce memory usage (8 rows per batch)
    batch_size = 8  # Draw 8 rows per batch

    for start_row in range(0, size, batch_size):
        end_row = min(start_row + batch_size, size)

        for i in range(start_row, end_row):
            for j in range(size):
                color_idx = matrix[i, j] % len(colors)
                # Note: Matrix row index opposite to y-axis (flip vertically)
                rect = patches.Rectangle(
                    (j, size - 1 - i), 1, 1,
                    facecolor=colors[color_idx],
                    edgecolor='black',
                    linewidth=0.3  # Thinner lines save memory
                )
                ax.add_patch(rect)

        # Force garbage collection after each batch
        import gc
        gc.collect()

    # Set axis range
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.axis('off')


def _draw_color_bar(ax, colors: typing.List, size: int) -> None:
    """
    Draw bottom color bar.

    Show index corresponding to each color.
    One-to-one match with upper grid colors.

    Args:
        ax: matplotlib axis object
        colors: Color list
        size: Grid size
    """
    # Draw color bar with same colors as grid (one-to-one mapping)
    num_colors = min(len(colors), size)  # Limit color count

    for i in range(num_colors):
        rect = patches.Rectangle(
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


def _add_number_labels(ax, size: int, position: str = 'top') -> None:
    """
    Add number labels.

    Add numbers 1-24 at top or bottom of grid.
    Facilitate positioning of each cell.

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


def _add_arrow_annotation(ax, size: int, target_col: float) -> None:
    """
    Add arrow annotation.

    Indicate correspondence between grid and color bar.

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


def main() -> None:
    """
    Main function: execute plotting workflow.

    Steps:
        1. Setup Chinese font
        2. Create diagonal matrix
        3. Generate color map
        4. Draw grid and color bar
        5. Add annotations and labels
        6. Save output files
    """
    # 1. Load configuration
    print('Loading configuration...')
    cfg = utils.load_config()

    # 2. Extract commonly used configuration values
    plot_cfg = cfg['plot']
    grid_size = plot_cfg['grid_size']
    fig_width = plot_cfg['fig_width']
    fig_height = plot_cfg['fig_height']
    color_map_name = plot_cfg['color_map']
    output_dpi = plot_cfg['dpi']
    out_dir = cfg['paths']['output_dir']

    # 3. Setup Chinese font and matplotlib parameters
    chinese_font_path = _setup_chinese_font(cfg['fonts'])
    _set_matplotlib_params(chinese_font_path)

    # 4. Create figure and define subplot areas
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax_grid = fig.add_axes([0.1, 0.15, 0.8, 0.75])      # Main grid area
    ax_colorbar = fig.add_axes([0.1, 0.05, 0.8, 0.08])  # Color bar area
    ax_labels = fig.add_axes([0.1, 0.15, 0.8, 0.75])    # Label overlay area

    # 5. Generate diagonal matrix and color map
    diagonal_matrix = _create_diagonal_matrix(grid_size)
    color_list = _generate_color_map(grid_size * 2 - 1, color_map_name)

    # 6. Draw main elements (grid and color bar)
    _draw_grid(ax_grid, diagonal_matrix, color_list, grid_size)
    _draw_color_bar(ax_colorbar, color_list, grid_size * 2 - 1)

    # 7. Add number labels and annotations
    _add_number_labels(ax_labels, grid_size, position='top')
    _add_number_labels(ax_labels, grid_size, position='bottom')
    _add_arrow_annotation(ax_labels, grid_size, target_col=12.5)

    # 8. Set title and save output files
    if chinese_font_path:
        # Use font property to specify font
        font_prop = fm.FontProperties(fname=chinese_font_path, size=14)
        fig.suptitle(
            'Diagonal Color Distribution',
            fontproperties=font_prop, y=0.98
        )
    else:
        fig.suptitle(
            'Diagonal Color Distribution Schematic', fontsize=14,
            y=0.98)

    # Save files (use lower DPI to save memory)
    output_path_pdf = str(pathlib.Path(out_dir) / 'diagonal_grid_fixed.pdf')
    plt.savefig(
        output_path_pdf, dpi=output_dpi, bbox_inches='tight',
        format='pdf', facecolor='white', edgecolor='none'
    )
    print(f'PDF figure saved to {output_path_pdf}')

    output_path_eps = str(pathlib.Path(out_dir) / 'diagonal_grid_fixed.eps')
    plt.savefig(
        output_path_eps, dpi=output_dpi, bbox_inches='tight',
        format='eps', facecolor='white', edgecolor='none'
    )
    print(f'EPS figure saved to {output_path_eps}')

    # Clean up memory
    plt.close(fig)


def _format_time(second: float, is_abbreviation: bool = False) -> str:
    """
    Format seconds to human-readable time string.

    Args:
        second (float): Seconds to format.
        is_abbreviation (bool): Use abbreviation format (e.g., '12.5m').

    Returns:
        str: Formatted time string.
    """
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
    elapsed_str = _format_time(total_elapsed)
    print(f'Program schematic.py finished, total time: {elapsed_str}')
