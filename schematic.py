#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagonal color grid plotting program
Display color distribution along diagonal
Fixed Chinese font and memory issues

Founded in 2026-04-04
Modified in 2026-10-01
@author: yinlb, space-bunny
"""

import colorsys
import gc
import pathlib
import random
import typing

import arrow
import numpy as np

from matplotlib import axes
from matplotlib import patches
from matplotlib import pyplot as plt

from src import utils


# The matplotlib backend, rcParams and the Chinese font are applied by
# src.utils.setup_plot_style(), which main() calls before plotting.


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


def _build_qualitative_pool(cmap_names: typing.List[str]) -> typing.List:
    """
    Build a color pool from multiple qualitative matplotlib colormaps.

    Args:
        cmap_names: List of matplotlib qualitative colormap names.

    Returns:
        pool: Flat list of RGBA colors.
    """
    pool = list()
    for cmap_name in cmap_names:
        cmap = plt.get_cmap(cmap_name)
        pool.extend([cmap(i) for i in range(cmap.N)])
    return pool


def _generate_color_map(
    num_colors: int, color_map_name: str, random_seed: int = 42
) -> typing.List[typing.Tuple[float, ...]]:
    """
    Generate color list of specified count.

    Use built-in matplotlib colormap, or one of the special 'distinct' modes:
      - 'distinct' / 'distinct_ordered': pick 24 colors evenly spaced in hue
        from a pool of qualitative colormaps. Looks ordered but still varied.
      - 'distinct_random': randomly shuffle the same qualitative pool for a
        non-gradient, irregular look.

    Args:
        num_colors: Number of colors needed
        color_map_name: Name of matplotlib colormap, or 'distinct' / etc.

    Returns:
        colors: Color list
    """
    distinct_mode = color_map_name.lower()
    if distinct_mode in ('distinct', 'distinct_ordered'):
        # Build a pool from several qualitative colormaps, then sample evenly
        # in hue space so adjacent diagonals are clearly different but not
        # arranged in a rigid alternating pattern.
        qualitative_cmaps = ['tab20', 'tab20b', 'tab20c', 'Set1', 'Set2',
                             'Set3', 'Dark2', 'Pastel1']
        pool = _build_qualitative_pool(qualitative_cmaps)

        # Sort pool by hue and pick evenly spaced colors
        pool_with_hue = [
            (colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])[0], rgb)
            for rgb in pool
        ]
        pool_with_hue.sort(key=lambda x: x[0])
        n_pool = len(pool_with_hue)
        step = n_pool / num_colors
        colors = [
            pool_with_hue[min(int(i * step), n_pool - 1)][1]
            for i in range(num_colors)
        ]
        return colors

    if distinct_mode == 'distinct_random':
        # Randomly shuffle a qualitative color pool for a non-gradient look.
        # Fixed seed guarantees reproducibility.
        qualitative_cmaps = ['tab20', 'tab20b', 'tab20c', 'Set1', 'Set2',
                             'Set3', 'Dark2', 'Pastel1']
        pool = _build_qualitative_pool(qualitative_cmaps)
        rng = random.Random(random_seed)
        rng.shuffle(pool)
        return pool[:num_colors]

    # Uniformly sample colors from the specified colormap
    base_cmap = plt.get_cmap(color_map_name)
    colors = [base_cmap(i / max(1, num_colors - 1)) for i in range(num_colors)]

    return colors


def _draw_grid(
    ax: axes.Axes,
    matrix: np.ndarray,
    colors: typing.List,
    size: int,
) -> None:
    """
    Draw colored grid for the TLE schematic.

    Only the right-lower triangle (i + j >= size - 1) is filled,
    because those cells represent valid (init_time, lead_time) pairs
    that shere the same forecast valid time along each anti-diagonal.
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
                # Keep only the right-lower triangle including anti-diagonal
                if i + j < size - 1:
                    continue
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
        gc.collect()

    # Set axis range
    ax.set_xlim(-0.5, size + 0.5)
    ax.set_ylim(-0.5, size + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')


def _draw_color_bar(
    ax: axes.Axes,
    colors: typing.List,
    size: int,
    font_size: int = 6,
) -> None:
    """
    Draw bottom color bar for the TLE schematic.

    The grid has size anti-diagonals (indices size-1 to 2*size-2),
    so the color bar contains exactly size segments, one per diagonal.

    Args:
        ax: matplotlib axis object
        colors: Full color list; the segment for diagonal k uses colors[k]
        size: Grid size (number of segments in the color bar)
        font_size: Font size for segment labels.
    """
    # Use the size diagonal colors that correspond to the filled triangle
    start_idx = size - 1
    num_colors = size

    for i in range(num_colors):
        color_idx = start_idx + i
        rect = patches.Rectangle(
            (i, 0), 1, 1,
            facecolor=colors[color_idx],
            edgecolor='black',
            linewidth=0.3
        )
        ax.add_patch(rect)

    # Add labels 1-24 beneath each color bar segment
    label_step = 1
    for i in range(0, num_colors, label_step):
        ax.text(
            i + 0.5, -0.15, str(i + 1),
            ha='center', va='top',
            fontsize=font_size,
            color='black',
            clip_on=False
        )

    # Set axis: y 0-1 exactly matches one segment height so each segment is
    # a square identical in size to a grid cell.
    ax.set_xlim(-0.5, num_colors + 0.5)
    ax.set_ylim(0, 1)
    ax.axis('off')


def _add_number_labels(
    ax: axes.Axes,
    size: int,
    position: str = 'top',
    font_size: int = 6,
) -> None:
    """
    Add number labels around the grid.

    Supports 'top', 'bottom', 'left', 'right'.
    Facilitate positioning of each cell.

    Args:
        ax: matplotlib axis object
        size: Grid size
        position: Label position ('top', 'bottom', 'left', 'right')
        font_size: Label font size in points.
    """
    # Show every number from 1 to size
    label_step = 1

    for i in range(0, size, label_step):
        value = str(i + 1)

        if position == 'top':
            x_pos = i + 0.5
            y_pos = size + 0.5
            ha, va = 'center', 'bottom'
        elif position == 'bottom':
            x_pos = i + 0.5
            y_pos = -0.5
            ha, va = 'center', 'top'
        elif position == 'left':
            x_pos = -0.5
            y_pos = size - 1 - i + 0.5
            ha, va = 'right', 'center'
        else:  # right
            x_pos = size + 0.5
            y_pos = size - 1 - i + 0.5
            ha, va = 'left', 'center'

        ax.text(
            x_pos, y_pos, value,
            ha=ha, va=va,
            fontsize=font_size,
            color='black'
        )


def _add_arrow_annotation(
    ax: axes.Axes,
    size: int,
    target_col: float,
) -> None:
    """
    Add downward arrow annotation in the gap between grid and color bar.

    The arrow is drawn in a dedicated axis between the grid and color bar,
    so it never overlaps with either the grid cells or the colorbar segments.

    Args:
        ax: matplotlib axis object (dedicated arrow axis)
        size: Grid size
        target_col: Target column position in grid coordinates
    """
    ax.annotate(
        '',
        xy=(target_col, 0.08),
        xytext=(target_col, 0.92),
        arrowprops=dict(
            arrowstyle='->',
            color='black',
            linewidth=3.0
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
    plot_cfg = cfg['draw']['plot']
    grid_size = plot_cfg['grid_size']
    cell_size_inches = plot_cfg['cell_size_inches']
    color_map_name = plot_cfg['color_map']
    random_seed = plot_cfg['random_seed']
    output_dpi = plot_cfg['dpi']
    out_dir = cfg['paths']['output_dir']

    # 3. Apply the shared plot style, Chinese font first
    utils.setup_plot_style(cfg['fonts'], chinese_first=True)

    # 4. Compute figure size from grid dimensions and cell size
    grid_width = grid_size * cell_size_inches
    grid_height = grid_width
    side_margin = 0.2
    label_gap = max(0.05, 0.5 * cell_size_inches)
    # Make each colorbar segment the same height as a grid cell
    colorbar_height = cell_size_inches
    # Dedicated gap between grid and colorbar for the arrow annotation
    arrow_gap = 0.6
    bottom_margin = 0.1

    fig_width = grid_width + 2 * side_margin
    fig_height = (
        label_gap + grid_height + label_gap +
        arrow_gap + colorbar_height + bottom_margin
    )

    fig = plt.figure(figsize=(fig_width, fig_height))

    grid_left = side_margin / fig_width
    grid_bottom = (bottom_margin + colorbar_height + arrow_gap) / fig_height
    grid_width_frac = grid_width / fig_width
    grid_height_frac = grid_height / fig_height

    colorbar_left = grid_left
    colorbar_bottom = bottom_margin / fig_height
    colorbar_width_frac = grid_width_frac
    colorbar_height_frac = colorbar_height / fig_height

    ax_grid = fig.add_axes(
        [grid_left, grid_bottom, grid_width_frac, grid_height_frac]
    )
    ax_colorbar = fig.add_axes(
        [colorbar_left, colorbar_bottom, colorbar_width_frac,
         colorbar_height_frac]
    )

    # Dedicated transparent axis for the arrow in the gap between grid and
    # colorbar
    arrow_bottom = (bottom_margin + colorbar_height) / fig_height
    arrow_height_frac = arrow_gap / fig_height
    ax_arrow = fig.add_axes(
        [grid_left, arrow_bottom, grid_width_frac, arrow_height_frac]
    )
    ax_arrow.set_xlim(-0.5, grid_size + 0.5)
    ax_arrow.set_ylim(0, 1)
    ax_arrow.axis('off')
    ax_arrow.set_zorder(10)

    label_font_size = max(4, min(10, cell_size_inches * 40))
    title_font_size = max(8, min(16, cell_size_inches * 80))

    # 5. Generate diagonal matrix and color map
    diagonal_matrix = _create_diagonal_matrix(grid_size)
    # We need colors for anti-diagonal indices 0 .. 2*size-2,
    # but only size-1 .. 2*size-2 are actually displayed.
    color_list = _generate_color_map(
        grid_size * 2 - 1, color_map_name, random_seed
    )

    # 6. Draw main elements (grid and color bar)
    _draw_grid(ax_grid, diagonal_matrix, color_list, grid_size)
    _draw_color_bar(
        ax_colorbar, color_list, grid_size, font_size=label_font_size
    )

    # 7. Add number labels and annotations
    # Top and right labels mark lead time and init time respectively
    _add_number_labels(ax_grid, grid_size, position='top',
                       font_size=label_font_size)
    _add_number_labels(ax_grid, grid_size, position='right',
                       font_size=label_font_size)
    _add_arrow_annotation(ax_arrow, grid_size, target_col=12.5)

    # 8. Save output files (no title per user request)
    # Create the output directory first: matplotlib does not create it
    output_dir = pathlib.Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save files in all configured formats
    for fmt in plot_cfg['output_formats']:
        output_path = str(output_dir / f'diagonal_grid_fixed.{fmt}')
        plt.savefig(
            output_path, dpi=output_dpi, bbox_inches='tight',
            format=fmt, facecolor='white', edgecolor='none'
        )
        print(f'{fmt.upper()} figure saved to {output_path}')

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
