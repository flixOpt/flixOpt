# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 09:43:09 2021
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""
import logging
from typing import Literal, Tuple, Union, Optional, List

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger('flixOpt')


def with_plotly(data: pd.DataFrame,
                mode: Literal['bar', 'line'] = 'bar',
                colors: Union[List[str], str] = 'viridis',
                fig: Optional[go.Figure] = None) -> go.Figure:
    """
    Plot a DataFrame with Plotly, using either stacked bars or stepped lines.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the data to plot, where the index represents
        time (e.g., hours), and each column represents a separate data series.
    mode : {'bar', 'line'}, default='bar'
        The plotting mode. Use 'bar' for stacked bar charts or 'line' for
        stepped lines.
    colors : List[str], str, default='viridis'
        A List of colors (as str) or a name of a colorscale (e.g., 'viridis', 'plasma') to use for
        coloring the data series.
    fig : go.Figure, optional
        A Plotly figure object to plot on. If not provided, a new figure
        will be created.

    Returns
    -------
    go.Figure
        A Plotly figure object containing the generated plot.

    Notes
    -----
    - If `mode` is 'bar', bars are stacked for each data series.
    - If `mode` is 'line', a stepped line is drawn for each data series.
    - The legend is positioned below the plot for a cleaner layout when many
      data series are present.

    Examples
    --------
    >>> fig = with_plotly(data, mode='bar', colorscale='plasma')
    >>> fig.show()
    """
    if isinstance(colors, str):
        colorscale = px.colors.get_colorscale(colors)
        colors = px.colors.sample_colorscale(colorscale, [i / (len(data.columns) - 1) for i in range(len(data.columns))])
    assert len(colors) == len(data.columns), (f'The number of colors does not match the provided data columns. '
                                              f'{len(colors)=}; {len(colors)=}')
    fig = fig if fig is not None else go.Figure()

    if mode == 'bar':
        for i, column in enumerate(data.columns):
            fig.add_trace(go.Bar(
                x=data.index,
                y=data[column],
                name=column,
                marker=dict(color=colors[i])
            ))

        fig.update_layout(
            barmode='relative' if mode == 'bar' else None,
            bargap=0,  # No space between bars
            bargroupgap=0,  # No space between groups of bars
        )
    elif mode == 'line':
        for i, column in enumerate(data.columns):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[column],
                mode='lines',
                name=column,
                line=dict(shape='hv', color=colors[i]),
            ))
    else:
        raise ValueError(f'mode must be either "bar" or "line"')

    # Update layout for better aesthetics
    fig.update_layout(
        yaxis=dict(
            showgrid=True,  # Enable grid lines on the y-axis
            gridcolor='lightgrey',  # Customize grid line color
            gridwidth=0.5,  # Customize grid line width
        ),
        xaxis=dict(
            title='Time in h',
            showgrid=True,  # Enable grid lines on the x-axis
            gridcolor='lightgrey',  # Customize grid line color
            gridwidth=0.5  # Customize grid line width
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
        font=dict(size=14),  # Increase font size for better readability
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=-0.3,  # Adjusts how far below the plot it appears
            xanchor="center",
            x=0.5,
            title_text=None  # Removes legend title for a cleaner look
        )
    )
    return fig


def with_matplotlib(data: pd.DataFrame,
                    mode: Literal['bar', 'line'] = 'bar',
                    colors: Union[List[str], str] = 'viridis',
                    figsize: Tuple[int, int] = (12, 6),
                    fig: Optional[plt.Figure] = None,
                    ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a DataFrame with Matplotlib using stacked bars or stepped lines.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the data to plot. The index should represent
        time (e.g., hours), and each column represents a separate data series.
    mode : {'bar', 'line'}, default='bar'
        Plotting mode. Use 'bar' for stacked bar charts or 'line' for stepped lines.
    colors : List[str], str, default='viridis'
        A List of colors (as str) or a name of a colorscale (e.g., 'viridis', 'plasma') to use for
        coloring the data series.
    figsize: Tuple[int, int], optional
        Specify the size of the figure
    fig : plt.Figure, optional
        A Matplotlib figure object to plot on. If not provided, a new figure
        will be created.
    ax : plt.Axes, optional
        A Matplotlib axes object to plot on. If not provided, a new axes
        will be created.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        A tuple containing the Matplotlib figure and axes objects used for the plot.

    Notes
    -----
    - If `mode` is 'bar', bars are stacked for both positive and negative values.
      Negative values are stacked separately without extra labels in the legend.
    - If `mode` is 'line', stepped lines are drawn for each data series.
    - The legend is placed below the plot to accommodate multiple data series.

    Examples
    --------
    >>> fig, ax = with_matplotlib(data, mode='bar', colorscale='plasma')
    >>> plt.show()
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if isinstance(colors, str):
        cmap = plt.get_cmap(colors, len(data.columns))
        colors = [cmap(i) for i in range(len(data.columns))]
    assert len(colors) == len(data.columns), (f'The number of colors does not match the provided data columns. '
                                              f'{len(colors)=}; {len(colors)=}')

    if mode == 'bar':
        cumulative_positive = np.zeros(len(data))
        cumulative_negative = np.zeros(len(data))
        width = data.index.to_series().diff().dropna().min()  # Minimum time difference

        for i, column in enumerate(data.columns):
            positive_values = np.clip(data[column], 0, None)  # Keep only positive values
            negative_values = np.clip(data[column], None, 0)  # Keep only negative values
            # Plot positive bars
            ax.bar(
                data.index,
                positive_values,
                bottom=cumulative_positive,
                color=colors[i],
                label=column,
                width=width,
                align='edge'
            )
            cumulative_positive += positive_values.values
            # Plot negative bars
            ax.bar(
                data.index,
                negative_values,
                bottom=cumulative_negative,
                color=colors[i],
                label="",  # No label for negative bars
                width=width,
                align='edge'
            )
            cumulative_negative += negative_values.values

    elif mode == 'line':
        for i, column in enumerate(data.columns):
            ax.step(
                data.index,
                data[column],
                where='post',
                color=colors[i],
                label=column
            )
    else:
        raise ValueError(f"mode must be either 'bar' or 'line'")

    # Aesthetics
    ax.set_xlabel('Time in h', fontsize=14)
    ax.set_ylabel('Values', fontsize=14)
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    ax.legend(
        loc='upper center',  # Place legend at the bottom center
        bbox_to_anchor=(0.5, -0.15),  # Adjust the position to fit below plot
        ncol=5,
        frameon=False  # Remove box around legend
    )
    fig.tight_layout()

    return fig, ax


def color_map_matplotlib(data: pd.DataFrame,
                              nr_of_periods: int,
                              time_steps_per_period: int,
                              color_map: str = 'viridis',
                              xlabel: str = '',
                              ylabel: str = '',
                              fontsize: float = 12,
                              figsize: Tuple[float, float] = (12, 6),
                              save_as: Optional[str] = None,
                              fig=None,
                              ax=None,
                              **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """ Plot values as a colormap. kwargs are passed to plt.subplots(**kwargs)"""

    color_bar_min, color_bar_max = data.min(), data.max()

    if not fig or not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)

    ax.pcolormesh(
        range(nr_of_periods + 1),
        range(time_steps_per_period + 1),
        data.index,
        cmap=color_map,
        vmin=color_bar_min,
        vmax=color_bar_max,
        **kwargs,
    )
    ax.axis([0, nr_of_periods, 0, time_steps_per_period])
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.xaxis.set_label_position("bottom"), ax.xaxis.set_ticks_position("bottom")

    sm1 = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=color_bar_min, vmax=color_bar_max))
    sm1._A = []
    cb1 = fig.colorbar(sm1, ax=ax, pad=0.12, aspect=15, fraction=0.2, orientation='horizontal')
    cb1.ax.tick_params(labelsize=12)
    cb1.ax.set_xlabel('Color Bar Label')
    cb1.ax.xaxis.set_label_position('top')

    fig.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi='300', bbox_inches="tight")

    return fig, ax


def color_map_plotly(data: pd.DataFrame,
                          nr_of_periods: int,
                          time_steps_per_period: int,
                          color_map: str = 'Viridis',
                          xlabel: str = 'period',
                          ylabel: str = 'period index',
                          fontsize: float = 12,
                          figsize: Tuple[float, float] = (1200, 600),
                          save_as: Optional[str] = None,
                          show: bool = True,
                          **kwargs
                          ) -> go.Figure:
    """ Plot values as a color map using Plotly. kwargs are passed to `go.Heatmap`."""

    color_bar_min, color_bar_max = data.min().min(), data.max().max()  # Min and max values for color scaling
    # Define the figure
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=list(range(nr_of_periods)),
        y=list(range(time_steps_per_period)),
        colorscale=color_map,
        zmin=color_bar_min,
        zmax=color_bar_max,
        colorbar=dict(
            title=dict(text='Color Bar Label', side='right'),
            tickfont=dict(size=fontsize),
            orientation='h',
            xref='container',
            yref='container',
            len=0.8,  # Color bar length relative to plot
            x=0.5,
            y=0.1
        ),
        **kwargs
    ))

    # Set axis labels and style
    # Set axis labels and style
    fig.update_layout(
        xaxis=dict(title=xlabel, tickfont=dict(size=fontsize), side='top'),
        yaxis=dict(title=ylabel, tickfont=dict(size=fontsize), autorange='reversed'),
        width=figsize[0],
        height=figsize[1],
    )

    # Save as file if specified
    if save_as:
        fig.write_image(save_as)

    if show:
        plotly.offline.plot(fig)

    return fig
