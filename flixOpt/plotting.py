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

import plotly

logger = logging.getLogger('flixOpt')


def with_plotly(data: pd.DataFrame,
                mode: Literal['bar', 'line'] = 'bar',
                colorscale: str = 'viridis',
                fig: Optional[go.Figure] = None) -> go.Figure:
    """
    Plot a DataFrame with plotly. Optionally, provide a custom color sequence of px.colors. ...
    DataFrame is expected to have a time stamp as the index
    """
    colorscale = px.colors.get_colorscale(colorscale)
    colors = px.colors.sample_colorscale(colorscale, [i / (len(data.columns) - 1) for i in range(len(data.columns))])
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
                    colorscale: str = 'viridis',
                    figsize: Tuple[int, int] = (12, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a DataFrame with Matplotlib using stacked bars or lines.
    Optionally provide a color scale name (e.g., 'viridis') for Matplotlib colormap.
    """
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap(colorscale, len(data.columns))
    colors = [cmap(i) for i in range(len(data.columns))]

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
        ncol=(len(data.columns) // 5) + 1,  # Adjust legend to have multiple columns if needed
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
