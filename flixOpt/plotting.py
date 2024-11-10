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
                mode: Literal['area', 'line', 'bar'] = 'bar',
                color_sequence=None) -> go.Figure:
    """
    Plot a DataFrame with plotly. Optionally, provide a custom color sequence of px.colors. ...
    DataFrame is expected to have a time stamp as the index
    """

    colors = color_sequence or px.colors.sequential.Turbo
    fig = go.Figure()
    if mode == 'area':
        for i, column in enumerate(data.columns):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[column],
                mode='lines',
                stackgroup='positive' if data[column].min() >= -1e-5 else 'negative',  # This ensures stacking
                name=column,
                line=dict(color=colors[i % len(colors)])

            ))
    elif mode == 'bar':
        for i, column in enumerate(data.columns):
            fig.add_trace(go.Bar(
                x=data.index,
                y=data[column],
                name=column,
                marker=dict(color=colors[i % len(colors)])
            ))
    elif mode == 'line':
        for i, column in enumerate(data.columns):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[column],
                mode='lines',
                name=column,
                line=dict(color=colors[i % len(colors)])
            ))
    else:
        raise TypeError(f'mode must be either "area" or "bar" or "line"')

    # Update layout for better aesthetics
    fig.update_layout(
        barmode='relative' if mode == 'bar' else None,
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
        font=dict(size=14)  # Increase font size for better readability
    )
    return fig


def with_matplotlib(data: pd.DataFrame,
                    mode: Literal['area', 'line', 'bar'] = 'bar',
                    color_sequence: Optional[List[str]] = None,
                    figsize: Tuple[int, int] = (10, 6),
                    fig: Optional[plt.Figure] = None,
                    ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a DataFrame with Matplotlib. Optionally, provide a custom color sequence.
    DataFrame is expected to have a timestamp as the index.
    """

    # Default color sequence if none provided
    color_sequence = color_sequence or plt.cm.viridis(np.linspace(0, 1, len(data.columns)))

    if not fig or not ax:
        fig, ax = plt.subplots(figsize=figsize)

    if mode == 'area':
        # Stacked area plot
        cumulated_data = data.clip(lower=0).cumsum(axis=1)
        for i, column in enumerate(data.columns):
            ax.fill_between(
                data.index,
                cumulated_data[column] - data[column],  # Bottom line for each layer
                cumulated_data[column],  # Top line for each layer
                label=column,
                color=color_sequence[i % len(color_sequence)]
            )
    elif mode == 'bar':
        # Stacked bar plot
        bottom = np.zeros(len(data))
        for i, column in enumerate(data.columns):
            ax.bar(
                data.index,
                data[column],
                bottom=bottom,
                label=column,
                color=color_sequence[i % len(color_sequence)]
            )
            bottom += data[column]
    elif mode == 'line':
        # Line plot
        for i, column in enumerate(data.columns):
            ax.plot(
                data.index,
                data[column],
                label=column,
                color=color_sequence[i % len(color_sequence)]
            )
    else:
        raise ValueError("mode must be either 'area', 'bar', or 'line'")

    # Customizing the plot aesthetics
    ax.set_xlabel('Time in h', fontsize=14)
    ax.set_ylabel('Values', fontsize=14)
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    ax.legend(loc='upper left')
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
