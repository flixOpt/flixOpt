"""
This module contains the core functionality of the flixOpt framework.
It provides Datatypes, logging functionality, and some functions to transform data structures.
"""

import inspect
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr
import pandas as pd

from . import utils

logger = logging.getLogger('flixOpt')

Skalar = Union[int, float]  # Datatype
Numeric = Union[int, float, np.ndarray]  # Datatype


class TimeSeriesData:
    # TODO: Move to Interface.py
    def __init__(self, data: Numeric, agg_group: Optional[str] = None, agg_weight: Optional[float] = None):
        """
        timeseries class for transmit timeseries AND special characteristics of timeseries,
        i.g. to define weights needed in calculation_type 'aggregated'
            EXAMPLE solar:
            you have several solar timeseries. These should not be overweighted
            compared to the remaining timeseries (i.g. heat load, price)!
            fixed_relative_profile_solar1 = TimeSeriesData(sol_array_1, type = 'solar')
            fixed_relative_profile_solar2 = TimeSeriesData(sol_array_2, type = 'solar')
            fixed_relative_profile_solar3 = TimeSeriesData(sol_array_3, type = 'solar')
            --> this 3 series of same type share one weight, i.e. internally assigned each weight = 1/3
            (instead of standard weight = 1)

        Parameters
        ----------
        data : Union[int, float, np.ndarray]
            The timeseries data, which can be a scalar, array, or numpy array.
        agg_group : str, optional
            The group this TimeSeriesData is a part of. agg_weight is split between members of a group. Default is None.
        agg_weight : float, optional
            The weight for calculation_type 'aggregated', should be between 0 and 1. Default is None.

        Raises
        ------
        Exception
            If both agg_group and agg_weight are set, an exception is raised.
        """
        self.data = data
        self.agg_group = agg_group
        self.agg_weight = agg_weight
        if (agg_group is not None) and (agg_weight is not None):
            raise Exception('Either <agg_group> or explicit <agg_weigth> can be used. Not both!')
        self.label: Optional[str] = None

    def __repr__(self):
        # Get the constructor arguments and their current values
        init_signature = inspect.signature(self.__init__)
        init_args = init_signature.parameters

        # Create a dictionary with argument names and their values
        args_str = ', '.join(f'{name}={repr(getattr(self, name, None))}' for name in init_args if name != 'self')
        return f'{self.__class__.__name__}({args_str})'

    def __str__(self):
        return str(self.data)


Numeric_TS = Union[
    Skalar, np.ndarray, TimeSeriesData
]  # TODO: This is not really correct throughozt the codebase. Sometimes its used for TimeSeries aswell?


class TimeSeries:
    """
    Class for data that applies to time series, stored as vector (np.ndarray) or scalar.

    This class represents a vector or scalar value that makes the handling of time series easier.
    It supports various operations such as activation of specific time indices, setting explicit active data, and
    aggregation weight management.

    Attributes
    ----------
    label : str
        The label for the time series.
    data : Optional[Numeric]
        The actual data for the time series. Can be None.
    aggregated_data : Optional[Numeric]
        aggregated_data to use instead of data if provided.
    active_indices : Optional[np.ndarray]
        Indices of the time steps to activate.
    aggregation_weight : float
        Weight for aggregation method, between 0 and 1, normally 1.
    aggregation_group : str
        Group for calculating the aggregation weigth for aggregation method.
    """

    def __init__(self,
                 label: str,
                 data: Numeric,
                 index: pd.Index,
                 aggregation_weight: Optional[float] = None):
        self.label: str = label
        self.data: pd.Series = self.as_series(data, index)
        self.aggregation_weight = aggregation_weight

        self.active_coords: Optional[Dict] = None
        self.aggregated_data: Optional[Numeric] = None

    def activate_indices(self, coords: Dict, aggregated_data: Optional[Numeric] = None):
        self.active_coords = coords

        if aggregated_data is not None:
            assert len(aggregated_data) == len(self.active_indices) or len(aggregated_data) == 1, (
                f'The aggregated_data has the wrong length for TimeSeries {self.label}. '
                f'Length should be: {len(self.active_indices)} or 1, but is {len(aggregated_data)}'
            )
            self.aggregated_data = self.make_scalar_if_possible(aggregated_data)

    def clear_indices_and_aggregated_data(self):
        self.active_coords = None
        self.aggregated_data = None

    @property
    def active_data(self) -> Union[int, float, xr.DataArray]:
        if self.aggregated_data is not None:  # Aggregated data is always active, if present
            return self.aggregated_data

        if np.isscalar(self.data) or (self.active_coords is None):
            return self.data
        else:
            return self.data.sel(self.active_coords)

    @staticmethod
    def as_series(data: Numeric, dims: Tuple[pd.Index, ...]) -> pd.Series:
        """
        Converts the given data to a pd.Series with the specified index.

        - Arrays and Series are stacked across the period index.
        - The length of the array must match the length of the time coordinate if applicable.
        - If a 1D array is given but two indices are provided, it is reshaped to 2D automatically.

        Parameters:
        - data: The input data (scalar, array, Series, or DataFrame).
        - dims: A Tuple of pd.Index objects specifying the index dimensions.

        Returns:
        - pd.Series: The resulting Series, possibly with a MultiIndex.
        """

        if not isinstance(dims, tuple) or not all(isinstance(idx, pd.Index) for idx in dims):
            raise TypeError("dims must be a tuple of pandas Index objects")

        expected_shape = tuple(len(idx) for idx in dims)
        index = pd.MultiIndex.from_product(dims) if len(dims) > 1 else dims[0]

        if isinstance(data, (int, float)):  # Scalar case
            return pd.Series(data, index=index)

        if isinstance(data, pd.DataFrame):
            if len(dims) == 1:
                if not data.index.equals(dims[0]):
                    raise ValueError("Series index does not match the provided index")
                data = data.values.ravel()
            else:
                data = data.stack().swaplevel(0,1).sort_index()
                if data.index != index:
                    raise ValueError("DataFrame index does not match the provided index")
                return data

        if isinstance(data, pd.Series):
            if len(dims) == 1:
                if not data.index.equals(dims[0]):
                    raise ValueError("Series index does not match the provided index")
            data =  data.ravel()  # Retrieve the data from the Series

        if isinstance(data, np.ndarray):

            if data.ndim == 1 and len(dims) == 2:  # If 1D data but 2D index is given
                if data.shape[0] == len(dims[0]):
                    data = np.tile(data[:, np.newaxis], (1, len(dims[1])))  # Expand along second dimension
                elif data.shape[0] == len(dims[1]):
                    data = np.tile(data[np.newaxis, :], (len(dims[0]), 1))  # Expand along first dimension
                else:
                    raise ValueError("1D array length does not match either dimension in dims")

            if data.shape != expected_shape:
                raise ValueError(f"Shape of data {data.shape} does not match expected shape {expected_shape}")

            return pd.Series(data.ravel(), index=index)

        elif isinstance(data, pd.Series):
            if not data.index.equals(dims[0]):
                raise ValueError("Series index does not match the provided index")
            return data

        elif isinstance(data, pd.DataFrame):
            if len(dims) != 2 or data.shape != (len(dims[0]), len(dims[1])):
                raise ValueError("DataFrame shape does not match provided indexes")
            return data.stack()

        else:
            raise TypeError("Unsupported data type. Must be scalar, np.ndarray, pd.Series, or pd.DataFrame.")

    @property
    def is_scalar(self) -> bool:
        return np.isscalar(self.data)

    @property
    def is_array(self) -> bool:
        return not self.is_scalar and self.data is not None

    def __repr__(self):
        # Retrieve all attributes and their values
        attrs = vars(self)
        # Format each attribute as 'key=value'
        attrs_str = ', '.join(f'{key}={value!r}' for key, value in attrs.items())
        # Format the output as 'ClassName(attr1=value1, attr2=value2, ...)'
        return f'{self.__class__.__name__}({attrs_str})'

    def __str__(self):
        return str(self.active_data)

    def _apply_op(self, other, op):
        """Helper function to apply an operation using active_data."""
        if isinstance(other, TimeSeries):
            return op(self.active_data, other.active_data)
        return op(self.active_data, other)

    def __add__(self, other):
        return self._apply_op(other, np.add)

    def __sub__(self, other):
        return self._apply_op(other, np.subtract)

    def __mul__(self, other):
        return self._apply_op(other, np.multiply)

    def __truediv__(self, other):
        return self._apply_op(other, np.divide)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self._apply_op(other, lambda x, y: x - y)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self._apply_op(other, lambda x, y: x / y)
