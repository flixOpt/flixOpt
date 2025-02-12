"""
This module contains the core functionality of the flixOpt framework.
It provides Datatypes, logging functionality, and some functions to transform data structures.
"""

import inspect
import logging
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import xarray as xr
import pandas as pd

from . import utils

logger = logging.getLogger('flixOpt')

Skalar = Union[int, float]  # Datatype
Numeric = Union[int, float, np.ndarray]  # Datatype


class DataConverter:
    @staticmethod
    def as_series(data: Union[Numeric, pd.Series, pd.DataFrame], dims: Tuple[pd.Index, ...]) -> pd.Series:
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

        index = pd.MultiIndex.from_product(dims) if len(dims) > 1 else dims[0]

        if isinstance(data, (int, float)):  # Scalar case
            return DataConverter._handle_scalar(data, index)

        if isinstance(data, np.ndarray):
            return DataConverter._handle_array(data, dims, index)

        if isinstance(data, pd.Series):
            return DataConverter._handle_series(data, dims, index)

        if isinstance(data, pd.DataFrame):
            return DataConverter._handle_dataframe(data, dims, index)

        raise TypeError("Unsupported data type. Must be scalar, np.ndarray, pd.Series, or pd.DataFrame.")

    @staticmethod
    def _handle_scalar(data: Union[int, float], index: pd.Index) -> pd.Series:
        """Handles scalar input."""
        return pd.Series(data, index=index)

    @staticmethod
    def _handle_array(data: np.ndarray, dims: Tuple[pd.Index, ...], index: pd.Index) -> pd.Series:
        """Handles NumPy array input."""
        expected_shape = tuple(len(idx) for idx in dims)

        if data.ndim == 1 and len(dims) == 2:  # Automatically reshape 1D arrays
            if data.shape[0] == len(dims[0]):
                data = np.tile(data[:, np.newaxis], (1, len(dims[1])))  # Expand along second dimension
            elif data.shape[0] == len(dims[1]):
                data = np.tile(data[np.newaxis, :], (len(dims[0]), 1))  # Expand along first dimension
            else:
                raise ValueError("1D array length does not match either dimension in dims")

        if data.shape != expected_shape:
            raise ValueError(f"Shape of data {data.shape} does not match expected shape {expected_shape}")

        return pd.Series(data.ravel(), index=index)

    @staticmethod
    def _handle_series(data: pd.Series, dims: Tuple[pd.Index, ...], index: pd.Index) -> pd.Series:
        """Handles pandas Series input."""
        if len(dims) == 1:
            if not data.index.equals(dims[0]):
                raise ValueError("Series index does not match the provided index")
            return data
        return pd.Series(data.values.ravel(), index=index)

    @staticmethod
    def _handle_dataframe(data: pd.DataFrame, dims: Tuple[pd.Index, ...], index: pd.Index) -> pd.Series:
        """Handles pandas DataFrame input."""
        if len(dims) != 2 or data.shape != (len(dims[0]), len(dims[1])):
            raise ValueError("DataFrame shape does not match provided indexes")

        # Stack and ensure columns become level 0
        stacked = data.stack().swaplevel(0, 1).sort_index()
        if not stacked.index.equals(index):
            raise ValueError("Stacked DataFrame index does not match the provided index")

        return stacked


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
    def __init__(self, data: pd.Series, aggregation_weight: Optional[float] = None):
        """
        Initialize the TimeSeriesManager with a Series.

        Parameters:
        - data (pd.Series): A Series with a DatetimeIndex and possibly a MultiIndex.
        - aggregation_weight (float, optional): The weight of the data in the aggregation. Defaults to None.
        """
        self._stored_data = data.copy()  # Store data
        self._backup: pd.Series = self.stored_data  # Single backup instance. Enables to temporarily overwrite the data.
        self._active_data = None
        self._active_index = None
        self.aggregation_weight = aggregation_weight

        self.active_index = None  # Initializes the active index and active data

    def restore_data(self):
        """Restore stored_data from the backup."""
        self._stored_data = self._backup.copy()
        self.active_index = None

    def as_dataarray(self) -> xr.DataArray:
        return self.active_data.to_xarray()

    @property
    def active_index(self) -> pd.Index:
        """Return the current active index."""
        return self._active_index

    @active_index.setter
    def active_index(self, index: Optional[pd.Index]):
        """Set a new active index and refresh active_data."""
        if index is None:
            self._active_index = self._stored_data.index
            self._active_data = self._stored_data
            return
        elif not isinstance(index, (pd.Index, pd.MultiIndex)):
            raise TypeError("active_index must be a pandas Index or MultiIndex or None")
        else:
            self._active_index = index
            self._active_data = self.stored_data.loc[self._active_index]  # Refresh view

    @property
    def active_data(self) -> pd.Series:
        """Return a view of stored_data based on active_index."""
        return self._active_data

    @active_data.setter
    def active_data(self, value):
        """Prevent direct modification of active_data."""
        raise AttributeError("active_data cannot be directly modified. Modify stored_data instead.")

    @property
    def stored_data(self) -> pd.Series:
        """Return a copy of stored_data. Prevents modification of stored data"""
        return self._stored_data.copy()

    @stored_data.setter
    def stored_data(self, value: pd.Series):
        """Set stored_data and refresh active_index and active_data."""
        self._backup = self._stored_data
        self._stored_data = value
        self.active_index = None

    @property
    def loc(self):
        """Access active_data using loc."""
        return self.active_data.loc

    @property
    def iloc(self):
        """Access active_data using iloc."""
        return self.active_data.iloc

    # Enable arithmetic operations using active_data
    def _apply_operation(self, other, op):
        if isinstance(other, TimeSeries):
            other = other.active_data
        if isinstance(other, xr.DataArray):
            return op(self.as_dataarray(), other)
        return op(self.active_data, other)

    def __add__(self, other):
        return self._apply_operation(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self._apply_operation(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self._apply_operation(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self._apply_operation(other, lambda x, y: x / y)

    def __floordiv__(self, other):
        return self._apply_operation(other, lambda x, y: x // y)

    def __pow__(self, other):
        return self._apply_operation(other, lambda x, y: x ** y)

    # Reflected arithmetic operations (to handle cases like `some_xarray + ts1`)
    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self._apply_operation(other, lambda x, y: y - x)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self._apply_operation(other, lambda x, y: y / x)

    def __rfloordiv__(self, other):
        return self._apply_operation(other, lambda x, y: y // x)

    def __rpow__(self, other):
        return self._apply_operation(other, lambda x, y: y ** x)

    # Unary operations. Not sure if this is the best way...
    def __neg__(self):
        return -self.as_dataarray()

    def __pos__(self):
        return +self.as_dataarray()

    def __abs__(self):
        return abs(self.as_dataarray())

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Ensures NumPy functions like np.add(TimeSeries, xarray) work correctly."""
        inputs = [x.as_dataarray() if isinstance(x, TimeSeries) else x for x in inputs]
        result = getattr(ufunc, method)(*inputs, **kwargs)

        # Ensure return type consistency
        if isinstance(result, xr.DataArray):
            return result
        elif isinstance(result, np.ndarray):  # Handles cases like np.exp(ts)
            return pd.Series(result, index=self.active_data.index)
        else:
            raise NotImplementedError(f"ufunc {ufunc} not implemented for TimeSeries")


