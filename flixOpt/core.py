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


logger = logging.getLogger('flixOpt')

Skalar = Union[int, float]  # Datatype
Numeric = Union[int, float, np.ndarray]  # Datatype


class DataConverter:
    """
    A utility class for converting various data types into an xarray.DataArray
    with specified time and optional period indexes.

    Supported input types:
    - int or float: Generates a DataArray filled with the given scalar value.
    - pd.Series: Index should be time steps; expands over periods if provided.
    - pd.DataFrame: Columns represent periods, and the index represents time steps.
      If a single column is passed but periods exist, the data is expanded over periods.
    - np.ndarray:
        - If 1D, attempts to reshape based on time steps and periods.
        - If 2D, ensures dimensions match time steps and periods, transposing if necessary.
        - Logs a warning if periods and time steps have the same length to prevent confusion.

    Raises:
    - TypeError if an unsupported data type is provided.
    - ValueError if data dimensions do not match expected time and period indexes.
    """
    @staticmethod
    def as_dataarray(data: Union[Numeric, pd.Series, pd.DataFrame, np.ndarray], time: pd.DatetimeIndex,
                     period: Optional[pd.Index] = None) -> xr.DataArray:
        """
        Converts the given data to an xarray.DataArray with the specified time and period indexes.
        """
        if period is not None:
            coords = [period, time]
            dims = ['period', 'time']
        else:
            coords = [time]
            dims = ['time']

        if isinstance(data, (int, float)):
            return DataConverter._handle_scalar(data, coords, dims)
        if isinstance(data, pd.DataFrame):
            return DataConverter._handle_dataframe(data, coords, dims)
        if isinstance(data, pd.Series):
            return DataConverter._handle_series(data, coords, dims)
        if isinstance(data, np.ndarray):
            return DataConverter._handle_array(data, coords, dims)

        raise TypeError("Unsupported data type. Must be scalar, np.ndarray, pd.Series, or pd.DataFrame.")

    @staticmethod
    def _handle_scalar(data: Numeric, coords: list, dims: list) -> xr.DataArray:
        """Handles scalar input by filling the array with the value."""
        return xr.DataArray(data, coords=coords, dims=dims)

    @staticmethod
    def _handle_dataframe(data: pd.DataFrame, coords: list, dims: list) -> xr.DataArray:
        """Handles pandas DataFrame input."""
        if len(coords) == 2:
            if data.shape[1] == 1:
                return DataConverter._handle_series(data.iloc[:, 0], coords, dims)
            elif data.shape != (len(coords[1]), len(coords[0])):
                raise ValueError("DataFrame shape does not match provided indexes")
        return xr.DataArray(data.T, coords=coords, dims=dims)

    @staticmethod
    def _handle_series(data: pd.Series, coords: list, dims: list) -> xr.DataArray:
        """Handles pandas Series input."""
        if len(coords) == 2:
            if data.shape[0] != len(coords[1]):
                raise ValueError(f"Series index does not match the shape of the provided timsteps: {data.shape[0]= } != {len(coords[1])=}")
            return xr.DataArray(np.tile(data.values, (len(coords[0]), 1)), coords=coords, dims=dims)
        return xr.DataArray(data.values, coords=coords, dims=dims)

    @staticmethod
    def _handle_array(data: np.ndarray, coords: list, dims: list) -> xr.DataArray:
        """Handles NumPy array input."""
        expected_shape = tuple(len(coord) for coord in coords)

        if data.ndim == 1 and len(coords) == 2:
            if data.shape[0] == len(coords[0]):
                data = np.tile(data[:, np.newaxis], (1, len(coords[1])))
            elif data.shape[0] == len(coords[1]):
                data = np.tile(data[np.newaxis, :], (len(coords[0]), 1))
            else:
                raise ValueError("1D array length does not match either dimension in coords")

        if data.shape != expected_shape:
            raise ValueError(f"Shape of data {data.shape} does not match expected shape {expected_shape}")

        return xr.DataArray(data, coords=coords, dims=dims)


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

    @classmethod
    def from_datasource(cls,
                        data: Union[int, float, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray],
                        name: str,
                        timesteps: pd.DatetimeIndex = None,
                        periods: Optional[pd.Index] = None,
                        aggregation_weight: Optional[float] = None,
                        aggregation_group: Optional[str] = None
                        ) -> 'TimeSeries':
        """
        Initialize the TimeSeries from multiple datasources.

        Parameters:
        - data (pd.Series): A Series with a DatetimeIndex and possibly a MultiIndex.
        - dims (Tuple[pd.Index, ...]): The dimensions of the TimeSeries.
        - aggregation_weight (float, optional): The weight of the data in the aggregation. Defaults to None.
        - aggregation_group (str, optional): The group this TimeSeries is a part of. agg_weight is split between members of a group. Default is None.
        """
        data = cls(DataConverter.as_dataarray(data, timesteps, periods), name, aggregation_weight, aggregation_group)
        return data

    def __init__(self,
                 data: xr.DataArray,
                 name: str,
                 aggregation_weight: Optional[float] = None,
                 aggregation_group: Optional[str] = None):
        """
        Initialize a TimeSeries with a DataArray.

        Parameters:
        - data (xr.DataArray): A Series with a DatetimeIndex and possibly a MultiIndex.
        - aggregation_weight (float, optional): The weight of the data in the aggregation. Defaults to None.
        - aggregation_group (str, optional): The group this TimeSeries is a part of. agg_weight is split between members of a group. Default is None.
        """
        if 'time' not in data.indexes:
            raise ValueError(f'DataArray must have a "time" index. Got {data.indexes}')
        if 'period' not in data.indexes and data.ndim > 1:
            raise ValueError(f'Second index of DataArray must be "period". Got {data.indexes}')

        self.name = name
        self.aggregation_weight = aggregation_weight
        self.aggregation_group = aggregation_group

        self._stored_data = data.copy()
        self._backup: xr.DataArray = self.stored_data  # Single backup instance. Enables to temporarily overwrite the data.
        self._active_data = None

        self._active_timesteps = self.stored_data.indexes['time']
        self._active_periods = self.stored_data.indexes['period'] if 'period' in self.stored_data.indexes else None
        self._update_active_data()

    def restore_data(self):
        """Restore stored_data from the backup."""
        self._stored_data = self._backup.copy()
        self.active_timesteps = None
        self.active_periods = None

    def _update_active_data(self):
        """Update the active data."""
        if 'period' in self._stored_data.indexes:
            self._active_data = self._stored_data.sel(time=self.active_timesteps, period=self.active_periods)
        else:
            self._active_data = self._stored_data.sel(time=self.active_timesteps)

    @property
    def all_equal(self) -> bool:
        """ Checks for all values in the being equal"""
        return np.unique(self.active_data.values).size == 1

    @property
    def active_timesteps(self) -> pd.DatetimeIndex:
        """Return the current active index."""
        return self._active_timesteps

    @active_timesteps.setter
    def active_timesteps(self, timesteps: Optional[pd.DatetimeIndex]):
        """Set active_timesteps and refresh active_data."""
        if timesteps is None:
            self._active_timesteps = self.stored_data.indexes['time']
        elif isinstance(timesteps, pd.DatetimeIndex):
            self._active_timesteps = timesteps
        else:
            raise TypeError("active_index must be a pandas Index or MultiIndex or None")

        self._update_active_data()  # Refresh view

    @property
    def active_periods(self) -> pd.Index:
        """Return the current active index."""
        return self._active_periods

    @active_periods.setter
    def active_periods(self, periods: Optional[pd.Index]):
        """Set new active periods and refresh active_data."""
        if periods is None:
            self._active_periods = self.stored_data.indexes['period'] if 'period' in self.stored_data.indexes else None
        elif isinstance(periods, pd.Index):
            self._active_periods = periods
        else:
            raise TypeError("periods must be a pd.Index or None")

        self._update_active_data()  # Refresh view

    @property
    def active_data(self) -> xr.DataArray:
        """Return a view of stored_data based on active_index."""
        return self._active_data

    @active_data.setter
    def active_data(self, value):
        """Prevent direct modification of active_data."""
        raise AttributeError("active_data cannot be directly modified. Modify stored_data instead.")

    @property
    def stored_data(self) -> xr.DataArray:
        """Return a copy of stored_data. Prevents modification of stored data"""
        return self._stored_data.copy()

    @stored_data.setter
    def stored_data(self, value: xr.DataArray):
        """Set stored_data and refresh active_index and active_data."""
        self._backup = self._stored_data
        self._stored_data = value
        self.active_timesteps = None
        self.active_periods = None

    @property
    def sel(self):
        return self.active_data.sel

    @property
    def isel(self):
        return self.active_data.isel

    # Enable arithmetic operations using active_data
    def _apply_operation(self, other, op):
        if isinstance(other, TimeSeries):
            other = other.active_data
        return op(self.active_data, other)

    def __add__(self, other):
        return self._apply_operation(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self._apply_operation(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self._apply_operation(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self._apply_operation(other, lambda x, y: x / y)

    # Reflected arithmetic operations (to handle cases like `some_xarray + ts1`)
    def __radd__(self, other):
        return other + self.active_data

    def __rsub__(self, other):
        return other - self.active_data

    def __rmul__(self, other):
        return other * self.active_data

    def __rtruediv__(self, other):
        return other / self.active_data

    # Unary operations. Not sure if this is the best way...
    def __neg__(self):
        return -self.active_data

    def __pos__(self):
        return +self.active_data

    def __abs__(self):
        return abs(self.active_data)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Ensures NumPy functions like np.add(TimeSeries, xarray) work correctly."""
        inputs = [x.active_data if isinstance(x, TimeSeries) else x for x in inputs]
        return getattr(ufunc, method)(*inputs, **kwargs)
