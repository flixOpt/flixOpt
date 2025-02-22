"""
This module contains the core functionality of the flixOpt framework.
It provides Datatypes, logging functionality, and some functions to transform data structures.
"""

import inspect
import logging
from collections import Counter
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger('flixOpt')

Skalar = Union[int, float]  # Datatype
Numeric = Union[int, float, np.ndarray]  # Datatype

NumericData = Union[int, float, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray]


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
    def as_dataarray(data: NumericData, time: pd.DatetimeIndex, period: Optional[pd.Index] = None) -> xr.DataArray:
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
        elif isinstance(data, pd.DataFrame):
            return DataConverter._handle_dataframe(data, coords, dims)
        elif isinstance(data, pd.Series):
            return DataConverter._handle_series(data, coords, dims)
        elif isinstance(data, np.ndarray):
            return DataConverter._handle_array(data, coords, dims)
        elif isinstance(data, xr.DataArray):
            return data

        raise TypeError(f"Unsupported data type. Must be scalar, np.ndarray, pd.Series, or pd.DataFrame."
                        f"Got {type(data)=}")

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

    @staticmethod
    def _handle_xr_dataarray(data: xr.DataArray, coords: list, dims: list) -> xr.DataArray:
        """Handles xr.DataArray input."""
        if data.ndim != len(coords):
            raise ValueError(f"DataArray must have {len(coords)} dimensions, got {data.ndim}")
        if data.dims != dims:
            raise ValueError(f"DataArray dimensions {data.dims} do not match expected dimensions {dims}")
        if data.shape != tuple(coord.size for coord in coords):
            raise ValueError(f"DataArray shape {data.shape} does not match expected shape {tuple(coord.size for coord in coords)}")
        # TODO: This is not really thought through or tested
        return data


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

    def reset(self):
        """Reset active timesteps and periods."""
        self.active_timesteps = None
        self.active_periods = None

    def restore_data(self):
        """Restore stored_data from the backup."""
        self._stored_data = self._backup.copy()
        self.reset()

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
    def active_periods(self) -> Optional[pd.Index]:
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
    def stored_data(self, value: Union[pd.Series, pd.DataFrame, xr.DataArray]):
        """
        Update stored_data and refresh active_index and active_data.

        Parameters
        ----------
        value: Union[int, float, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray]
            Data to update stored_data with.
        """
        new_data = DataConverter.as_dataarray(value, time=self.active_timesteps, period=self.active_periods)
        if new_data.equals(self._stored_data):
            return  # No change in stored_data. Do nothing. This prevents pushing out the backup
        self._backup = self._stored_data
        self._stored_data = new_data
        self.active_timesteps = None
        self.active_periods = None

    @property
    def sel(self):
        return self.active_data.sel

    @property
    def isel(self):
        return self.active_data.isel

    def _apply_operation(self, other, op):
        # Enable arithmetic operations using active_data
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


class TimeSeriesCollection:
    def __init__(
            self,
            timesteps: pd.DatetimeIndex,
            hours_of_last_timestep: Optional[float],
            hours_of_previous_timesteps: Optional[Union[int, float, np.ndarray]],
            periods: Optional[List[int]]
    ):
        self.hours_of_last_timestep = hours_of_last_timestep
        (
            self._timesteps,
            self._timesteps_extra,
            self._hours_per_timestep,
            self._hours_of_previous_timesteps,
            self._periods
        ) = TimeSeriesCollection.allign_dimensions(timesteps,
                                                   periods,
                                                   hours_of_last_timestep,
                                                   hours_of_previous_timesteps)

        self._active_timesteps = None
        self._active_timesteps_extra = None
        self._active_periods = None
        self._active_hours_per_timestep = None

        self.group_weights: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}
        self.time_serieses: List[TimeSeries] = []
        self._timeserieses_longer: List[TimeSeries] = []  # All part of self.time_serieses, but with extra timestep

    def _add_time_series(self, time_series: TimeSeries, extra_timestep: bool):
        self.time_serieses.append(time_series)
        if extra_timestep:
            self._timeserieses_longer.append(time_series)
        self._check_unique_labels()

    def create_time_series(
        self,
        data: Union[NumericData, TimeSeriesData],
        name: str,
        extra_timestep: bool=False
    ) -> TimeSeries:
        """
        Creates a TimeSeries from the given data and adds it to the list of time_serieses of an Element.
        If the data already is a TimeSeries, nothing happens.

        Parameters
        ----------
        data: Union[int, float, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray]
            The data to create the TimeSeries from.
        name: str
            The name of the TimeSeries.
        extra_timestep: bool, optional
            Whether to create an additional timestep at the end of the timesteps.

        Returns
        -------
        TimeSeries
            The created TimeSeries.

        """
        if isinstance(data, TimeSeries):
            if data not in self.time_serieses:
                self._add_time_series(data, extra_timestep)
            return data

        time_series = TimeSeries.from_datasource(
            name=name,
            data=data if not isinstance(data, TimeSeriesData) else data.data,
            timesteps=self.timesteps if not extra_timestep else self.timesteps_extra,
            periods=self.periods,
            aggregation_weight=data.agg_weight if isinstance(data, TimeSeriesData) else None,
            aggregation_group=data.agg_group if isinstance(data, TimeSeriesData) else None
        )

        if isinstance(data, TimeSeriesData):
            data.label = time_series.name  # Connecting User_time_series to TimeSeries

        self._add_time_series(time_series, extra_timestep)
        return time_series

    def calculate_aggregation_weights(self) -> Dict[str, float]:
        self.group_weights = self._calculate_group_weights()
        self.weights = self._calculate_aggregation_weights()

        if np.all(np.isclose(list(self.weights.values()), 1, atol=1e-6)):
            logger.info('All Aggregation weights were set to 1')

        return self.weights

    def update_data(self,
                    active_timesteps: Optional[pd.DatetimeIndex] = None,
                    active_periods: Optional[pd.Index] = None):
        """
        Update active timesteps, periods, and data of the TimeSeriesCollection.

        Parameters
        ----------
        active_timesteps : Optional[pd.DatetimeIndex]
            The active timesteps of the model.
            If None, the all timesteps of the TimeSeriesCollection are taken.
        active_periods : Optional[pd.Index]
            The active periods of the model.
            If None, all periods from the TimeSeriesCollection are taken.
        """

        if active_timesteps is None and active_periods is None:
            raise ValueError('Either active_timesteps or active_periods must be provided.'
                             'Else use .reset() to reset the active timesteps and periods.')

        active_timesteps = active_timesteps if active_timesteps is not None else self._timesteps
        active_periods = active_periods if active_periods is not None else self._periods

        if not active_timesteps.isin(self._timesteps):
            raise ValueError('active_timesteps must be a subset of the timesteps of the TimeSeriesCollection')
        if not active_periods.isin(self._periods):
            raise ValueError('active_periods must be a subset of the periods of the TimeSeriesCollection')


        (
            self._active_timesteps,
            self._active_timesteps_extra,
            self._active_hours_per_timestep,
            _,
            self._active_periods
        ) = TimeSeriesCollection.allign_dimensions(
            active_timesteps, active_periods, self.hours_of_last_timestep, self._hours_of_previous_timesteps
        )

        self._active_timeserieses()

    def reset(self):
        """Reset active timesteps and periods of all TimeSeries."""
        self._active_timesteps = None
        self._active_timesteps_extra = None
        self._active_hours_per_timestep = None
        self._active_periods = None
        for time_series in self.time_serieses:
            time_series.reset()

    def insert_new_data(self, data: pd.DataFrame):
        """Insert new data into the TimeSeriesCollection.

        Parameters
        ----------
        data : pd.DataFrame
            The new data to insert.
            Must have the same columns as the TimeSeries in the TimeSeriesCollection.
            Must have the same index as the timesteps of the TimeSeriesCollection.
        """
        #TODO: Sanitize the values for timeseries that are one step longer!
        for time_series in self.time_serieses:
            if time_series.name in data.columns:
                time_series.stored_data = data[time_series.name]
                logger.debug(f'Inserted data for {time_series.name}')

    def _active_timeserieses(self):
        for time_series in self.time_serieses:
            time_series.active_periods = self.periods
            if time_series in self._timeserieses_longer:
                time_series.active_timesteps = self.timesteps_extra
            else:
                time_series.active_timesteps = self.timesteps

    def to_dataframe(self, filtered: Literal['all', 'constant', 'non_constant'] = 'non_constant'):
        if filtered == 'all':
            return pd.concat([time_series.active_data.to_dataframe(time_series.name)
                              for time_series in self.time_serieses],
                             axis=1)
        elif filtered == 'constant':
            return pd.concat([time_series.active_data.to_dataframe(time_series.name)
                              for time_series in self.constants],
                             axis=1)
        elif filtered == 'non_constant':
            return pd.concat([time_series.active_data.to_dataframe(time_series.name)
                              for time_series in self.non_constants],
                             axis=1)
        else:
            raise ValueError('Not supported argument for "filtered".')

    @staticmethod
    def allign_dimensions(
        timesteps: pd.DatetimeIndex,
        periods: Optional[pd.Index] = None,
        hours_of_last_timestep: Optional[float] = None,
        hours_of_previous_timesteps: Optional[Union[int, float, np.ndarray]] = None
    ) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, xr.DataArray, Union[int, float, np.ndarray], Optional[pd.Index]]:
        """ Converts the given timesteps, periods and hours_of_last_timestep to the right format

        Parameters
        ----------
        timesteps : pd.DatetimeIndex
            The timesteps of the model.
        hours_of_last_timestep : Optional[float], optional
            The duration of the last time step. Uses the last time interval if not specified
        hours_of_previous_timesteps: Un
        periods : Optional[pd.Index], optional
            The periods of the model. Every period has the same timesteps.
            Usually years are used as periods.

        Returns
        -------
        Tuple[pd.DatetimeIndex, pd.DatetimeIndex, xr.DataArray, Optional[pd.Index]]
            The timesteps, timesteps_extra, hours_per_timestep and periods

            - timesteps: pd.DatetimeIndex
                The timesteps of the model.
            - timesteps_extra: pd.DatetimeIndex
                The timesteps of the model, including an extra timestep at the end.
            - hours_per_timestep: xr.DataArray
                The duration of each timestep in hours.
            - hours_of_previous_timesteps: Optional[Union[int, float, np.ndarray]]
                The duration of the previous timesteps in hours.
            - periods: Optional[pd.Index]
                The periods of the model. Every period has the same timesteps.
                Usually years are used as periods.
        """
        if not isinstance(timesteps, pd.DatetimeIndex):
            raise TypeError('timesteps must be a pandas DatetimeIndex')
        if not timesteps.name == 'time':
            logger.warning('timesteps must be a pandas DatetimeIndex with name "time". Renamed it to "time".')
            timesteps.name = 'time'

        if periods is not None and not isinstance(periods, pd.Index):
            raise TypeError('periods must be a pandas Index or None')
        if periods is not None and periods.name != 'period':
            logger.warning('periods must be a pandas Index with name "period". Renamed it.')
            periods.name = 'period'

        if hours_of_last_timestep:
            last_date = pd.DatetimeIndex(
                [timesteps[-1] + pd.to_timedelta(hours_of_last_timestep, 'h')])
        else:
            last_date = pd.DatetimeIndex([timesteps[-1] + (timesteps[-1] - timesteps[-2])])

        timesteps_extra = pd.DatetimeIndex(timesteps.append(last_date), name='time')

        hours_of_previous_timesteps: Union[int, float, np.ndarray] = (
            ((timesteps[1] - timesteps[0]) / np.timedelta64(1, 'h'))
            if hours_of_previous_timesteps is None
            else hours_of_previous_timesteps
        )

        hours_per_step = timesteps_extra.to_series().diff()[1:].values / pd.to_timedelta(1, 'h')
        hours_per_step = xr.DataArray(
            data=np.tile(hours_per_step, (len(periods), 1)) if periods is not None else hours_per_step,
            coords=(periods, timesteps) if periods is not None else (timesteps,),
            dims=('period', 'time') if periods is not None else ('time',),
            name='hours_per_step'
        )
        return timesteps, timesteps_extra , hours_per_step, hours_of_previous_timesteps, periods

    @property
    def non_constants(self) -> List[TimeSeries]:
        return [time_series for time_series in self.time_serieses if not time_series.all_equal]

    @property
    def constants(self) -> List[TimeSeries]:
        return [time_series for time_series in self.time_serieses if time_series.all_equal]

    @property
    def coords(self):
        return [self.periods, self.timesteps] if self.periods is not None else [self.timesteps]

    @property
    def coords_extra(self):
        return [self.periods, self.timesteps_extra] if self.periods is not None else [self.timesteps_extra]

    @property
    def timesteps(self):
        return self._timesteps if self._active_timesteps is None else self._active_timesteps

    @property
    def timesteps_extra(self):
        return self._timesteps_extra if self._active_timesteps_extra is None else self._active_timesteps_extra

    @property
    def periods(self):
        return self._periods if self._active_periods is None else self._active_periods

    @property
    def hours_per_timestep(self):
        return self._hours_per_timestep if self._active_hours_per_timestep is None else self._active_hours_per_timestep

    def description(self) -> str:
        # TODO:
        result = f'{len(self.time_serieses)} TimeSeries used for aggregation:\n'
        for time_series in self.time_serieses:
            result += f' -> {time_series.name} (weight: {self.weights[time_series.name]:.4f}; group: "{time_series.aggregation_group}")\n'
        if self.group_weights:
            result += f'Aggregation_Groups: {list(self.group_weights.keys())}\n'
        else:
            result += 'Warning!: no agg_types defined, i.e. all TS have weight 1 (or explicitly given weight)!\n'
        return result

    def _calculate_group_weights(self) -> Dict[str, float]:
        """Calculates the aggregation weights of each group"""
        groups = [
            time_series.aggregation_group
            for time_series in self.time_serieses
            if time_series.aggregation_group is not None
        ]
        group_size = dict(Counter(groups))
        group_weights = {group: 1 / size for group, size in group_size.items()}
        return group_weights

    def _calculate_aggregation_weights(self) -> Dict[str, float]:
        """Calculates the aggregation weight for each TimeSeries. Default is 1"""
        return {
            time_series.name: self.group_weights.get(time_series.aggregation_group, time_series.aggregation_weight or 1)
            for time_series in self.time_serieses
        }

    def _check_unique_labels(self):
        """Makes sure every label of the TimeSeries in time_series_list is unique"""
        label_counts = Counter([time_series.name for time_series in self.time_serieses])
        duplicates = [label for label, count in label_counts.items() if count > 1]
        assert duplicates == [], 'Duplicate TimeSeries labels found: {}.'.format(', '.join(duplicates))
