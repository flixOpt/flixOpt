"""
This module contains the core functionality of the flixOpt framework.
It provides Datatypes, logging functionality, and some functions to transform data structures.
"""

from typing import Union, Optional, List, Dict, Any, Literal
import logging
import inspect

import numpy as np
from rich.logging import RichHandler
from rich.console import Console

from . import utils

logger = logging.getLogger('flixOpt')

Skalar = Union[int, float]  # Datatype
Numeric = Union[int, float, np.ndarray]  # Datatype
# zeitreihenbezogene Input-Daten:
Numeric_TS = Union[Skalar, np.ndarray, 'TimeSeries']
# Datatype Numeric_TS:
#   Skalar      --> wird später dann in array ("Zeitreihe" mit length=nrOfTimeIndexe) übersetzt
#   np.ndarray  --> muss length=nrOfTimeIndexe haben ("Zeitreihe")
#   TimeSeriesData      --> wie obige aber zusätzliche Übergabe aggWeight (für Aggregation)


class Config:
    """
    Configuration class for global settings.
    The values are used as defaults in several classes.
    They can be overwritten by the user via the .update() - method.
    Use with care, and make sure to adjust them in the beginning of the script.
    """
    BIG_M: Union[int, float] = 1e7
    EPSILON: Union[int, float] = 1e-5
    OFFSET_TO_BIG_M: Union[int, float] = 100
    BIG_BINARY_BOUND: Union[int, float] = BIG_M / OFFSET_TO_BIG_M

    @classmethod
    def update(cls, big_m: Optional[int] = None,
               epsilon: Optional[float] = None,
               offset_to_big_m: Optional[int] = None,
               big_binary_bound: Optional[int] = None) -> None:
        """
        Update the configuration with the given values.
        -----
        Parameters
        -----------
        big_m: int, optional
            The value of the big M constant. Defaults to 1e7.
        epsilon: float, optional
            The value of the epsilon constant. Defaults to 1e-5.
        offset_to_big_m: int, optional
            The value of the offset to big M constant for the big binary bound. Defaults to 100.
        big_binary_bound: int, optional
            The value of the big binary bound. Defaults to the value of big M minus the offset to big M.
            Use either this or the offset!
        """
        if big_binary_bound is not None and offset_to_big_m is not None:
            raise ValueError(f'Either use "offset_to_big_m" or set the "big_binary_bound" directly. Not Both')
        if big_m is not None:
            cls.BIG_M = big_m
        if epsilon is not None:
            cls.EPSILON = epsilon
        if big_binary_bound is not None:
            cls.BIG_BINARY_BOUND = big_binary_bound
        if offset_to_big_m is not None:
            cls.BIG_BINARY_BOUND = cls.BIG_M / offset_to_big_m


class TimeSeriesData:
    # TODO: Move to Interface.py
    def __init__(self,
                 data: Numeric,
                 agg_group: Optional[str] = None,
                 agg_weight: Optional[float] = None):
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
        args_str = ', '.join(
            f"{name}={repr(getattr(self, name, None))}"
            for name in init_args if name != 'self'
        )
        return f"{self.__class__.__name__}({args_str})"

    def __str__(self):
        return str(self.data)


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

    def __init__(self, label: str, data: Optional[Numeric_TS]):
        self.label: str = label
        if isinstance(data, TimeSeriesData):
            self.data = self.make_scalar_if_possible(data.data)
            self.aggregation_weight, self.aggregation_group = data.agg_weight, data.agg_group
            data.label = self.label  # Connecting User_time_series to real Time_series
        else:
            self.data = self.make_scalar_if_possible(data)
            self.aggregation_weight, self.aggregation_group = None, None

        self.active_indices: Optional[Union[range, List[int]]] = None
        self.aggregated_data: Optional[Numeric] = None

    def activate_indices(self, indices: Optional[Union[range, List[int]]], aggregated_data: Optional[Numeric] = None):
        self.active_indices = indices

        if aggregated_data is not None:
            assert len(aggregated_data) == len(self.active_indices) or len(aggregated_data) == 1, \
                (f'The aggregated_data has the wrong length for TimeSeries {self.label}. '
                 f'Length should be: {len(self.active_indices)} or 1, but is {len(aggregated_data)}')
            self.aggregated_data = self.make_scalar_if_possible(aggregated_data)

    def clear_indices_and_aggregated_data(self):
        self.active_indices = None
        self.aggregated_data = None

    @property
    def active_data(self) -> Numeric:
        if self.aggregated_data is not None:  # Aggregated data is always active, if present
            return self.aggregated_data

        indices_not_applicable = np.isscalar(self.data) or (self.data is None) or (self.active_indices is None)
        if indices_not_applicable:
            return self.data
        else:
            return self.data[self.active_indices]

    @property
    def active_data_vector(self) -> np.ndarray:
        # Always returns the active data as a vector.
        return utils.as_vector(self.active_data, len(self.active_indices))

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
        attrs_str = ', '.join(f"{key}={value!r}" for key, value in attrs.items())
        # Format the output as 'ClassName(attr1=value1, attr2=value2, ...)'
        return f"{self.__class__.__name__}({attrs_str})"

    def __str__(self):
        return str(self.active_data)

    @staticmethod
    def make_scalar_if_possible(data: Optional[Numeric]) -> Optional[Numeric]:
        """
        Convert an array to a scalar if all values are equal, or return the array as-is.
        Can Return None if the passed data is None

        Parameters
        ----------
        data : Numeric, None
            The data to process.

        Returns
        -------
        Numeric
            A scalar if all values in the array are equal, otherwise the array itself. None, if the passed value is None
        """
        #TODO: Should this really return None Values?
        if np.isscalar(data) or data is None:
            return data
        data = np.array(data)
        if np.all(data == data[0]):
            return data[0]
        return data


def as_effect_dict(effect_values: Union[Numeric, TimeSeries, Dict]) -> Optional[Dict]:
    """
    Converts effect values into a dictionary. If a scalar value is provided, it is associated with a standard effect type.

    Examples
    --------
    If costs are given without effectType, a standard effect is related:
      costs = 20                        -> {None: 20}
      costs = None                      -> no change
      costs = {effect1: 20, effect2: 0.3} -> no change

    Parameters
    ----------
    effect_values : None, int, float, TimeSeries, or dict
        The effect values to convert can be a scalar, a TimeSeries, or a dictionary with an effectas key

    Returns
    -------
    dict or None
        Converted values in from of dict with either None or Effect as key. if values is None, None is returned
    """
    if isinstance(effect_values, dict):
        return effect_values
    elif effect_values is None:
        return None
    else:
        return {None: effect_values}


def effect_values_to_ts(name_of_param: str, effect_dict: Optional[Dict[Any, Union[Numeric, TimeSeries]]], owner) -> Optional[Dict[Any, TimeSeries]]:
    """
    Transforms values in a dictionary to instances of TimeSeries.

    Parameters
    ----------
    name_of_param : str
        The base name of the parameter.
    effect_dict : dict
        A dictionary with effect-value pairs.
    owner : object
        The owner object where TimeSeries belongs to.

    Returns
    -------
    dict
        A dictionary with effect types as keys and TimeSeries instances as values.
    """
    if effect_dict is None:
        return None

    transformed_dict = {}
    for effect, value in effect_dict.items():
        if not isinstance(value, TimeSeries):
            subname = 'standard' if effect is None else effect.label
            full_name = f"{name_of_param}_{subname}"
            transformed_dict[effect] = TimeSeries(full_name, value, owner)

    return transformed_dict


def as_effect_dict_with_ts(name_of_param: str,
                           effect_values: Union[Numeric, TimeSeries, Dict],
                           owner
                           ) -> Optional[Dict[Any, TimeSeries]]:
    """
    Transforms effect or cost input to a dictionary of TimeSeries instances.

    If only a value is given, it is associated with a standard effect type.

    Parameters
    ----------
    name_of_param : str
        The base name of the parameter.
    effect_values : int, float, TimeSeries, or dict
        The effect values to transform.
    owner : object
        The owner object where cTS_vector belongs to.

    Returns
    -------
    dict
        A dictionary with effect types as keys and cTS_vector instances as values.
    """
    effect_dict = as_effect_dict(effect_values)
    effect_ts_dict = effect_values_to_ts(name_of_param, effect_dict, owner)
    return effect_ts_dict

class Commodity:
    """
    Class for commodity objects.
    """
    def __init__(self, unit: str, label: Optional[str] = None, description: Optional[str] = None):
        """
        Parameters
        ----------
        unit : str
            The unit of the commodity.
        label : str, optional
            The label of the commodity.
        description : str, optional
            A description of the commodity.
        """
        self.unit = unit
        self.label = label
        self.description = description


class MultilineFormater(logging.Formatter):

    def format(self, record):
        message_lines = record.getMessage().split('\n')

        # Prepare the log prefix (timestamp + log level)
        timestamp = self.formatTime(record, self.datefmt)
        log_level = record.levelname.ljust(8)  # Align log levels for consistency
        log_prefix = f"{timestamp} | {log_level} |"

        # Format all lines
        first_line = [f'{log_prefix} {message_lines[0]}']
        if len(message_lines) > 1:
            lines = first_line + [f"{log_prefix} {line}" for line in message_lines[1:]]
        else:
            lines = first_line

        return '\n'.join(lines)


class ColoredMultilineFormater(MultilineFormater):
    # ANSI escape codes for colors
    COLORS = {
        'DEBUG': '\033[32m',  # Green
        'INFO': '\033[34m',  # Blue
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[1m\033[31m',  # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        lines = super().format(record).splitlines()
        log_color = self.COLORS.get(record.levelname, self.RESET)

        # Create a formatted message for each line separately
        formatted_lines = []
        for line in lines:
            formatted_lines.append(f"{log_color}{line}{self.RESET}")

        return '\n'.join(formatted_lines)


def _get_logging_handler(log_file: Optional[str] = None,
                         use_rich_handler: bool = False) -> logging.Handler:
    """Returns a logging handler for the given log file."""
    if use_rich_handler and log_file is None:
        # RichHandler for console output
        console = Console(width=120)
        rich_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            omit_repeated_times=True,
            show_path=False,
            log_time_format="%Y-%m-%d %H:%M:%S",
        )
        rich_handler.setFormatter(logging.Formatter("%(message)s"))  # Simplified formatting

        return rich_handler
    elif log_file is None:
        # Regular Logger with custom formating enabled
        file_handler = logging.StreamHandler()
        file_handler.setFormatter(ColoredMultilineFormater(
            fmt="%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        return file_handler
    else:
        # FileHandler for file output
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(MultilineFormater(
            fmt="%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        return file_handler

def setup_logging(default_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO',
                  log_file: Optional[str] = 'flixOpt.log',
                  use_rich_handler: bool = False):
    """Setup logging configuration"""
    logger = logging.getLogger('flixOpt')  # Use a specific logger name for your package
    logger.setLevel(get_logging_level_by_name(default_level))
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(_get_logging_handler(use_rich_handler=use_rich_handler))
    if log_file is not None:
        logger.addHandler(_get_logging_handler(log_file, use_rich_handler=False))

    return logger


def get_logging_level_by_name(level_name: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']) -> int:
    possible_logging_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if level_name.upper() not in possible_logging_levels:
        raise ValueError(f'Invalid logging level {level_name}')
    else:
        logging_level = getattr(logging, level_name.upper(), logging.WARNING)
        return logging_level


def change_logging_level(level_name: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']):
    logger = logging.getLogger('flixOpt')
    logging_level = get_logging_level_by_name(level_name)
    logger.setLevel(logging_level)
    for handler in logger.handlers:
        handler.setLevel(logging_level)
