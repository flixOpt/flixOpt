# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:51:38 2021
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""
from typing import Union, Optional, List, Dict, Any, Literal
import logging

import numpy as np

from flixOpt import utils

logger = logging.getLogger('flixOpt')

Skalar = Union[int, float]  # Datatype
Numeric = Union[int, float, np.ndarray]  # Datatype
# zeitreihenbezogene Input-Daten:
Numeric_TS = Union[Skalar, np.ndarray, 'TimeSeries']
# Datatype Numeric_TS:
#   Skalar      --> wird später dann in array ("Zeitreihe" mit length=nrOfTimeIndexe) übersetzt
#   np.ndarray  --> muss length=nrOfTimeIndexe haben ("Zeitreihe")
#   TimeSeriesData      --> wie obige aber zusätzliche Übergabe aggWeight (für Aggregation)


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
            fixed_relative_value_solar1 = TimeSeriesData(sol_array_1, type = 'solar')
            fixed_relative_value_solar2 = TimeSeriesData(sol_array_2, type = 'solar')
            fixed_relative_value_solar3 = TimeSeriesData(sol_array_3, type = 'solar')
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
        return f"TimeSeriesData(value={self.data}, agg_group={self.agg_group}, agg_weight={self.agg_weight})"


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
        return (f"TimeSeries(label={self.label}, "
                f"aggregation_weight={self.aggregation_weight}, "
                f"data={self.data}, active_indices={self.active_indices}, "
                f"aggregated_data={self.aggregated_data}")

    def __str__(self):
        active_data = self.active_data
        if isinstance(active_data, Skalar):
            data_stats = active_data
            all_indices_active = None
        else:
            data_stats = (f"[min={np.min(active_data):.2f}, max={np.max(active_data):.2f}, "
                          f"mean={np.mean(active_data):.2f}, std={np.std(active_data):.2f}]")
            if len(active_data) == len(self.data):
                all_indices_active = 'all'
            else:
                all_indices_active = 'some'

        further_infos = []
        if all_indices_active:
            further_infos.append(f"indices='{all_indices_active}'")
        if self.aggregation_weight is not None:
            further_infos.append(f"aggregation_weight={self.aggregation_weight}")
        if self.aggregation_group is not None:
            further_infos.append(f"aggregation_group= '{self.aggregation_group}'")
        if self.explicit_active_data is not None:
            further_infos.append(f"'Explicit Active Data was used'")

        if further_infos:
            infos = f"TimeSeries(active_data={data_stats}, {', '.join(further_infos)})"
        else:
            infos = f"TimeSeries({data_stats})"

        return infos

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


# TODO: Move logging to utils.py
class CustomFormatter(logging.Formatter):
    # ANSI escape codes for colors
    COLORS = {
        'DEBUG': '\033[96m',    # Cyan
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        original_message = record.getMessage()
        message_lines = original_message.split('\n')

        # Create a formatted message for each line separately
        formatted_lines = []
        for line in message_lines:
            temp_record = logging.LogRecord(
                record.name, record.levelno, record.pathname, record.lineno,
                line, record.args, record.exc_info, record.funcName, record.stack_info
            )
            formatted_line = super().format(temp_record)
            formatted_lines.append(f"{log_color}{formatted_line}{self.RESET}")

        formatted_message = '\n'.join(formatted_lines)
        return formatted_message


def setup_logging(level_name: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']):
    """Setup logging configuration"""
    logger = logging.getLogger('flixOpt')  # Use a specific logger name for your package
    logging_level = get_logging_level_by_name(level_name)
    logger.setLevel(logging_level)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create console handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging_level)

    # Create a clean and aligned formatter
    log_format = '%(asctime)s - %(levelname)-8s : %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    c_format = CustomFormatter(log_format, datefmt=date_format)
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

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
