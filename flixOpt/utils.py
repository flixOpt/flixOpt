# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:13:31 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import logging
from typing import Union, List, Optional, Dict, Literal, Any, Tuple

import numpy as np

logger = logging.getLogger('flixOpt')


def as_vector(value: Union[int, float, np.ndarray, List], length: int) -> np.ndarray:
    """
    Macht aus Skalar einen Vektor. Vektor bleibt Vektor.
    -> Idee dahinter: Aufruf aus abgespeichertem Vektor schneller, als für jede i-te Gleichung zu Checken ob Vektor oder Skalar)

    Parameters
    ----------

    aValue: skalar, list, np.array
    aLen  : skalar
    """
    # dtype = 'float64' # -> muss mit übergeben werden, sonst entstehen evtl. int32 Reihen (dort ist kein +/-inf möglich)
    # TODO: as_vector() -> int32 Vektoren möglich machen

    # Wenn Skalar oder None, return directly as array
    if value is None:
        return np.array([None] * length)
    if np.isscalar(value):
        return np.ones(length) * value

    if len(value) != length:   # Wenn Vektor nicht richtige Länge
        raise Exception(f'error in changing to {length=}; vector has already {length=}')

    if isinstance(value, np.ndarray):
        return value
    else:
        return np.array(value)


def is_number(number_alias: Union[int, float, str]):
    """ Returns True is string is a number. """
    try:
        float(number_alias)
        return True
    except ValueError:
        return False


def check_time_series(label: str,
                      time_series: np.ndarray[np.datetime64]):
    # check sowohl für globale Zeitreihe, als auch für chosenIndexe:

    # Zeitdifferenz:
    #              zweites bis Letztes            - erstes bis Vorletztes
    dt = time_series[1:] - time_series[0:-1]
    # dt_in_hours    = dt.total_seconds() / 3600
    dt_in_hours = dt / np.timedelta64(1, 'h')

    # unterschiedliche dt:
    if np.max(dt_in_hours) - np.min(dt_in_hours) != 0:
        logger.warning(f'{label}: !! Achtung !! unterschiedliche delta_t von {min(dt)} h bis  {max(dt)} h')
    # negative dt:
    if np.min(dt_in_hours) < 0:
        raise Exception(label + ': Zeitreihe besitzt Zurücksprünge - vermutlich Zeitumstellung nicht beseitigt!')


def apply_formating(data_dict: Dict[str, Union[int, float]],
                    key_format: str = "<17",
                    value_format: str = ">10.2f",
                    indent: int = 0,
                    sort_by: Optional[Literal['key', 'value']] = None) -> str:
    if sort_by == 'key':
        sorted_keys = sorted(data_dict.keys(), key=str.lower)
    elif sort_by == 'value':
        sorted_keys = sorted(data_dict, key=lambda k: data_dict[k], reverse=True)
    else:
        sorted_keys = data_dict.keys()

    lines = [f'{indent*" "}{key:{key_format}}: {data_dict[key]:{value_format}}' for key in sorted_keys]
    return '\n'.join(lines)


def label_is_valid(label: str) -> bool:
    """ Function to make shure '__' is reserved for internal splitting of labels"""
    if label.startswith('_') or label.endswith('_') or '__' in label:
        return False
    return True


def convert_numpy_array(array: np.ndarray) -> List[Any]:
    """Convert a numpy array to a list with native types."""
    return [convert_to_native_types(item) for item in array.tolist()]

def convert_list_or_tuple(sequence: Union[List[Any], Tuple]) -> List[Any]:
    """Convert lists or tuples to lists with native types, preserving tuples."""
    return [convert_to_native_types(item) for item in sequence]

def convert_dictionary(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert all items in a dictionary to native types."""
    return {key: convert_to_native_types(value) for key, value in d.items()}

def convert_to_native_types(value: Any) -> Any:
    """Identify the type of `value` and apply the appropriate conversion function."""
    from .core import TimeSeries, TimeSeriesData
    if isinstance(value, np.ndarray):
        return convert_numpy_array(value)
    elif isinstance(value, (list, tuple)):
        return convert_list_or_tuple(value)
    elif isinstance(value, dict):
        return convert_dictionary(value)
    elif isinstance(value, TimeSeries):
        return convert_to_native_types(value.active_data)
    elif isinstance(value, TimeSeriesData):
       return convert_to_native_types(value.data)
    elif isinstance(value, (np.integer, np.int_)):
        return int(value)
    elif isinstance(value, (np.floating, np.float_)):
        return float(value)
    else:
        raise TypeError(f'Type {type(value)} is not supported yet.')

def convert_arrays_to_lists(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert all numpy arrays, lists, and tuples in a nested dictionary to lists with native types.
    """
    d_copy: Dict[str, Any] = d.copy()
    return convert_dictionary(d_copy)

def convert_numeric_lists_to_arrays(d: Union[Dict[str, Any], List[Any], tuple]) -> Union[Dict[str, Any], List[Any], tuple]:
    """
    Recursively converts all lists of numeric values in a nested dictionary to numpy arrays.
    Handles nested lists, tuples, and dictionaries. Does not alter the original dictionary.
    """
    if isinstance(d, dict):
        # Make a copy of the dictionary to avoid modifying the original
        d_copy = {}
        for key, value in d.items():
            if isinstance(value, (list, tuple)):
                d_copy[key] = convert_list_to_array_if_numeric(value)
            elif isinstance(value, dict):
                d_copy[key] = convert_numeric_lists_to_arrays(value)  # Recursively process nested dictionaries
        return d_copy
    elif isinstance(d, (list, tuple)):
        # If the input itself is a list or tuple, process it as a sequence
        return convert_list_to_array_if_numeric(d)
    else:
        return d


def convert_list_to_array_if_numeric(sequence: Union[List[Any], tuple]) -> Union[np.ndarray, List[Any], tuple]:
    """
    Converts a list to a numpy array if all elements are numeric.
    Preserves tuples and recursively processes each element.
    """
    # Check if all elements are numeric in the list
    if isinstance(sequence, list) and all(isinstance(item, (int, float)) for item in sequence):
        return np.array(sequence)
    else:
        converted_sequence = [convert_numeric_lists_to_arrays(item) if isinstance(item, (dict, list, tuple)) else item for item in sequence]
        return tuple(converted_sequence) if isinstance(sequence, tuple) else converted_sequence
