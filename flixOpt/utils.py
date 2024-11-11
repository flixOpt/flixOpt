# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:13:31 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import logging
from typing import Union, List, Optional, Dict, Literal

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


def convert_arrays_to_lists(d: dict) -> dict:
    """Recursively converts all numpy arrays in a nested dictionary to lists. Does not alter the original dictionary."""
    d_copy = d.copy()  # Make a copy of the dictionary to avoid modifying in-place
    for key, value in d_copy.items():
        if isinstance(value, np.ndarray):
            d_copy[key] = value.tolist()
        elif isinstance(value, dict):
            d_copy[key] = convert_arrays_to_lists(value)
    return d_copy


def convert_numeric_lists_to_arrays(d: dict) -> dict:
    """Recursively converts all numpy arrays in a nested dictionary to lists. Does not alter the original dictionary."""
    d_copy = d.copy()  # Make a copy of the dictionary to avoid modifying in-place
    for key, value in d_copy.items():
        if isinstance(value, list) and isinstance(value[0], (int, float)):
            d_copy[key] = np.array(value)
        elif isinstance(value, dict):
            d_copy[key] = convert_numeric_lists_to_arrays(value)
    return d_copy