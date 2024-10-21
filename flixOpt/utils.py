# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:13:31 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import logging
from typing import Union, List, Optional, Dict, Literal

import numpy as np
import math  # für nan

from flixOpt.core import TimeSeriesRaw
from flixOpt.core import Numeric, Skalar

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


def check_bounds(value: Union[int, float, np.ndarray, TimeSeriesRaw],
                 label: str,
                 lower_bound: Numeric,
                 upper_bound: Numeric):
    if isinstance(value, TimeSeriesRaw):
        value = value.value
    if np.any(value < lower_bound):
        raise Exception(f'{label} is below its {lower_bound=}!')
    if np.any(value >= upper_bound):
        raise Exception(f'{label} is above its {upper_bound=}!')


def is_number(number_alias: Union[Skalar, str]):
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
