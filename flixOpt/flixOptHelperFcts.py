# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:13:31 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

# TODO: as_vector() -> int32 Vektoren möglich machen

import numpy as np
import re
import math  # für nan
import matplotlib.pyplot as plt

from flixOpt.flixBasicsPublic import TimeSeriesRaw
from flixOpt.flixBasics import Numeric_TS, Numeric, Skalar
from typing import Union, List, Tuple, Optional


def as_vector(value: Union[int, float, np.ndarray, List], length: int) -> np.ndarray:
    '''
    Macht aus Skalar einen Vektor. Vektor bleibt Vektor.
    -> Idee dahinter: Aufruf aus abgespeichertem Vektor schneller, als für jede i-te Gleichung zu Checken ob Vektor oder Skalar)

    Parameters
    ----------

    aValue: skalar, list, np.array
    aLen  : skalar
    '''
    # dtype = 'float64' # -> muss mit übergeben werden, sonst entstehen evtl. int32 Reihen (dort ist kein +/-inf möglich)

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


def zero_to_nan(vector: np.ndarray) -> np.ndarray:
    # changes zeros to Nans in Vector:
    nan_vector = vector.astype(float)  # Binär ist erstmal int8-Vektor, Nan gibt es da aber nicht
    nan_vector[nan_vector == 0] = math.nan
    return nan_vector


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


def check_name_for_conformity(label: str):
    # löscht alle in Attributen ungültigen Zeichen: todo: Vollständiger machen!
    char_map = {ord('ä'): 'ae',
                ord('ü'): 'ue',
                ord('ö'): 'oe',
                ord('ß'): 'ss',
                ord('-'): '_'}
    new_label = label.translate(char_map)
    if new_label != label:
        print(f'{label=} doesnt allign with name restrictions and is changed to {new_label=}')

    # check, ob jetzt valid variable name: (für Verwendung in results_struct notwendig)
    import re
    if not re.search(r'^[a-zA-Z_]\w*$', new_label):
        raise Exception('label \'' + label + '\' is not valid for variable name \n .\
                     (no number first, no special characteres etc.)')
    return new_label


def check_exists(exists: Union[int, list, np.ndarray])-> Union[int, list,np.ndarray]:
    # type checking for argument "exist"
    if np.all(np.isin(exists, [0, 1])):
        return exists
    else:
        raise ValueError(f"Argument 'exists' must be int, list or np.ndarray with values 0 or 1")


class InfiniteFullSet(object):
    def __and__(self, item):  # left operand of &
        return item

    def __rand__(self, item):  # right operand of &
        return item

    def __str__(self):
        return ('<InfiniteFullSet>')


def is_number(number_alias:str):
    """ Returns True is string is a number. """
    try:
        float(number_alias)
        return True
    except ValueError:
        return False

    # Macht aus verschachteltem Dict ein "matlab-struct"-like object

def createStructFromDictInDict(aDict):
    # --> als FeldNamen wird key verwendet wenn string, sonst key.label

    # --> dict[key1]['key1_1'] -> struct.key1.key1_1

    # z.B.:
    #      {Kessel_object : {'Q_th':{'on':[..], 'val': [..]}
    #                        'Pel' :{'on':[..], 'val': [..]} }
    #       Last_object   : {'Q_th':{'val': [..]           } } }

    aStruct = cDataBox2()
    if isinstance(aDict, dict):
        for key, val in aDict.items():

            ## 1. attr-name :

            # Wenn str (z.b. 'Q_th') :
            if isinstance(key, str):
                name = key
            # sonst (z.B. bei Kessel_object):
            else:
                try:
                    name = key.label
                except:
                    raise Exception('key has no label!')

            ## 2. value:
            # Wenn Wert wiederum dict, dann rekursiver Aufruf:
            if isinstance(val, dict):
                value = createStructFromDictInDict(val)  # rekursiver Aufruf!
            else:
                value = val

            if hasattr(aStruct, name):
                name = name + '_2'
            setattr(aStruct, name, value)
    else:
        raise Exception('fct needs dict!')

    return aStruct


# emuliert matlab - struct-datentyp. (-> durch hinzufügen von Attributen)
class cDataBox2:
    # pass
    def __str__(self):
        astr = ('<cDataBox with ' + str(len(self.__dict__)) + ' values: ')
        for aAttr in self.__dict__.keys():
            astr += aAttr + ', '
        astr += '>'
        return astr


def get_time_series_with_end(time_series: np.ndarray[np.datetime64],
                             dt_last: Optional[np.timedelta64] = None):
    ## letzten Zeitpunkt hinzufügen:
    if dt_last is None:
        dt_last = time_series[-1] - time_series[-2]
    t_end = time_series[-1] + dt_last
    return np.append(time_series, t_end)


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
        print(label + ': !! Achtung !! unterschiedliche delta_t von ' + str(min(dt)) + 'h bis ' + str(max(dt)) + ' h')
    # negative dt:
    if np.min(dt_in_hours) < 0:
        raise Exception(label + ': Zeitreihe besitzt Zurücksprünge - vermutlich Zeitumstellung nicht beseitigt!')


def printDictAndList(aDictOrList):
    import yaml
    print(yaml.dump(aDictOrList,
                    default_flow_style=False,
                    width=1000,  # verhindern von zusätzlichen Zeilenumbrüchen
                    allow_unicode=True))

def max_args(*args):
    # max from num-lists and skalars
    # arg = list, array, skalar
    # example: max_of_lists_and_scalars([1,2],3) --> 3
    array = _mergeToArray(args)
    return array.max()

def min_args(*args):
    # example: min_of_lists_and_scalars([1,2],3) --> 1
    array = _mergeToArray(args)
    return array.min()


def _mergeToArray(args):
    array = np.array([])
    for i in range(len(args)):

        if np.isscalar(args[i]):
            arg = [args[i]]  # liste draus machen
        else:
            arg = args[i]
        array = np.append(array, arg)
    return array
