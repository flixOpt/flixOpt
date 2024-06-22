# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:51:38 2021
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import numpy as np
from . import flixOptHelperFcts as helpers
from .flixBasicsPublic import TimeSeriesRaw
from typing import Union, Optional, List, Dict, Any

Skalar = Union[int, float]  # Datatype
Numeric = Union[int, float, np.ndarray]  # Datatype
# zeitreihenbezogene Input-Daten:
Numeric_TS = Union[Skalar, np.ndarray, TimeSeriesRaw]
# Datatype Numeric_TS:
#   Skalar      --> wird später dann in array ("Zeitreihe" mit len=nrOfTimeIndexe) übersetzt
#   np.ndarray  --> muss len=nrOfTimeIndexe haben ("Zeitreihe")
#   TimeSeriesRaw      --> wie obige aber zusätzliche Übergabe aggWeight (für Aggregation)


class TimeSeries:
    '''
    Class for data that applies to time series, stored as vector (np.ndarray) or scalar.

    This class represents a vector or scalar value that makes the handling of time series easier.
    It supports various operations such as activation of specific time indices, setting explicit active data, and
    aggregation weight management.

    Attributes
    ----------
    label : str
        The label for the time series.
    owner : object
        The owner object which holds the time series list.
    TSraw : Optional[TimeSeriesRaw]
        The raw time series data if provided as cTSraw.
    data : Optional[Numeric]
        The actual data for the time series. Can be None.
    explicit_active_data : Optional[Numeric]
        Explicit data to use instead of raw data if provided.
    active_time_indices : Optional[np.ndarray]
        Indices of the time steps to activate.
    aggregation_weight : float
        Weight for aggregation method, between 0 and 1, normally 1.
    '''

    def __init__(self, label: str, data: Optional[Numeric_TS], owner):
        self.label: str = label
        self.owner: object = owner

        if isinstance(data, TimeSeriesRaw):
            self.TSraw: Optional[TimeSeriesRaw] = data
            data = self.TSraw.value  # extract value
            #TODO: Instead of storing the TimeSeriesRaw object, storing the underlying data directly would be preferable.
        else:
            self.TSraw = None

        self.data: Optional[Numeric] = self.make_scalar_if_possible(data)  # (data wie data), data so knapp wie möglich speichern
        self.explicit_active_data: Optional[Numeric] = None  # Shortcut fneeded for aggregation. TODO: Improve this!

        self.active_time_indices = None  # aktuelle timeIndexe der modBox

        owner.TS_list.append(self)  # Register TimeSeries in owner

        self._aggregation_weight = 1  # weight for Aggregation method # between 0..1, normally 1

    def __repr__(self):
        return f"{self.active_data}"

    @property
    def active_data_vector(self) -> np.ndarray:
        # Always returns the active data as a vector.
        return helpers.getVector(self.active_data, len(self.active_time_indices))

    @property
    def active_data(self) -> Numeric:
        # wenn explicit_active_data gesetzt wurde:
        if self.explicit_active_data is not None:
            return self.explicit_active_data

        indices_not_applicable = np.isscalar(self.data) or (self.data is None) or (self.active_time_indices is None)
        if indices_not_applicable:
            return self.data
        else:
            return self.data[self.active_time_indices]

    @property
    def is_scalar(self) -> bool:
        return np.isscalar(self.data)

    @property
    def is_array(self) -> bool:
        return not self.is_scalar and self.data is not None

    @property
    def label_full(self) -> str:
        return self.owner.label_full + '_' + self.label

    @property
    def aggregation_weight(self):
        return self._aggregation_weight

    @aggregation_weight.setter
    def aggregation_weight(self, weight: Union[int, float]):
        if weight > 1 or weight < 0:
            raise Exception('Aggregation weight must not be below 0 or above 1!')
        self._aggregation_weight = weight

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

    def activate(self, time_indices, explicit_active_data: Optional = None):
        self.active_time_indices = time_indices

        if explicit_active_data is not None:
            assert len(explicit_active_data) == len(self.active_time_indices) or len(explicit_active_data) == 1, \
                'explicit_active_data has incorrect length!'
            self.explicit_active_data = self.make_scalar_if_possible(explicit_active_data)


class TimeSeriesCollection:
    '''
    calculates weights of TimeSeries for being in that collection (depending on)
    '''

    @property
    def addPeak_Max_labels(self):
        if self._addPeakMax_labels == []:
            return None
        else:
            return self._addPeakMax_labels

    @property
    def addPeak_Min_labels(self):
        if self._addPeakMin_labels == []:
            return None
        else:
            return self._addPeakMin_labels

    def __init__(self,
                 time_series_list: List[TimeSeries],
                 addPeakMax_TSraw: Optional[List[TimeSeriesRaw]] = None,
                 addPeakMin_TSraw: Optional[List[TimeSeriesRaw]] = None):
        self.time_series_list = time_series_list
        self.addPeakMax_TSraw = addPeakMax_TSraw or []
        self.addPeakMin_TSraw = addPeakMin_TSraw or []
        # i.g.: self.agg_type_count = {'solar': 3, 'price_el' = 2}
        self.agg_type_count = self._get_agg_type_count()

        self._checkPeak_TSraw(addPeakMax_TSraw)
        self._checkPeak_TSraw(addPeakMin_TSraw)

        # these 4 attributes are now filled:
        self.seriesDict = {}
        self.weightDict = {}
        self._addPeakMax_labels = []
        self._addPeakMin_labels = []
        self.calculateParametersForTSAM()

    def calculateParametersForTSAM(self):
        for i in range(len(self.time_series_list)):
            aTS: TimeSeries
            aTS = self.time_series_list[i]
            # check uniqueness of label:
            if aTS.label_full in self.seriesDict.keys():
                raise Exception('label of TS \'' + str(aTS.label_full) + '\' exists already!')
            # add to dict:
            self.seriesDict[
                aTS.label_full] = aTS.active_data_vector  # Vektor zuweisen!# TODO: müsste doch active_data sein, damit abhängig von Auswahlzeitraum, oder???
            self.weightDict[aTS.label_full] = self._getWeight(aTS)  # Wichtung ermitteln!
            if (aTS.TSraw is not None):
                if aTS.TSraw in self.addPeakMax_TSraw:
                    self._addPeakMax_labels.append(aTS.label_full)
                if aTS.TSraw in self.addPeakMin_TSraw:
                    self._addPeakMin_labels.append(aTS.label_full)

    def _get_agg_type_count(self):
        # count agg_types:
        from collections import Counter

        TSlistWithAggType = []
        for TS in self.time_series_list:
            if self._get_agg_type(TS) is not None:
                TSlistWithAggType.append(TS)
        agg_types = (aTS.TSraw.agg_type for aTS in TSlistWithAggType)
        return Counter(agg_types)

    def _get_agg_type(self, aTS: TimeSeries):
        if (aTS.TSraw is not None):
            agg_type = aTS.TSraw.agg_type
        else:
            agg_type = None
        return agg_type

    def _getWeight(self, aTS: TimeSeries):
        if aTS.TSraw is None:
            # default:
            weight = 1
        elif aTS.TSraw.agg_weight is not None:
            # explicit:
            weight = aTS.TSraw.agg_weight
        elif aTS.TSraw.agg_type is not None:
            # via agg_type:
            # i.g. n=3 -> weight=1/3
            weight = 1 / self.agg_type_count[aTS.TSraw.agg_type]
        else:
            weight = 1
            # raise Exception('TSraw is without weight definition.')
        return weight

    def _checkPeak_TSraw(self, aTSrawlist):
        if aTSrawlist is not None:
            for aTSraw in aTSrawlist:
                if not isinstance(aTSraw, TimeSeriesRaw):
                    raise Exception('addPeak_max/min must be list of TimeSeriesRaw-objects!')

    def print(self):
        print('used ' + str(len(self.time_series_list)) + ' TS for aggregation:')
        for TS in self.time_series_list:
            aStr = ' ->' + TS.label_full + ' (weight: {:.4f}; agg_type: ' + str(self._get_agg_type(TS)) + ')'
            print(aStr.format(self._getWeight(TS)))
        if len(self.agg_type_count.keys()) > 0:
            print('agg_types: ' + str(list(self.agg_type_count.keys())))
        else:
            print('Warning!: no agg_types defined, i.e. all TS have weigth 1 (or explicit given weight)!')


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
        Converted values in from of dict with either None or cEffectType as key. if values is None, None is returned
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
