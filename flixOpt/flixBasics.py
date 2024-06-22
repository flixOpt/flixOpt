# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:51:38 2021
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import numpy as np
from . import flixOptHelperFcts as helpers
from .flixBasicsPublic import cTSraw
from typing import Union, Optional

Skalar = Union[int, float]  # Datatype
Numeric = Union[int, float, np.ndarray]  # Datatype
# zeitreihenbezogene Input-Daten:
Numeric_TS = Union[Skalar, np.ndarray, cTSraw]
# Datatype Numeric_TS:
#   Skalar      --> wird später dann in array ("Zeitreihe" mit len=nrOfTimeIndexe) übersetzt
#   np.ndarray  --> muss len=nrOfTimeIndexe haben ("Zeitreihe")
#   cTSraw      --> wie obige aber zusätzliche Übergabe aggWeight (für Aggregation)


class cArgsClass:
    '''
    stellt Infrastruktur getInitArgs() etc zur Verfügung: 
        TODO: geht das nicht irgendwie nativ? noch notwendig?
    gibt Warnung, falls unbenutzte kwargs vorhanden! 
    '''

    @classmethod
    def getInitArgs(cls):
        '''
        diese (Klassen-)Methode holt aus dieser und den Kindklassen
        alle zulässigen Argumente der Kindklasse!
        '''

        ### 1. Argumente der Mutterklasse (rekursiv) ###
        # wird rekursiv aufgerufen bis man bei Mutter-Klasse cModelingElement ankommt.
        # nur bis zu cArgsClass zurück gehen:
        if hasattr(cls.__base__, 'getInitArgs'):  # man könnte auch schreiben: if cls.__name__ == cArgsClass
            allArgsFromMotherClass = cls.__base__.getInitArgs()  # rekursiv in Mutterklasse aufrufen

        # wenn cls.__base__ also bereits eine Ebene UNTER cArgsClass:
        else:
            allArgsFromMotherClass = []

            # checken, dass die zwei class-Atributes auch wirklich für jede Klasse (und nicht nur für Mutterklasse) existieren (-> die nimmt er sonst einfach automatisch)
        if (not ('not_used_args' in cls.__dict__)) | (not ('new_init_args' in cls.__dict__)):
            raise Exception(
                'class ' + cls.__name__ + ': you forgot to implement class attribute <not_used_args> or/and <new_int_args>')
        notTransferedMotherArgs = cls.not_used_args

        ### 2. Abziehen der nicht durchgereichten Argumente ###
        # delete not Transfered Args:
        allArgsFromMotherClass = [prop for prop in allArgsFromMotherClass if prop not in notTransferedMotherArgs]

        ### 3. Ergänzen der neuen Argumente ###
        myArgs = cls.new_init_args.copy()  # get all new arguments of __init__() (as a copy)
        # melt lists:
        myArgs.extend(allArgsFromMotherClass)
        return myArgs

    # Diese Variablen muss jede Kindklasse auch haben:
    new_init_args = []
    not_used_args = []

    def __init__(self, **kwargs):
        # wenn hier kwargs auftauchen, dann wurde zuviel übergeben:
        if len(kwargs) > 0:
            raise Exception('class and its motherclasses have no allowed arguments for:' + str(kwargs)[:200])


class TimeSeries:
    '''
    Klasse für Timeseries-Vektoren bzw. Skalare, die für Zeitreihe gelten
    '''

    # create and register in List:

    def __init__(self, label: str, data: Numeric_TS, owner):
        '''
        Parameters
        ----------
        data :
            scalar, array or cTSraw!
        owner :
        '''
        self.label = label
        self.owner = owner

        # if value is cTSraw, then extract value:
        if isinstance(data, cTSraw):
            self.TSraw = data
            data = self.TSraw.value  # extract value
        else:
            self.TSraw = None

        self.data = self.make_scalar_if_possible(data)  # (data wie data), data so knapp wie möglich speichern
        self.d_i_explicit = None  #

        self.__timeIndexe_actual = None  # aktuelle timeIndexe der modBox

        owner.TS_list.append(self)

        self.weight_agg = 1  # weight for Aggregation method # between 0..1, normally 1

    def __repr__(self):
        return f"{self.data}"

    # Vektor:
    @property
    def active_data_vector(self) -> np.ndarray:
        # Always returns the active data as a vector.
        return helpers.getVector(self.active_data, len(self.__timeIndexe_actual))

    @property
    def active_data(self) -> Numeric:
        # wenn d_i_explicit gesetzt wurde:
        if self.d_i_explicit is not None:
            return self.d_i_explicit

        indices_not_applicable = np.isscalar(self.data) or (self.data is None) or (self.__timeIndexe_actual is None)
        if indices_not_applicable:
            return self.data
        else:
            return self.data[self.__timeIndexe_actual]

    @property
    def is_scalar(self) -> bool:
        return np.isscalar(self.data)

    @property
    def isArray(self):
        return (not (self.is_scalar)) & (not (self.data is None))

    @property
    def label_full(self):
        return self.owner.label_full + '_' + self.label

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

    # define, which timeStep-Set should be transfered in data-request self.active_data()
    def activate(self, dataTimeIndexe, d_i_explicit=None):
        # time-Index:
        self.__timeIndexe_actual = dataTimeIndexe

        # explicitData:
        if d_i_explicit is not None:
            assert ((len(d_i_explicit) == len(self.__timeIndexe_actual)) or \
                    (len(d_i_explicit) == 1)), 'd_i_explicit has not right length!'

        self.d_i_explicit = self.make_scalar_if_possible(d_i_explicit)

    def setAggWeight(self, aWeight):
        '''
        only for aggregation: set weight of timeseries for creating of typical periods!
        '''
        self.weight_agg = aWeight
        if (aWeight > 1) or (aWeight < 0):
            raise Exception('weigth must be between 0 and 1!')

    # Rückgabe Maximum
    def max(self):
        return TimeSeries.__getMax(self.d)

        # Maximum für indexe:

    def max_i(self):
        return TimeSeries.__getMax(self.active_data)

    def __getMax(aValue):
        if np.isscalar(aValue):
            return aValue
        else:
            return max(aValue)


class cTS_collection():
    '''
    calculates weights of TS_vector for being in that collection (depending on)
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

    def __init__(self, listOfTS_vectors, addPeakMax_TSraw=[], addPeakMin_TSraw=[]):
        self.listOfTS_vectors = listOfTS_vectors
        self.addPeakMax_TSraw = addPeakMax_TSraw
        self.addPeakMin_TSraw = addPeakMin_TSraw
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
        for i in range(len(self.listOfTS_vectors)):
            aTS: TimeSeries
            aTS = self.listOfTS_vectors[i]
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
        for TS in self.listOfTS_vectors:
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
                if not isinstance(aTSraw, cTSraw):
                    raise Exception('addPeak_max/min must be list of cTSraw-objects!')

    def print(self):
        print('used ' + str(len(self.listOfTS_vectors)) + ' TS for aggregation:')
        for TS in self.listOfTS_vectors:
            aStr = ' ->' + TS.label_full + ' (weight: {:.4f}; agg_type: ' + str(self._get_agg_type(TS)) + ')'
            print(aStr.format(self._getWeight(TS)))
        if len(self.agg_type_count.keys()) > 0:
            print('agg_types: ' + str(list(self.agg_type_count.keys())))
        else:
            print('Warning!: no agg_types defined, i.e. all TS have weigth 1 (or explicit given weight)!')


def getEffectDictOfEffectValues(effect_values):
    '''
    if costs is given without effectType, standardeffect is related
    examples:
      costs = 20                        -> {None:20}
      costs = None                      -> no change
      costs = {effect1:20, effect2:0.3} -> no change

    Parameters
    ----------
    effect_values : None, scalar or TS, dict
        see examples

    Returns
    -------
    effect_values_dict : dict
        see examples
    '''

    ## Umwandlung in dict:
    # Wenn schon dict:
    if isinstance(effect_values, dict):
        # nur übergeben, nix machen
        effect_values_dict = effect_values
    elif effect_values is None:
        effect_values_dict = None
        # Wenn Skalar oder TS:
    else:
        # dict bauen mit standard-effect:
        effect_values_dict = {
            None: effect_values}  # standardType noch nicht bekannt, dann None. Wird später Standard-Effekt-Type

    return effect_values_dict


def transformDictValuesToTS(nameOfParam, aDict, owner):
    '''
      transformiert Wert -> TimeSeries
      für alle {Effekt:Wert}-couples in dict,

      Parameters
      ----------
      nameOfParam : str

      aDict : dict
          {Effect:value}-couples
      owner : class
          class where TimeSeries belongs to

      Returns
      -------
      aDict_TS : dict
         {Effect:TS_value}-couples

      '''

    # Einzelne Faktoren zu Vektoren:
    aDict_TS = {}  #
    # für jedes Dict -> Values (=Faktoren) zu Vektoren umwandeln:
    if aDict is None:
        aDict_TS = None
    else:
        for effect, value in aDict.items():
            if not isinstance(value, TimeSeries):
                # Subnamen aus key:
                if effect is None:
                    subname = 'standard'  # Standard-Effekt o.ä. # todo: das ist nicht schön, weil costs in Namen nicht auftaucht
                else:
                    subname = effect.label  # z.B. costs, Q_th,...
                nameOfParam_full = nameOfParam + '_' + subname  # name ergänzen mit key.label
                aDict_TS[effect] = TimeSeries(nameOfParam_full, value, owner)  # Transform to TS
        return aDict_TS


def transFormEffectValuesToTSDict(nameOfParam, aEffectsValue, ownerOfParam):
    '''
    Transforms effect/cost-input to dict of TS,
      wenn nur wert gegeben, dann wird gegebener effect zugeordnet
      effectToUseIfOnlyValue = None -> Standard-EffektType wird genommen
    Fall 1:
        output = {effect1 : TS1, effects2: TS2}
    Fall2 (falls Skalar übergeben):
        output = {standardEffect : TS1}

    Parameters
    ----------
    nameOfParam : str
    aEffectsValue : TYPE
        DESCRIPTION.
    ownerOfParam : TYPE
        DESCRIPTION.

    Returns
    -------
    effectsDict_TS : TYPE
        DESCRIPTION.

    '''

    # add standardeffect if only value is given:
    effectsDict = getEffectDictOfEffectValues(aEffectsValue)
    # dict-values zu TimeSeries:
    effectsDict_TS = transformDictValuesToTS(nameOfParam, effectsDict, ownerOfParam)
    return effectsDict_TS
