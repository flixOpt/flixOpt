# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 10:23:49 2021
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden

Modul zur aggregierten Berechnung eines Energiesystemmodells.
"""

import copy
import timeit
from typing import Optional, List, Dict, Union
import warnings

import pandas as pd
import numpy as np
import tsam.timeseriesaggregation as tsam

from flixOpt.core import Skalar, TimeSeries
from flixOpt.elements import Global, Flow
from flixOpt.system import System
from flixOpt.components import Storage
from flixOpt.flixBasicsPublic import TimeSeriesRaw
from flixOpt.structure import Element, SystemModel
from flixOpt.modeling import Equation, Variable, VariableTS, LinearModel

warnings.filterwarnings("ignore", category=DeprecationWarning)


class Aggregation:
    """
    aggregation organizing class
    """

    def __init__(self,
                 name: str,
                 timeseries: pd.DataFrame,
                 hours_per_time_step: Skalar,
                 hours_per_period: Skalar,
                 hasTSA=False,  # TODO: Remove unused parameter
                 nr_of_typical_periods: int = 8,
                 use_extreme_periods: bool = True,
                 weights: Optional[dict] = None,
                 addPeakMax: Optional[List[TimeSeries]] = None,
                 addPeakMin: Optional[List[TimeSeries]] = None
                 ):

        """ 
        Konstruktor für die Klasse EnergySystemModel
        """

        self.results = None
        self.time_for_clustering = None
        self.aggregation: Optional[tsam.TimeSeriesAggregation] = None

        self.name = name
        self.timeseries = copy.deepcopy(timeseries)
        self.hours_per_time_step = hours_per_time_step
        self.hours_per_period = hours_per_period
        self.hasTSA = hasTSA
        self.nr_of_typical_periods = nr_of_typical_periods
        self.use_extreme_periods = use_extreme_periods
        self.weights = weights
        self.addPeakMax = addPeakMax
        self.addPeakMin = addPeakMin

        # Wenn Extremperioden eingebunden werden sollen, nutze die Methode 'new_cluster_center' aus tsam
        self.extreme_period_method = 'new_cluster_center' if self.use_extreme_periods else 'None'
        if self.use_extreme_periods and not (self.addPeakMax or self.addPeakMin):   # Check
            raise Exception('addPeakMax or addPeakMin timeseries given if useExtremValues=True!')

        # Initiales Setzen von Zeitreiheninformationen; werden überschrieben, falls Zeitreihenaggregation
        self.nr_of_time_steps = len(self.timeseries.index)
        self.nr_of_time_steps_per_period = self.nr_of_time_steps   # TODO: Remove unused parameter
        self.total_periods = 1      # TODO: Remove unused parameter

        # Timeseries Index anpassen, damit gesamter Betrachtungszeitraum als eine lange Periode + entsprechender Zeitschrittanzahl interpretiert wird
        self.original_timeseries_index = self.timeseries.index  # ursprünglichen Index in Array speichern für späteres Speichern
        period_index = [0] * self.nr_of_time_steps
        step_index = list(range(self.nr_of_time_steps))
        self.timeseries.index = pd.MultiIndex.from_arrays([period_index, step_index],
                                                          names=['Period', 'TimeStep'])

        # Setzen der Zeitreihendaten für Modell
        # werden später überschrieben, falls Zeitreihenaggregation
        self.typical_periods = [0]
        self.periods, self.periods_order, self.period_occurrences = [0], [0], [1]
        self.total_time_steps = list(range(self.nr_of_time_steps))  # gesamte Anzahl an Zeitschritten
        self.time_steps_per_period = list(range(self.nr_of_time_steps))  # Zeitschritte pro Periode, ohne ZRA = gesamte Anzahl
        self.inter_period_time_steps = list(range(int(len(self.total_time_steps) / len(self.time_steps_per_period)) + 1))

    def cluster(self) -> None:

        """
        Durchführung der Zeitreihenaggregation
        """

        start_time = timeit.default_timer()

        # Neu berechnen der nr_of_time_steps_per_period
        self.nr_of_time_steps_per_period = int(self.hours_per_period / self.hours_per_time_step)

        # Erstellen des aggregation objects
        self.aggregation = tsam.TimeSeriesAggregation(self.timeseries,
                                                      noTypicalPeriods=self.nr_of_typical_periods,
                                                      hoursPerPeriod=self.hours_per_period,
                                                      resolution=self.hours_per_time_step,
                                                      clusterMethod='k_means',
                                                      extremePeriodMethod=self.extreme_period_method,
                                                      weightDict=self.weights,
                                                      addPeakMax=self.addPeakMax,
                                                      addPeakMin=self.addPeakMin
                                                      )

        self.aggregation.createTypicalPeriods()   # Ausführen der Aggregation/Clustering

        # ERGEBNISSE:
        self.results = self.aggregation.predictOriginalData()

        self.time_for_clustering = timeit.default_timer() - start_time   # Zeit messen:
        print(self.describe_clusters())

    @property
    def results_original_index(self):
        return self.results.set_index(self.original_timeseries_index, inplace=False)  # neue DF erstellen

    @property
    def index_vectors_of_clusters(self):
        # TODO: make more performant? using self._index_vectors_of_clusters maybe?
        ###############
        # Zuordnung der Indexe erstellen: 
        # {cluster 0: [index_vector_3, index_vector_7]
        #  cluster 1: [index_vector_1]
        #  cluster 2: ...} 

        # Beachte: letzte Periode muss nicht vollgefüllt sein!
        clusterList = self.aggregation.clusterPeriodNoOccur.keys()
        # Leerer Dict:
        index_vectors_of_clusters = {cluster: [] for cluster in clusterList}
        period_len = len(self.aggregation.stepIdx)
        for period in range(len(self.aggregation.clusterOrder)):
            clusterNr = self.aggregation.clusterOrder[period]

            periodStartIndex = period * period_len
            periodEndIndex = min(periodStartIndex + period_len - 1,
                                 len(self.timeseries) - 1)  # Beachtet auch letzte Periode
            indexVector = np.array(range(periodStartIndex, periodEndIndex + 1))

            index_vectors_of_clusters[clusterNr].append(indexVector)

        return index_vectors_of_clusters

    def describe_clusters(self) -> str:
        aVisual = {}
        for cluster in self.index_vectors_of_clusters.keys():
            aVisual[cluster] = [str(indexVector[0]) + '...' + str(indexVector[-1]) for indexVector in
                                self.index_vectors_of_clusters[cluster]]

        if self.use_extreme_periods:
            # Zeitreihe rauslöschen:
            extremePeriods = self.aggregation.extremePeriods.copy()
            for key, val in extremePeriods.items():
                del (extremePeriods[key]['profile'])
        else:
            extremePeriods = {}

        return (f'#########################\n'
                f'###### Clustering #######\n'
                f'periods_order:\n'
                f'{self.aggregation.clusterOrder}\n'
                f'clusterPeriodNoOccur:\n'
                f'{self.aggregation.clusterPeriodNoOccur}\n'
                f'index_vectors_of_clusters:\n'
                f'{aVisual}\n'
                f'########################\n'
                f'extremePeriods:\n'
                f'{extremePeriods}\n'
                f'########################')


class AggregationModeling(Element):
    # ModelingElement mit Zusatz-Glg. und Variablen für aggregierte Berechnung
    def __init__(self,
                 label: str,
                 system: System,
                 index_vectors_of_clusters: Dict[int, List[np.ndarray]],
                 fix_storage_flows: bool = True,
                 fix_binary_vars_only: bool = True,
                 elements_to_clusterize: Optional[List] = None,  # TODO: List[Element|
                 percentage_of_period_freedom=0,
                 costs_of_period_freedom=0,
                 **kwargs):
        '''
        Modeling-Element for "index-equating"-equations


        Parameters
        ----------
        label : TYPE
            DESCRIPTION.
        es : TYPE
            DESCRIPTION.
        index_vectors_of_clusters : TYPE
            DESCRIPTION.
        fix_storage_flows : TYPE, optional
            DESCRIPTION. The default is True.
        fix_binary_vars_only : TYPE, optional
            DESCRIPTION. The default is True.
        elements_to_clusterize : TYPE, optional
            DESCRIPTION. The default is None.
        percentage_of_period_freedom : TYPE, optional
            DESCRIPTION. The default is 0.
        costs_of_period_freedom : TYPE, optional
            DESCRIPTION. The default is 0.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.system = system
        self.index_vectors_of_clusters = index_vectors_of_clusters
        self.fix_storage_flows = fix_storage_flows
        self.fix_binary_vars_only = fix_binary_vars_only
        self.elements_to_clusterize = elements_to_clusterize

        self.percentage_of_period_freedom = percentage_of_period_freedom
        self.costs_of_period_freedom = costs_of_period_freedom

        super().__init__(label, **kwargs)
        # invest_parameters to attributes:

        self.var_K_list = []

        # self.sub_elements.append(self.featureOn)

    def declare_vars_and_eqs(self, system_model: SystemModel):
        super().declare_vars_and_eqs(system_model)

    def do_modeling(self, system_model: SystemModel, time_indices: Union[list[int], range]):

        if self.elements_to_clusterize is None:
            # Alle:
            compSet = set(self.system.components)
            flowSet = self.system.flows
        else:
            # Ausgewählte:
            compSet = set(self.elements_to_clusterize)
            flowSet = {flow for flow in self.system.flows if flow.comp in self.elements_to_clusterize}

        flow: Flow

        # todo: hier anstelle alle Elemente durchgehen, nicht nur flows und comps:
        for element in flowSet | compSet:
            # Wenn StorageFlows nicht gefixt werden sollen und flow ein storage-Flow ist:
            if (not self.fix_storage_flows) and hasattr(element, 'comp') and (
            isinstance(element.comp, Storage)):
                pass  # flow hier nicht fixen!
            else:
                all_vars_of_element = element.model.variables
                for sub_element in element.all_sub_elements:
                    all_vars_of_sub_element = sub_element.model.variables
                    duplicate_var_names = set(all_vars_of_element.keys()) & set(all_vars_of_sub_element.keys())
                    if duplicate_var_names:
                        raise Exception(f'Variables {duplicate_var_names} already exists in system model')
                    all_vars_of_element.update(all_vars_of_sub_element)

                if 'on' in all_vars_of_element:
                    self.equate_indices(all_vars_of_element['on'], system_model, fix_first_index_of_period=True)
                if 'switchOn' in all_vars_of_element:
                    self.equate_indices(all_vars_of_element['switchOn'], system_model, fix_first_index_of_period=True)
                if 'switchOff' in all_vars_of_element:
                    self.equate_indices(all_vars_of_element['switchOff'], system_model, fix_first_index_of_period=True)

                if not self.fix_binary_vars_only and 'val' in all_vars_of_element:
                    self.equate_indices(all_vars_of_element['val'], system_model, fix_first_index_of_period=True)

    def equate_indices(self, aVar: Variable, system_model, fix_first_index_of_period: bool) -> Equation:
        aVar: Variable
        # todo!: idx_var1/2 wird jedes mal gemacht! nicht schick

        # Gleichung:
        # eq1: x(p1,t) - x(p3,t) = 0 # wobei p1 und p3 im gleichen Cluster sind und t = 0..N_p
        idx_var1 = np.array([])
        idx_var2 = np.array([])
        for clusterNr in self.index_vectors_of_clusters.keys():
            listOfIndexVectors = self.index_vectors_of_clusters[clusterNr]
            # alle Indexvektor-Tupel durchgehen:
            for i in range(len(listOfIndexVectors) - 1):  # ! Nur wenn cluster mehr als eine Periode enthält:
                # Falls eine Periode nicht ganz voll (eigl. nur bei letzter Periode möglich)
                v1 = listOfIndexVectors[0]
                v2 = listOfIndexVectors[i + 1]
                if not fix_first_index_of_period:
                    v1 = v1[1:]  # erstes Element immer weglassen
                    v2 = v2[1:]
                minLen = min(len(v1), len(v2))
                idx_var1 = np.append(idx_var1, v1[:minLen])
                idx_var2 = np.append(idx_var2, v2[:minLen])

        eq = Equation('equalIdx_' + aVar.label_full, self, system_model, eqType='eq')
        self.model.add_equation(eq)
        eq.add_summand(aVar, 1, indices_of_variable=idx_var1)
        eq.add_summand(aVar, -1, indices_of_variable=idx_var2)

        # Korrektur: (bisher nur für Binärvariablen:)
        if aVar.is_binary and self.percentage_of_period_freedom > 0:
            # correction-vars (so viele wie Indexe in eq:)
            var_K1 = Variable('Korr1_' + aVar.label_full.replace('.', '_'), eq.nr_of_single_equations, self.label_full,
                              system_model,
                              is_binary=True)
            self.model.add_variable(var_K1)
            var_K0 = Variable('Korr0_' + aVar.label_full.replace('.', '_'), eq.nr_of_single_equations, self.label_full,
                              system_model,
                              is_binary=True)
            self.model.add_variable(var_K0)
            # equation extends ...
            # --> On(p3) can be 0/1 independent of On(p1,t)!
            # eq1: On(p1,t) - On(p3,t) + K1(p3,t) - K0(p3,t) = 0
            # --> correction On(p3) can be:
            #  On(p1,t) = 1 -> On(p3) can be 0 -> K0=1 (,K1=0)
            #  On(p1,t) = 0 -> On(p3) can be 1 -> K1=1 (,K0=1)
            eq.add_summand(var_K1, +1)
            eq.add_summand(var_K0, -1)
            self.var_K_list.append(var_K1)
            self.var_K_list.append(var_K0)

            # interlock var_K1 and var_K2:
            # eq: var_K0(t)+var_K1(t) <= 1.1
            eq_lock = Equation('lock_K0andK1' + aVar.label_full, self, system_model, eqType='ineq')
            self.model.add_equation(eq_lock)
            eq_lock.add_summand(var_K0, 1)
            eq_lock.add_summand(var_K1, 1)
            eq_lock.add_constant(1.1)

            # Begrenzung der Korrektur-Anzahl:
            # eq: sum(K) <= n_Corr_max
            self.noOfCorrections = round(self.percentage_of_period_freedom / 100 * var_K1.length)
            eq_max = Equation('maxNoOfCorrections_' + aVar.label_full, self, system_model, eqType='ineq')
            self.model.add_equation(eq_max)
            eq_max.add_summand(var_K1, 1, as_sum=True)
            eq_max.add_summand(var_K0, 1, as_sum=True)
            eq_max.add_constant(self.noOfCorrections)  # Maximum
        return eq

    def add_share_to_globals(self, globalComp: Global, system_model):

        # einzelne Stellen korrigierbar machen (aber mit Kosten)
        if (self.percentage_of_period_freedom > 0) & (self.costs_of_period_freedom != 0):
            for var_K in self.var_K_list:
                # todo: Krücke, weil muss eigentlich sowas wie Strafkosten sein!!!
                globalComp.objective.add_summand(var_K, self.costs_of_period_freedom, as_sum=True)


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
        agg_types = (aTS.TSraw.agg_group for aTS in TSlistWithAggType)
        return Counter(agg_types)

    def _get_agg_type(self, aTS: TimeSeries):
        if (aTS.TSraw is not None):
            agg_type = aTS.TSraw.agg_group
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
        elif aTS.TSraw.agg_group is not None:
            # via agg_group:
            # i.g. n=3 -> weight=1/3
            weight = 1 / self.agg_type_count[aTS.TSraw.agg_group]
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
            aStr = ' ->' + TS.label_full + ' (weight: {:.4f}; agg_group: ' + str(self._get_agg_type(TS)) + ')'
            print(aStr.format(self._getWeight(TS)))
        if len(self.agg_type_count.keys()) > 0:
            print('agg_types: ' + str(list(self.agg_type_count.keys())))
        else:
            print('Warning!: no agg_types defined, i.e. all TS have weigth 1 (or explicit given weight)!')
