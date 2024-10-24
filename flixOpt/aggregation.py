# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 10:23:49 2021
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden

Modul zur aggregierten Berechnung eines Energiesystemmodells.
"""

import copy
import timeit
from typing import Optional, List, Dict, Union, TYPE_CHECKING
import warnings
import logging
from collections import Counter

import pandas as pd
import numpy as np
import tsam.timeseriesaggregation as tsam

from flixOpt.core import Skalar, TimeSeries
from flixOpt.elements import Flow
from flixOpt.flow_system import FlowSystem
from flixOpt.components import Storage
from flixOpt.core import TimeSeriesData
from flixOpt.structure import Element, SystemModel
from flixOpt.math_modeling import Equation, Variable

if TYPE_CHECKING:
    from flixOpt.effects import EffectCollection


warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger('flixOpt')


class Aggregation:
    """
    aggregation organizing class
    """
    def __init__(self,
                 original_data: pd.DataFrame,
                 hours_per_time_step: Skalar,
                 hours_per_period: Skalar,
                 nr_of_periods: int = 8,
                 weights: Optional[Dict[str, float]] = None,
                 time_series_for_high_peaks: Optional[List[str]] = None,
                 time_series_for_low_peaks: Optional[List[str]] = None
                 ):
        """
        Write a docstring please

        Parameters
        ----------
        timeseries: pd.DataFrame
            timeseries of the data with a datetime index
        """
        self.original_data = copy.deepcopy(original_data)
        self.hours_per_time_step = hours_per_time_step
        self.hours_per_period = hours_per_period
        self.nr_of_periods = nr_of_periods
        self.nr_of_time_steps = len(self.original_data.index)
        self.weights = weights
        self.time_series_for_high_peaks = time_series_for_high_peaks
        self.time_series_for_low_peaks = time_series_for_low_peaks

        self.aggregated_data: Optional[pd.DataFrame] = None
        self.clustering_duration_seconds = None
        self.tsam: Optional[tsam.TimeSeriesAggregation] = None

        self.original_data.index = pd.MultiIndex.from_arrays(
            [[0] * self.nr_of_time_steps, list(range(self.nr_of_time_steps))],
            names=['Period', 'TimeStep'])

    def cluster(self) -> None:
        """
        Durchführung der Zeitreihenaggregation
        """
        start_time = timeit.default_timer()
        # Erstellen des aggregation objects
        self.tsam = tsam.TimeSeriesAggregation(self.original_data,
                                                      noTypicalPeriods=self.nr_of_periods,
                                                      hoursPerPeriod=self.hours_per_period,
                                                      resolution=self.hours_per_time_step,
                                                      clusterMethod='k_means',
                                                      extremePeriodMethod='new_cluster_center' if self.use_extreme_periods else 'None',  # Wenn Extremperioden eingebunden werden sollen, nutze die Methode 'new_cluster_center' aus tsam
                                                      weightDict=self.weights,
                                                      addPeakMax=self.time_series_for_high_peaks,
                                                      addPeakMin=self.time_series_for_low_peaks
                                                      )

        self.tsam.createTypicalPeriods()   # Ausführen der Aggregation/Clustering
        self.aggregated_data = self.tsam.predictOriginalData()

        self.clustering_duration_seconds = timeit.default_timer() - start_time   # Zeit messen:
        logger.info(self.describe_clusters())

    def describe_clusters(self) -> str:
        aVisual = {}
        for cluster in self.get_cluster_indices(self.tsam).keys():
            aVisual[cluster] = [str(indexVector[0]) + '...' + str(indexVector[-1]) for indexVector in
                                self.get_cluster_indices(self.tsam)[cluster]]

        if self.use_extreme_periods:
            # Zeitreihe rauslöschen:
            extremePeriods = self.tsam.extremePeriods.copy()
            for key, val in extremePeriods.items():
                del (extremePeriods[key]['profile'])
        else:
            extremePeriods = {}

        return (f'{"":#^80}\n'
                f'{" Clustering ":#^80}\n'
                f'periods_order:\n'
                f'{self.tsam.clusterOrder}\n'
                f'clusterPeriodNoOccur:\n'
                f'{self.tsam.clusterPeriodNoOccur}\n'
                f'index_vectors_of_clusters:\n'
                f'{aVisual}\n'
                f'{"":#^80}\n'
                f'extremePeriods:\n'
                f'{extremePeriods}\n'
                f'{"":#^80}')

    @property
    def use_extreme_periods(self):
        return self.time_series_for_high_peaks or self.time_series_for_low_peaks

    @staticmethod
    def get_cluster_indices(aggregation: tsam.TimeSeriesAggregation) -> Dict[str, List[np.ndarray]]:
        """
        Generates a dictionary that maps each cluster to a list of index vectors representing the time steps
        assigned to that cluster for each period.

        Returns:
            dict: {cluster_0: [index_vector_3, index_vector_7, ...],
                   cluster_1: [index_vector_1],
                   ...}
        """
        clusters = aggregation.clusterPeriodNoOccur.keys()
        index_vectors = {cluster: [] for cluster in clusters}

        period_length = len(aggregation.stepIdx)
        total_steps = len(aggregation.timeSeries)

        for period, cluster_id in enumerate(aggregation.clusterOrder):
            start_idx = period * period_length
            end_idx = np.min([start_idx + period_length, total_steps])
            index_vectors[cluster_id].append(np.arange(start_idx, end_idx))

        return index_vectors


class AggregationModeling(Element):
    # ModelingElement mit Zusatz-Glg. und Variablen für aggregierte Berechnung
    def __init__(self,
                 label: str,
                 flow_system: FlowSystem,
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
        self.flow_system = flow_system
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

    def do_modeling(self, system_model: SystemModel):

        if self.elements_to_clusterize is None:
            # Alle:
            compSet = set(self.flow_system.components)
            flowSet = self.flow_system.all_flows
        else:
            # Ausgewählte:
            compSet = set(self.elements_to_clusterize)
            flowSet = {flow for flow in self.flow_system.all_flows if flow.comp in self.elements_to_clusterize}

        flow: Flow

        # todo: hier anstelle alle Elemente durchgehen, nicht nur flows und comps:
        for element in flowSet | compSet:
            # Wenn StorageFlows nicht gefixt werden sollen und flow ein storage-Flow ist:
            if (not self.fix_storage_flows) and hasattr(element, 'comp') and (
            isinstance(element.comp, Storage)):
                pass  # flow hier nicht fixen!
            else:
                all_vars_of_element = element.all_variables_with_sub_elements
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
        self.model.add_equations(eq)
        eq.add_summand(aVar, 1, indices_of_variable=idx_var1)
        eq.add_summand(aVar, -1, indices_of_variable=idx_var2)

        # Korrektur: (bisher nur für Binärvariablen:)
        if aVar.is_binary and self.percentage_of_period_freedom > 0:
            # correction-vars (so viele wie Indexe in eq:)
            var_K1 = Variable('Korr1_' + aVar.label_full.replace('.', '_'), eq.nr_of_single_equations, self.label_full,
                              system_model,
                              is_binary=True)
            self.model.add_variables(var_K1)
            var_K0 = Variable('Korr0_' + aVar.label_full.replace('.', '_'), eq.nr_of_single_equations, self.label_full,
                              system_model,
                              is_binary=True)
            self.model.add_variables(var_K0)
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
            self.model.add_equations(eq_lock)
            eq_lock.add_summand(var_K0, 1)
            eq_lock.add_summand(var_K1, 1)
            eq_lock.add_constant(1.1)

            # Begrenzung der Korrektur-Anzahl:
            # eq: sum(K) <= n_Corr_max
            self.noOfCorrections = round(self.percentage_of_period_freedom / 100 * var_K1.length)
            eq_max = Equation('maxNoOfCorrections_' + aVar.label_full, self, system_model, eqType='ineq')
            self.model.add_equations(eq_max)
            eq_max.add_summand(var_K1, 1, as_sum=True)
            eq_max.add_summand(var_K0, 1, as_sum=True)
            eq_max.add_constant(self.noOfCorrections)  # Maximum
        return eq

    def add_share_to_globals(self, effect_collection: 'EffectCollection', system_model):
        # TODO: BUGFIX
        # einzelne Stellen korrigierbar machen (aber mit Kosten)
        if (self.percentage_of_period_freedom > 0) & (self.costs_of_period_freedom != 0):
            for var_K in self.var_K_list:
                # todo: Krücke, weil muss eigentlich sowas wie Strafkosten sein!!!
                effect_collection.objective.add_summand(var_K, self.costs_of_period_freedom, as_sum=True)


class TimeSeriesCollection:
    def __init__(self,
                 time_series_list: List[TimeSeries]):
        self.time_series_list = time_series_list
        self.group_weights: Dict[str, float] = {}
        self._unique_labels()
        self._calculate_aggregation_weigths()
        self.weights: Dict[str, float] = {time_series.label: time_series.aggregation_weight for
                                          time_series in self.time_series_list}
        self.data: Dict[str, np.ndarray] = {time_series.label: time_series.active_data for
                                            time_series in self.time_series_list}

        if np.all(np.isclose(list(self.weights.values()), 1, atol=1e-6)):
            logger.info(f'All Aggregation weights were set to 1')

    def _calculate_aggregation_weigths(self):
        """ Calculates the aggergation weights of all TimeSeries. Necessary to use groups"""
        groups = [time_series.aggregation_group for time_series in self.time_series_list if
                  time_series.aggregation_group is not None]
        group_size = dict(Counter(groups))
        self.group_weights = {group: 1 / size for group, size in group_size.items()}
        for time_series in self.time_series_list:
            time_series.aggregation_weight = self.group_weights.get(time_series.aggregation_group, 1)

    def _unique_labels(self):
        """ Makes sure every label of the TimeSeries in time_series_list is unique """
        label_counts = Counter([time_series.label for time_series in self.time_series_list])
        duplicates = [label for label, count in label_counts.items() if count > 1]
        assert duplicates == [], "Duplicate TimeSeries labels found: {}.".format(', '.join(duplicates))

    def description(self) -> str:
        #TODO:
        result = f'{len(self.time_series_list)} TimeSeries used for aggregation:\n'
        for time_series in self.time_series_list:
            result += f' -> {time_series.label} (weight: {time_series.aggregation_weight:.4f}; group: "{time_series.aggregation_group}")\n'
        if self.group_weights:
            result += f'Aggregation_Groups: {list(self.group_weights.keys())}\n'
        else:
            result += 'Warning!: no agg_types defined, i.e. all TS have weight 1 (or explicitly given weight)!\n'
        return result
