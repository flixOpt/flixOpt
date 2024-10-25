# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 10:23:49 2021
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden

Modul zur aggregierten Berechnung eines Energiesystemmodells.
"""

import copy
import timeit
from typing import Optional, List, Dict, Union, TYPE_CHECKING, Tuple
import warnings
import logging
from collections import Counter

import pandas as pd
import numpy as np
import tsam.timeseriesaggregation as tsam

from flixOpt.core import Skalar, TimeSeries
from flixOpt.elements import Flow, Component
from flixOpt.flow_system import FlowSystem
from flixOpt.components import Storage
from flixOpt.core import TimeSeriesData
from flixOpt.structure import Element, SystemModel, ElementModel, create_variable, create_equation
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

    def plot(self, colormap: str = 'viridis', show: bool = True) -> tuple:
        import matplotlib.pyplot as plt
        # Get the column names from the DataFrame
        column_names = self.original_data.columns

        # Handle colormap
        cmap = plt.get_cmap(colormap or 'viridis')  # Use default colormap if not provided

        # Generate color palette
        colors = cmap(np.linspace(0, 1, len(column_names)))

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the original and aggregated data with different line styles
        for i, col in enumerate(column_names):
            ax.plot(self.original_data.index, self.original_data[col],
                    label=f'Original - {col}', color=colors[i], linestyle='-')
            ax.plot(self.aggregated_data.index, self.aggregated_data[col],
                    label=f'Aggregated - {col}', color=colors[i], linestyle='--')

        # Add title and labels
        ax.set_title('Original vs Aggregated Data (dashed = aggregated)')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add legend
        ax.legend(loc='best')

        # Adjust layout
        plt.tight_layout()
        if show:
            plt.show()

        # Return fig, ax for further use
        return fig, ax

    def get_cluster_indices(self) -> Dict[str, List[np.ndarray]]:
        """
        Generates a dictionary that maps each cluster to a list of index vectors representing the time steps
        assigned to that cluster for each period.

        Returns:
            dict: {cluster_0: [index_vector_3, index_vector_7, ...],
                   cluster_1: [index_vector_1],
                   ...}
        """
        clusters = self.tsam.clusterPeriodNoOccur.keys()
        index_vectors = {cluster: [] for cluster in clusters}

        period_length = len(self.tsam.stepIdx)
        total_steps = len(self.tsam.timeSeries)

        for period, cluster_id in enumerate(self.tsam.clusterOrder):
            start_idx = period * period_length
            end_idx = np.min([start_idx + period_length, total_steps])
            index_vectors[cluster_id].append(np.arange(start_idx, end_idx))

        return index_vectors

    def get_equation_indices(self, fix_first_index_of_period: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates pairs of indices for the equations by comparing index vectors of the same cluster.
        If `fix_first_index_of_period` is True, the first index of each period is skipped.

        Args:
            fix_first_index_of_period (bool): Whether to fix or to skip the first index of each period in the comparison.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two arrays of indices.
        """
        idx_var1 = []
        idx_var2 = []

        # Iterate through cluster index vectors
        for cluster_id, index_vectors in self.get_cluster_indices().items():
            if len(index_vectors) <= 1:  # Only proceed if cluster has more than one period
                continue

            # Process the first vector, optionally skip first index
            first_vector = index_vectors[0]
            if fix_first_index_of_period:
                first_vector = first_vector[1:]

            # Compare first vector to others in the cluster
            for other_vector in index_vectors[1:]:
                if fix_first_index_of_period:
                    other_vector = other_vector[1:]

                # Compare elements up to the minimum length of both vectors
                min_len = min(len(first_vector), len(other_vector))
                idx_var1.extend(first_vector[:min_len])
                idx_var2.extend(other_vector[:min_len])

        # Convert lists to numpy arrays
        return np.array(idx_var1), np.array(idx_var2)


class AggregationModel(ElementModel):
    """ The AggregationModel interacts directly with the SystemModel, inserting its Variables and Equations there """
    def __init__(self,
                 aggregation_parameters: AggregationParameters,
                 flow_system: FlowSystem,
                 aggregation_data: Aggregation,
                 components_to_clusterize: Optional[List[Component]]):
        """
        Modeling-Element for "index-equating"-equations
        """
        super().__init__(Element("Aggregation"), "Model")
        self.flow_system = flow_system
        self.aggregation_parameters = aggregation_parameters
        self.aggregation_data = aggregation_data
        self.components_to_clusterize = components_to_clusterize

    def do_modeling(self, system_model: SystemModel):
        if self.components_to_clusterize is None:
            components = self.flow_system.components
        else:
            components = [component for component in self.components_to_clusterize]

        for component in components:
            if isinstance(component, Storage) and not self.aggregation_parameters.fix_storage_flows:
                continue  # Fix Nothing in The Storage
            all_variables_of_component = component.model.all_variables
            print(all_variables_of_component.keys())
            """
            if 'on' in all_variables_of_component:
                self.equate_indices(all_vars_of_element['on'], system_model, fix_first_index_of_period=True)
            if 'switchOn' in all_variables_of_component:
                self.equate_indices(all_vars_of_element['switchOn'], system_model, fix_first_index_of_period=True)
            if 'switchOff' in all_variables_of_component:
                self.equate_indices(all_vars_of_element['switchOff'], system_model, fix_first_index_of_period=True)

            if not self.fix_binary_vars_only and 'val' in all_vars_of_element:
                self.equate_indices(all_vars_of_element['val'], system_model, fix_first_index_of_period=True)

            """
    def equate_indices(self, variable: Variable,
                       indices: Tuple[np.ndarray, np.ndarray],
                       system_model: SystemModel) -> Equation:
        # Gleichung:
        # eq1: x(p1,t) - x(p3,t) = 0 # wobei p1 und p3 im gleichen Cluster sind und t = 0..N_p
        length = len(indices[0])
        assert len(indices[0]) == len(indices[1]), f'The length of the indices must match!!'

        eq = create_equation(f'Equate_indices_of_{variable.label}', self, system_model)
        eq.add_summand(variable, 1, indices_of_variable=indices[0])
        eq.add_summand(variable, -1, indices_of_variable=indices[1])

        # Korrektur: (bisher nur für Binärvariablen:)
        if variable.is_binary and self.aggregation_parameters.percentage_of_period_freedom > 0:
            # correction-vars (so viele wie Indexe in eq:)
            var_K1 = create_variable(f'Korr1_{variable.label}', self, length, system_model, is_binary=True)
            var_K0 = create_variable(f'Korr0_{variable.label}', self, length, system_model, is_binary=True)
            # equation extends ...
            # --> On(p3) can be 0/1 independent of On(p1,t)!
            # eq1: On(p1,t) - On(p3,t) + K1(p3,t) - K0(p3,t) = 0
            # --> correction On(p3) can be:
            #  On(p1,t) = 1 -> On(p3) can be 0 -> K0=1 (,K1=0)
            #  On(p1,t) = 0 -> On(p3) can be 1 -> K1=1 (,K0=1)
            eq.add_summand(var_K1, +1)
            eq.add_summand(var_K0, -1)

            # interlock var_K1 and var_K2:
            # eq: var_K0(t)+var_K1(t) <= 1.1
            eq_lock = create_equation(f'lock_K0andK1_{variable.label}', self, system_model, eq_type='ineq')
            eq_lock.add_summand(var_K0, 1)
            eq_lock.add_summand(var_K1, 1)
            eq_lock.add_constant(1.1)

            # Begrenzung der Korrektur-Anzahl:
            # eq: sum(K) <= n_Corr_max
            eq_max = create_equation(f'Nr_of_Corrections_{variable.label}', self, system_model, eq_type='ineq')
            eq_max.add_summand(var_K1, 1, as_sum=True)
            eq_max.add_summand(var_K0, 1, as_sum=True)
            eq_max.add_constant(self.nr_of_corrections)  # Maximum
        return eq

    def add_share_to_globals(self, system_model: SystemModel):
        # TODO: percentage_of_period_freedom is not used??
        penalty = self.aggregation_parameters.costs_of_period_freedom
        if (self.aggregation_parameters.percentage_of_period_freedom > 0) and penalty != 0:
            for label, variable in self.variables.items():
                system_model.effect_collection_model.add_share_to_penalty(f'Penalty_{label}', self.element, variable, penalty)

    @property
    def nr_of_corrections(self) -> int:
        return round(self.aggregation_parameters.percentage_of_period_freedom / 100 * len(self.variables.values()))


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


class AggregationParameters:
    def __init__(self,
                 hours_per_period: float,
                 nr_of_periods: int,
                 fix_storage_flows: bool,
                 fix_binary_vars_only: bool,
                 fix_first_index_of_period: bool = False,
                 percentage_of_period_freedom: float = 0,
                 costs_of_period_freedom: float = 0,
                 time_series_for_high_peaks: Optional[List[TimeSeriesData]] = None,
                 time_series_for_low_peaks: Optional[List[TimeSeriesData]] = None
                 ):
        self.hours_per_period = hours_per_period
        self.nr_of_periods = nr_of_periods
        self.fix_storage_flows = fix_storage_flows
        self.fix_binary_vars_only = fix_binary_vars_only
        self.fix_first_index_of_period = fix_first_index_of_period
        self.percentage_of_period_freedom = percentage_of_period_freedom
        self.costs_of_period_freedom = costs_of_period_freedom
        self.time_series_for_high_peaks: List[TimeSeriesData] = time_series_for_high_peaks or []
        self.time_series_for_low_peaks: List[TimeSeriesData] = time_series_for_low_peaks or []

    @property
    def use_extreme_periods(self):
        return self.time_series_for_high_peaks or self.time_series_for_low_peaks

    @property
    def labels_for_high_peaks(self) -> List[str]:
        return [ts.label for ts in self.time_series_for_high_peaks]

    @property
    def labels_for_low_peaks(self) -> List[str]:
        return [ts.label for ts in self.time_series_for_low_peaks]

    @property
    def use_low_peaks(self):
        return self.time_series_for_low_peaks is not None
