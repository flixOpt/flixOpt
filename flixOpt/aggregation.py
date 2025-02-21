"""
This module contains the Aggregation functionality for the flixOpt framework.
Through this, aggregating TimeSeriesData is possible.
"""

import copy
import logging
import timeit
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import linopy
import numpy as np
import pandas as pd
import tsam.timeseriesaggregation as tsam

from .components import Storage
from .core import Skalar, TimeSeries, TimeSeriesData, DataConverter
from .elements import Component
from .flow_system import FlowSystem
from .math_modeling import Equation, Variable, VariableTS
from .structure import (
    Element,
    Model,
    SystemModel,
    create_equation,
    create_variable,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go

warnings.filterwarnings('ignore', category=DeprecationWarning)
logger = logging.getLogger('flixOpt')


class Aggregation:
    """
    aggregation organizing class
    """

    def __init__(
        self,
        original_data: pd.DataFrame,
        hours_per_time_step: Skalar,
        hours_per_period: Skalar,
        nr_of_periods: int = 8,
        weights: Dict[str, float] = None,
        time_series_for_high_peaks: List[str] = None,
        time_series_for_low_peaks: List[str] = None,
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
        self.weights = weights or {}
        self.time_series_for_high_peaks = time_series_for_high_peaks or []
        self.time_series_for_low_peaks = time_series_for_low_peaks or []

        self.aggregated_data: Optional[pd.DataFrame] = None
        self.clustering_duration_seconds = None
        self.tsam: Optional[tsam.TimeSeriesAggregation] = None

    def cluster(self) -> None:
        """
        Durchführung der Zeitreihenaggregation
        """
        start_time = timeit.default_timer()
        # Erstellen des aggregation objects
        self.tsam = tsam.TimeSeriesAggregation(
            self.original_data,
            noTypicalPeriods=self.nr_of_periods,
            hoursPerPeriod=self.hours_per_period,
            resolution=self.hours_per_time_step,
            clusterMethod='k_means',
            extremePeriodMethod='new_cluster_center'
            if self.use_extreme_periods
            else 'None',  # Wenn Extremperioden eingebunden werden sollen, nutze die Methode 'new_cluster_center' aus tsam
            weightDict=self.weights,
            addPeakMax=self.time_series_for_high_peaks,
            addPeakMin=self.time_series_for_low_peaks,
        )

        self.tsam.createTypicalPeriods()  # Ausführen der Aggregation/Clustering
        self.aggregated_data = self.tsam.predictOriginalData()

        self.clustering_duration_seconds = timeit.default_timer() - start_time  # Zeit messen:
        logger.info(self.describe_clusters())

    def describe_clusters(self) -> str:
        description = {}
        for cluster in self.get_cluster_indices().keys():
            description[cluster] = [
                str(indexVector[0]) + '...' + str(indexVector[-1])
                for indexVector in self.get_cluster_indices()[cluster]
            ]

        if self.use_extreme_periods:
            # Zeitreihe rauslöschen:
            extreme_periods = self.tsam.extremePeriods.copy()
            for key in extreme_periods:
                del extreme_periods[key]['profile']
        else:
            extreme_periods = {}

        return (
            f'{"":#^80}\n'
            f'{" Clustering ":#^80}\n'
            f'periods_order:\n'
            f'{self.tsam.clusterOrder}\n'
            f'clusterPeriodNoOccur:\n'
            f'{self.tsam.clusterPeriodNoOccur}\n'
            f'index_vectors_of_clusters:\n'
            f'{description}\n'
            f'{"":#^80}\n'
            f'extreme_periods:\n'
            f'{extreme_periods}\n'
            f'{"":#^80}'
        )

    @property
    def use_extreme_periods(self):
        return self.time_series_for_high_peaks or self.time_series_for_low_peaks

    def plot(self, colormap: str = 'viridis', show: bool = True) -> 'go.Figure':
        from . import plotting

        df_org = self.original_data.copy().rename(
            columns={col: f'Original - {col}' for col in self.original_data.columns}
        )
        df_agg = self.aggregated_data.copy().rename(
            columns={col: f'Aggregated - {col}' for col in self.aggregated_data.columns}
        )
        fig = plotting.with_plotly(df_org, 'line', colors=colormap)
        for trace in fig.data:
            trace.update(dict(line=dict(dash='dash')))
        fig = plotting.with_plotly(df_agg, 'line', colors=colormap, show=show, fig=fig)

        fig.update_layout(
            title='Original vs Aggregated Data (original = ---)', xaxis_title='Index', yaxis_title='Value'
        )
        return fig

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

    def get_equation_indices(self, skip_first_index_of_period: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates pairs of indices for the equations by comparing index vectors of the same cluster.
        If `skip_first_index_of_period` is True, the first index of each period is skipped.

        Args:
            skip_first_index_of_period (bool): Whether to include or skip the first index of each period.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two arrays of indices.
        """
        idx_var1 = []
        idx_var2 = []

        # Iterate through cluster index vectors
        for index_vectors in self.get_cluster_indices().values():
            if len(index_vectors) <= 1:  # Only proceed if cluster has more than one period
                continue

            # Process the first vector, optionally skip first index
            first_vector = index_vectors[0][1:] if skip_first_index_of_period else index_vectors[0]

            # Compare first vector to others in the cluster
            for other_vector in index_vectors[1:]:
                if skip_first_index_of_period:
                    other_vector = other_vector[1:]

                # Compare elements up to the minimum length of both vectors
                min_len = min(len(first_vector), len(other_vector))
                idx_var1.extend(first_vector[:min_len])
                idx_var2.extend(other_vector[:min_len])

        # Convert lists to numpy arrays
        return np.array(idx_var1), np.array(idx_var2)


class TimeSeriesCollection:
    def __init__(self, *time_series):
        self.time_serieses: List[TimeSeries] = list(time_series)
        self._check_unique_labels()
        self.group_weights = self._calculate_group_weights()
        self.weights = self._calculate_aggregation_weights()

        if np.all(np.isclose(list(self.weights.values()), 1, atol=1e-6)):
            logger.info('All Aggregation weights were set to 1')

    def insert_data(self, data: pd.DataFrame):
        for time_series in self.time_serieses:
            if time_series.name in data.columns:
                time_series.stored_data = DataConverter.as_dataarray(
                    data[time_series.name],
                    time_series.active_timesteps,
                    time_series.active_periods
                )
                logger.debug(f'Inserted data for {time_series.name}')

    def description(self) -> str:
        # TODO:
        result = f'{len(self.time_serieses)} TimeSeries used for aggregation:\n'
        for time_series in self.time_serieses:
            result += f' -> {time_series.name} (weight: {self.weights[time_series.name]:.4f}; group: "{time_series.aggregation_group}")\n'
        if self.group_weights:
            result += f'Aggregation_Groups: {list(self.group_weights.keys())}\n'
        else:
            result += 'Warning!: no agg_types defined, i.e. all TS have weight 1 (or explicitly given weight)!\n'
        return result

    def _calculate_group_weights(self) -> Dict[str, float]:
        """Calculates the aggregation weights of each group"""
        groups = [
            time_series.aggregation_group
            for time_series in self.time_serieses
            if time_series.aggregation_group is not None
        ]
        group_size = dict(Counter(groups))
        group_weights = {group: 1 / size for group, size in group_size.items()}
        return group_weights

    def _calculate_aggregation_weights(self) -> Dict[str, float]:
        """Calculates the aggregation weight for each TimeSeries. Default is 1"""
        return {
            time_series.name: self.group_weights.get(time_series.aggregation_group, time_series.aggregation_weight or 1)
            for time_series in self.time_serieses
        }

    def _check_unique_labels(self):
        """Makes sure every label of the TimeSeries in time_series_list is unique"""
        label_counts = Counter([time_series.name for time_series in self.time_serieses])
        duplicates = [label for label, count in label_counts.items() if count > 1]
        assert duplicates == [], 'Duplicate TimeSeries labels found: {}.'.format(', '.join(duplicates))


class AggregationParameters:
    def __init__(
        self,
        hours_per_period: float,
        nr_of_periods: int,
        fix_storage_flows: bool,
        aggregate_data_and_fix_non_binary_vars: bool,
        percentage_of_period_freedom: float = 0,
        penalty_of_period_freedom: float = 0,
        time_series_for_high_peaks: List[TimeSeriesData] = None,
        time_series_for_low_peaks: List[TimeSeriesData] = None,
    ):
        """
        Initializes aggregation parameters for time series data

        Parameters
        ----------
        hours_per_period : float
            Duration of each period in hours.
        nr_of_periods : int
            Number of typical periods to use in the aggregation.
        fix_storage_flows : bool
            Whether to aggregate storage flows (load/unload); if other flows
            are fixed, fixing storage flows is usually not required.
        aggregate_data_and_fix_non_binary_vars : bool
            Whether to aggregate all time series data, which allows to fix all time series variables (like flow_rate),
            or only fix binary variables. If False non time_series data is changed!! If True, the mathematical Problem
            is simplified even further.
        percentage_of_period_freedom : float, optional
            Specifies the maximum percentage (0–100) of binary values within each period
            that can deviate as "free variables", chosen by the solver (default is 0).
            This allows binary variables to be 'partly equated' between aggregated periods.
        penalty_of_period_freedom : float, optional
            The penalty associated with each "free variable"; defaults to 0. Added to Penalty
        time_series_for_high_peaks : list of TimeSeriesData
            List of time series to use for explicitly selecting periods with high values.
        time_series_for_low_peaks : list of TimeSeriesData
            List of time series to use for explicitly selecting periods with low values.
        """
        self.hours_per_period = hours_per_period
        self.nr_of_periods = nr_of_periods
        self.fix_storage_flows = fix_storage_flows
        self.aggregate_data_and_fix_non_binary_vars = aggregate_data_and_fix_non_binary_vars
        self.percentage_of_period_freedom = percentage_of_period_freedom
        self.penalty_of_period_freedom = penalty_of_period_freedom
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


class AggregationModel(Model):
    """The AggregationModel holds equations and variables related to the Aggregation of a FLowSystem.
    It creates Equations that equates indices of variables, and introduces penalties related to binary variables, that
    escape the equation to their related binaries in other periods"""

    def __init__(
        self,
        model: SystemModel,
        aggregation_parameters: AggregationParameters,
        flow_system: FlowSystem,
        aggregation_data: Aggregation,
        components_to_clusterize: Optional[List[Component]],
    ):
        """
        Modeling-Element for "index-equating"-equations
        """
        super().__init__(model, label_of_element='No Element', label_full='Aggregation')
        self.flow_system = flow_system
        self.aggregation_parameters = aggregation_parameters
        self.aggregation_data = aggregation_data
        self.components_to_clusterize = components_to_clusterize

    def do_modeling(self, system_model: SystemModel):
        if not self.components_to_clusterize:
            components = self.flow_system.components.values()
        else:
            components = [component for component in self.components_to_clusterize]

        indices = self.aggregation_data.get_equation_indices(skip_first_index_of_period=True)

        time_variables: List[str] = [k for k, v in self._model.variables.data.items() if 'time' in v.indexes]
        binary_time_variables: List[str] = [
            k for k, v in self._model.variables.data.items() if 'time' in v.indexes and k in self._model.binaries
        ]

        for component in components:
            if isinstance(component, Storage) and not self.aggregation_parameters.fix_storage_flows:
                continue  # Fix Nothing in The Storage

            all_variables_of_component = component.model.variables
            if self.aggregation_parameters.aggregate_data_and_fix_non_binary_vars:

                relevant_variables = all_variables_of_component[v for v in all_variables_of_component if v in self._model.]
            else:
                relevant_variables = [
                    v for v in all_variables_of_component.values() if isinstance(v, VariableTS) and v.is_binary
                ]
            for variable in relevant_variables:
                self.equate_indices(variable, indices)

        penalty = self.aggregation_parameters.penalty_of_period_freedom
        if (self.aggregation_parameters.percentage_of_period_freedom > 0) and penalty != 0:
            for label, variable in self.variables_direct.items():
                system_model.effects.add_share_to_penalty(
                    system_model,
                    f'Aggregation_penalty__{label}',
                    variable * penalty
                )

    def equate_indices(self, variable: linopy.Variable, indices: Tuple[np.ndarray, np.ndarray]) -> None:
        assert len(indices[0]) == len(indices[1]), 'The length of the indices must match!!'
        length = len(indices[0])

        # Gleichung:
        # eq1: x(p1,t) - x(p3,t) = 0 # wobei p1 und p3 im gleichen Cluster sind und t = 0..N_p
        con = self.add(self._model.add_constraints(
            variable.isel(time=indices[0]) - variable.isel(time=indices[1]) == 0,
            name=f'Equate_indices_of_{variable.name}'),
            variable.name)

        # Korrektur: (bisher nur für Binärvariablen:)
        if variable in self._model.binaries and self.aggregation_parameters.percentage_of_period_freedom > 0:
            var_k1 = self.add(self._model.add_variables(
                binary=True,
                coords={'time': variable.isel(time=indices[0]).indexes['time']},
                name=f'{self.label_full}|Korr1|{variable.name}'), f'Korr1|{variable.name}')

            var_k0 = self.add(self._model.add_variables(
                binary=True,
                coords={'time': variable.isel(time=indices[0]).indexes['time']},
                name=f'{self.label_full}|Korr0|{variable.name}'), f'Korr0|{variable.name}')

            # equation extends ...
            # --> On(p3) can be 0/1 independent of On(p1,t)!
            # eq1: On(p1,t) - On(p3,t) + K1(p3,t) - K0(p3,t) = 0
            # --> correction On(p3) can be:
            #  On(p1,t) = 1 -> On(p3) can be 0 -> K0=1 (,K1=0)
            #  On(p1,t) = 0 -> On(p3) can be 1 -> K1=1 (,K0=1)
            con.lhs += 1 * var_k1 - 1 * var_k0

            # interlock var_k1 and var_K2:
            # eq: var_k0(t)+var_k1(t) <= 1.1
            self.add(self._model.add_constraints(
                var_k0 + var_k1 <= 1.1,
                name=f'{self.label_full}|lock_K0andK1|{variable.name}'),
                f'lock_K0andK1|{variable.name}'
            )

            # Begrenzung der Korrektur-Anzahl:
            # eq: sum(K) <= n_Corr_max
            self.add(self._model.add_constraints(
                sum(var_k0) + sum(var_k1) <= round(self.aggregation_parameters.percentage_of_period_freedom / 100 * length),
                name=f'{self.label_full}|Nr_of_Corrections|{variable.name}'),
                f'Nr_of_Corrections|{variable.name}'
            )
