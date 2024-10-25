# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:40:23 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import datetime
import logging
import math
import pathlib
import timeit
from typing import List, Dict, Optional, Literal, Tuple, Union, TYPE_CHECKING

import numpy as np
import yaml

from flixOpt.aggregation import TimeSeriesCollection, AggregationParameters, AggregationModel
from flixOpt.core import TimeSeriesData
from flixOpt.structure import SystemModel
from flixOpt.flow_system import FlowSystem
from flixOpt.elements import Component
from flixOpt.features import InvestmentModel


logger = logging.getLogger('flixOpt')


class Calculation:
    """
    class for defined way of solving a flow_system optimization
    """
    def __init__(self, name, flow_system: FlowSystem,
                 modeling_language: Literal["pyomo", "cvxpy"] = "pyomo",
                 time_indices: Optional[Union[range, List[int]]] = None):
        """
        Parameters
        ----------
        name : str
            name of calculation
        flow_system : FlowSystem
            flow_system which should be calculated
        modeling_language : 'pyomo','cvxpy' (not implemeted yet)
            choose optimization modeling language
        time_indices : List[int] or None
            list with indices, which should be used for calculation. If None, then all timesteps are used.
        """
        self.name = name
        self.flow_system = flow_system
        self.modeling_language = modeling_language
        self.time_indices = time_indices

        self.system_model: Optional[SystemModel] = None
        self.durations = {'modeling': 0.0, 'solving': 0.0}  # Dauer der einzelnen Dinge

        self._paths: Dict[str, Optional[Union[pathlib.Path, List[pathlib.Path]]]] = {'log': None, 'data': None, 'info': None}
        self._results = None

    def description_of_equations_as_dict(self, system_model: int = 0) -> Dict:
        return {'Components': {comp.label: comp.model.description_of_equations for comp in self.flow_system.components},
                'Buses': {bus.label: bus.model.description_of_equations for bus in self.flow_system.all_buses},
                'Objective': 'MISSING AFTER REWORK',
                'Effects': self.flow_system.effect_collection.model.description_of_equations}

    def description_of_variables_as_dict(self, system_model: int = 0) -> Dict:
        return {'Components': {comp.label: comp.model.description_of_variables + [{flow.label: flow.model.description_of_variables
                                                                         for flow in comp.inputs + comp.outputs}]
                          for comp in self.flow_system.components},
                'Buses': {bus.label: bus.model.description_of_variables for bus in self.flow_system.all_buses},
                'Objective': 'MISSING AFTER REWORK',
                'Effects': self.flow_system.effect_collection.model.description_of_variables}

    def describe_equations(self, system_model: int = 0) -> str:
        return (f'\n'
                f'{"":#^80}\n'
                f'{" Equations of FlowSystem ":#^80}\n\n'
                f'{yaml.dump(self.description_of_equations_as_dict(system_model), default_flow_style=False, allow_unicode=True)}')

    def describe_variables(self, system_model: int = 0) -> str:
        return (f'\n'
                f'{"":#^80}\n'
                f'{" Variables of FlowSystem ":#^80}\n\n'
                f'{yaml.dump(self.description_of_variables_as_dict(system_model))}')

    def _define_path_names(self, path: str, save_results: bool, include_timestamp: bool = True,
                           nr_of_system_models: int = 1):
        """
        Creates the path for saving results and alters the name of the calculation to have a timestamp
        """
        if include_timestamp:
            timestamp = datetime.datetime.now()
            self.name = f'{timestamp.strftime("%Y-%m-%d")}_{self.name.replace(" ", "")}'

        if save_results:
            path = pathlib.Path.cwd() / path  # absoluter Pfad:
            path.mkdir(parents=True, exist_ok=True)  # Pfad anlegen, fall noch nicht vorhanden:

            self._paths["log"] = [path / f'{self.name}_solver_{i}.log' for i in range(nr_of_system_models)]
            self._paths["data"] = path / f'{self.name}_data.pickle'
            self._paths["info"] = path / f'{self.name}_solvingInfos.yaml'

    def _save_solve_infos(self):
        message = f' Saved Calculation: {self.name} '
        logger.info(f'{"":#^80}\n'
                    f'{message:#^80}\n'
                    f'{"":#^80}')

    def results(self):
        if self._results is None:
            self._results = self.system_model.results()
        return self._results


class FullCalculation(Calculation):
    """
    class for defined way of solving a flow_system optimization
    """

    def do_modeling(self) -> SystemModel:
        t_start = timeit.default_timer()

        self.flow_system.transform_data()
        for time_series in self.flow_system.all_time_series:
            time_series.activate_indices(self.time_indices)

        self.system_model = SystemModel(self.name, self.modeling_language, self.flow_system, self.time_indices)
        self.system_model.do_modeling()
        self.system_model.to_math_model()

        self.durations['modeling'] = round(timeit.default_timer() - t_start, 2)
        return self.system_model

    def solve(self, solverProps: dict, path='results/', save_results=True):
        self._define_path_names(path, save_results, nr_of_system_models=1)
        t_start = timeit.default_timer()
        self.system_model.solve(**solverProps, logfile_name=self._paths['log'][0])
        self.durations['solving'] = round(timeit.default_timer() - t_start, 2)

        if save_results:
            self._save_solve_infos()


class AggregatedCalculation(Calculation):
    """
    class for defined way of solving a flow_system optimization
    """

    def __init__(self, name, flow_system: FlowSystem,
                 aggregation_parameters: AggregationParameters,
                 components_to_clusterize: Optional[List[Component]] = None,
                 modeling_language: Literal["pyomo", "cvxpy"] = "pyomo",
                 time_indices: Optional[Union[range, List[int]]] = None):
        """
        Class for Optimizing the FLowSystem including:
            1. Aggregating TImeSeriesData via typical periods using tsam.
            2. Equalizing variables of typical periods.
        Parameters
        ----------
        name : str
            name of calculation
        aggregation_parameters : AggregationParameters
            Parameters for aggregation. See documentation of AggregationParameters class.
        components_to_clusterize: List[Component] or None
            List of Components to perform aggregation on. If None, then all components are aggregated.
            This means, teh variables in the components are equalized to each other, according to the typical periods
            computed in the DataAggregation
        flow_system : FlowSystem
            flow_system which should be calculated
        modeling_language : 'pyomo','cvxpy' (not implemeted yet)
            choose optimization modeling language
        time_indices : List[int] or None
            list with indices, which should be used for calculation. If None, then all timesteps are used.
        """
        super().__init__(name, flow_system, modeling_language, time_indices)
        self.aggregation_parameters = aggregation_parameters
        self.components_to_clusterize = components_to_clusterize
        self.time_series_for_aggregation = None
        self.aggregation = None
        self.time_series_collection: Optional[TimeSeriesCollection] = None

    def do_modeling(self) -> SystemModel:
        self.flow_system.transform_data()
        for time_series in self.flow_system.all_time_series:
            time_series.activate_indices(self.time_indices)

        from flixOpt.aggregation import Aggregation

        (chosenTimeSeries, chosenTimeSeriesWithEnd, dt_in_hours, dt_in_hours_total) = (
            self.flow_system.get_time_data_from_indices(self.time_indices))

        t_start_agg = timeit.default_timer()

        # Validation
        dt_min, dt_max = np.min(dt_in_hours), np.max(dt_in_hours)
        if not dt_min == dt_max:
            raise ValueError(f"Aggregation failed due to inconsistent time step sizes:"
                             f"delta_t varies from {dt_min} to {dt_max} hours.")
        steps_per_period = self.aggregation_parameters.hours_per_period / dt_in_hours[0]
        if not steps_per_period.is_integer():
            raise Exception(f"The selected {self.aggregation_parameters.hours_per_period=} does not match the time "
                            f"step size of {dt_in_hours[0]} hours). It must be a multiple of {dt_in_hours[0]} hours.")

        logger.info(f'{"":#^80}')
        logger.info(f'{" Aggregating TimeSeries Data ":#^80}')

        self.time_series_collection = TimeSeriesCollection([ts for ts in self.flow_system.all_time_series if ts.is_array])

        import pandas as pd
        original_data = pd.DataFrame(self.time_series_collection.data, index=chosenTimeSeries)

        # Aggregation - creation of aggregated timeseries:
        self.aggregation = Aggregation(original_data=original_data,
                                            hours_per_time_step=dt_min,
                                            hours_per_period=self.aggregation_parameters.hours_per_period,
                                            nr_of_periods=self.aggregation_parameters.nr_of_periods,
                                            weights=self.time_series_collection.weights,
                                            time_series_for_high_peaks=self.aggregation_parameters.labels_for_high_peaks,
                                            time_series_for_low_peaks=self.aggregation_parameters.labels_for_low_peaks)

        self.aggregation.cluster()
        self.aggregation.plot()
        self.time_series_collection.insert_data(  # Converting it into a dict with labels as keys
            {col: np.array(values) for col, values in self.aggregation.aggregated_data.to_dict(orient='list').items()})
        self.durations['aggregation'] = round(timeit.default_timer() - t_start_agg, 2)

        # Model the System
        t_start = timeit.default_timer()

        self.system_model = SystemModel(self.name, self.modeling_language, self.flow_system, self.time_indices)
        self.system_model.do_modeling()
        #Add Aggregation Model after modeling the rest
        aggregation_model = AggregationModel(self.aggregation_parameters, self.flow_system, self.aggregation,
                                             self.components_to_clusterize)
        self.system_model.other_models.append(aggregation_model)
        aggregation_model.do_modeling(self.system_model)

        self.system_model.to_math_model()

        self.durations['modeling'] = round(timeit.default_timer() - t_start, 2)
        return self.system_model

    def solve(self, solverProps: dict, path='results/', save_results=True):
        self._define_path_names(path, save_results, nr_of_system_models=1)
        self.system_model.solve(**solverProps, logfile_name=self._paths['log'][0])

        if save_results:
            self._save_solve_infos()


class SegmentedCalculation(Calculation):
    def __init__(self, name, flow_system: FlowSystem,
                 segment_length: int,
                 overlap_length: int,
                 modeling_language: Literal["pyomo", "cvxpy"] = "pyomo",
                 time_indices: Optional[Union[range, list[int]]] = None):
        """
        Dividing and Modeling the problem in (overlapping) segments.
        Storage values as result of segment n are overtaken
        to the next segment n+1 for timestep, which is first in segment n+1

        Afterwards timesteps of segments (without overlap)
        are put together to the full timeseries

        Because the result of segment n is used in segment n+1, modeling and
        solving is done in one step

        Take care:
        Parameters like invest_parameters, loadfactor etc. do not make sense in
        segmented modeling, because they are newly defined in each segment.
        This is not yet explicitly checked for...

        Parameters
        ----------
        name : str
            name of calculation
        flow_system : FlowSystem
            flow_system which should be calculated
        segment_length : int
            The number of time_steps per individual segment (without the overlap)
        overlap_length : int
            The number of time_steps that are added to each individual model. Used for better
            results of storages)
        modeling_language : 'pyomo', 'cvxpy' (not implemeted yet)
            choose optimization modeling language
        time_indices : List[int] or None
            list with indices, which should be used for calculation. If None, then all timesteps are used.

        """
        super().__init__(name, flow_system, modeling_language, time_indices)
        self.segment_length = segment_length
        self.overlap_length = overlap_length
        self._total_length = len(self.time_indices) if self.time_indices is not None else len(flow_system.time_series)
        self.number_of_segments = math.ceil(self._total_length / self.segment_length)
        self.sub_calculations: List[FullCalculation] = []

    def do_modeling_and_solve(self, solverProps: dict, path='results/'):
        logger.info(f'{"":#^80}')
        logger.info(f'{" Segmented Solving ":#^80}')

        assert self.segment_length_with_overlap <= self._total_length, \
            f'{self.segment_length_with_overlap=} cant be greater than the total length {self._total_length}'

        self._define_path_names(path, save_results=True, nr_of_system_models=self.number_of_segments)

        for i in range(self.number_of_segments):
            time_indices = self._get_indices(i)
            logger.info(f'{i}. Segment (flow_system indices {time_indices.start}...{time_indices.stop-1}):')
            calculation = FullCalculation(f'Segment_{i+1}', self.flow_system, self.modeling_language, time_indices)
            # TODO: Add Before Values if available
            self.sub_calculations.append(calculation)
            calculation.do_modeling()
            invest_elements = [model.element.label_full for model in calculation.system_model.sub_models
                               if isinstance(model, InvestmentModel)]
            if invest_elements:
                logger.critical(f'Investments are not supported in Segmented Calculation! '
                                f'Following elements Contain Investments: {invest_elements}')
            calculation.solve(solverProps, path=path)

        self.durations = {calculation.name: calculation.durations for calculation in self.sub_calculations}

    def results(self, individual_results: bool = False) -> dict:
        if individual_results:
            return {f'Segment_{i+1}': calculation.results() for i, calculation in enumerate(self.sub_calculations)}
        else:
            logger.warning('This is not yet implemented')

    def _get_indices(self, segment_index: int) -> range:
        start = segment_index * self.segment_length
        return range(start, min(start + self.segment_length + self.overlap_length, self._total_length))

    @property
    def segment_length_with_overlap(self):
        return self.segment_length + self.overlap_length
