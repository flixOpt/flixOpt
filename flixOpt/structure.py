# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:40:23 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

from typing import List, Tuple, Dict, Union, Optional, Literal, TYPE_CHECKING
import logging

import numpy as np

from flixOpt import utils
from flixOpt.math_modeling import MathModel, Variable, Equation
from flixOpt.core import TimeSeries, Skalar, Numeric, Numeric_TS, as_effect_dict

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from flixOpt.flow_system import FlowSystem
    from flixOpt.modeling import ComponentModel, BusModel


logger = logging.getLogger('flixOpt')


class SystemModel(MathModel):
    '''
    Hier kommen die ModellingLanguage-spezifischen Sachen rein
    '''

    def __init__(self,
                 label: str,
                 modeling_language: Literal['pyomo', 'cvxpy'],
                 flow_system: 'FlowSystem',
                 time_indices: Union[List[int], range]):
        super().__init__(label, modeling_language)
        self.flow_system = flow_system
        self.nr_of_time_steps = len(time_indices)
        self.time_indices = range(self.nr_of_time_steps)

        # Zeitdaten generieren:
        self.time_series, self.time_series_with_end, self.dt_in_hours, self.dt_in_hours_total = (
            flow_system.get_time_data_from_indices(time_indices))

        self.effect_collection_model = flow_system.effect_collection.create_model(self)
        self.component_models: List['ComponentModel'] = []
        self.bus_models: List['BusModel'] = []

    def do_modeling(self):
        self.effect_collection_model.do_modeling(self)
        self.component_models = [component.create_model() for component in self.flow_system.components]
        self.bus_models = [bus.create_model() for bus in self.flow_system.all_buses]
        for component_model in self.component_models:
            component_model.do_modeling(self)
        for bus_model in self.bus_models:  # Buses after Components, because FlowModels are created in ComponentModels
            bus_model.do_modeling(self)

    def solve(self,
              mip_gap: float = 0.02,
              time_limit_seconds: int = 3600,
              solver_name: str = 'highs',
              solver_output_to_console: bool = True,
              excess_threshold: Union[int, float] = 0.1,
              logfile_name: str = 'solver_log.log',
              **kwargs):
        """
        Parameters
        ----------
        mip_gap : TYPE, optional
            DESCRIPTION. The default is 0.02.
        time_limit_seconds : TYPE, optional
            DESCRIPTION. The default is 3600.
        solver_name : TYPE, optional
            DESCRIPTION. The default is 'highs'.
        solver_output_to_console : TYPE, optional
            DESCRIPTION. The default is True.
        excess_threshold : float, positive!
            threshold for excess: If sum(Excess)>excess_threshold a warning is raised, that an excess occurs
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        main_results_str : TYPE
            DESCRIPTION.
        """

        # check valid solver options:
        if len(kwargs) > 0:
            for key in kwargs.keys():
                if key not in ['threads']:
                    raise Exception(
                        'no allowed arguments for kwargs: ' + str(key) + '(all arguments:' + str(kwargs) + ')')

        logger.info(f'{" starting solving ":#^80}')
        logger.info(f'{self.describe()}')

        super().solve(mip_gap, time_limit_seconds, solver_name, solver_output_to_console, logfile_name, **kwargs)

        if solver_name == 'gurobi':
            termination_message = self.solver_results['Solver'][0]['Termination message']
        elif solver_name == 'glpk':
            termination_message = self.solver_results['Solver'][0]['Status']
        else:
            termination_message = f'not implemented for solver "{solver_name}" yet'
        logger.info(f'Termination message: "{termination_message}"')

        # Variablen-Ergebnisse abspeichern:
        # 1. dict:
        (self.results, self.results_var) = self.flow_system.get_results_after_solve()
        # 2. struct:
        self.results_struct = utils.createStructFromDictInDict(self.results)

        def extract_main_results() -> Dict[str, Union[Skalar, Dict]]:
            main_results = {}
            effect_results = {}
            main_results['Effects'] = effect_results
            for effect in self.flow_system.effect_collection.effects:
                effect_results[f'{effect.label} [{effect.unit}]'] = {
                    'operation': float(effect.operation.model.variables['sum'].result),
                    'invest': float(effect.invest.model.variables['sum'].result),
                    'sum': float(effect.all.model.variables['sum'].result)}
            main_results['penalty'] = float(self.flow_system.effect_collection.penalty.model.variables['sum'].result)
            main_results['Result of objective'] = self.objective_result
            if self.solver_name == 'highs':
                main_results['lower bound'] = self.solver_results.best_objective_bound
            else:
                main_results['lower bound'] = self.solver_results['Problem'][0]['Lower bound']
            busesWithExcess = []
            main_results['buses with excess'] = busesWithExcess
            for aBus in self.flow_system.all_buses:
                if aBus.with_excess:
                    if (
                            np.sum(self.results[aBus.label]['excess_input']) > excess_threshold or
                            np.sum(self.results[aBus.label]['excess_output']) > excess_threshold
                    ):
                        busesWithExcess.append(aBus.label)

            invest_decisions = {'invested': {}, 'not invested': {}}
            main_results['Invest-Decisions'] = invest_decisions
            for invest_feature in self.flow_system.all_investments:
                invested_size = invest_feature.model.variables[invest_feature.name_of_investment_size].result
                invested_size = float(invested_size)  # bei np.floats Probleme bei Speichern
                label = invest_feature.owner.label_full
                if invested_size > 1e-3:
                    invest_decisions['invested'][label] = invested_size
                else:
                    invest_decisions['not invested'][label] = invested_size

            return main_results

        self.main_results_str = extract_main_results()

        logger.info(f'{" finished solving ":#^80}')
        logger.info(f'{" Main Results ":#^80}')
        for effect_name, effect_results in self.main_results_str['Effects'].items():
            logger.info(f'{effect_name}:\n'
                        f'  {"operation":<15}: {effect_results["operation"]:>10.2f}\n'
                        f'  {"invest":<15}: {effect_results["invest"]:>10.2f}\n'
                        f'  {"sum":<15}: {effect_results["sum"]:>10.2f}')

        logger.info(
            # f'{"SUM":<15}: ...todo...\n'
            f'{"penalty":<17}: {self.main_results_str["penalty"]:>10.2f}\n'
            f'{"":-^80}\n'
            f'{"Objective":<17}: {self.main_results_str["Result of objective"]:>10.2f}\n'
            f'{"":-^80}')

        logger.info(f'Investment Decisions:')
        logger.info(utils.apply_formating(data_dict={**self.main_results_str["Invest-Decisions"]["invested"],
                                                     **self.main_results_str["Invest-Decisions"]["not invested"]},
                                          key_format="<30", indent=2, sort_by='value'))

        for bus in self.main_results_str['buses with excess']:
            logger.warning(f'Excess Value in Bus {bus}!')

        if self.main_results_str["penalty"] > 10:
            logger.warning(f'A total penalty of {self.main_results_str["penalty"]} occurred.'
                           f'This might distort the results')
        logger.info(f'{" End of Main Results ":#^80}')

    @property
    def infos(self):
        infos = super().infos
        infos['str_Eqs'] = self.flow_system.description_of_equations()
        infos['str_Vars'] = self.flow_system.description_of_variables()
        infos['main_results'] = self.main_results_str   # Hauptergebnisse:
        infos.update(self._infos)
        return infos

    @property
    def all_variables(self) -> Dict[str, Variable]:
        all_vars = {}
        for sub_model in self.sub_models:
            for key, value in sub_model.all_variables.items():
                if key in all_vars:
                    raise KeyError(f"Duplicate key found: '{key}' in both main model and submodel!")
                all_vars[key] = value
        return all_vars

    @property
    def all_equations(self) -> Dict[str, Equation]:
        all_eqs = {}
        for sub_model in self.sub_models:
            for key, value in sub_model.all_equations.items():
                if key in all_eqs:
                    raise KeyError(f"Duplicate key found: '{key}' in both main model and submodel!")
                all_eqs[key] = value
        return all_eqs

    @property
    def all_inequations(self) -> Dict[str, Equation]:
        return {name: eq for name, eq in self.all_equations.items() if eq.eqType == 'ineq'}

    @property
    def sub_models(self) -> List['ElementModel']:
        direct_models = [self.effect_collection_model] + self.component_models
        sub_models = [sub_model for direct_model in direct_models for sub_model in direct_model.all_sub_models]
        return direct_models + sub_models


class Element:
    """ Basic Element of flixOpt"""
    def __init__(self, label: str):
        self.label = label
        self.used_time_series: List[TimeSeries] = []  # Used for better access
        self.model: Optional[ElementModel] = None

    def _plausibility_checks(self) -> None:
        """ This function is used to do some basic plausibility checks for each Element during initialization """
        raise NotImplementedError(f'Every Element needs a _plausibility_checks() method')
    
    def transform_to_time_series(self) -> None:
        """ This function is used to transform the time series data from the User to proper TimeSeries Objects """
        raise NotImplementedError(f'Every Element needs a transform_to_time_series() method')

    def create_model(self) -> None:
        raise NotImplementedError(f'Every Element needs a create_model() method')

    def get_results(self) -> Tuple[Dict, Dict]:
        """Get results after the solve"""
        return self.model.results

    def __repr__(self):
        return f"<{self.__class__.__name__}> {self.label}"

    @property
    def label_full(self) -> str:
        return self.label


class ElementModel:
    """ Interface to create the mathematical Models for Elements """

    def __init__(self, element: Element):
        logger.debug(f'Created Model for {element.label_full}')
        self.element = element
        self.variables = {}
        self.eqs = {}
        self.sub_models = []

    def add_variables(self, *variables: Variable) -> None:
        for variable in variables:
            if variable.label not in self.variables.keys():
                self.variables[variable.label] = variable
            elif variable in self.variables.values():
                raise Exception(f'Variable "{variable.label}" already exists')
            else:
                raise Exception(f'A Variable with the label "{variable.label}" already exists')

    def add_equations(self, *equations: Equation) -> None:
        for equation in equations:
            if equation.label not in self.eqs.keys():
                self.eqs[equation.label] = equation
            else:
                raise Exception(f'Equation "{equation.label}" already exists')

    @property
    def description_of_variables(self) -> List[str]:
        if self.all_variables:
            return [var.description() for var in self.all_variables.values()]
        return []

    @property
    def description_of_equations(self) -> List[str]:
        if self.all_equations:
            return [eq.description() for eq in self.all_equations.values()]
        return []

    @property
    def overview_of_model_size(self) -> Dict[str, int]:
        all_vars, all_eqs, all_ineqs = self.all_variables, self.all_equations, self.all_inequations
        return {'no eqs': len(all_eqs),
                'no eqs single': sum(eq.nr_of_single_equations for eq in all_eqs.values()),
                'no inEqs': len(all_ineqs),
                'no inEqs single': sum(ineq.nr_of_single_equations for ineq in all_ineqs.values()),
                'no vars': len(all_vars),
                'no vars single': sum(var.length for var in all_vars.values())}

    @property
    def ineqs(self) -> Dict[str, Equation]:
        return {name: eq for name, eq in self.eqs.items() if eq.eqType == 'ineq'}

    @property
    def all_variables(self) -> Dict[str, Variable]:
        all_vars = self.variables.copy()
        for sub_model in self.sub_models:
            for key, value in sub_model.all_variables.items():
                if key in all_vars:
                    raise KeyError(f"Duplicate key found: '{key}' in both main model and submodel!")
                all_vars[key] = value
        return all_vars

    @property
    def all_equations(self) -> Dict[str, Equation]:
        all_eqs = self.eqs.copy()
        for sub_model in self.sub_models:
            for key, value in sub_model.all_equations.items():
                if key in all_eqs:
                    raise KeyError(f"Duplicate key found: '{key}' in both main model and submodel!")
                all_eqs[key] = value
        return all_eqs

    @property
    def all_inequations(self) -> Dict[str, Equation]:
        all_ineqs = self.ineqs.copy()
        for sub_model in self.sub_models:
            for key, value in sub_model.all_equations.items():
                if key in all_ineqs:
                    raise KeyError(f"Duplicate key found: '{key}' in both main model and submodel!")
                if value.eqType == 'ineq':
                    all_ineqs[key] = value
        return all_ineqs

    @property
    def all_sub_models(self) -> List['ElementModel']:
        all_subs = []
        to_process = self.sub_models.copy()
        for model in to_process:
            all_subs.append(model)
            to_process.extend(model.sub_models)
        return all_subs


def _create_time_series(label: str, data: Optional[Numeric_TS], element: Element) -> TimeSeries:
    """Creates a TimeSeries from Numeric Data and adds it to the list of time_series of an Element"""
    time_series = TimeSeries(label=label, data=data)
    element.TS_list.append(time_series)
    return time_series
