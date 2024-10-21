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
from flixOpt.math_modeling import MathModel, Variable, Equation, VariableTS
from flixOpt.core import TimeSeries, Skalar, Numeric, Numeric_TS, as_effect_dict

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from flixOpt.flow_system import FlowSystem
    from flixOpt.elements import ComponentModel, BusModel


logger = logging.getLogger('flixOpt')


class SystemModel(MathModel):
    '''
    Hier kommen die ModellingLanguage-spezifischen Sachen rein
    '''

    def __init__(self,
                 label: str,
                 modeling_language: Literal['pyomo', 'cvxpy'],
                 flow_system: 'FlowSystem',
                 time_indices: Optional[Union[List[int], range]]):
        super().__init__(label, modeling_language)
        self.flow_system = flow_system
        # Zeitdaten generieren:
        self.time_series, self.time_series_with_end, self.dt_in_hours, self.dt_in_hours_total = (
            flow_system.get_time_data_from_indices(time_indices))
        self.nr_of_time_steps = len(self.time_series)
        self.indices = range(self.nr_of_time_steps)

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

        logger.info(f'{" finished solving ":#^80}')
        logger.info(f'{" Main Results ":#^80}')
        for effect_name, effect_results in self.main_results['Effects'].items():
            logger.info(f'{effect_name}:\n'
                        f'  {"operation":<15}: {effect_results["operation"]:>10.2f}\n'
                        f'  {"invest":<15}: {effect_results["invest"]:>10.2f}\n'
                        f'  {"sum":<15}: {effect_results["sum"]:>10.2f}')

        logger.info(
            # f'{"SUM":<15}: ...todo...\n'
            f'{"penalty":<17}: {self.main_results["penalty"]:>10.2f}\n'
            f'{"":-^80}\n'
            f'{"Objective":<17}: {self.main_results["Objective"]:>10.2f}\n'
            f'{"":-^80}')

        logger.info(f'Investment Decisions:')
        logger.info(utils.apply_formating(data_dict={**self.main_results["Invest-Decisions"]["invested"],
                                                     **self.main_results["Invest-Decisions"]["not invested"]},
                                          key_format="<30", indent=2, sort_by='value'))

        for bus in self.main_results['buses with excess']:
            logger.warning(f'Excess in Bus {bus}!')

        if self.main_results["penalty"] > 10:
            logger.warning(f'A total penalty of {self.main_results["penalty"]} occurred.'
                           f'This might distort the results')
        logger.info(f'{" End of Main Results ":#^80}')

    @property
    def infos(self):
        infos = super().infos
        #infos['str_Eqs'] = self.description_of_equations()
        #infos['str_Vars'] = self.description_of_variables()
        infos['main_results'] = self.main_results
        infos.update(self._infos)
        return infos

    @property
    def all_variables(self) -> Dict[str, Variable]:
        all_vars = {}
        for model in self.sub_models:
            for label, variable in model.variables.items():
                if label in all_vars:
                    raise KeyError(f"Duplicate Variable found in SystemModel:{model=} {label=}; {variable=}")
                all_vars[label] = variable
        return all_vars

    @property
    def all_equations(self) -> Dict[str, Equation]:
        all_eqs = {}
        for model in self.sub_models:
            for label, equation in model.eqs.items():
                if label in all_eqs:
                    raise KeyError(f"Duplicate Equation found in SystemModel: {label=}; {equation=}")
                all_eqs[label] = equation
        return all_eqs

    @property
    def all_inequations(self) -> Dict[str, Equation]:
        return {name: eq for name, eq in self.all_equations.items() if eq.eqType == 'ineq'}

    @property
    def sub_models(self) -> List['ElementModel']:
        direct_models = [self.effect_collection_model] + self.component_models + self.bus_models
        sub_models = [sub_model for direct_model in direct_models for sub_model in direct_model.all_sub_models]
        return direct_models + sub_models

    @property
    def variables(self) -> List[Variable]:
        """ Needed for Mother class """
        return list(self.all_variables.values())

    @property
    def eqs(self) -> List[Equation]:
        """ Needed for Mother class """
        return list(self.all_equations.values())

    @property
    def main_results(self) -> Dict[str, Union[Skalar, Dict]]:
        main_results = {}
        effect_results = {}
        main_results['Effects'] = effect_results
        for effect in self.flow_system.effect_collection.effects:
            effect_results[f'{effect.label} [{effect.unit}]'] = {
                'operation': float(effect.model.operation.sum.result),
                'invest': float(effect.model.invest.sum.result),
                'sum': float(effect.model.all.sum.result)}
        main_results['penalty'] = float(self.effect_collection_model.penalty.sum.result)
        main_results['Objective'] = self.objective_result
        if self.solver_name == 'highs':
            main_results['lower bound'] = self.solver_results.best_objective_bound
        else:
            main_results['lower bound'] = self.solver_results['Problem'][0]['Lower bound']
        buses_with_excess = []
        main_results['buses with excess'] = buses_with_excess
        for bus in self.flow_system.all_buses:
            if bus.with_excess:
                if np.sum(bus.model.excess_input.result) > 1e-3 or np.sum(bus.model.excess_output.result) > 1e-3:
                    buses_with_excess.append(bus.label)

        invest_decisions = {'invested': {}, 'not invested': {}}
        main_results['Invest-Decisions'] = invest_decisions
        from flixOpt.features import InvestmentModel
        for sub_model in self.sub_models:
            if isinstance(sub_model, InvestmentModel):
                invested_size = float(sub_model.size.result)  # bei np.floats Probleme bei Speichern
                if invested_size > 1e-3:
                    invest_decisions['invested'][sub_model.element.label_full] = invested_size
                else:
                    invest_decisions['not invested'][sub_model.element.label_full] = invested_size

        return main_results

    def results(self):
        return {'Components': {model.element.label: model.results() for model in self.component_models},
                'Effects': self.effect_collection_model.results(),
                'Objective': self.objective_result
                }


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

    def __init__(self, element: Element, label: Optional[str] = None):
        logger.debug(f'Created {self.__class__.__name__} for {element.label_full}')
        self.element = element
        self.variables = {}
        self.eqs = {}
        self.sub_models = []
        self._label = label

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

    def results(self):
        return {**{variable.label_short: variable.result for variable in self.variables.values()},
                **{model.label: model.results() for model in self.sub_models}}

    @property
    def label_full(self) -> str:
        return f'{self.element.label_full}_{self._label}' if self._label else self.element.label_full

    @property
    def label(self):
        return self._label or self.element.label


def _create_time_series(label: str, data: Optional[Numeric_TS], element: Element) -> TimeSeries:
    """Creates a TimeSeries from Numeric Data and adds it to the list of time_series of an Element"""
    time_series = TimeSeries(label=label, data=data)
    element.TS_list.append(time_series)
    return time_series


def create_equation(label: str, element_model: ElementModel, system_model: SystemModel,
                     eq_type: Literal['eq', 'ineq', 'objective'] = 'eq') -> Equation:
    """ Creates an Equation and adds it to the model of the Element """
    eq = Equation(f'{element_model.label_full}_{label}', label, system_model, eq_type)
    element_model.add_equations(eq)
    return eq


def create_variable(label: str,
                       element_model: ElementModel,
                       length: int,
                       system_model: SystemModel,
                       is_binary: bool = False,
                       value: Optional[Numeric] = None,
                       lower_bound: Optional[Numeric] = None,
                       upper_bound: Optional[Numeric] = None,
                       before_value: Optional[Numeric] = None,
                       ) -> VariableTS:
    """ Creates a VariableTS and adds it to the model of the Element """
    variable_label = f'{element_model.label_full}_{label}'
    if length > 1:
        var = VariableTS(variable_label, label, length, system_model,
                         is_binary, value, lower_bound, upper_bound, before_value)
        logger.debug(f'Created VariableTS "{variable_label}": [{length}]')
    else:
        var = Variable(variable_label, label, length, system_model,
                       is_binary, value, lower_bound, upper_bound)
        logger.debug(f'Created Variable "{variable_label}"')
    element_model.add_variables(var)
    return var
