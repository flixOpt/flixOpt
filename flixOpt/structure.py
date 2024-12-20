"""
This module contains the core structure of the flixOpt framework.
These classes are not directly used by the end user, but are used by other modules.
"""

from typing import List, Dict, Union, Optional, Literal, TYPE_CHECKING, Any
import logging
import inspect
import textwrap

import numpy as np

from . import utils
from .math_modeling import MathModel, Variable, Equation, Inequation, VariableTS, Solver
from .core import TimeSeries, Skalar, Numeric, Numeric_TS, TimeSeriesData

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from .flow_system import FlowSystem
    from .elements import ComponentModel, BusModel

logger = logging.getLogger('flixOpt')


class SystemModel(MathModel):
    """
    Hier kommen die ModellingLanguage-spezifischen Sachen rein
    """

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
        self.other_models: List[ElementModel] = []

    def do_modeling(self):
        self.effect_collection_model.do_modeling(self)
        self.component_models = [component.create_model() for component in self.flow_system.components]
        self.bus_models = [bus.create_model() for bus in self.flow_system.all_buses]
        for component_model in self.component_models:
            component_model.do_modeling(self)
        for bus_model in self.bus_models:  # Buses after Components, because FlowModels are created in ComponentModels
            bus_model.do_modeling(self)

    def solve(self, solver: Solver, excess_threshold: Union[int, float] = 0.1):
        """
        Parameters
        ----------
        solver : Solver
            An Instance of the class Solver. Choose from flixOpt.solvers
        excess_threshold : float, positive!
            threshold for excess: If sum(Excess)>excess_threshold a warning is raised, that an excess occurs
        """

        logger.info(f'{" starting solving ":#^80}')
        logger.info(f'{self.describe_size()}')

        super().solve(solver)

        logger.info(f'Termination message: "{self.solver.termination_message}"')

        logger.info(f'{" finished solving ":#^80}')
        logger.info(f'{" Main Results ":#^80}')
        for effect_name, effect_results in self.main_results['Effects'].items():
            logger.info(f'{effect_name}:\n'
                        f'  {"operation":<15}: {effect_results["operation"]:>10.2f}\n'
                        f'  {"invest":<15}: {effect_results["invest"]:>10.2f}\n'
                        f'  {"sum":<15}: {effect_results["sum"]:>10.2f}')

        logger.info(
            # f'{"SUM":<15}: ...todo...\n'
            f'{"Penalty":<17}: {self.main_results["penalty"]:>10.2f}\n'
            f'{"":-^80}\n'
            f'{"Objective":<17}: {self.main_results["Objective"]:>10.2f}\n'
            f'{"":-^80}')

        logger.info(f'Investment Decisions:')
        logger.info(utils.apply_formating(data_dict={**self.main_results["Invest-Decisions"]["invested"],
                                                     **self.main_results["Invest-Decisions"]["not invested"]},
                                          key_format="<30", indent=2, sort_by='value'))

        for bus in self.main_results['buses with excess']:
            logger.warning(f'A penalty occurred in Bus "{bus}"!')

        if self.main_results["penalty"] > 10:
            logger.warning(f'A total penalty of {self.main_results["penalty"]} occurred.'
                           f'This might distort the results')
        logger.info(f'{" End of Main Results ":#^80}')

    def description_of_variables(self, structured: bool = True) -> Dict[str, Union[str, List[str]]]:
        return {'Components': {comp.label: comp.model.description_of_variables(structured)
                               for comp in self.flow_system.components},
                'Buses': {bus.label: bus.model.description_of_variables(structured)
                          for bus in self.flow_system.all_buses},
                'Effects': self.flow_system.effect_collection.model.description_of_variables(structured),
                'Others': {model.element.label: model.description_of_variables(structured)
                           for model in self.other_models}}

    def description_of_constraints(self, structured: bool = True) -> Dict[str, Union[str, List[str]]]:
        return {'Components': {comp.label: comp.model.description_of_constraints(structured)
                               for comp in self.flow_system.components},
                'Buses': {bus.label: bus.model.description_of_constraints(structured)
                          for bus in self.flow_system.all_buses},
                'Objective': self.objective.description(),
                'Effects': self.flow_system.effect_collection.model.description_of_constraints(structured),
                'Others': {model.element.label: model.description_of_constraints(structured)
                           for model in self.other_models}}

    def results(self):
        return {'Components': {model.element.label: model.results() for model in self.component_models},
                'Effects': self.effect_collection_model.results(),
                'Buses': {model.element.label: model.results() for model in self.bus_models},
                'Others': {model.element.label: model.results() for model in self.other_models},
                'Objective': self.result_of_objective,
                'Time': self.time_series_with_end,
                'Time intervals in hours': self.dt_in_hours
                }

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
        main_results['Objective'] = self.result_of_objective
        main_results['lower bound'] = self.solver.best_bound
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

    @property
    def infos(self) -> Dict:
        infos = super().infos
        infos['Constraints'] = self.description_of_constraints()
        infos['Variables'] = self.description_of_variables()
        infos['Main Results'] = self.main_results
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
    def all_constraints(self) -> Dict[str, Union[Equation, Inequation]]:
        all_constr = {}
        for model in self.sub_models:
            for label, constr in model.constraints.items():
                if label in all_constr:
                    raise KeyError(f"Duplicate Constraint found in SystemModel: {label=}; {constr=}")
                else:
                    all_constr[label] = constr
        return all_constr

    @property
    def all_equations(self) -> Dict[str, Equation]:
        return {key: value for key, value in self.all_constraints.items() if isinstance(value, Equation)}

    @property
    def all_inequations(self) -> Dict[str, Inequation]:
        return {key: value for key, value in self.all_constraints.items() if isinstance(value, Inequation)}

    @property
    def sub_models(self) -> List['ElementModel']:
        direct_models = [self.effect_collection_model] + self.component_models + self.bus_models + self.other_models
        sub_models = [sub_model for direct_model in direct_models for sub_model in direct_model.all_sub_models]
        return direct_models + sub_models

    @property
    def variables(self) -> List[Variable]:
        """ Needed for Mother class """
        return list(self.all_variables.values())

    @property
    def equations(self) -> List[Equation]:
        """ Needed for Mother class """
        return list(self.all_equations.values())

    @property
    def inequations(self) -> List[Inequation]:
        """ Needed for Mother class """
        return list(self.all_inequations.values())

    @property
    def objective(self) -> Equation:
        return self.effect_collection_model.objective


class Element:
    """ Basic Element of flixOpt"""

    def __init__(self, label: str, meta_data: Optional[Dict] = None):
        """
        Parameters
        ----------
        label : str
            label of the element
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        """
        if not utils.label_is_valid(label):
            logger.critical(f"'{label}' cannot be used as a label. Leading or Trailing '_' and '__' are reserved. "
                            f"Use any other symbol instead")
        self.label = label
        self.meta_data = meta_data
        self.used_time_series: List[TimeSeries] = []  # Used for better access
        self.model: Optional[ElementModel] = None

    def _plausibility_checks(self) -> None:
        """ This function is used to do some basic plausibility checks for each Element during initialization """
        raise NotImplementedError(f'Every Element needs a _plausibility_checks() method')

    def transform_data(self) -> None:
        """ This function is used to transform the time series data from the User to proper TimeSeries Objects """
        raise NotImplementedError(f'Every Element needs a transform_data() method')

    def create_model(self) -> None:
        raise NotImplementedError(f'Every Element needs a create_model() method')

    def __repr__(self):
        # Get the constructor arguments and their current values
        init_signature = inspect.signature(self.__init__)
        init_args = init_signature.parameters

        # Create a dictionary with argument names and their values
        args_str = ', '.join(
            f"{name}={repr(getattr(self, name, None))}"
            for name in init_args if name != 'self'
        )
        return f"{self.__class__.__name__}({args_str})"

    def __str__(self):
        return get_object_infos_as_str(self)

    def infos(self) -> Dict:
        return get_object_infos_as_dict(self)

    @property
    def label_full(self) -> str:
        return self.label


class ElementModel:
    """ Interface to create the mathematical Models for Elements """

    def __init__(self, element: Element, label: Optional[str] = None):
        logger.debug(f'Created {self.__class__.__name__} for {element.label_full}')
        self.element = element
        self.variables = {}
        self.constraints = {}
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

    def add_constraints(self, *constraints: Union[Equation, Inequation]) -> None:
        for constraint in constraints:
            if constraint.label not in self.constraints.keys():
                self.constraints[constraint.label] = constraint
            else:
                raise Exception(f'Constraint "{constraint.label}" already exists')

    def description_of_variables(self, structured: bool = True) -> Union[Dict[str, Union[List[str], Dict]], List[str]]:
        if structured:
            # Gather descriptions of this model's variables
            descriptions = {'_self': [var.description() for var in self.variables.values()]}

            # Recursively gather descriptions from sub-models
            for sub_model in self.sub_models:
                descriptions[sub_model.label] = sub_model.description_of_variables(structured=structured)

            return descriptions
        else:
            return [var.description() for var in self.all_variables.values()]

    def description_of_constraints(self, structured: bool = True) -> Union[Dict[str, str], List[str]]:
        if structured:
            # Gather descriptions of this model's variables
            descriptions = {'_self': [constr.description() for constr in self.constraints.values()]}

            # Recursively gather descriptions from sub-models
            for sub_model in self.sub_models:
                descriptions[sub_model.label] = sub_model.description_of_constraints(structured=structured)

            return descriptions
        else:
            return [eq.description() for eq in self.all_equations.values()]

    @property
    def overview_of_model_size(self) -> Dict[str, int]:
        all_vars, all_eqs, all_ineqs = self.all_variables, self.all_equations, self.all_inequations
        return {'no of Euations': len(all_eqs),
                'no of Equations single': sum(eq.nr_of_single_equations for eq in all_eqs.values()),
                'no of Inequations': len(all_ineqs),
                'no of Inequations single': sum(ineq.nr_of_single_equations for ineq in all_ineqs.values()),
                'no of Variables': len(all_vars),
                'no of Variables single': sum(var.length for var in all_vars.values())}

    @property
    def inequations(self) -> Dict[str, Inequation]:
        return {name: ineq for name, ineq in self.constraints.items() if isinstance(ineq, Inequation)}

    @property
    def equations(self) -> Dict[str, Equation]:
        return {name: eq for name, eq in self.constraints.items() if isinstance(eq, Equation)}

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
    def all_constraints(self) -> Dict[str, Union[Equation, Inequation]]:
        all_constr = self.constraints.copy()
        for sub_model in self.sub_models:
            for key, value in sub_model.all_constraints.items():
                if key in all_constr:
                    raise KeyError(f"Duplicate key found: '{key}' in both main model and submodel!")
                all_constr[key] = value
        return all_constr

    @property
    def all_equations(self) -> Dict[str, Equation]:
        all_eqs = self.equations.copy()
        for sub_model in self.sub_models:
            for key, value in sub_model.all_equations.items():
                if key in all_eqs:
                    raise KeyError(f"Duplicate key found: '{key}' in both main model and submodel!")
                all_eqs[key] = value
        return all_eqs

    @property
    def all_inequations(self) -> Dict[str, Inequation]:
        all_ineqs = self.inequations.copy()
        for sub_model in self.sub_models:
            for key, value in sub_model.all_inequations.items():
                if key in all_ineqs:
                    raise KeyError(f"Duplicate key found: '{key}' in both main model and submodel!")
        return all_ineqs

    @property
    def all_sub_models(self) -> List['ElementModel']:
        all_subs = []
        to_process = self.sub_models.copy()
        for model in to_process:
            all_subs.append(model)
            to_process.extend(model.sub_models)
        return all_subs

    def results(self) -> Dict:
        return {**{variable.label_short: variable.result for variable in self.variables.values()},
                **{model.label: model.results() for model in self.sub_models}}

    @property
    def label_full(self) -> str:
        return f'{self.element.label_full}__{self._label}' if self._label else self.element.label_full

    @property
    def label(self):
        return self._label or self.element.label


def _create_time_series(label: str, data: Optional[Union[Numeric_TS, TimeSeries]], element: Element) -> Optional[TimeSeries]:
    """Creates a TimeSeries from Numeric Data and adds it to the list of time_series of an Element.
    If the data already is a TimeSeries, nothing happens and the TimeSeries gets cleaned and returned"""
    if data is None:
        return None
    elif isinstance(data, TimeSeries):
        data.clear_indices_and_aggregated_data()
        return data
    else:
        time_series = TimeSeries(label=f'{element.label_full}__{label}', data=data)
        element.used_time_series.append(time_series)
        return time_series


def create_equation(label: str,
                    element_model: ElementModel,
                    eq_type: Literal['eq', 'ineq'] = 'eq') -> Union[Equation, Inequation]:
    """ Creates an Equation and adds it to the model of the Element """
    if eq_type == 'eq':
        constr = Equation(f'{element_model.label_full}_{label}', label)
    elif eq_type == 'ineq':
        constr = Inequation(f'{element_model.label_full}_{label}', label)
    element_model.add_constraints(constr)
    return constr


def create_variable(label: str,
                    element_model: ElementModel,
                    length: int, is_binary: bool = False,
                    fixed_value: Optional[Numeric] = None,
                    lower_bound: Optional[Numeric] = None,
                    upper_bound: Optional[Numeric] = None,
                    previous_values: Optional[Numeric] = None,
                    avoid_use_of_variable_ts: bool = False) -> VariableTS:
    """ Creates a VariableTS and adds it to the model of the Element """
    variable_label = f'{element_model.label_full}_{label}'
    if length > 1 and not avoid_use_of_variable_ts:
        var = VariableTS(variable_label, length, label,
                         is_binary, fixed_value, lower_bound, upper_bound, previous_values)
        logger.debug(f'Created VariableTS "{variable_label}": [{length}]')
    else:
        var = Variable(variable_label, length, label,
                       is_binary, fixed_value, lower_bound, upper_bound)
        logger.debug(f'Created Variable "{variable_label}": [{length}]')
    element_model.add_variables(var)
    return var


def get_object_infos_as_str(obj) -> str:
    """
    Returns a string representation of an object's constructor arguments,
    excluding default values. Formats dictionaries with nested child class objects, displaying their labels.

    Args:
        obj: The object whose constructor arguments will be formatted and returned as a string.

    Returns:
        str: A string representation of the object's constructor arguments,
             with properly formatted dictionaries, and nested objects' labels.
    """

    def numeric_as_str(item: Union[Numeric, TimeSeries, TimeSeriesData], max_length: int = 100) -> str:
        """
        Returns a short oneline string of numeric data.
        If something else than the expected datatypes is passes, it returns the item aas a str
        """
        if isinstance(item, TimeSeries):
            item = item.active_data
        elif isinstance(item, TimeSeriesData):
            item = item.data

        if isinstance(item, np.ndarray):
            text = str(item).replace('\n', '')
            if len(text) > max_length:
                return text[:max_length-3]+'...'
            else:
                return text
        elif isinstance(item, (Skalar, np.integer, np.floating)):
            return str(item)
        else:
            raise TypeError(f' Wrong type passed to function numeric_as_str(): {type(item)}')

    def dict_as_str(d: Dict, current_indent_level: int = 1, indent_depth: int = 3) -> str:
        """
        Recursively formats a dictionary, skipping {None: some_value} dictionaries by returning only the value.

        Args:
            d (dict): The dictionary to format.
            current_indent_level (int): The current indentation level (default is 1).
            indent_depth (int): The number of spaces per indent (default is 3).

        Returns:
            str: A string representation of the dictionary with the keys' labels and appropriate indentation.
        """
        # If the dictionary has a single {None: value} entry, return the value directly
        if len(d) == 1 and None in d:
            return str(d[None])

        formatted_items = []
        for k, v in d.items():
            key_str = k.label if hasattr(k, 'label') else str(k)  # Use labels instead of __str__ if availlable
            if isinstance(v, dict):
                v_str = dict_as_str(v, current_indent_level + 1)  # Recursively format nested dictionaries
            else:
                v_str = item_as_str(v)
            formatted_items.append(f"{key_str}: {v_str}")
        return '{\n' + textwrap.indent(",\n".join(formatted_items), ' ' * current_indent_level * indent_depth) + '}'

    def item_as_str(item: Any, indent_level: int = 1, indent_depth: int = 3) -> str:
        """ Convert Any item to a string"""
        if isinstance(item, dict):
            return dict_as_str(item, indent_level, indent_depth)  # Recursively format nested dictionaries
        elif isinstance(item, (Numeric, TimeSeries, TimeSeriesData)):
            return numeric_as_str(item)  # Special formatting for numerical data
        else:
            return str(item)  # All others as str

    # Get the constructor arguments and their default values
    init_params = sorted(inspect.signature(obj.__init__).parameters.items(),
                         key=lambda x: (x[0].lower() != 'label', x[0].lower()))

    # Build a list of attribute=value pairs, excluding defaults
    details = []
    for name, param in init_params:
        if name == 'self':
            continue
        value, default = getattr(obj, name, None), param.default

        if isinstance(value, (dict, list)) and not value:  # Ignore empty dicts and lists
            pass
        elif np.all(value == default):  # Ignore default values
            pass
        else:
            details.append(f"{name}={item_as_str(value)}")  # Formatted numeric-string

    # Join all relevant parts and format them in the output
    full_str = ',\n'.join(details)
    return f"{obj.__class__.__name__}(\n{textwrap.indent(full_str, ' '*3)})"


def get_object_infos_as_dict(obj) -> Dict[str, Optional[Union[int, float, str, bool, list, dict]]]:
    """
    Returns a dictionary representation of an object's constructor arguments,
    excluding default values. Formats dictionaries with nested child class objects, displaying their labels.
    Converts data to python native objects. Further, converts Tuples to Lists.

    Args:
        obj: The object whose constructor arguments will be formatted and returned as a dictionary.

    Returns:
        dict: A dictionary representation of the object's constructor arguments,
              with properly formatted dictionaries and nested objects' labels.
    """

    def format_dict(d: Dict[Any, Any]) -> Dict[str, Optional[Union[int, float, str, bool, Dict, List]]]:
        """ Recursively converts datatypes in a dictionary. Converts keys to string """
        return {k.label if hasattr(k, 'label') else str(k): format_item(v) for k, v in d.items()}

    def format_item(item: Any) -> Optional[Union[int, float, str, bool, List, Dict]]:
        """ Convert any item to a format that is usable by json and yaml"""
        if isinstance(item, (int, float, str, bool, type(None), np.integer, np.floating, np.ndarray)):
            return utils.convert_to_native_types(item)
        elif isinstance(item, (tuple, list)):
            return [format_item(i) for i in item]
        elif isinstance(item, dict):  # Format dictionaries with custom formatting
            if len(item) == 1 and None in item:
                return format_item(item[None])
            else:
                return format_dict(item)
        elif isinstance(item, TimeSeries):
            return format_item(item.active_data)
        elif isinstance(item, TimeSeriesData):
            return format_item(item.data)
        elif hasattr(item, 'infos') and callable(getattr(item, 'infos')):
            return item.infos()
        else:
            raise TypeError(f'Unexpected type passed to function format_item(). {type(item)=}; {item=}')

    init_params = sorted(inspect.signature(obj.__init__).parameters.items(),
                         key=lambda x: (x[0].lower() != 'label', x[0].lower()))

    # Build a dict of attribute=value pairs, excluding defaults
    details = {'class': ':'.join([cls.__name__ for cls in obj.__class__.__mro__])}
    for name, param in init_params:
        if name == 'self':
            continue
        value, default = getattr(obj, name, None), param.default
        if isinstance(value, (dict, list)) and not value:  # Ignore empty dicts and lists
            pass
        elif np.all(value == default):  # Ignore default values
            pass
        else:
            details[name] = format_item(value)

    return details
