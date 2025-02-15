"""
This module contains the core structure of the flixOpt framework.
These classes are not directly used by the end user, but are used by other modules.
"""

import inspect
import json
import logging
import pathlib
from datetime import datetime
from io import StringIO
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union, Tuple

import numpy as np
from rich.console import Console
from rich.pretty import Pretty
import xarray as xr
import linopy
import pandas as pd

from . import utils
from .config import CONFIG
from .core import Numeric, Numeric_TS, Skalar, TimeSeries, TimeSeriesData
from .math_modeling import Equation, Inequation, MathModel, Solver, Variable, VariableTS

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from .elements import BusModel, ComponentModel
    from .flow_system import FlowSystem
    from .effects import EffectCollection

logger = logging.getLogger('flixOpt')


class SystemModel(linopy.Model):

    def __init__(self, flow_system: 'FlowSystem'):
        super().__init__(force_dim_names=True)
        self.flow_system = flow_system
        self.effects: Optional[EffectCollection] = None

    def do_modeling(self):
        from .effects import EffectCollection
        self.effects = EffectCollection(self, list(self.flow_system.effects.values()))
        self.effects.do_modeling(self)
        component_models = [component.create_model(self) for component in self.flow_system.components.values()]
        bus_models = [bus.create_model(self) for bus in self.flow_system.buses.values()]
        for component_model in component_models:
            component_model.do_modeling(self)
        for bus_model in bus_models:  # Buses after Components, because FlowModels are created in ComponentModels
            bus_model.do_modeling(self)

        self.add_objective(
            self.effects.objective_effect.model.total + self.effects.penalty.total
        )

    @property
    def main_results(self) -> Dict[str, Union[Skalar, Dict]]:
        main_results = {}
        effect_results = {}
        main_results['Effects'] = effect_results
        for effect in self.flow_system.effects.values():
            effect_results[f'{effect.label} [{effect.unit}]'] = {
                'operation': float(effect.model.operation.total.solution.values),
                'invest': float(effect.model.invest.total.solution.values),
                'total': float(effect.model.total.solution.values),
            }
        main_results['penalty'] = float(self.effects.penalty.total.solution.values)
        main_results['Objective'] = self.objective.value
        main_results['lower bound'] = 'Not available'
        buses_with_excess = []
        main_results['buses with excess'] = buses_with_excess
        for bus in self.flow_system.buses.values():
            if bus.with_excess:
                excess_in = np.sum(bus.model.excess_input.solution.values)
                excess_out = np.sum(bus.model.excess_output.solution.values)
                if excess_in > 1e-3 or excess_out > 1e-3:
                    buses_with_excess.append({bus.label_full: {'input': excess_in, 'output': excess_out}})

        invest_decisions = {'invested': {}, 'not invested': {}}
        main_results['Invest-Decisions'] = invest_decisions
        from flixOpt.features import InvestmentModel

        for component in self.flow_system.components.values():
            for model in component.model.all_sub_models:
                if isinstance(model, InvestmentModel):
                    invested_size = float(model.size.result)  # bei np.floats Probleme bei Speichern
                    if invested_size >= CONFIG.modeling.EPSILON:
                        invest_decisions['invested'][model.element.label_full] = invested_size
                    else:
                        invest_decisions['not invested'][model.element.label_full] = invested_size

        return main_results

    @property
    def infos(self) -> Dict:
        return {'Constraints': self.constraints.ncons,
                'Variables': self.variables.nvars,
                'Main Results': self.main_results,
                'Config': CONFIG.to_dict()}

    @property
    def hours_per_step(self):
        return self.flow_system.hours_per_step

    @property
    def hours_of_previous_timesteps(self):
        return self.flow_system.hours_of_previous_timesteps

    @property
    def coords(self):
        return self.flow_system.coords


class Interface:
    """
    This class is used to collect arguments about a Model.
    """

    def transform_data(self, flow_system: 'FlowSystem'):
        """ Transforms the data of the interface to match the FlowSystem's dimensions"""
        raise NotImplementedError('Every Interface needs a transform_data() method')

    def infos(self, use_numpy=True, use_element_label=False) -> Dict:
        """
        Generate a dictionary representation of the object's constructor arguments.
        Excludes default values and empty dictionaries and lists.
        Converts data to be compatible with JSON.

        Parameters:
        -----------
        use_numpy bool:
            Whether to convert NumPy arrays to lists. Defaults to True.
            If True, numeric numpy arrays (`np.ndarray`) are preserved as-is.
            If False, they are converted to lists.
        use_element_label bool:
            Whether to use the element label instead of the infos of the element. Defaults to False.
            Note that Elements used as keys in dictionaries are always converted to their labels.

        Returns:
            Dict: A dictionary representation of the object's constructor arguments.

        """
        # Get the constructor arguments and their default values
        init_params = sorted(
            inspect.signature(self.__init__).parameters.items(),
            key=lambda x: (x[0].lower() != 'label', x[0].lower()),  # Prioritize 'label'
        )
        # Build a dict of attribute=value pairs, excluding defaults
        details = {'class': ':'.join([cls.__name__ for cls in self.__class__.__mro__])}
        for name, param in init_params:
            if name == 'self':
                continue
            value, default = getattr(self, name, None), param.default
            # Ignore default values and empty dicts and list
            if np.all(value == default) or (isinstance(value, (dict, list)) and not value):
                continue
            details[name] = copy_and_convert_datatypes(value, use_numpy, use_element_label)
        return details

    def to_json(self, path: Union[str, pathlib.Path]):
        """
        Saves the element to a json file.
        This not meant to be reloaded and recreate the object, but rather used to document or compare the object.

        Parameters:
        -----------
        path : Union[str, pathlib.Path]
            The path to the json file.
        """
        data = get_compact_representation(self.infos(use_numpy=True, use_element_label=True))
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def __repr__(self):
        # Get the constructor arguments and their current values
        init_signature = inspect.signature(self.__init__)
        init_args = init_signature.parameters

        # Create a dictionary with argument names and their values
        args_str = ', '.join(f'{name}={repr(getattr(self, name, None))}' for name in init_args if name != 'self')
        return f'{self.__class__.__name__}({args_str})'

    def __str__(self):
        return get_str_representation(self.infos(use_numpy=True, use_element_label=True))

    @staticmethod
    def _create_time_series(
        element: 'Element',
        name: str,
        data: Optional[Union[Numeric_TS, TimeSeries]],
        timesteps: pd.DatetimeIndex,
        periods: Optional[pd.Index],
    ) -> Optional[TimeSeries]:
        """Creates a TimeSeries from Numeric Data and adds it to the list of time_series of an Element.
        If the data already is a TimeSeries, nothing happens and the TimeSeries gets reset and returned"""

        if data is None:
            return None
        elif isinstance(data, TimeSeries):
            data.restore_data()
            return data

        time_series = TimeSeries.from_datasource(
            name=f'{element.label_full}__{name}',
            data=data.data if isinstance(data, TimeSeriesData) else data,
            timesteps=timesteps,
            periods=periods,
            aggregation_weight=data.agg_weight if isinstance(data, TimeSeriesData) else None,
        )
        element.used_time_series.append(time_series)
        if isinstance(data, TimeSeriesData):
            data.label = time_series.name  # Connecting User_time_series to TimeSeries
        return time_series


class Element(Interface):
    """Basic Element of flixOpt"""

    def __init__(self, label: str, meta_data: Dict = None):
        """
        Parameters
        ----------
        label : str
            label of the element
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        """
        if not utils.label_is_valid(label):
            logger.critical(
                f"'{label}' cannot be used as a label. Leading or Trailing '_' and '__' are reserved. "
                f'Use any other symbol instead'
            )
        self.label = label
        self.meta_data = meta_data if meta_data is not None else {}
        self.used_time_series: List[TimeSeries] = []  # Used for better access
        self.model: Optional[ElementModel] = None

    def solution_numeric(
        self,
        use_numpy: bool = True,
        all_variables: bool = True
    ) -> Union[Dict[str, np.ndarray], Dict[str, Union[List, int, float]]]:
        vars = self.model.all_variables if all_variables else self.model.variables
        results = {var: vars.solution[var].values for var in vars.solution.data_vars}
        if use_numpy:
            return {k: v.item() if v.ndim == 0 else v for k, v in results.items()}
        return {k: v.tolist() for k, v in results.items()}

    def _plausibility_checks(self) -> None:
        """This function is used to do some basic plausibility checks for each Element during initialization"""
        raise NotImplementedError('Every Element needs a _plausibility_checks() method')

    def create_model(self, model: SystemModel) -> 'ElementModel':
        raise NotImplementedError('Every Element needs a create_model() method')

    @property
    def label_full(self) -> str:
        return self.label

    def _create_time_series(
        self,
        name: str,
        data: Optional[Union[Numeric_TS, TimeSeries]],
        timesteps: pd.DatetimeIndex,
        periods: Optional[pd.Index],
    ) -> Optional[TimeSeries]:
        return super()._create_time_series(self, name, data, timesteps, periods)


class Model:
    """Stores Variables and Constraints"""

    def __init__(self, model: SystemModel, label_of_parent: str, label: str, label_full: Optional[str] = None):
        """
        Parameters
        ----------
        interface : Interface
            The interface this model is created for.
        label_of_parent : str
            The label of the parent (Element). Used to construct the full label of the model.
        label : str
            Used to construct the label of the model. If None, the interface label is used.
        label_full : str
            The full label of the model. If None, the full label is constructed using the other given labels.
        """

        self._model = model
        self._label = label
        self._label_of_parent = label_of_parent
        self._label_full = label_full

        self._variables: List[str] = []
        self._constraints: List[str] = []
        self.sub_models: List[Model] = []

        self._variables_short: Dict[str, str] = {}
        self._constraints_short: Dict[str, str] = {}
        self._sub_models_short: Dict[str, str] = {}
        logger.debug(f'Created {self.__class__.__name__}  "{self._label}"')

    def add(
        self,
        item: Union[linopy.Variable, linopy.Constraint, 'Model'],
        short_name: Optional[str] = None
    ) -> Union[linopy.Variable, linopy.Constraint, 'Model']:
        """
        Add a variable, constraint or sub-model to the model

        Parameters
        ----------
        item : linopy.Variable, linopy.Constraint, InterfaceModel
            The variable, constraint or sub-model to add to the model
        short_name : str, optional
            The short name of the variable, constraint or sub-model. If not provided, the full name is used.
        """
        # TODO: Check uniquenes of short names
        if isinstance(item, linopy.Variable):
            self._variables.append(item.name)
            self._variables_short[item.name] = short_name or item.name
        elif isinstance(item, linopy.Constraint):
            self._constraints.append(item.name)
            self._constraints_short[item.name] = short_name or item.name
        elif isinstance(item, Model):
            self.sub_models.append(item)
            self._constraints_short[item.label_full] = short_name or item.label_full
        else:
            raise ValueError(
                f'Item must be a linopy.Variable, linopy.Constraint or flixOpt.structure.Model, got {type(item)}')
        return item

    def filter_variables(self,
                         filter_by: Optional[Literal['binary', 'continuous', 'integer']] = None,
                         length: Literal['scalar', 'time'] = None):
        if filter_by is None:
            all_variables = self.variables
        elif filter_by == 'binary':
            all_variables = self.variables.binaries
        elif filter_by == 'integer':
            all_variables = self.variables.integers
        elif filter_by == 'continuous':
            all_variables = self.variables.continuous
        else:
            raise ValueError(f'Invalid filter_by "{filter_by}", must be one of "binary", "continous", "integer"')
        if length is None:
            return all_variables
        elif length == 'scalar':
            return all_variables[[name for name in all_variables if all_variables[name].ndim == 0]]
        elif length == 'time':
            return all_variables[[name for name in all_variables if 'time' in all_variables[name].dims]]
        raise ValueError(f'Invalid length "{length}", must be one of "scalar", "time" or None')

    def solution_structured(
        self,
        use_numpy: bool = True,
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        results = {
            self._variables_short[var_name]: var.values
            for var_name, var in self.variables.solution.data_vars.items()
        }
        return {
            **results,
            **{sub_model.label: sub_model.solution_structured(use_numpy) for sub_model in self.sub_models}
        }

    @property
    def label(self) -> str:
        return self._label

    @property
    def label_full(self) -> str:
        return self._label_full or f'{self._label_of_parent}__{self.label}'

    @property
    def variables(self) -> linopy.Variables:
        return self._model.variables[self._variables]

    @property
    def constraints(self) -> linopy.Constraints:
        return self._model.constraints[self._constraints]

    @property
    def _all_variables(self) -> List[str]:
        all_variables = self._variables.copy()
        for sub_model in self.sub_models:
            for variable in sub_model._all_variables:
                if variable in all_variables:
                    raise KeyError(
                        f"Duplicate key found: '{variable}' in both {self.label_full} and {sub_model.label_full}!"
                    )
                all_variables.append(variable)
        return all_variables

    @property
    def _all_constraints(self) -> List[str]:
        all_constraints = self._constraints.copy()
        for sub_model in self.sub_models:
            for constraint in sub_model._all_constraints:
                if constraint in all_constraints:
                    raise KeyError(f"Duplicate key found: '{constraint}' in both main model and submodel!")
                all_constraints.append(constraint)
        return all_constraints

    @property
    def all_variables(self) -> linopy.Variables:
        return self._model.variables[self._all_variables]

    @property
    def all_constraints(self) -> linopy.Constraints:
        return self._model.constraints[self._all_constraints]

    @property
    def all_sub_models(self) -> List['Model']:
        return [model for sub_model in self.sub_models for model in [sub_model] + sub_model.all_sub_models]


class InterfaceModel(Model):
    """Stores the mathematical Variables and Constraints related to an Interface"""

    def __init__(self, model: SystemModel, interface: Optional[Interface] = None, label_of_parent: Optional[str] = None, label: Optional[str] = None):
        """
        Parameters
        ----------
        interface : Interface
            The interface this model is created for.
        label_of_parent : str
            The label of the parent. Used to construct the full label of the model.
        label : str
            Used to construct the label of the model. If None, the interface label is used.
        """
        if label_of_parent is None and label is None:
            raise ValueError('Either label_of_parent or label must be set')
        super().__init__(model, label, f'{label_of_parent}__{label}' if label_of_parent else None)

        self.interface = interface
        logger.debug(f'Created {self.__class__.__name__}  "{self.label_full}"')


class ElementModel(Model):
    """Interface to create the mathematical Variables and Constraints for Elements"""

    def __init__(self, model: SystemModel, element: Element):
        """
        Parameters
        ----------
        element : Element
            The element this model is created for.
        """
        super().__init__(model, label=element.label, label_of_parent=element.label_full)
        self.element = element


def create_equation(
    label: str, element_model: ElementModel, eq_type: Literal['eq', 'ineq'] = 'eq'
) -> Union[Equation, Inequation]:
    """Creates an Equation and adds it to the model of the Element"""
    if eq_type == 'eq':
        constr = Equation(f'{element_model.label_full}_{label}', label)
    elif eq_type == 'ineq':
        constr = Inequation(f'{element_model.label_full}_{label}', label)
    element_model.add_constraints(constr)
    return constr


def create_variable(
    label: str,
    element_model: ElementModel,
    length: int,
    is_binary: bool = False,
    fixed_value: Optional[Numeric] = None,
    lower_bound: Optional[Numeric] = None,
    upper_bound: Optional[Numeric] = None,
    previous_values: Optional[Numeric] = None,
    avoid_use_of_variable_ts: bool = False,
) -> VariableTS:
    """Creates a VariableTS and adds it to the model of the Element"""
    variable_label = f'{element_model.label_full}_{label}'
    if length > 1 and not avoid_use_of_variable_ts:
        var = VariableTS(
            variable_label, length, label, is_binary, fixed_value, lower_bound, upper_bound, previous_values
        )
        logger.debug(f'Created VariableTS "{variable_label}": [{length}]')
    else:
        var = Variable(variable_label, length, label, is_binary, fixed_value, lower_bound, upper_bound)
        logger.debug(f'Created Variable "{variable_label}": [{length}]')
    element_model.add_variables(var)
    return var


def copy_and_convert_datatypes(data: Any, use_numpy: bool = True, use_element_label: bool = False) -> Any:
    """
    Converts values in a nested data structure into JSON-compatible types while preserving or transforming numpy arrays
    and custom `Element` objects based on the specified options.

    The function handles various data types and transforms them into a consistent, readable format:
    - Primitive types (`int`, `float`, `str`, `bool`, `None`) are returned as-is.
    - Numpy scalars are converted to their corresponding Python scalar types.
    - Collections (`list`, `tuple`, `set`, `dict`) are recursively processed to ensure all elements are compatible.
    - Numpy arrays are preserved or converted to lists, depending on `use_numpy`.
    - Custom `Element` objects can be represented either by their `label` or their initialization parameters as a dictionary.
    - Timestamps (`datetime`) are converted to ISO 8601 strings.

    Parameters
    ----------
    data : Any
        The input data to process, which may be deeply nested and contain a mix of types.
    use_numpy : bool, optional
        If `True`, numeric numpy arrays (`np.ndarray`) are preserved as-is. If `False`, they are converted to lists.
        Default is `True`.
    use_element_label : bool, optional
        If `True`, `Element` objects are represented by their `label`. If `False`, they are converted into a dictionary
        based on their initialization parameters. Default is `False`.

    Returns
    -------
    Any
        A transformed version of the input data, containing only JSON-compatible types:
        - `int`, `float`, `str`, `bool`, `None`
        - `list`, `dict`
        - `np.ndarray` (if `use_numpy=True`. This is NOT JSON-compatible)

    Raises
    ------
    TypeError
        If the data cannot be converted to the specified types.

    Examples
    --------
    >>> copy_and_convert_datatypes({'a': np.array([1, 2, 3]), 'b': Element(label='example')})
    {'a': array([1, 2, 3]), 'b': {'class': 'Element', 'label': 'example'}}

    >>> copy_and_convert_datatypes({'a': np.array([1, 2, 3]), 'b': Element(label='example')}, use_numpy=False)
    {'a': [1, 2, 3], 'b': {'class': 'Element', 'label': 'example'}}

    Notes
    -----
    - The function gracefully handles unexpected types by issuing a warning and returning a deep copy of the data.
    - Empty collections (lists, dictionaries) and default parameter values in `Element` objects are omitted from the output.
    - Numpy arrays with non-numeric data types are automatically converted to lists.
    """
    if isinstance(data, np.integer):  # This must be checked before checking for regular int and float!
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)

    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    elif isinstance(data, datetime):
        return data.isoformat()

    elif isinstance(data, (tuple, set)):
        return copy_and_convert_datatypes([item for item in data], use_numpy, use_element_label)
    elif isinstance(data, dict):
        return {
            copy_and_convert_datatypes(key, use_numpy, use_element_label=True): copy_and_convert_datatypes(
                value, use_numpy, use_element_label
            )
            for key, value in data.items()
        }
    elif isinstance(data, list):  # Shorten arrays/lists to be readable
        if use_numpy and all([isinstance(value, (int, float)) for value in data]):
            return np.array([item for item in data])
        else:
            return [copy_and_convert_datatypes(item, use_numpy, use_element_label) for item in data]

    elif isinstance(data, np.ndarray):
        if not use_numpy:
            return copy_and_convert_datatypes(data.tolist(), use_numpy, use_element_label)
        elif use_numpy and np.issubdtype(data.dtype, np.number):
            return data
        else:
            logger.critical(
                f'An np.array with non-numeric content was found: {data=}.It will be converted to a list instead'
            )
            return copy_and_convert_datatypes(data.tolist(), use_numpy, use_element_label)

    elif isinstance(data, TimeSeries):
        return copy_and_convert_datatypes(data.active_data, use_numpy, use_element_label)
    elif isinstance(data, TimeSeriesData):
        return copy_and_convert_datatypes(data.data, use_numpy, use_element_label)

    elif isinstance(data, Interface):
        if use_element_label and isinstance(data, Element):
            return data.label
        return data.infos(use_numpy, use_element_label)
    elif isinstance(data, xr.DataArray):
        #TODO: This is a temporary basic work around
        return copy_and_convert_datatypes(data.values, use_numpy, use_element_label)
    else:
        raise TypeError(f'copy_and_convert_datatypes() did get unexpected data of type "{type(data)}": {data=}')


def get_compact_representation(data: Any, array_threshold: int = 50, decimals: int = 2) -> Dict:
    """
    Generate a compact json serializable representation of deeply nested data.
    Numpy arrays are statistically described if they exceed a threshold and converted to lists.

    Args:
        data (Any): The data to format and represent.
        array_threshold (int): Maximum length of NumPy arrays to display. Longer arrays are statistically described.
        decimals (int): Number of decimal places in which to describe the arrays.

    Returns:
        Dict: A dictionary representation of the data
    """

    def format_np_array_if_found(value: Any) -> Any:
        """Recursively processes the data, formatting NumPy arrays."""
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        elif isinstance(value, np.ndarray):
            return describe_numpy_arrays(value)
        elif isinstance(value, dict):
            return {format_np_array_if_found(k): format_np_array_if_found(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple, set)):
            return [format_np_array_if_found(v) for v in value]
        else:
            logger.warning(
                f'Unexpected value found when trying to format numpy array numpy array: {type(value)=}; {value=}'
            )
            return value

    def describe_numpy_arrays(arr: np.ndarray) -> Union[str, List]:
        """Shortens NumPy arrays if they exceed the specified length."""

        def normalized_center_of_mass(array: Any) -> float:
            # position in array (0 bis 1 normiert)
            positions = np.linspace(0, 1, len(array))  # weights w_i
            # mass center
            if np.sum(array) == 0:
                return np.nan
            else:
                return np.sum(positions * array) / np.sum(array)

        if arr.size > array_threshold:  # Calculate basic statistics
            fmt = f'.{decimals}f'
            return (
                f'Array (min={np.min(arr):{fmt}}, max={np.max(arr):{fmt}}, mean={np.mean(arr):{fmt}}, '
                f'median={np.median(arr):{fmt}}, std={np.std(arr):{fmt}}, len={len(arr)}, '
                f'center={normalized_center_of_mass(arr):{fmt}})'
            )
        else:
            return np.around(arr, decimals=decimals).tolist()

    # Process the data to handle NumPy arrays
    formatted_data = format_np_array_if_found(copy_and_convert_datatypes(data, use_numpy=True))

    return formatted_data


def get_str_representation(data: Any, array_threshold: int = 50, decimals: int = 2) -> str:
    """
    Generate a string representation of deeply nested data using `rich.print`.
    NumPy arrays are shortened to the specified length and converted to strings.

    Args:
        data (Any): The data to format and represent.
        array_threshold (int): Maximum length of NumPy arrays to display. Longer arrays are statistically described.
        decimals (int): Number of decimal places in which to describe the arrays.

    Returns:
        str: The formatted string representation of the data.
    """

    formatted_data = get_compact_representation(data, array_threshold, decimals)

    # Use Rich to format and print the data
    with StringIO() as output_buffer:
        console = Console(file=output_buffer, width=1000)  # Adjust width as needed
        console.print(Pretty(formatted_data, expand_all=True, indent_guides=True))
        return output_buffer.getvalue()
