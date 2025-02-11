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

    def __init__(
            self,
            flow_system: 'FlowSystem',
            active_time_steps: Optional = None,
    ):
        super().__init__(force_dim_names=True)
        self.flow_system = flow_system
        self.active_time_steps = active_time_steps
        self._order_dimensions()

        self.effects: Optional[EffectCollection] = None

    def do_modeling(self):
        from .effects import EffectCollection
        self.effects = EffectCollection(list(self.flow_system.effects.values()))
        component_models = [component.create_model() for component in self.flow_system.components.values()]
        bus_models = [bus.create_model() for bus in self.flow_system.buses.values()]
        for component_model in component_models:
            component_model.do_modeling(self)
        for bus_model in bus_models:  # Buses after Components, because FlowModels are created in ComponentModels
            bus_model.do_modeling(self)

    def _order_dimensions(self):
        if self.flow_system.timesteps.dtype == np.dtype('datetime64[ns]'):
            self.timesteps = self.flow_system.timesteps.astype('datetime64[us]')
        else:
            self.timesteps = self.flow_system.timesteps
        self.timesteps.name = 'time'

        self.periods = pd.Index(self.flow_system.periods, name='period') if self.flow_system.periods is not None else None

        if self.flow_system.hours_of_last_step:
            last_date = pd.DatetimeIndex([self.timesteps[-1] + pd.to_timedelta(self.flow_system.hours_of_last_step, 'h')])
        else:
            last_date = pd.DatetimeIndex([self.timesteps[-1] + (self.timesteps[-1] - self.timesteps[-2])])
        self.timesteps_extra = self.timesteps.append(last_date)
        self.timesteps_extra.name = 'time'
        hours_per_step = self.timesteps_extra.to_series().diff()[1:].values / pd.to_timedelta(1, 'h')
        self.hours_per_step = xr.DataArray(
            data=np.tile(hours_per_step, (len(self.periods), 1)) if self.periods is not None else hours_per_step,
            coords=self.coords,
            name='hours_per_step'
        )

        # utils.check_time_series('time series of FlowSystem', self.timesteps_extra)

    @property
    def snapshots(self):
        return xr.Dataset(
            coords={'period': list(self.periods), 'time': list(self.timesteps)} if self.periods is not None else {'time': list(self.timesteps)},
        )

    @property
    def coords(self):
        return self.snapshots.coords

    @property
    def variables_filtered(self, filter_by: Optional[Literal['binary', 'continous', 'integer']] = None):
        if filter_by is None:
            all_variables = super().variables
        elif filter_by == 'binary':
            all_variables = super().binaries
        elif filter_by == 'integer':
            all_variables = super().integers
        elif filter_by == 'continous':
            all_variables = super().continuous
        else:
            raise ValueError(f'Invalid filter_by "{filter_by}", must be one of "binary", "continous", "integer"')
        return all_variables[[name for name in all_variables if 'time' in all_variables[name].dims]]

    @property
    def index_shape(self) -> Tuple[int, int]:
        return len(self.periods) if self.periods is not None else 1, len(self.timesteps)

    @property
    def infos(self) -> Dict:
        infos = super().infos
        infos['Constraints'] = self.description_of_constraints()
        infos['Variables'] = self.description_of_variables()
        infos['Main Results'] = self.main_results
        infos['Config'] = CONFIG.to_dict()
        return infos


class Interface:
    """
    This class is used to collect arguments about a Model.
    """

    def transform_data(self):
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

    def _plausibility_checks(self) -> None:
        """This function is used to do some basic plausibility checks for each Element during initialization"""
        raise NotImplementedError('Every Element needs a _plausibility_checks() method')

    def create_model(self) -> 'ElementModel':
        raise NotImplementedError('Every Element needs a create_model() method')

    @property
    def label_full(self) -> str:
        return self.label


class ElementModel:
    """Interface to create the mathematical Variables and Constraints for Elements"""

    def __init__(self, element: Element, labels: Optional[Union[str, List[str]]] = None):
        """
        Parameters
        ----------
        element : Element
            The element this model is created for.
        labels : Optional[Union[str, List[str]]], optional
            Used to construct the label of the model. If None, the element label is used.
            The labels are used as suffixes
        """
        self.element = element
        self.variables = {}
        self.constraints = {}
        self.sub_models = []
        self._labels = labels
        logger.debug(f'Created {self.__class__.__name__}  "{self.label_full}"')

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
        return {
            'no of Euations': len(all_eqs),
            'no of Equations single': sum(eq.nr_of_single_equations for eq in all_eqs.values()),
            'no of Inequations': len(all_ineqs),
            'no of Inequations single': sum(ineq.nr_of_single_equations for ineq in all_ineqs.values()),
            'no of Variables': len(all_vars),
            'no of Variables single': sum(var.length for var in all_vars.values()),
        }

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
    def all_sub_models(self) -> List['ElementModel']:
        all_subs = []
        to_process = self.sub_models.copy()
        for model in to_process:
            all_subs.append(model)
            to_process.extend(model.sub_models)
        return all_subs

    def results(self) -> Dict:
        return {
            **{variable.label_short: variable.result for variable in self.variables.values()},
            **{model.label: model.results() for model in self.sub_models},
        }

    @property
    def label_full(self) -> str:
        if self._labels is not None and self.element is not None:
            if isinstance(self._labels, str) :
                return f'{self.element.label_full}__{self._labels}'
            else:
                return f'{self.element.label_full}__' + '__'.join(self._labels)
        if self._labels is not None and self.element is None:
            if isinstance(self._labels, str):
                return self._labels
            else:
                return '__'.join(self._labels)
        if self.element is not None:
            return self.element.label_full
        raise Exception('This should not happen! Internal Error. Please create Issue on GitHub')

    @property
    def label(self) -> str:
        if self._labels is not None:
            if isinstance(self._labels, str):
                return self._labels
            else:
                return self._labels[-1]
        else:
            return self.element.label


def _create_time_series(
    label: str, data: Optional[Union[Numeric_TS, TimeSeries]], element: Element
) -> Optional[TimeSeries]:
    """Creates a TimeSeries from Numeric Data and adds it to the list of time_series of an Element.
    If the data already is a TimeSeries, nothing happens and the TimeSeries gets reset and returned"""
    if data is None:
        return None
    elif isinstance(data, TimeSeries):
        data.clear_indices_and_aggregated_data()
        return data
    else:
        time_series = TimeSeries(label=f'{element.label_full}__{label}', data=data)
        element.used_time_series.append(time_series)
        return time_series


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
