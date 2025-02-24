import datetime
import json
import logging
import pathlib
from typing import Dict, List, Literal, Union, Optional, TYPE_CHECKING

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from . import plotting, utils
from .core import TimeSeriesCollection

from .io import _results_structure

if TYPE_CHECKING:
    from .calculation import Calculation


logger = logging.getLogger('flixOpt')


class CalculationResults:
    """
    Results for a Calculation.
    This class is used to collect the results of a Calculation.
    It is used to analyze the results and to visualize the results.

    Parameters
    ----------
    model : linopy.Model
        The linopy model that was used to solve the calculation.
    flow_system_structure : Dict[str, Dict[str, Dict]]
        The structure of the flow_system that was used to solve the calculation.

    Attributes
    ----------
    model : linopy.Model
        The linopy model that was used to solve the calculation.
    components : Dict[str, ComponentResults]
        A dictionary of ComponentResults for each component in the flow_system.
    buses : Dict[str, BusResults]
        A dictionary of BusResults for each bus in the flow_system.
    effects : Dict[str, EffectResults]
        A dictionary of EffectResults for each effect in the flow_system.
    timesteps_extra : pd.DatetimeIndex
        The extra timesteps of the flow_system.
    periods : pd.Index
        The periods of the flow_system.
    hours_per_timestep : xr.DataArray
        The duration of each timestep in hours.

    Class Methods
    -------
    from_file(folder: Union[str, pathlib.Path], name: str)
        Create CalculationResults directly from file.
    from_calculation(calculation: Calculation)
        Create CalculationResults directly from a Calculation.

    """
    @classmethod
    def from_file(cls, folder: Union[str, pathlib.Path], name: str):
        """ Create CalculationResults directly from file"""
        folder = pathlib.Path(folder)
        path = folder / name
        model = linopy.read_netcdf(path.with_suffix('.nc'))
        with open(path.with_suffix('.json'), 'r', encoding='utf-8') as f:
            flow_system_structure = json.load(f)
        logger.info(f'Loaded calculation "{name}" from file ({path})')
        return cls(model, flow_system_structure, name)

    @classmethod
    def from_calculation(cls, calculation: 'Calculation'):
        """Create CalculationResults directly from a Calculation"""
        return cls(calculation.model, _results_structure(calculation.flow_system), calculation.name)

    def __init__(self, model: linopy.Model, flow_system_structure: Dict[str, Dict[str, Dict]], name: str):
        self.model = model
        self._flow_system_structure = flow_system_structure
        self.name = name
        self.components = {label: ComponentResults.from_json(self, infos)
                           for label, infos in flow_system_structure['Components'].items()}

        self.buses = {label: BusResults.from_json(self, infos)
                      for label, infos in flow_system_structure['Buses'].items()}

        self.effects = {label: EffectResults.from_json(self, infos)
                        for label, infos in flow_system_structure['Effects'].items()}

        self.timesteps_extra = pd.DatetimeIndex([datetime.datetime.fromisoformat(date) for date in flow_system_structure['Time']], name='time')
        self.periods = pd.Index(flow_system_structure['Periods'], name = 'period') if flow_system_structure['Periods'] is not None else None
        self.hours_per_timestep = TimeSeriesCollection.create_hours_per_timestep(self.timesteps_extra, self.periods)

    def __getitem__(self, key: str) -> Union['ComponentResults', 'BusResults', 'EffectResults']:
        if key in self.components:
            return self.components[key]
        if key in self.buses:
            return self.buses[key]
        if key in self.effects:
            return self.effects[key]
        raise KeyError(f'No element with label {key} found.')

    def to_file(self, folder: Union[str, pathlib.Path], name: Optional[str] = None, *args, **kwargs):
        """Save the results to a file"""
        folder = pathlib.Path(folder)
        name = self.name if name is None else name
        path = folder / name
        if not folder.exists():
            try:
                folder.mkdir(parents=False)
            except FileNotFoundError as e:
                raise FileNotFoundError(f'Folder {folder} and its parent do not exist. Please create them first.') from e

        self.model.to_netcdf(path.with_suffix('.nc'), *args, **kwargs)
        with open(path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(self._flow_system_structure, f, indent=4, ensure_ascii=False)
        logger.info(f'Saved calculation "{name}" to {path}')


class _ElementResults:
    @classmethod
    def from_json(cls, calculation_results, json_data: Dict) -> '_ElementResults':
        return cls(calculation_results,
                   json_data['label'],
                   json_data['variables'],
                   json_data['constraints'])

    def __init__(self,
                 calculation_results: CalculationResults,
                 label: str,
                 variables: List[str],
                 constraints: List[str]):
        self._calculation_results = calculation_results
        self.label = label
        self._variables = variables
        self._constraints = constraints

        self.variables = self._calculation_results.model.variables[self._variables]
        self.constraints = self._calculation_results.model.constraints[self._constraints]

    @property
    def variables_time(self):
        return self.variables[[name for name in self._variables if 'time' in self.variables[name].dims]]


class _NodeResults(_ElementResults):
    @classmethod
    def from_json(cls, calculation_results, json_data: Dict)  -> '_NodeResults':
        return cls(calculation_results,
                   json_data['label'],
                   json_data['variables'],
                   json_data['constraints'],
                   json_data['inputs'],
                   json_data['outputs'])

    def __init__(self,
                 calculation_results: CalculationResults,
                 label: str,
                 variables: List[str],
                 constraints: List[str],
                 inputs: List[str],
                 outputs: List[str]):
        super().__init__(calculation_results, label, variables, constraints)
        self.inputs = inputs
        self.outputs = outputs

    def plot_balance(self, show: bool = True):
        return plotting.with_plotly(self.operation_balance(),
                                    mode='area',
                                    title=f'Operation Balance of {self.label}',
                                    show=show)

    def operation_balance(self, negate_inputs: bool = True, negate_outputs: bool = False):
        df = self.variables_time.solution.to_dataframe()
        if negate_outputs:
            df[self.outputs] = -df[self.outputs]
        if negate_inputs:
            df[self.inputs] = -df[self.inputs]
        return df


class BusResults(_NodeResults):
    """Results for a Bus"""


class ComponentResults(_NodeResults):
    """Results for a Component"""


class EffectResults(_ElementResults):
    """Results for an Effect"""

    def get_shares_from(self, element: str):
        return self.variables[[name for name in self._variables if name.startswith(f'{element}->')]]

