import linopy
import json
import pathlib
import xarray as xr
from typing import Dict, Union, List, Literal
import logging


class CalculationResults:
    @classmethod
    def read_from_file(cls, folder: Union[str, pathlib.Path], name: str):
        folder = pathlib.Path(folder)
        path = folder / name
        model = linopy.read_netcdf(path.with_suffix('.nc'))
        with open(path.with_suffix('.json'), 'r', encoding='utf-8') as f:
            flow_system_structure = json.load(f)
        return cls(model, flow_system_structure)

    def __init__(self, model: linopy.Model, flow_system_structure: Dict[str, Dict[str, Dict]]):
        self.model = model
        self._flow_system_structure = flow_system_structure
        self.components = {label: ComponentResults.from_json(self, infos)
                           for label, infos in flow_system_structure['Components'].items()}

        self.buses = {label: BusResults.from_json(self, infos)
                      for label, infos in flow_system_structure['Buses'].items()}

        self.effects = {label: EffectResults.from_json(self, infos)
                        for label, infos in flow_system_structure['Effects'].items()}

    def __getitem__(self, key: str) -> Union['ComponentResults', 'BusResults', 'EffectResults']:
        if key in self.components:
            return self.components[key]
        if key in self.buses:
            return self.buses[key]
        if key in self.effects:
            return self.effects[key]
        raise KeyError(f'No element with label {key} found.')


class _ElementResults:
    @classmethod
    def from_json(cls, calculation_results, json_data: Dict):
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


class _NodeResults(_ElementResults):
    @classmethod
    def from_json(cls, calculation_results, json_data: Dict):
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
                 inputs: Dict[str, xr.DataArray],
                 outputs: Dict[str, xr.DataArray]):
        super().__init__(calculation_results, label, variables, constraints)
        self.inputs = inputs
        self.outputs = outputs


class BusResults(_NodeResults):
    """Results for a Bus"""


class ComponentResults(_NodeResults):
    """Results for a Component"""


class EffectResults(_ElementResults):
    """Results for an Effect"""
