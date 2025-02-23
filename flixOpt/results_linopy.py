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

    def __init__(self, model: linopy.Model, flow_system_structure: Dict[str, Dict[str, str]]):
        self.model = model
        self._flow_system_structure = flow_system_structure
        if False:
            self.components = {label: ComponentResults(self, label, variables, constraints)
                               for label, variables, constraints in flow_system_structure['Components'].items()}


class ElementResults:
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

class _NodeResults(ElementResults):
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
    def __init__(self,
                 calculation_results: CalculationResults,
                 label: str,
                 variables: List[str],
                 constraints: List[str],
                 inputs: Dict[str, xr.DataArray],
                 outputs: Dict[str, xr.DataArray]
                 ):
        super().__init__(calculation_results, label, variables, constraints, inputs, outputs)


class ComponentResults(_NodeResults):
    def __init__(self,
                 calculation_results: CalculationResults,
                 label: str,
                 variables: List[str],
                 constraints: List[str],
                 inputs: Dict[str, xr.DataArray],
                 outputs: Dict[str, xr.DataArray]
                 ):
        super().__init__(calculation_results, label, variables, constraints, inputs, outputs)


class EffectResults(ElementResults):
    def __init__(self,
                 calculation_results: CalculationResults,
                 label: str,
                 variables: List[str],
                 constraints: List[str]):
        super().__init__(calculation_results, label, variables, constraints)
