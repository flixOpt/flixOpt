import linopy
import json
import pathlib
import xarray as xr
from typing import Dict, Union
import logging


logger = logging.getLogger('flixOpt')


def _solution_structure_basic(self):
    return {
        'Buses': {
            bus.label_full: [f':::{var}' for var in bus.model.variables]
            for bus in sorted(self.flow_system.buses.values(), key=lambda bus: bus.label_full.upper())
        },
        'Components': {
            comp.label_full: [f':::{var}' for var in comp.model.variables]
            for comp in sorted(self.flow_system.components.values(), key=lambda component: component.label_full.upper())
        },
        'Effects': {
            effect.label_full: [f':::{var}' for var in effect.model.variables]
            for effect in sorted(self.flow_system.effects.values(), key=lambda effect: effect.label_full.upper())
        },
        'Penalty': float(self.effects.penalty.total.solution.values),
        'Objective': self.objective.value
    }


def model_to_netcdf(model: linopy.Model, path: Union[str, pathlib.Path] = 'system_model.nc', *args, **kwargs):
    """
    Save the linopy model to a netcdf file.
    """
    model.to_netcdf(path, *args, **kwargs)
    logger.info(f'Saved linopy model to {path}')


def solution_to_netcdf(self, path: Union[str, pathlib.Path] = 'system_model.nc'):
    """
    Save the model to a netcdf file.
    """
    ds = self.solution
    ds = ds.rename_vars({var: var.replace('/', '-slash-') for var in ds.data_vars})
    ds.attrs["structure"] = json.dumps(self._solution_structure_basic())  # Convert dict to JSON string
    ds.to_netcdf(path)


def model_from_netcdf(path: Union[str, pathlib.Path] = 'system_model.nc', *args, **kwargs) -> linopy.Model:
    """
    Read a linopy model from a netcdf file.
    """
    return linopy.read_netcdf(path)


def solution_from_netcdf(path: Union[str, pathlib.Path] = 'flow_system.nc') -> Dict[str, Union[str, Dict, xr.DataArray]]:
    """
    Load a linopy model from a netcdf file.
    """
    results = xr.open_dataset(path)
    return {
        **_insert_dataarrays(results, json.loads(results.attrs['structure'])),
        'Solution': results
    }


def _insert_dataarrays(dataset: xr.Dataset, structure: Dict[str, Union[str, Dict]]):
    dataset = dataset.rename_vars({var: var.replace('-slash-', '/') for var in dataset.data_vars})
    result = {}

    def insert_data(value_part):
        if isinstance(value_part, dict):  # If the value is another nested dictionary
            return _insert_dataarrays(dataset, value_part)  # Recursively handle it
        elif isinstance(value_part, list):
            return [insert_data(v) for v in value_part]
        elif isinstance(value_part, str) and value_part.startswith(':::'):
            return dataset[value_part.removeprefix(':::')]
        elif isinstance(value_part, str):
            return value_part
        elif isinstance(value_part, (int, float)):
            return value_part
        else:
            raise ValueError(f'Loading the Dataset failed. Not able to handle {value_part}')

    for key, value in structure.items():
        result[key] = insert_data(value)

    return result
