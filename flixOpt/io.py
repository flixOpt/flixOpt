import datetime
import json
import logging
import pathlib
from typing import Dict, Union

import linopy
import xarray as xr

from .flow_system import FlowSystem

logger = logging.getLogger('flixOpt')


def _results_structure(flow_system: FlowSystem) -> Dict[str, Dict[str, str]]:
    return {
        'Components': {
            comp.label_full: comp.model.results_structure()
            for comp in sorted(flow_system.components.values(), key=lambda component: component.label_full.upper())
        },
        'Buses': {
            bus.label_full: bus.model.results_structure()
            for bus in sorted(flow_system.buses.values(), key=lambda bus: bus.label_full.upper())
        },
        'Effects': {
            effect.label_full: effect.model.results_structure()
            for effect in sorted(flow_system.effects.values(), key=lambda effect: effect.label_full.upper())
        },
        'Time': [datetime.datetime.isoformat(date) for date in flow_system.time_series_collection.timesteps_extra],
        'Periods': flow_system.time_series_collection.periods.tolist() if flow_system.time_series_collection.periods is not None else None
    }


def model_to_netcdf(model: linopy.Model, path: Union[str, pathlib.Path] = 'system_model.nc', *args, **kwargs):
    """
    Save the linopy model to a netcdf file.
    """
    model.to_netcdf(path, *args, **kwargs)


def structure_to_json(flow_system: FlowSystem, path: Union[str, pathlib.Path] = 'system_model.json'):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_results_structure(flow_system), f, indent=4, ensure_ascii=False)
