import datetime
import json
import logging
import pathlib
from typing import Dict, Literal, Union

import xarray as xr

from .core import TimeSeries
from .flow_system import FlowSystem

logger = logging.getLogger('flixOpt')


def _results_structure(flow_system: FlowSystem) -> Dict[str, Dict]:
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
            for effect in sorted(flow_system.effects, key=lambda effect: effect.label_full.upper())
        },
        'Time': [datetime.datetime.isoformat(date) for date in flow_system.time_series_collection.timesteps_extra],
    }


def structure_to_json(flow_system: FlowSystem, path: Union[str, pathlib.Path] = 'system_model.json'):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_results_structure(flow_system), f, indent=4, ensure_ascii=False)


def replace_timeseries(obj, mode: Literal['name', 'stats', 'data'] = 'name'):
    """Recursively replaces TimeSeries objects with their names prefixed by '::::'."""
    if isinstance(obj, dict):
        return {k: replace_timeseries(v, mode) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_timeseries(v, mode) for v in obj]
    elif isinstance(obj, TimeSeries):  # Adjust this based on the actual class
        if obj.all_equal:
            return obj.active_data.values[0].item()
        elif mode == 'name':
            return f"::::{obj.name}"
        elif mode == 'stats':
            return obj.stats
        elif mode == 'data':
            return obj
        else:
            raise ValueError(f"Invalid mode {mode}")
    else:
        return obj

def insert_dataarray(obj, ds: xr.Dataset):
    """Recursively inserts TimeSeries objects into a dataset."""
    if isinstance(obj, dict):
        return {k: insert_dataarray(v, ds) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [insert_dataarray(v, ds) for v in obj]
    elif isinstance(obj, str) and obj.startswith("::::"):
        da = ds[obj[4:]]
        if da.isel(time=-1).isnull():
            return da.isel(time=slice(0, -1))
        return da
    else:
        return obj


def remove_none_and_empty(obj):
    """Recursively removes None and empty dicts and lists values from a dictionary or list."""

    if isinstance(obj, dict):
        return {k: remove_none_and_empty(v) for k, v in obj.items() if
                not (v is None or (isinstance(v, (list, dict)) and not v))}

    elif isinstance(obj, list):
        return [remove_none_and_empty(v) for v in obj if
                not (v is None or (isinstance(v, (list, dict)) and not v))]

    else:
        return obj
