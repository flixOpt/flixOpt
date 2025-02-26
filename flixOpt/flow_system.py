"""
This module contains the FlowSystem class, which is used to collect instances of many other classes by the end User.
"""

import json
import warnings
import logging
import pathlib
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from . import utils
from .core import TimeSeries, TimeSeriesCollection, NumericData, NumericDataTS, TimeSeriesData
from .effects import Effect, EffectCollection, EffectTimeSeries, EffectValuesUser, EffectValuesDict
from .elements import Bus, Component, Flow
from .structure import Element, SystemModel, get_compact_representation, get_str_representation

if TYPE_CHECKING:
    import pyvis

logger = logging.getLogger('flixOpt')


class FlowSystem:
    """
    A FlowSystem organizes the high level Elements (Components & Effects).
    """

    def __init__(
            self,
            timesteps: pd.DatetimeIndex,
            hours_of_last_timestep: Optional[float] = None,
            hours_of_previous_timesteps: Optional[Union[int, float, np.ndarray]] = None,
            periods: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        timesteps : pd.DatetimeIndex
            The timesteps of the model.
        hours_of_last_timestep : Optional[float], optional
            The duration of the last time step. Uses the last time interval if not specified
        hours_of_previous_timesteps : Union[int, float, np.ndarray]
            The duration of previous timesteps.
            If None, the first time increment of time_series is used.
            This is needed to calculate previous durations (for example consecutive_on_hours).
            If you use an array, take care that its long enough to cover all previous values!
        periods : Optional[List[int]], optional
            The periods of the model. Every period has the same timesteps.
            Usually years are used as periods.
        """
        self.time_series_collection = TimeSeriesCollection(
            timesteps=timesteps,
            hours_of_last_timestep=hours_of_last_timestep,
            hours_of_previous_timesteps=hours_of_previous_timesteps,
            periods=periods
        )

        # defaults:
        self.components: Dict[str, Component] = {}
        self.buses: Dict[str, Bus] = {}
        self.effects: EffectCollection = EffectCollection()
        self.model: Optional[SystemModel] = None

    def add_elements(self, *elements: Element) -> None:
        """
        add all modeling elements, like storages, boilers, heatpumps, buses, ...

        Parameters
        ----------
        *elements : childs of  Element like Boiler, HeatPump, Bus,...
            modeling Elements

        """
        for new_element in list(elements):
            if isinstance(new_element, Component):
                self._add_components(new_element)
            elif isinstance(new_element, Effect):
                self._add_effects(new_element)
            elif isinstance(new_element, Bus):
                self._add_buses(new_element)
            else:
                raise Exception('argument is not instance of a modeling Element (Element)')

    def _add_effects(self, *args: Effect) -> None:
        self.effects.add_effects(*args)

    def _add_components(self, *components: Component) -> None:
        for new_component in list(components):
            logger.info(f'Registered new Component: {new_component.label}')
            self._check_if_element_is_unique(new_component)  # check if already exists:
            self.components[new_component.label] = new_component  # Add to existing components

    def _add_buses(self, *buses: Bus):
        for new_bus in list(buses):
            logger.info(f'Registered new Bus: {new_bus.label}')
            self._check_if_element_is_unique(new_bus)  # check if already exists:
            self.buses[new_bus.label] = new_bus  # Add to existing components

    def _connect_network(self):
        """Connects the network of components and buses. Can be rerun without changes if no elements were added"""
        for component in self.components.values():
            for flow in component.inputs + component.outputs:
                flow.component = component.label_full
                flow.is_input_in_component = True if flow in component.inputs else False

                # Add Bus if not already added (deprecated)
                if flow._bus_object is not None and flow._bus_object not in self.buses.values():
                    self._add_buses(flow._bus_object)
                    warnings.warn(
                        f'Bus {flow._bus_object.label} was passed as a Bus object to {flow.label_full} and not added to the FlowSystem.'
                        f' Add the Bus to the FlowSystem instead and pass its label to the Flow.',
                        DeprecationWarning,
                        stacklevel=2)

                # Connect Buses
                bus = self.buses.get(flow.bus)
                if bus is None:
                    raise KeyError(f'Bus {flow.bus} not found in the FlowSystem, but used by "{flow.label_full}". '
                                   f'Please add it first.')
                if flow.is_input_in_component and flow not in bus.outputs:
                    bus.outputs.append(flow)
                elif not flow.is_input_in_component and flow not in bus.inputs:
                    bus.inputs.append(flow)

    def transform_data(self):
        self._connect_network()
        for element in self.all_elements.values():
            element.transform_data(self)

    def create_time_series(
            self,
            name: str,
            data: Optional[Union[NumericData, TimeSeriesData, TimeSeries]],
            extra_timestep: bool = False,
    ) -> Optional[TimeSeries]:
        """
        Tries to create a TimeSeries from NumericData Data and adds it to the time_series_collection
        If the data already is a TimeSeries, nothing happens and the TimeSeries gets reset and returned
        If the data is a TimeSeriesData, it is converted to a TimeSeries, and the aggregation weights are applied.
        If the data is None, nothing happens.
        """

        if data is None:
            return None
        elif isinstance(data, TimeSeries):
            data.restore_data()
            return data
        return self.time_series_collection.create_time_series(
            data=data,
            name=name,
            extra_timestep=extra_timestep
        )

    def create_effect_time_series(self,
                                  label_prefix: Optional[str],
                                  effect_values: EffectValuesUser,
                                  label_suffix: Optional[str] = None,
                                  ) -> Optional[EffectTimeSeries]:
        """
        Transform EffectValues to EffectTimeSeries.
        Creates a TimeSeries for each key in the nested_values dictionary, using the value as the data.

        The resulting label of the TimeSeries is the label of the parent_element,
        followed by the label of the Effect in the nested_values and the label_suffix.
        If the key in the EffectValues is None, the alias 'Standard_Effect' is used
        """
        effect_values: Optional[EffectValuesDict] = self.effects.create_effect_values_dict(effect_values)
        if effect_values is None:
            return None

        return {
            effect: self.create_time_series(
                '|'.join(filter(None, [label_prefix, effect, label_suffix])),
                value
            )
            for effect, value in effect_values.items()
        }

    def network_infos(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
        self._connect_network()
        nodes = {
            node.label_full: {
                'label': node.label,
                'class': 'Bus' if isinstance(node, Bus) else 'Component',
                'infos': node.__str__(),
            }
            for node in list(self.components.values()) + list(self.buses.values())
        }

        edges = {
            flow.label_full: {
                'label': flow.label,
                'start': flow.bus if flow.is_input_in_component else flow.component,
                'end': flow.component if flow.is_input_in_component else flow.bus,
                'infos': flow.__str__(),
            }
            for flow in self.flows.values()
        }

        return nodes, edges

    def infos(self, use_numpy=True, use_element_label=False) -> Dict:
        infos = {
            'Components': {
                comp.label: comp.infos(use_numpy, use_element_label)
                for comp in sorted(self.components.values(), key=lambda component: component.label.upper())
            },
            'Buses': {
                bus.label: bus.infos(use_numpy, use_element_label)
                for bus in sorted(self.buses.values(), key=lambda bus: bus.label.upper())
            },
            'Effects': {
                effect.label: effect.infos(use_numpy, use_element_label)
                for effect in sorted(self.effects, key=lambda effect: effect.label.upper())
            },
        }
        return infos

    def infos_compact(self):
        return get_compact_representation(self.infos(use_numpy=True, use_element_label=True)),

    def to_json(self, path: Union[str, pathlib.Path]):
        """
        Saves the flow system to a json file.
        This not meant to be reloaded and recreate the object, but rather used to document or compare the object.

        Parameters:
        -----------
        path : Union[str, pathlib.Path]
            The path to the json file.
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.infos_compact(), f, indent=4, ensure_ascii=False)

    def plot_network(
        self,
        path: Union[bool, str, pathlib.Path] = 'flow_system.html',
        controls: Union[
            bool,
            List[
                Literal['nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer']
            ],
        ] = True,
        show: bool = False,
    ) -> Optional['pyvis.network.Network']:
        """
        Visualizes the network structure of a FlowSystem using PyVis, saving it as an interactive HTML file.

        Parameters:
        - path (Union[bool, str, pathlib.Path], default='flow_system.html'):
          Path to save the HTML visualization.
            - `False`: Visualization is created but not saved.
            - `str` or `Path`: Specifies file path (default: 'flow_system.html').

        - controls (Union[bool, List[str]], default=True):
          UI controls to add to the visualization.
            - `True`: Enables all available controls.
            - `List`: Specify controls, e.g., ['nodes', 'layout'].
            - Options: 'nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer'.

        - show (bool, default=True):
          Whether to open the visualization in the web browser.

        Returns:
        - Optional[pyvis.network.Network]: The `Network` instance representing the visualization, or `None` if `pyvis` is not installed.

        Usage:
        - Visualize and open the network with default options:
          >>> self.plot_network()

        - Save the visualization without opening:
          >>> self.plot_network(show=False)

        - Visualize with custom controls and path:
          >>> self.plot_network(path='output/custom_network.html', controls=['nodes', 'layout'])

        Notes:
        - This function requires `pyvis`. If not installed, the function prints a warning and returns `None`.
        - Nodes are styled based on type (e.g., circles for buses, boxes for components) and annotated with node information.
        """
        from . import plotting

        node_infos, edge_infos = self.network_infos()
        return plotting.plot_network(node_infos, edge_infos, path, controls, show)

    def create_model(self) -> SystemModel:
        self.model = SystemModel(self)
        return self.model

    def _check_if_element_is_unique(self, element: Element) -> None:
        """
        checks if element or label of element already exists in list

        Parameters
        ----------
        element : Element
            new element to check
        """
        if element in self.all_elements.values():
            raise Exception(f'Element {element.label} already added to FlowSystem!')
        # check if name is already used:
        if element.label_full in self.all_elements:
            raise Exception(f'Label of Element {element.label} already used in another element!')

    def __repr__(self):
        return f'<{self.__class__.__name__} with {len(self.components)} components and {len(self.effects)} effects>'

    def __str__(self):
        return get_str_representation(self.infos(use_numpy=True, use_element_label=True))

    @property
    def flows(self) -> Dict[str, Flow]:
        set_of_flows = {flow for comp in self.components.values() for flow in comp.inputs + comp.outputs}
        return {flow.label_full: flow for flow in set_of_flows}

    @property
    def all_elements(self) -> Dict[str, Element]:
        return {**self.components, **self.effects.effects, **self.flows, **self.buses}
