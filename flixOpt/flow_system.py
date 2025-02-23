"""
This module contains the FlowSystem class, which is used to collect instances of many other classes by the end User.
"""

import json
import logging
import pathlib
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from . import utils
from .core import TimeSeries, TimeSeriesCollection
from .effects import Effect
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
        self.effects: Dict[str, Effect] = {}
        self.model: Optional[SystemModel] = None

    def add_effects(self, *args: Effect) -> None:
        for new_effect in list(args):
            if new_effect.label in self.effects:
                raise Exception(f'Effect with label "{new_effect.label=}" already added!')
            self.effects[new_effect.label] = new_effect
            logger.info(f'Registered new Effect: {new_effect.label}')

    def add_components(self, *args: Component) -> None:
        # Komponenten registrieren:
        new_components = list(args)
        for new_component in new_components:
            logger.info(f'Registered new Component: {new_component.label}')
            self._check_if_element_is_unique(new_component)  # check if already exists:
            new_component.register_component_in_flows()  # Komponente in Flow registrieren
            new_component.register_flows_in_bus()  # Flows in Bus registrieren:
            self.components[new_component.label] = new_component  # Add to existing components

    def add_elements(self, *args: Element) -> None:
        """
        add all modeling elements, like storages, boilers, heatpumps, buses, ...

        Parameters
        ----------
        *args : childs of  Element like Boiler, HeatPump, Bus,...
            modeling Elements

        """
        for new_element in list(args):
            if isinstance(new_element, Component):
                self.add_components(new_element)
            elif isinstance(new_element, Effect):
                self.add_effects(new_element)
            else:
                raise Exception('argument is not instance of a modeling Element (Element)')

    def transform_data(self):
        for element in self.all_elements.values():
            element.transform_data(self.time_series_collection)

    def network_infos(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
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
                'start': flow.bus.label_full if flow.is_input_in_comp else flow.comp.label_full,
                'end': flow.comp.label_full if flow.is_input_in_comp else flow.bus.label_full,
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
                for effect in sorted(self.effects.values(), key=lambda effect: effect.label.upper())
            },
        }
        return infos

    def to_json(self, path: Union[str, pathlib.Path]):
        """
        Saves the flow system to a json file.
        This not meant to be reloaded and recreate the object, but rather used to document or compare the object.

        Parameters:
        -----------
        path : Union[str, pathlib.Path]
            The path to the json file.
        """
        data = get_compact_representation(self.infos(use_numpy=True, use_element_label=True))
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def results(self):
        #TODO: Remove this function, as access through the FLowSystem is not correct if another calucaltion was made.
        return {
            'Components': {
                comp.label: comp.model.solution_structured(mode='numpy')
                for comp in sorted(self.components.values(), key=lambda component: component.label.upper())
            },
            'Buses': {
                bus.label: bus.model.solution_structured(mode='numpy')
                for bus in sorted(self.buses.values(), key=lambda bus: bus.label.upper())
            },
            'Effects': {
                effect.label: effect.model.solution_structured(mode='numpy')
                for effect in sorted(self.effects.values(), key=lambda effect: effect.label.upper())
            },
            'Time': self.time_series_collection.timesteps_extra.tolist(),
            'Time intervals in hours': self.time_series_collection.hours_per_timestep,
        }

    def visualize_network(
        self,
        path: Union[bool, str, pathlib.Path] = 'flow_system.html',
        controls: Union[
            bool,
            List[
                Literal['nodes', 'edges', 'layout', 'interaction', 'manipulation', 'physics', 'selection', 'renderer']
            ],
        ] = True,
        show: bool = True,
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
          >>> self.visualize_network()

        - Save the visualization without opening:
          >>> self.visualize_network(show=False)

        - Visualize with custom controls and path:
          >>> self.visualize_network(path='output/custom_network.html', controls=['nodes', 'layout'])

        Notes:
        - This function requires `pyvis`. If not installed, the function prints a warning and returns `None`.
        - Nodes are styled based on type (e.g., circles for buses, boxes for components) and annotated with node information.
        """
        from . import plotting

        node_infos, edge_infos = self.network_infos()
        return plotting.visualize_network(node_infos, edge_infos, path, controls, show)

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
    def buses(self) -> Dict[str, Bus]:
        return {flow.bus.label: flow.bus for flow in self.flows.values()}

    @property
    def all_elements(self) -> Dict[str, Element]:
        return {**self.components, **self.effects, **self.flows, **self.buses}

    @property
    def all_time_series(self) -> List[TimeSeries]:
        return [ts for element in self.all_elements.values() for ts in element.used_time_series]
