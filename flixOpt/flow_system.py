"""
This module contains the FlowSystem class, which is used to collect instances of many other classes by the end User.
"""

import pathlib
from typing import List, Set, Tuple, Dict, Union, Optional, Literal
import logging

import numpy as np

from . import utils
from .core import TimeSeries
from .structure import Element, SystemModel, Commodity
from .elements import Bus, Flow, Component
from .effects import Effect, EffectCollection

logger = logging.getLogger('flixOpt')


class FlowSystem:
    """
    A FlowSystem organizes the high level Elements (Components & Effects).
    """
    def __init__(self,
                 time_series: np.ndarray[np.datetime64],
                 last_time_step_hours: Optional[Union[int, float]] = None,
                 use_default_comodities: bool = True):
        """
          Parameters
          ----------
          time_series : np.ndarray of datetime64
              timeseries of the data. Must be in datetime64 format. Don't use precisions below 'us'. !np.datetime64[ns]!
          last_time_step_hours :
              The duration of last time step.
              Storages needs this time-duration for calculation of charge state
              after last time step.
              If None, then last time increment of time_series is used.
        """
        self.time_series = time_series if isinstance(time_series, np.ndarray) else np.array(time_series)
        if self.time_series.dtype == np.dtype('datetime64[ns]'):
            self.time_series = self.time_series.astype('datetime64[us]')

        self.last_time_step_hours = self.time_series[-1] - self.time_series[-2] if last_time_step_hours is None else last_time_step_hours
        self.time_series_with_end = np.append(self.time_series, self.time_series[-1] + self.last_time_step_hours)

        utils.check_time_series('time series of FlowSystem', self.time_series_with_end)

        # defaults:
        self.components: List[Component] = []
        self.effect_collection: EffectCollection = EffectCollection('Effects')  # Organizes Effects, Penalty & Objective
        self.commodities: Dict[str, Commodity] = {
            'default': Commodity('default', 'None', description='THe default commodity', color='#222831')
        }
        if use_default_comodities:
            self.commodities.update({
                'electricity': Commodity('electricity', 'MWh', 'Electricity', color='#00A0E9'),
                'heat': Commodity('heat', 'MWh', 'Heat', color='#FFA500'),
                'fuel': Commodity('fuel', 'MWh', 'Fuel', color='#A52A2A'),
                'money': Commodity('money', '€', 'Money', color='#228B22'),
            })
        self.model: Optional[SystemModel] = None

    def add_effects(self, *args: Effect) -> None:
        for new_effect in list(args):
            logger.info(f'Registered new Effect: {new_effect.label}')
            self.effect_collection.add_effect(new_effect)
            if new_effect.commodity not in self.commodities:
                raise ValueError(
                    f'Commodity with the label "{new_effect.commodity}" was not found. Please add it to '
                    f'the commodities of the FlowSystem before adding "{new_effect.label_full}".')

    def add_components(self, *args: Component) -> None:
        # Komponenten registrieren:
        new_components = list(args)
        for new_component in new_components:
            logger.info(f'Registered new Component: {new_component.label}')
            self._check_if_element_is_unique(new_component)  # check if already exists:
            new_component.register_component_in_flows()  # Komponente in Flow registrieren
            new_component.register_flows_in_bus()  # Flows in Bus registrieren:
            for commodity in new_component.commodities + [flow.bus.commodity
                                                          for flow in new_component.inputs + new_component.outputs
                                                          if flow.bus.commodity is not None]:
                if commodity not in self.commodities:
                    raise ValueError(
                        f'Commodity with the label "{commodity}" was not found. Please add it to the commodities of '
                        f'the FlowSystem before adding "{new_component.label_full}".')

        self.components.extend(new_components)  # Add to existing list of components

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

    def add_commodities(self, *commodities: Commodity) -> None:
        """
        Add new commodities to the flow system.
        If a commodity with the same label already exists, it is not added again.
        """
        for commodity in list(commodities):
            if commodity.label in self.commodities:
                logger.critical(f'Commodity with the label "{commodity.label}" is already present in the FlowSystem. '
                                f'Old commodity will be overwritten.')
            self.commodities[commodity.label] = commodity

    def transform_data(self):
        for element in self.all_elements:
            element.transform_data()

    def network_infos(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
        nodes = {node.label_full: {'label': node.label,
                                   'class': 'Bus' if isinstance(node, Bus) else 'Component',
                                   'color': self.commodities.get(node.commodity, self.commodities['default']).color,
                                   'infos':  node.__str__()}
                 for node in self.components + list(self.all_buses)}

        edges = {flow.label_full: {'label': flow.label,
                                   'start': flow.bus.label_full if flow.is_input_in_comp else flow.comp.label_full,
                                   'end': flow.comp.label_full if flow.is_input_in_comp else flow.bus.label_full,
                                   'color': self.commodities.get(flow.commodity, self.commodities['default']).color,
                                   'infos': flow.__str__()}
                 for flow in self.all_flows}

        return nodes, edges

    def infos(self):
        infos = {'Components': {comp.label: comp.infos() for comp in
                                sorted(self.components, key=lambda component: component.label.upper())},
                 'Buses': {bus.label: bus.infos() for bus in
                           sorted(self.all_buses, key=lambda bus: bus.label.upper())},
                 'Effects': {effect.label: effect.infos() for effect in
                             sorted(self.effect_collection.effects, key=lambda effect: effect.label.upper())},
                 'Commodities': {commodity.label: commodity.infos() for commodity in self.commodities.values()}}
        return infos

    def visualize_network(self,
                          path: Union[bool, str, pathlib.Path] = 'flow_system.html',
                          controls: Union[bool, List[Literal[
                              'nodes', 'edges', 'layout', 'interaction', 'manipulation',
                              'physics', 'selection', 'renderer']]] = True,
                          show: bool = True
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

    def _check_if_element_is_unique(self, element: Element) -> None:
        """
        checks if element or label of element already exists in list

        Parameters
        ----------
        element : Element
            new element to check
        """
        if element in self.all_elements:
            raise Exception(f'Element {element.label} already added to FlowSystem!')
        # check if name is already used:
        if element.label_full in [elem.label_full for elem in self.all_elements]:
            raise Exception(f'Label of Element {element.label} already used in another element!')

    def get_time_data_from_indices(self, time_indices: Optional[Union[List[int], range]] = None
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.float64]:
        """
        Computes time series data based on the provided time indices.

        Args:
            time_indices: A list of indices or a range object indicating which time steps to extract.
                          If None, the entire time series is used.

        Returns:
            A tuple containing:
            - Extracted time series
            - Time series with the "end time" appended
            - Differences between consecutive timestamps in hours
            - Total time in hours
        """
        # If time_indices is None, use the full time series range
        if time_indices is None:
            time_indices = range(len(self.time_series))

        # Extract the time series for the provided indices
        time_series = self.time_series[time_indices]

        # Ensure the next timestamp for end time is within bounds
        last_index = time_indices[-1]
        if last_index + 1 < len(self.time_series_with_end):
            end_time = self.time_series_with_end[last_index + 1]
        else:
            raise IndexError(f"Index {last_index + 1} out of bounds for 'self.time_series_with_end'.")

        # Append end time to the time series
        time_series_with_end = np.append(time_series, end_time)

        # Calculate time differences (time deltas) in hours
        time_deltas = time_series_with_end[1:] - time_series_with_end[:-1]
        dt_in_hours = time_deltas / np.timedelta64(1, 'h')

        # Calculate the total time in hours
        dt_in_hours_total = np.sum(dt_in_hours)

        return time_series, time_series_with_end, dt_in_hours, dt_in_hours_total

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self.components)} components and {len(self.effect_collection.effects)} effects>"

    def __str__(self):
        components = '\n'.join(component.__str__() for component in
                               sorted(self.components, key=lambda component: component.label.upper()))
        effects = '\n'.join(effect.__str__() for effect in
                            sorted(self.effect_collection.effects, key=lambda effect: effect.label.upper()))
        return f"FlowSystem with components:\n{components}\nand effects:\n{effects}"

    @property
    def all_flows(self) -> Set[Flow]:
        return {flow for comp in self.components for flow in comp.inputs + comp.outputs}

    @property
    def all_buses(self) -> Set[Bus]:
        return {flow.bus for flow in self.all_flows}

    @property
    def all_elements(self) -> List[Element]:
        return self.components + self.effect_collection.effects + list(self.all_flows) + list(self.all_buses)

    @property
    def all_time_series(self) -> List[TimeSeries]:
        return [ts for element in self.all_elements for ts in element.used_time_series]


def create_datetime_array(start: str,
                          steps: Optional[int] = None,
                          freq: str = '1h',
                          end: Optional[str] = None) -> np.ndarray[np.datetime64]:
    """
    Create a NumPy array with datetime64 values.

    Parameters
    ----------
    start : str
        Start date in 'YYYY-MM-DD' format or a full timestamp (e.g., 'YYYY-MM-DD HH:MM').
    steps : int, optional
        Number of steps in the datetime array. If `end` is provided, `steps` is ignored.
    freq : str, optional
        Frequency for the datetime64 array. Supports flexible intervals:
        - 'Y', 'M', 'W', 'D', 'h', 'm', 's' (e.g., '1h', '15m', '2h').
        Defaults to 'h' (hourly).
    end : str, optional
        End date in 'YYYY-MM-DD' format or a full timestamp (e.g., 'YYYY-MM-DD HH:MM').
        If provided, the function generates an array from `start` to `end` using `freq`.

    Returns
    -------
    np.ndarray
        NumPy array of datetime64 values.

    Examples
    --------
    Create an array with 15-minute intervals:
    >>> create_datetime_array('2023-01-01', steps=5, freq='15m')
    array(['2023-01-01T00:00', '2023-01-01T00:15', '2023-01-01T00:30', ...], dtype='datetime64[m]')

    Create 2-hour intervals:
    >>> create_datetime_array('2023-01-01T00', steps=4, freq='2h')
    array(['2023-01-01T00', '2023-01-01T02', '2023-01-01T04', ...], dtype='datetime64[h]')

    Generate minute intervals until a specified end time:
    >>> create_datetime_array('2023-01-01T00:00', end='2023-01-01T01:00', freq='m')
    array(['2023-01-01T00:00', '2023-01-01T00:01', ..., '2023-01-01T00:59'], dtype='datetime64[m]')
    """
    # Parse the frequency and interval
    unit = freq[-1]  # Get the time unit (e.g., 'h', 'm', 's')
    interval = int(freq[:-1]) if freq[:-1].isdigit() else 1  # Default to interval=1 if not specified
    step_size = np.timedelta64(interval, unit)  # Create the timedelta step size

    # Convert the start time to a datetime64 object
    start_dt = np.datetime64(start)

    # Generate the array based on the parameters
    if end:  # If `end` is specified, create a range from start to end
        end_dt = np.datetime64(end)
        return np.arange(start_dt, end_dt, step_size)

    elif steps:  # If `steps` is specified, create a range with the given number of steps
        return np.array([start_dt + i * step_size for i in range(steps)], dtype='datetime64')

    else:  # If neither `steps` nor `end` is provided, raise an error
        raise ValueError("Either `steps` or `end` must be provided.")



