# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:40:23 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

from typing import List, Set, Tuple, Dict, Union, Optional
import logging

import numpy as np
import yaml  # (für json-Schnipsel-print)

from flixOpt import utils
from flixOpt.core import TimeSeries
from flixOpt.structure import Element, SystemModel
from flixOpt.elements import Bus, Flow, Component
from flixOpt.effects import Effect, EffectCollection

logger = logging.getLogger('flixOpt')


class FlowSystem:
    """
    A FlowSystem organizes the high level Elements (Components & Effects).
    """
    def __init__(self,
                 time_series: np.ndarray[np.datetime64],
                 last_time_step_hours: Optional[Union[int, float]] = None):
        """
          Parameters
          ----------
          time_series : np.ndarray of datetime64
              timeseries of the data
          last_time_step_hours :
              The duration of last time step.
              Storages needs this time-duration for calculation of charge state
              after last time step.
              If None, then last time increment of time_series is used.
        """
        self.time_series = time_series
        self.last_time_step_hours = self.time_series[-1] - self.time_series[-2] if last_time_step_hours is None else last_time_step_hours
        self.time_series_with_end = np.append(self.time_series, self.time_series[-1] + self.last_time_step_hours)

        utils.check_time_series('time series of FlowSystem', self.time_series_with_end)

        # defaults:
        self.components: List[Component] = []
        self.effect_collection: EffectCollection = EffectCollection('Effects')  # Organizes Effects, Penalty & Objective
        self.model: Optional[SystemModel] = None

    def add_effects(self, *args: Effect) -> None:
        for new_effect in list(args):
            logger.info(f'Registered new Effect {new_effect.label}')
            self.effect_collection.add_effect(new_effect)

    def add_components(self, *args: Component) -> None:
        # Komponenten registrieren:
        new_components = list(args)
        for new_component in new_components:
            logger.info(f'Registered new Component {new_component.label}')
            self._check_if_element_is_unique(new_component)  # check if already exists:
            new_component.register_component_in_flows()  # Komponente in Flow registrieren
            new_component.register_flows_in_bus()  # Flows in Bus registrieren:
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
