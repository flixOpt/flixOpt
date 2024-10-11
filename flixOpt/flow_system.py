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
          last_time_step_hours : for calc
              The duration of last time step.
              Storages needs this time-duration for calculation of charge state
              after last time step.
              If None, then last time increment of time_series is used.
        """
        self.time_series = time_series
        self.last_time_step_hours = last_time_step_hours

        self.time_series_with_end = utils.get_time_series_with_end(time_series, last_time_step_hours)
        utils.check_time_series('global esTimeSeries', self.time_series_with_end)

        # defaults:
        self.components: List[Component] = []
        self.effect_collection: EffectCollection = EffectCollection('Effects')  # Organizes Effects, Penalty & Objective
        self.other_elements: Set[Element] = set()  ## hier kommen zusätzliche Elements rein, z.B. aggregation
        self.temporary_elements = []  # temporary elements, only valid for one calculation (i.g. aggregation modeling)
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
        '''
        add all modeling elements, like storages, boilers, heatpumps, buses, ...

        Parameters
        ----------
        *args : childs of   Element like cBoiler, HeatPump, Bus,...
            modeling Elements

        '''

        for new_element in list(args):
            if isinstance(new_element, Component):
                self.add_components(new_element)
            elif isinstance(new_element, Effect):
                self.add_effects(new_element)
            elif isinstance(new_element, Element):
                # check if already exists:
                self._check_if_element_is_unique(new_element)
                # register Element:
                self.other_elements.add(new_element)
            else:
                raise Exception('argument is not instance of a modeling Element (Element)')

    def add_temporary_elements(self, *args: Element) -> None:
        '''
        add temporary modeling elements, only valid for one calculation,
        i.g. AggregationModeling-Element

        Parameters
        ----------
        *args : Element
            temporary modeling Elements.

        '''

        self.add_elements(*args)
        self.temporary_elements += args  # Register temporary Elements

    def delete_temporary_elements(self):  # function just implemented, still not used
        '''
        deletes all registered temporary Elements
        '''
        for temporary_element in self.temporary_elements:
            # delete them again in the lists:
            self.components.remove(temporary_element)
            self.other_elements.remove(temporary_element)
            self.effect_collection.effects.remove(temporary_element)

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
        if element.label in [elem.label_full for elem in self.all_elements]:
            raise Exception(f'Label of Element {element.label} already used in another element!')

    # aktiviere in TS die gewählten Indexe: (wird auch direkt genutzt, nicht nur in activate_system_model)
    def activate_indices_in_time_series(
            self, indices: Union[List[int], range],
            alternative_data_for_time_series: Optional[Dict[TimeSeries, np.ndarray]] = None) -> None:
        # TODO: Aggreagation functionality to other part of framework?
        aTS: TimeSeries
        if alternative_data_for_time_series is None:
            alternative_data_for_time_series = {}

        for aTS in self.all_time_series_in_elements:
            # Wenn explicitData vorhanden:
            if aTS in alternative_data_for_time_series.keys():
                explicitData = alternative_data_for_time_series[aTS]
            else:
                explicitData = None
                # Aktivieren:
            aTS.activate(indices, explicitData)

    def description_of_equations(self) -> Dict:
        return {'Components': {comp.label: comp.model.description_of_equations for comp in self.components},
                'buses': {bus.label: bus.model.description_of_equations for bus in self.all_buses},
                'objective': 'MISSING AFTER REWORK',
                'effects': self.effect_collection.model.description_of_equations,
                'flows': {flow.label_full: flow.model.description_of_equations
                          for comp in self.components for flow in (comp.inputs + comp.outputs)},
                'others': {element.label: element.model.description_of_equations for element in self.other_elements}}

    def description_of_variables(self) -> Dict:
        return {'comps': {comp.label: comp.model.description_of_variables + [{flow.label: flow.model.description_of_variables
                                                                         for flow in comp.inputs + comp.outputs}]
                          for comp in self.components},
                'buses': {bus.label: bus.model.description_of_variables for bus in self.all_buses},
                'objective': 'MISSING AFTER REWORK',
                'effects': self.effect_collection.model.description_of_variables,
                'others': {element.label: element.model.description_of_variables for element in self.other_elements}
                }

    def description_of_variables_unstructured(self) -> List:
        return [var.description() for var in self.model.variables]

    def print_equations(self) -> str:
        return (f'\n'
                f'{"":#^80}\n'
                f'{" Equations of FlowSystem ":#^80}\n\n'
                f'{yaml.dump(self.description_of_equations(), default_flow_style=False, allow_unicode=True)}')

    def print_variables(self) -> str:
        return (f'\n'
                f'{"":#^80}\n'
                f'{" Variables of FlowSystem ":#^80}\n\n'
                f'{" a) as list ":#^80}\n\n'
                f'{yaml.dump(self.description_of_variables_unstructured())}\n\n'
                f'{" b) structured ":#^80}\n\n'
                f'{yaml.dump(self.description_of_variables())}')

    # Datenzeitreihe auf Basis gegebener time_indices aus globaler extrahieren:
    def get_time_data_from_indices(
            self,
            time_indices: Union[List[int], range]
    ) -> Tuple[np.ndarray[np.datetime64],np.ndarray[np.datetime64], np.ndarray[np.float64], np.float64]:
        # if time_indices is None, dann alle : time_indices = range(length(self.time_series))
        # Zeitreihen:
        time_series = self.time_series[time_indices]
        # next timestamp as endtime:
        endTime = self.time_series_with_end[time_indices[-1] + 1]
        time_series_with_end = np.append(time_series, endTime)

        # Zeitdifferenz:
        #              zweites bis Letztes            - erstes bis Vorletztes
        dt = time_series_with_end[1:] - time_series_with_end[0:-1]
        dt_in_hours = dt / np.timedelta64(1, 'h')
        # dt_in_hours    = dt.total_seconds() / 3600
        dt_in_hours_total = sum(dt_in_hours)  # Gesamtzeit
        return (time_series, time_series_with_end, dt_in_hours, dt_in_hours_total)

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
    def all_time_series_in_elements(self) -> List[TimeSeries]:
        element: Element
        all_TS = []
        for element in self.all_first_level_elements_with_flows:
            all_TS += element.TS_list
        return all_TS

    @property
    def all_buses(self) -> Set[Bus]:
        return {flow.bus for flow in self.all_flows}

    @property
    def all_elements(self) -> List[Element]:
        return (self.components + self.effect_collection.effects + self.temporary_elements +
                list(self.other_elements) + list(self.all_flows) + list(self.all_buses))
