"""
This module contains the basic elements of the flixOpt framework.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import linopy
import pandas as pd

from .config import CONFIG
from .core import Numeric, Numeric_TS, Skalar
from .effects import EffectValuesUser, effect_values_to_time_series
from .features import InvestmentModel, OnOffModel, PreventSimultaneousUsageModel
from .interface import InvestParameters, OnOffParameters
from .structure import (
    Element,
    ElementModel,
    SystemModel,
)

if TYPE_CHECKING:
    from .flow_system import FlowSystem


logger = logging.getLogger('flixOpt')


class Component(Element):
    """
    basic component class for all components
    """

    def __init__(
        self,
        label: str,
        inputs: Optional[List['Flow']] = None,
        outputs: Optional[List['Flow']] = None,
        on_off_parameters: Optional[OnOffParameters] = None,
        prevent_simultaneous_flows: Optional[List['Flow']] = None,
        meta_data: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        label : str
            name.
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        inputs : input flows.
        outputs : output flows.
        on_off_parameters: Information about on and off state of Component.
            Component is On/Off, if all connected Flows are On/Off.
            Induces On-Variable in all FLows!
            See class OnOffParameters.
        prevent_simultaneous_flows: Define a Group of Flows. Only one them can be on at a time.
            Induces On-Variable in all FLows!
        """
        super().__init__(label, meta_data=meta_data)
        self.inputs: List['Flow'] = inputs or []
        self.outputs: List['Flow'] = outputs or []
        self.on_off_parameters = on_off_parameters
        self.prevent_simultaneous_flows: List['Flow'] = prevent_simultaneous_flows or []

        self.flows: Dict[str, Flow] = {flow.label: flow for flow in self.inputs + self.outputs}

    def create_model(self, model: SystemModel) -> 'ComponentModel':
        self.model = ComponentModel(model, self)
        return self.model

    def transform_data(self, flow_system: 'FlowSystem') -> None:
        if self.on_off_parameters is not None:
            self.on_off_parameters.transform_data(flow_system.timesteps, flow_system.periods, self)

    def register_component_in_flows(self) -> None:
        for flow in self.inputs + self.outputs:
            flow.comp = self

    def register_flows_in_bus(self) -> None:
        for flow in self.inputs:
            flow.bus.add_output(flow)
        for flow in self.outputs:
            flow.bus.add_input(flow)

    def infos(self, use_numpy=True, use_element_label=False) -> Dict:
        infos = super().infos(use_numpy, use_element_label)
        infos['inputs'] = [flow.infos(use_numpy, use_element_label) for flow in self.inputs]
        infos['outputs'] = [flow.infos(use_numpy, use_element_label) for flow in self.outputs]
        return infos


class Bus(Element):
    """
    realizing balance of all linked flows
    (penalty flow is excess can be activated)
    """

    def __init__(
        self, label: str, excess_penalty_per_flow_hour: Optional[Numeric_TS] = 1e5, meta_data: Optional[Dict] = None
    ):
        """
        Parameters
        ----------
        label : str
            name.
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        excess_penalty_per_flow_hour : none or scalar, array or TimeSeriesData
            excess costs / penalty costs (bus balance compensation)
            (none/ 0 -> no penalty). The default is 1e5.
            (Take care: if you use a timeseries (no scalar), timeseries is aggregated if calculation_type = aggregated!)
        """
        super().__init__(label, meta_data=meta_data)
        self.excess_penalty_per_flow_hour = excess_penalty_per_flow_hour
        self.inputs: List[Flow] = []
        self.outputs: List[Flow] = []

    def create_model(self, model: SystemModel) -> 'BusModel':
        self.model = BusModel(model, self)
        return self.model

    def transform_data(self, flow_system: 'FlowSystem'):
        self.excess_penalty_per_flow_hour = self._create_time_series(
            'excess_penalty_per_flow_hour', self.excess_penalty_per_flow_hour, flow_system.timesteps, flow_system.periods
        )

    def add_input(self, flow) -> None:
        flow: Flow
        self.inputs.append(flow)

    def add_output(self, flow) -> None:
        flow: Flow
        self.outputs.append(flow)

    def _plausibility_checks(self) -> None:
        if self.excess_penalty_per_flow_hour == 0:
            logger.warning(f'In Bus {self.label}, the excess_penalty_per_flow_hour is 0. Use "None" or a value > 0.')

    @property
    def with_excess(self) -> bool:
        return False if self.excess_penalty_per_flow_hour is None else True


class Connection:
    # input/output-dock (TODO:
    # -> wäre cool, damit Komponenten auch auch ohne Knoten verbindbar
    # input wären wie Flow,aber statt bus : connectsTo -> hier andere Connection oder aber Bus (dort keine Connection, weil nicht notwendig)

    def __init__(self):
        raise NotImplementedError()


class Flow(Element):
    """
    flows are inputs and outputs of components
    """

    def __init__(
        self,
        label: str,
        bus: Bus,
        size: Union[Skalar, InvestParameters] = None,
        fixed_relative_profile: Optional[Numeric_TS] = None,
        relative_minimum: Numeric_TS = 0,
        relative_maximum: Numeric_TS = 1,
        effects_per_flow_hour: EffectValuesUser = None,
        on_off_parameters: Optional[OnOffParameters] = None,
        flow_hours_total_max: Optional[Skalar] = None,
        flow_hours_total_min: Optional[Skalar] = None,
        load_factor_min: Optional[Skalar] = None,
        load_factor_max: Optional[Skalar] = None,
        previous_flow_rate: Optional[Numeric] = None,
        meta_data: Optional[Dict] = None,
    ):
        r"""
        Parameters
        ----------
        label : str
            name of flow
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
        bus : Bus, optional
            bus to which flow is linked
        size : scalar, InvestmentParameters, optional
            size of the flow. If InvestmentParameters is used, size is optimized.
            If size is None, a default value is used.
        relative_minimum : scalar, array, TimeSeriesData, optional
            min value is relative_minimum multiplied by size
        relative_maximum : scalar, array, TimeSeriesData, optional
            max value is relative_maximum multiplied by size. If size = max then relative_maximum=1
        load_factor_min : scalar, optional
            minimal load factor  general: avg Flow per nominalVal/investSize
            (e.g. boiler, kW/kWh=h; solarthermal: kW/m²;
             def: :math:`load\_factor:= sumFlowHours/ (nominal\_val \cdot \Delta t_{tot})`
        load_factor_max : scalar, optional
            maximal load factor (see minimal load factor)
        effects_per_flow_hour : scalar, array, TimeSeriesData, optional
            operational costs, costs per flow-"work"
        on_off_parameters : OnOffParameters, optional
            If present, flow can be "off", i.e. be zero (only relevant if relative_minimum > 0)
            Therefore a binary var "on" is used. Further, several other restrictions and effects can be modeled
            through this On/Off State (See OnOffParameters)
        flow_hours_total_max : TYPE, optional
            maximum flow-hours ("flow-work")
            (if size is not const, maybe load_factor_max fits better for you!)
        flow_hours_total_min : TYPE, optional
            minimum flow-hours ("flow-work")
            (if size is not const, maybe load_factor_min fits better for you!)
        fixed_relative_profile : scalar, array, TimeSeriesData, optional
            fixed relative values for flow (if given).
            val(t) := fixed_relative_profile(t) * size(t)
            With this value, the flow_rate is no opt-variable anymore;
            (relative_minimum u. relative_maximum are making sense anymore)
            used for fixed load profiles, i.g. heat demand, wind-power, solarthermal
            If the load-profile is just an upper limit, use relative_maximum instead.
        previous_flow_rate : scalar, array, optional
            previous flow rate of the component.
        """
        super().__init__(label, meta_data=meta_data)
        self.size = size or CONFIG.modeling.BIG  # Default size
        self.relative_minimum = relative_minimum
        self.relative_maximum = relative_maximum
        self.fixed_relative_profile = fixed_relative_profile

        self.load_factor_min = load_factor_min
        self.load_factor_max = load_factor_max
        # self.positive_gradient = TimeSeries('positive_gradient', positive_gradient, self)
        self.effects_per_flow_hour = effects_per_flow_hour if effects_per_flow_hour is not None else {}
        self.flow_hours_total_max = flow_hours_total_max
        self.flow_hours_total_min = flow_hours_total_min
        self.on_off_parameters = on_off_parameters

        self.previous_flow_rate = previous_flow_rate

        self.bus = bus
        self.comp: Optional[Component] = None

        self._plausibility_checks()

    def create_model(self, model: SystemModel) -> 'FlowModel':
        self.model = FlowModel(model, self)
        return self.model

    def transform_data(self, flow_system: 'FlowSystem'):
        self.relative_minimum = self._create_time_series('relative_minimum', self.relative_minimum, flow_system.timesteps, flow_system.periods)
        self.relative_maximum = self._create_time_series('relative_maximum', self.relative_maximum, flow_system.timesteps, flow_system.periods)
        self.fixed_relative_profile = self._create_time_series('fixed_relative_profile', self.fixed_relative_profile, flow_system.timesteps, flow_system.periods)
        self.effects_per_flow_hour = effect_values_to_time_series('per_flow_hour', self.effects_per_flow_hour, self, flow_system.timesteps, flow_system.periods)
        if self.on_off_parameters is not None:
            self.on_off_parameters.transform_data(flow_system, self)
        if isinstance(self.size, InvestParameters):
            self.size.transform_data(flow_system)

    def infos(self, use_numpy=True, use_element_label=False) -> Dict:
        infos = super().infos(use_numpy, use_element_label)
        infos['is_input_in_component'] = self.is_input_in_comp
        return infos

    def _plausibility_checks(self) -> None:
        # TODO: Incorporate into Variable? (Lower_bound can not be greater than upper bound
        if np.any(self.relative_minimum > self.relative_maximum):
            raise Exception(self.label_full + ': Take care, that relative_minimum <= relative_maximum!')

        if (
            self.size == CONFIG.modeling.BIG and self.fixed_relative_profile is not None
        ):  # Default Size --> Most likely by accident
            logger.warning(
                f'Flow "{self.label}" has no size assigned, but a "fixed_relative_profile". '
                f'The default size is {CONFIG.modeling.BIG}. As "flow_rate = size * fixed_relative_profile", '
                f'the resulting flow_rate will be very high. To fix this, assign a size to the Flow {self}.'
            )

    @property
    def label_full(self) -> str:
        # Wenn im Erstellungsprozess comp noch nicht bekannt:
        comp_label = 'unknownComp' if self.comp is None else self.comp.label
        return f'{self.label} ({comp_label})'

    @property  # Richtung
    def is_input_in_comp(self) -> bool:
        return True if self in self.comp.inputs else False

    @property
    def size_is_fixed(self) -> bool:
        # Wenn kein InvestParameters existiert --> True; Wenn Investparameter, den Wert davon nehmen
        return False if (isinstance(self.size, InvestParameters) and self.size.fixed_size is None) else True

    @property
    def invest_is_optional(self) -> bool:
        # Wenn kein InvestParameters existiert: # Investment ist nicht optional -> Keine Variable --> False
        return False if (isinstance(self.size, InvestParameters) and not self.size.optional) else True


class FlowModel(ElementModel):
    def __init__(self, model: SystemModel, element: Flow):
        super().__init__(model, element)
        self.element: Flow = element
        self.flow_rate: Optional[linopy.Variable] = None
        self.total_flow_hours: Optional[linopy.Variable] = None

        self.on_off: Optional[OnOffModel] = None
        self._investment: Optional[InvestmentModel] = None

    def do_modeling(self, system_model: SystemModel):
        # eq relative_minimum(t) * size <= flow_rate(t) <= relative_maximum(t) * size
        self.flow_rate = self.add(
            system_model.add_variables(
                lower=self.absolute_flow_rate_bounds[0] if self.element.on_off_parameters is None else 0,
                upper=self.absolute_flow_rate_bounds[1] if self.element.on_off_parameters is None else np.inf,
                coords=system_model.coords,
                name=f'{self.label_full}__flow_rate'
            ),
            'flow_rate'
        )
        if self.element.fixed_relative_profile is not None:
            self.add(
                system_model.add_constraints(
                    self.flow_rate == self.element.fixed_relative_profile.active_data,
                    name=f'{self.element.label}_fix_flow_rate'
                ),
                'flow_rate (fix)'
            )

        # OnOff
        if self.element.on_off_parameters is not None:
            self.on_off = self.add(
                OnOffModel(
                    self.element, self.element.on_off_parameters, [self.flow_rate], [self.absolute_flow_rate_bounds]
                ),
                'on_off'
            )
            self.on_off.do_modeling(system_model)

        # Investment
        if isinstance(self.element.size, InvestParameters):
            self._investment = self.add(
                InvestmentModel(
                    self.element,
                    self.element.size,
                    self.flow_rate,
                    self.relative_flow_rate_bounds,
                    fixed_relative_profile=self.fixed_relative_flow_rate,
                    on_variable=self.on_off.on if self.on_off is not None else None,
                ),
                'investment'
            )
            self._investment.do_modeling(system_model)

        self.total_flow_hours = self.add(
            system_model.add_variables(
                lower=self.element.flow_hours_total_min if self.element.flow_hours_total_min is not None else -np.inf,
                upper=self.element.flow_hours_total_max if self.element.flow_hours_total_max is not None else np.inf,
                coords=None,
                name=f'{self.element.label_full}__total_flow_hours'
            ),
            'total_flow_hours'
        )

        self.add(
            system_model.add_constraints(
                self.total_flow_hours == (self.flow_rate * system_model.hours_per_step).sum(),
                name=f'{self.element.label_full}__total_flow_hours'
            ),
            'total_flow_hours'
        )

        # Load factor
        self._create_bounds_for_load_factor(system_model)

        # Shares
        self._create_shares(system_model)

    def _create_shares(self, system_model: SystemModel):
        # Arbeitskosten:
        if self.element.effects_per_flow_hour != {}:
            system_model.effects.add_share_to_effects(
                system_model,
                name=self.label_full,  # Use the full label of the element
                expressions={
                    effect: self.flow_rate * system_model.hours_per_step * factor.active_data
                    for effect, factor in self.element.effects_per_flow_hour.items()
                },
                target='operation',
            )

    def _create_bounds_for_load_factor(self, system_model: SystemModel):
        # TODO: Add Variable load_factor for better evaluation?

        # eq: var_sumFlowHours <= size * dt_tot * load_factor_max
        if self.element.load_factor_max is not None:
            name_short = 'load_factor_max'
            name = f'{self.element.label_full}__{name_short}'
            flow_hours_per_size_max = system_model.hours_per_step * self.element.load_factor_max

            if self._investment is not None:
                self.add(
                    system_model.add_constraints(
                        self.total_flow_hours <= self._investment.size * flow_hours_per_size_max, name=name,
                    ),
                    name_short
                )
            else:
                self.add(
                    system_model.add_constraints(
                        self.total_flow_hours <= self.element.size * flow_hours_per_size_max, name=name,
                    ),
                    name_short
                )

        #  eq: size * sum(dt)* load_factor_min <= var_sumFlowHours
        if self.element.load_factor_min is not None:
            name_short = 'load_factor_min'
            name = f'{self.element.label_full}__{name_short}'
            flow_hours_per_size_in = system_model.hours_per_step * self.element.load_factor_min

            if self._investment is not None:
                self.add(
                    system_model.add_constraints(
                        self.total_flow_hours >= self._investment.size * flow_hours_per_size_in,
                        name=name
                    ),
                    name_short
                )
            else:
                self.add(
                    system_model.add_constraints(
                        self.total_flow_hours >= self.element.size * flow_hours_per_size_in,
                        name=name
                    ),
                    name_short
                )

    @property
    def with_investment(self) -> bool:
        """Checks if the element's size is investment-driven."""
        return isinstance(self.element.size, InvestParameters)

    @property
    def fixed_relative_flow_rate(self) -> Optional[np.ndarray]:
        """Returns a fixed flow rate if defined by the element."""
        if self.element.fixed_relative_profile is not None:
            return self.element.fixed_relative_profile.active_data
        return None

    @property
    def absolute_flow_rate_bounds(self) -> Tuple[Numeric, Numeric]:
        """Returns absolute flow rate bounds. Iportant for OnOffModel"""
        rel_min, rel_max = self.relative_flow_rate_bounds
        size = self.element.size
        if self.with_investment:
            if size.fixed_size is not None:
                return rel_min * size.fixed_size, rel_max * size.fixed_size
            return rel_min * size.minimum_size, rel_max * size.maximum_size
        else:
            return rel_min * size, rel_max * size

    @property
    def relative_flow_rate_bounds(self) -> Tuple[Numeric, Numeric]:
        """Returns relative flow rate bounds."""
        fixed_profile = self.element.fixed_relative_profile
        if fixed_profile is None:
            return self.element.relative_minimum.active_data, self.element.relative_maximum.active_data
        return (
            np.minimum(fixed_profile.active_data, self.element.relative_minimum.active_data),
            np.maximum(fixed_profile.active_data, self.element.relative_maximum.active_data),
        )


class BusModel(ElementModel):
    def __init__(self, model: SystemModel, element: Bus):
        super().__init__(model, element)
        self.element: Bus = element
        self.excess_input: Optional[linopy.Variable] = None
        self.excess_output: Optional[linopy.Variable] = None

    def do_modeling(self, system_model: SystemModel) -> None:
        # inputs == outputs
        inputs = sum([flow.model.flow_rate for flow in self.element.inputs])
        outputs = sum([flow.model.flow_rate for flow in self.element.outputs])
        eq_bus_balance = self.add(system_model.add_constraints(
            inputs == outputs,
            name=f'{self.label_full}__balance'
        ))

        # Fehlerplus/-minus:
        if self.element.with_excess:
            excess_penalty = np.multiply(
                system_model.hours_per_step, self.element.excess_penalty_per_flow_hour.active_data
            )
            self.excess_input = system_model.add_variables(
                lower=0, coords=system_model.coords, name=f'{self.label_full}__excess_input'
            )
            self.excess_output = system_model.add_variables(
                lower=0, coords=system_model.coords, name=f'{self.label_full}__excess_output'
            )
            eq_bus_balance.lhs -= -self.excess_input + self.excess_output

            system_model.effects.add_share_to_penalty(
                system_model, self.element.label_full, (self.excess_input * excess_penalty).sum()
            )
            system_model.effects.add_share_to_penalty(
                system_model, self.element.label_full, (self.excess_output * excess_penalty).sum()
            )


class ComponentModel(ElementModel):
    def __init__(self, model: SystemModel, element: Component):
        super().__init__(model, element)
        self.element: Component = element
        self.on_off: Optional[OnOffModel] = None

    def do_modeling(self, system_model: SystemModel):
        """Initiates all FlowModels"""
        all_flows = self.element.inputs + self.element.outputs
        if self.element.on_off_parameters:
            for flow in all_flows:
                if flow.on_off_parameters is None:
                    flow.on_off_parameters = OnOffParameters()

        if self.element.prevent_simultaneous_flows:
            for flow in self.element.prevent_simultaneous_flows:
                if flow.on_off_parameters is None:
                    flow.on_off_parameters = OnOffParameters()

        self.sub_models.extend([flow.create_model(system_model) for flow in all_flows])
        for sub_model in self.sub_models:
            sub_model.do_modeling(system_model)

        if self.element.on_off_parameters:
            flow_rates: List[linopy.Variable] = [flow.model.flow_rate for flow in all_flows]
            bounds: List[Tuple[Numeric, Numeric]] = [flow.model.absolute_flow_rate_bounds for flow in all_flows]
            self.on_off = OnOffModel(self.element, self.element.on_off_parameters, flow_rates, bounds)
            self.sub_models.append(self.on_off)
            self.on_off.do_modeling(system_model)

        if self.element.prevent_simultaneous_flows:
            # Simultanious Useage --> Only One FLow is On at a time, but needs a Binary for every flow
            on_variables = [flow.model.on_off.on for flow in self.element.prevent_simultaneous_flows]
            simultaneous_use = PreventSimultaneousUsageModel(self.element, on_variables)
            self.sub_models.append(simultaneous_use)
            simultaneous_use.do_modeling(system_model)
