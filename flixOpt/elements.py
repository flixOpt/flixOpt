# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:40:23 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import textwrap
from typing import List, Set, Tuple, Dict, Union, Optional, Literal, TYPE_CHECKING
import logging

import numpy as np

from flixOpt import utils
from flixOpt.math_modeling import Variable, VariableTS, Equation
from flixOpt.core import TimeSeries, Numeric, Numeric_TS, Skalar
from flixOpt.interface import InvestParameters, OnOffParameters
from flixOpt.modeling import OnOffModel, InvestmentModel, PreventSimultaneousUsageModel
from flixOpt.structure import SystemModel, Element, ElementModel, _create_time_series, create_equation, create_variable, \
    create_ts_variable
from flixOpt.effects import EffectValues, _effect_values_to_ts, EffectCollectionModel

logger = logging.getLogger('flixOpt')


class Component(Element):
    """
    basic component class for all components
    """
    def __init__(self,
                 label: str,
                 inputs: Optional[List['Flow']] = None,
                 outputs: Optional[List['Flow']] = None,
                 on_off_parameters: Optional[OnOffParameters] = None,
                 prevent_simultaneous_flows: bool = False):
        """ Old Docstring"""
        super().__init__(label)
        self.inputs: List['Flow'] = inputs or []
        self.outputs: List['Flow'] = outputs or []
        self.on_off_parameters = on_off_parameters
        self.prevent_simultaneous_flows = prevent_simultaneous_flows

    def create_model(self) -> 'ComponentModel':
        self.model = ComponentModel(self)
        return self.model

    def __str__(self):
        # Representing inputs and outputs by their labels
        inputs_str = ",\n".join([flow.__str__() for flow in self.inputs])
        outputs_str = ",\n".join([flow.__str__() for flow in self.outputs])
        inputs_str = f"inputs=\n{textwrap.indent(inputs_str, ' ' * 3)}" if self.inputs != [] else "inputs=[]"
        outputs_str = f"outputs=\n{textwrap.indent(outputs_str, ' ' * 3)}" if self.outputs != [] else "outputs=[]"

        remaining_data = {
            key: value for key, value in self.__dict__.items()
            if value and
               not isinstance(value, Flow) and key in self.get_init_args() and key != "label"
        }

        remaining_data_keys = sorted(remaining_data.keys())
        remaining_data_values = [remaining_data[key] for key in remaining_data_keys]

        remaining_data_str = ""
        for key, value in zip(remaining_data_keys, remaining_data_values):
            if hasattr(value, '__str__'):
                remaining_data_str += f"{key}: {value}\n"
            elif hasattr(value, '__repr__'):
                remaining_data_str += f"{key}: {repr(value)}\n"
            else:
                remaining_data_str += f"{key}: {value}\n"

        str_desc = (f"<{self.__class__.__name__}> {self.label}:\n"
                    f"{textwrap.indent(inputs_str, ' ' * 3)}\n"
                    f"{textwrap.indent(outputs_str, ' ' * 3)}\n"
                    f"{textwrap.indent(remaining_data_str, ' ' * 3)}"
                    )

        return str_desc

    def register_component_in_flows(self) -> None:
        for flow in self.inputs + self.outputs:
            flow.comp = self

    def register_flows_in_bus(self) -> None:
        for flow in self.inputs:
            flow.bus.add_output(flow)
        for flow in self.outputs:
            flow.bus.add_input(flow)


class Bus(Element):
    """
    realizing balance of all linked flows
    (penalty flow is excess can be activated)
    """

    def __init__(self,
                 label: str,
                 excess_penalty_per_flow_hour: Optional[Numeric_TS] = 1e5):
        """
        Parameters
        ----------
        label : str
            name.
        excess_penalty_per_flow_hour : none or scalar, array or TimeSeriesRaw
            excess costs / penalty costs (bus balance compensation)
            (none/ 0 -> no penalty). The default is 1e5.
            (Take care: if you use a timeseries (no scalar), timeseries is aggregated if calculation_type = aggregated!)
        """
        super().__init__(label)
        self.excess_penalty_per_flow_hour = excess_penalty_per_flow_hour
        self.inputs: List[Flow] = []
        self.outputs: List[Flow] = []

    def create_model(self) -> 'BusModel':
        self.model = BusModel(self)
        return self.model

    def transform_to_time_series(self):
        self.excess_penalty_per_flow_hour = _create_time_series(f'{self.label_full}_relative_minimum',
                                                                self.excess_penalty_per_flow_hour, self)

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
    '''
    flows are inputs and outputs of components
    '''

    # static var:
    _default_size = 1e9  # Großer Gültigkeitsbereich als Standard

    def __init__(self,
                 label: str,
                 bus: Bus = None,  # TODO: Is this for sure Optional?
                 size: Union[Skalar, InvestParameters] = _default_size,
                 fixed_relative_value: Optional[Numeric_TS] = None,  # TODO: Rename?
                 relative_minimum: Numeric_TS = 0,
                 relative_maximum: Numeric_TS = 1,
                 effects_per_flow_hour: Optional[EffectValues] = None,
                 can_be_off: Optional[OnOffParameters] = None,
                 flow_hours_total_max: Optional[Skalar] = None,
                 flow_hours_total_min: Optional[Skalar] = None,
                 load_factor_min: Optional[Skalar] = None,
                 load_factor_max: Optional[Skalar] = None):
        """
        Parameters
        ----------
        label : str
            name of flow
        bus : Bus, optional
            bus to which flow is linked
        relative_minimum : scalar, array, TimeSeriesRaw, optional
            min value is relative_minimum multiplied by size
        relative_maximum : scalar, array, TimeSeriesRaw, optional
            max value is relative_maximum multiplied by size. If size = max then relative_maximum=1
        size : scalar. None if is a nominal value is a opt-variable, optional
            nominal value/ invest size (linked to relative_minimum, relative_maximum and others).
            i.g. kW, area, volume, pieces,
            möglichst immer so stark wie möglich einschränken
            (wg. Rechenzeit bzw. Binär-Ungenauigkeits-Problem!)
        load_factor_min : scalar, optional
            minimal load factor  general: avg Flow per nominalVal/investSize
            (e.g. boiler, kW/kWh=h; solarthermal: kW/m²;
             def: :math:`load\_factor:= sumFlowHours/ (nominal\_val \cdot \Delta t_{tot})`
        load_factor_max : scalar, optional
            maximal load factor (see minimal load factor)
        effects_per_flow_hour : scalar, array, TimeSeriesRaw, optional
            operational costs, costs per flow-"work"
        can_be_off : OnOffParameters, optional
            flow can be "off", i.e. be zero (only relevant if relative_minimum > 0)
            Then a binary var "on" is used.. Further, several other restrictions and effects can be modeled
            through this On and Off State (See OnOffParameters)
        flow_hours_total_max : TYPE, optional
            maximum flow-hours ("flow-work")
            (if size is not const, maybe load_factor_max fits better for you!)
        flow_hours_total_min : TYPE, optional
            minimum flow-hours ("flow-work")
            (if size is not const, maybe load_factor_min fits better for you!)
        fixed_relative_value : scalar, array, TimeSeriesRaw, optional
            fixed relative values for flow (if given).
            val(t) := fixed_relative_value(t) * size(t)
            With this value, the flow-value is no opt-variable anymore;
            (relative_minimum u. relative_maximum are making sense anymore)
            used for fixed load profiles, i.g. heat demand, wind-power, solarthermal
            If the load-profile is just an upper limit, use relative_maximum instead.
        """
        super().__init__(label)
        self.size = size
        self.relative_minimum = relative_minimum
        self.relative_maximum = relative_maximum
        self.fixed_relative_value = fixed_relative_value

        self.load_factor_min = load_factor_min
        self.load_factor_max = load_factor_max
        # self.positive_gradient = TimeSeries('positive_gradient', positive_gradient, self)
        self.effects_per_flow_hour = effects_per_flow_hour
        self.flow_hours_total_max = flow_hours_total_max
        self.flow_hours_total_min = flow_hours_total_min
        self.on_off_parameters = can_be_off

        self.bus = bus
        self.comp: Optional[Component] = None

        self._plausibility_checks()

    def create_model(self) -> 'FlowModel':
        self.model = FlowModel(self)
        return self.model

    def transform_to_time_series(self):
        self.relative_minimum = _create_time_series(f'{self.label_full}_relative_minimum', self.relative_minimum, self)
        self.relative_maximum = _create_time_series(f'{self.label_full}_relative_maximum', self.relative_maximum, self)
        self.fixed_relative_value = _create_time_series(f'{self.label_full}_fixed_relative_value', self.fixed_relative_value, self)
        self.effects_per_flow_hour = _effect_values_to_ts(f'{self.label_full}_effects_per_flow_hour', self.effects_per_flow_hour, self)
        # TODO: self.on_off_parameters ??

    def _plausibility_checks(self) -> None:
        # TODO: Incorporate into Variable? (Lower_bound can not be greater than upper bound
        if np.any(self.relative_minimum > self.relative_maximum):
            raise Exception(self.label_full + ': Take care, that relative_minimum <= relative_maximum!')

        if self.size == Flow._default_size and self.fixed_relative_value is not None:  #Default Size --> Most likely by accident
            raise Exception('Achtung: Wenn fixed_relative_value genutzt wird, muss zugehöriges size definiert werden, '
                            'da: value = fixed_relative_value * size!')

    def __str__(self):
        details = [
            f"bus={self.bus.label if self.bus else 'None'}",
            f"size={self.size.__str__() if isinstance(self.size, InvestParameters) else self.size}",
            f"relative_minimum={self.relative_minimum}",
            f"relative_maximum={self.relative_maximum}",
            f"fixed_relative_value={self.fixed_relative_value}" if self.fixed_relative_value else "",
            f"effects_per_flow_hour={self.effects_per_flow_hour}" if self.effects_per_flow_hour else "",
            f"on_off_parameters={self.on_off_parameters.__str__()}" if self.on_off_parameters else "",
        ]

        all_relevant_parts = [part for part in details if part != ""]

        full_str = f"{', '.join(all_relevant_parts)}"

        return f"<{self.__class__.__name__}> {self.label}: {full_str}"

    @property
    def label_full(self) -> str:
        # Wenn im Erstellungsprozess comp noch nicht bekannt:
        comp_label = 'unknownComp' if self.comp is None else self.comp.label
        return f'{comp_label}__{self.label}'  # z.B. für results_struct (deswegen auch _  statt . dazwischen)

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
    def __init__(self, element: Flow):
        super().__init__(element)
        self.element: Flow = element
        self.flow_rate: Optional[VariableTS] = None
        self.sum_flow_hours: Optional[Variable] = None

        self._on: Optional[OnOffModel] = None
        self._investment: Optional[InvestmentModel] = None

    def do_modeling(self, system_model: SystemModel):
        # eq relative_minimum(t) * size <= flow_rate(t) <= relative_maximum(t) * size
        if self.element.on_off_parameters is None and not isinstance(self.element.size, InvestParameters):
            if self.element.fixed_relative_value is None:
                fixed_flow_rate = None
            else:
                fixed_flow_rate = self.element.fixed_relative_value * self.element.size
            self.flow_rate = create_ts_variable('flow_rate', self, system_model.nr_of_time_steps,
                                                system_model,
                                                lower_bound=self.element.relative_minimum * self.element.size,
                                                upper_bound=self.element.relative_maximum * self.element.size,
                                                value=fixed_flow_rate)
        else:  # Bounds are created later and in sub_models
            self.flow_rate = create_ts_variable('flow_rate', self, system_model.nr_of_time_steps,
                                                system_model, lower_bound=0)

        # OnOff
        if self.element.on_off_parameters is not None:
            self._on = OnOffModel(self.element, self.element.on_off_parameters,
                                  [self.flow_rate],
                                  [self.flow_rate_bounds])
            self._on.do_modeling(system_model)
            self.sub_models.append(self._on)

        # Investment
        if isinstance(self.element.size, InvestParameters):
            self._investment = InvestmentModel(self.element, self.element.size,
                                               self.flow_rate,
                                               (self.element.relative_minimum, self.element.size.minimum_size),
                                               on_variable=self._on.on if self._on is not None else None)
            self._investment.do_modeling(system_model)
            self.sub_models.append(self._investment)

        # sumFLowHours
        self.sum_flow_hours = create_variable('sumFlowHours', self, 1, system_model,
                                              lower_bound=self.element.flow_hours_total_min,
                                              upper_bound=self.element.flow_hours_total_max)
        eq_sum_flow_hours = create_equation('sumFlowHours', self, system_model, 'eq')
        eq_sum_flow_hours.add_summand(self.flow_rate, system_model.dt_in_hours, as_sum=True)
        eq_sum_flow_hours.add_summand(self.sum_flow_hours, -1)

        self._create_bounds_for_load_factor(system_model)
        self._create_shares(system_model)

    def _create_shares(self, system_model: SystemModel):
        # Arbeitskosten:
        if self.element.effects_per_flow_hour is not None:
            system_model.effect_collection_model.add_share_to_operation(
                name=f'{self.element.label_full}__effects_per_flow_hour',
                variable=self.flow_rate,
                effect_values=self.element.effects_per_flow_hour,
                factor=system_model.dt_in_hours
            )

    def _create_bounds_for_load_factor(self, system_model: SystemModel):
        # TODO: Add Variable load_factor for better evaluation?

        # eq: var_sumFlowHours <= size * dt_tot * load_factor_max
        if self.element.load_factor_max is not None:
            flow_hours_per_size_max = system_model.dt_in_hours_total * self.element.load_factor_max
            eq_load_factor_max = create_equation('load_factor_max', self, system_model, 'ineq')
            eq_load_factor_max.add_summand(self.sum_flow_hours, 1)
            # if investment:
            if self._investment is not None:
                eq_load_factor_max.add_summand(self._investment.size, -1 * flow_hours_per_size_max)
            else:
                eq_load_factor_max.add_constant(self.element.size * flow_hours_per_size_max)

        #  eq: size * sum(dt)* load_factor_min <= var_sumFlowHours
        if self.element.load_factor_min is not None:
            flow_hours_per_size_min = system_model.dt_in_hours_total * self.element.load_factor_min
            eq_load_factor_min = create_equation('load_factor_min', self, system_model, 'ineq')
            eq_load_factor_min.add_summand(self.sum_flow_hours, 1)
            if self._investment is not None:
                eq_load_factor_min.add_summand(self._investment.size, flow_hours_per_size_min)
            else:
                eq_load_factor_min.add_constant(-1 * self.element.size * flow_hours_per_size_min)

    @property
    def flow_rate_bounds(self) -> Tuple[Numeric, Numeric]:
        if not isinstance(self.element.size, InvestParameters):
            return (self.element.relative_minimum * self.element.size,
                    self.element.relative_maximum * self.element.size)
        else:
            return (self.element.relative_minimum * self.element.size.minimum_size,
                    self.element.relative_maximum * self.element.size.maximum_size)


class BusModel(ElementModel):
    def __init__(self, element: Bus):
        super().__init__(element)
        self.element: Bus
        self.excess_input: Optional[VariableTS] = None
        self.excess_output: Optional[VariableTS] = None

    def do_modeling(self, system_model: SystemModel) -> None:
        self.element: Bus
        # inputs = outputs
        eq_bus_balance = create_equation('busBalance', self, system_model)
        for flow in self.element.inputs:
            eq_bus_balance.add_summand(flow.model.flow_rate, 1)
        for flow in self.element.outputs:
            eq_bus_balance.add_summand(flow.model.flow_rate, -1)

        # Fehlerplus/-minus:
        if self.element.with_excess:
            excess_penalty = np.multiply(system_model.dt_in_hours, self.element.excess_penalty_per_flow_hour)
            self.excess_input = create_ts_variable('excess_input', self, system_model.nr_of_time_steps,
                                                   system_model, lower_bound=0)
            self.excess_output = create_ts_variable('excess_output', self, system_model.nr_of_time_steps,
                                                    system_model, lower_bound=0)

            eq_bus_balance.add_summand(self.excess_output, -1)
            eq_bus_balance.add_summand(self.excess_input, 1)

            fx_collection = system_model.effect_collection_model

            fx_collection.add_share_to_penalty('penalty', self.element, self.excess_input, excess_penalty)
            fx_collection.add_share_to_penalty('penalty', self.element, self.excess_output, excess_penalty)


class ComponentModel(ElementModel):
    def __init__(self, element: Component):
        super().__init__(element)
        self.element: Component = element
        self._on: Optional[OnOffModel] = None

    def do_modeling(self, system_model: SystemModel):
        """ Initiates all FlowModels """
        all_flows = self.element.inputs + self.element.outputs
        if self.element.prevent_simultaneous_flows or self.element.on_off_parameters:
            for flow in all_flows:
                if flow.on_off_parameters is None:
                    flow.on_off_parameters = OnOffParameters(force_on=True)
                else:
                    flow.on_off_parameters.force_on = True

        self.sub_models.extend([flow.create_model() for flow in all_flows])
        for sub_model in self.sub_models:
            sub_model.do_modeling(system_model)

        if self.element.on_off_parameters:
            flow_rates: List[VariableTS] = [flow.model.flow_rate for flow in all_flows]
            bounds: List[Tuple[Numeric, Numeric]] = [flow.model.flow_rate_bounds for flow in all_flows]
            self._on = OnOffModel(self.element, self.element.on_off_parameters,
                                  flow_rates, bounds)
            self.sub_models.append(self._on)
            self._on.do_modeling(system_model)

        if self.element.prevent_simultaneous_flows:
            # Simultanious Useage --> Only One FLow is On at a time, but needs a Binary for every flow
            on_variables = [flow.model._on.on for flow in all_flows]
            simultaneous_use = PreventSimultaneousUsageModel(self.element, on_variables)
            self.sub_models.append(simultaneous_use)
            simultaneous_use.do_modeling(system_model)
