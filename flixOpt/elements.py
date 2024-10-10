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
from flixOpt.structure import Element, SystemModel

logger = logging.getLogger('flixOpt')

EffectDict = Dict[Optional['Effect'], Numeric_TS]
EffectDictInvest = Dict[Optional['Effect'], Skalar]

EffectValues = Optional[Union[Numeric_TS, EffectDict]]  # Datatype for User Input
EffectValuesInvest = Optional[Union[Skalar, EffectDictInvest]]  # Datatype for User Input

EffectTimeSeries = Dict[Optional['Effect'], TimeSeries]  # Final Internal Data Structure


def _create_time_series(label: str, data: Optional[Numeric_TS], element: Element) -> TimeSeries:
    """Creates a TimeSeries from Numeric Data and adds it to the list of time_series of an Element"""
    time_series = TimeSeries(label=label, data=data)
    element.TS_list.append(time_series)
    return time_series


class Effect(Element):
    """
    Effect, i.g. costs, CO2 emissions, area, ...
    Components, FLows, and so on can contribute to an Effect. One Effect is chosen as the Objective of the Optimization
    """

    def __init__(self,
                 label: str,
                 unit: str,
                 description: str,
                 is_standard: bool = False,
                 is_objective: bool = False,
                 specific_share_to_other_effects_operation: Optional[EffectValues] = None,
                 specific_share_to_other_effects_invest: Optional[EffectValuesInvest] = None,
                 minimum_operation: Optional[Skalar] = None,
                 maximum_operation: Optional[Skalar] = None,
                 minimum_invest: Optional[Skalar] = None,
                 maximum_invest: Optional[Skalar] = None,
                 minimum_operation_per_hour: Optional[Numeric_TS] = None,
                 maximum_operation_per_hour: Optional[Numeric_TS] = None,
                 minimum_total: Optional[Skalar] = None,
                 maximum_total: Optional[Skalar] = None):
        """
        Parameters
        ----------
        label : str
            name
        unit : str
            unit of effect, i.g. €, kg_CO2, kWh_primaryEnergy
        description : str
            long name
        is_standard : boolean, optional
            true, if Standard-Effect (for direct input of value without effect (alternatively to dict)) , else false
        is_objective : boolean, optional
            true, if optimization target
        specific_share_to_other_effects_operation : {effectType: TS, ...}, i.g. 180 €/t_CO2, input as {costs: 180}, optional
            share to other effects (only operation)
        specific_share_to_other_effects_invest : {effectType: TS, ...}, i.g. 180 €/t_CO2, input as {costs: 180}, optional
            share to other effects (only invest).
        minimum_operation : scalar, optional
            minimal sum (only operation) of the effect
        maximum_operation : scalar, optional
            maximal sum (nur operation) of the effect.
        minimum_operation_per_hour : scalar or TS
            maximum value per hour (only operation) of effect (=sum of all effect-shares) for each timestep!
        maximum_operation_per_hour : scalar or TS
            minimum value per hour (only operation) of effect (=sum of all effect-shares) for each timestep!
        minimum_invest : scalar, optional
            minimal sum (only invest) of the effect
        maximum_invest : scalar, optional
            maximal sum (only invest) of the effect
        minimum_total : sclalar, optional
            min sum of effect (invest+operation).
        maximum_total : scalar, optional
            max sum of effect (invest+operation).
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().__init__(label)
        self.label = label
        self.unit = unit
        self.description = description
        self.is_standard = is_standard
        self.is_objective = is_objective
        self.specific_share_to_other_effects_operation = specific_share_to_other_effects_operation
        self.specific_share_to_other_effects_invest = specific_share_to_other_effects_invest
        self.minimum_operation = minimum_operation
        self.maximum_operation = maximum_operation
        self.minimum_operation_per_hour = minimum_operation_per_hour
        self.maximum_operation_per_hour = maximum_operation_per_hour
        self.minimum_invest = minimum_invest
        self.maximum_invest = maximum_invest
        self.minimum_total = minimum_total
        self.maximum_total = maximum_total

    def transform_to_time_series(self):
        self.minimum_operation_per_hour = _create_time_series(
            f'{self.label_full}_minimum_operation_per_hour', self.minimum_operation_per_hour, self)
        self.maximum_operation_per_hour = _create_time_series(
            f'{self.label_full}_maximum_operation_per_hour', self.maximum_operation_per_hour, self)

        self.specific_share_to_other_effects_operation = _effect_values_to_ts(
            f'{self.label_full}_specific_share_to_other_effects_operation',
            self.specific_share_to_other_effects_operation, self)

    def __str__(self):
        objective = "Objective" if self.is_objective else ""
        standart = "Standardeffect" if self.is_standard else ""
        op_sum = f"OperationSum={self.minimum_operation}-{self.maximum_operation}" \
            if self.minimum_operation is not None or self.maximum_operation is not None else ""
        inv_sum = f"InvestSum={self.minimum_invest}-{self.maximum_invest}" \
            if self.minimum_invest is not None or self.maximum_invest is not None else ""
        tot_sum = f"TotalSum={self.minimum_total}-{self.maximum_total}" \
            if self.minimum_total is not None or self.maximum_total is not None else ""
        label_unit = f"{self.label} [{self.unit}]:"
        desc = f"({self.description})"
        shares_op = f"Operation Shares={self.specific_share_to_other_effects_operation}" \
            if self.specific_share_to_other_effects_operation != {} else ""
        shares_inv = f"Invest Shares={self.specific_share_to_other_effects_invest}" \
            if self.specific_share_to_other_effects_invest != {} else ""

        all_relevant_parts = [info for info in [objective, tot_sum, inv_sum, op_sum, shares_inv, shares_op, standart, desc ] if info != ""]

        full_str =f"{label_unit} {', '.join(all_relevant_parts)}"

        return f"<{self.__class__.__name__}> {full_str}"


def _as_effect_dict(effect_values: EffectValues) -> Optional[EffectDict]:
    """
    Converts effect values into a dictionary. If a scalar value is provided, it is associated with a standard effect type.

    Examples
    --------
    If costs are given without specifying the effect, the standard effect is used (see class Effect):
      costs = 20                        -> {None: 20}
      costs = None                      -> no change
      costs = {effect1: 20, effect2: 0.3} -> no change

    Parameters
    ----------
    effect_values : None, int, float, TimeSeries, or dict
        The effect values to convert can be a scalar, a TimeSeries, or a dictionary with an effectas key

    Returns
    -------
    dict or None
        Converted values in from of dict with either None or Effect as key. if values is None, None is returned
    """
    if isinstance(effect_values, dict):
        return effect_values
    elif effect_values is None:
        return None
    else:
        return {None: effect_values}


def _effect_values_to_ts(label: str, effect_dict: EffectDict, element: Element) -> Optional[EffectTimeSeries]:
    """
    Transforms values in a dictionary to instances of TimeSeries.

    Parameters
    ----------
    label : str
        The name of the parameter. (the effect_label gets added)
    effect_dict : dict
        A dictionary with effect-value pairs.
    element : object
        The owner object where TimeSeries belongs to.

    Returns
    -------
    dict
        A dictionary with Effects (or None {= standard effect}) as keys and TimeSeries instances as values. On
    """
    if effect_dict is None:
        return None

    transformed_dict = {}
    for effect, values in effect_dict.items():
        if not isinstance(values, TimeSeries):
            subname = 'standard' if effect is None else effect.label
            transformed_dict[effect] = _create_time_series(f"{label}_{subname}", values, element)
    return transformed_dict


class EffectCollection(Element):
    """
    Handling all Effects
    """

    def __init__(self, label: str):
        super().__init__(label)
        self.effects: List[Effect] = []

    def add_effect(self, effect: Effect) -> None:
        if effect.is_standard and self.standard_effect is not None:
            raise Exception(f'A standard-effect already exists! ({self.standard_effect.label=})')
        if effect.is_objective and self.objective_effect is not None:
            raise Exception(f'A objective-effect already exists! ({self.objective_effect.label=})')
        if effect in self.effects:
            raise Exception(f'Effect already added! ({effect.label=})')
        if effect.label in [existing_effect.label for existing_effect in self.effects]:
            raise Exception(f'Effect with label "{effect.label=}" already added!')
        self.effects.append(effect)

    @property
    def standard_effect(self) -> Optional[Effect]:
        for effect in self.effects:
            if effect.is_standard:
                return effect

    @property
    def objective_effect(self) -> Optional[Effect]:
        for effect in self.effects:
            if effect.is_objective:
                return effect


class Component(Element):
    """
    basic component class for all components
    """
    def __init__(self,
                 label: str,
                 inputs: List['Flow'],
                 outputs: List['Flow'],
                 on_off_parameters: OnOffParameters):
        """ Old Docstring"""
        super().__init__(label)
        self.inputs = inputs
        self.outputs = outputs
        self.on_off_parameters = on_off_parameters

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

    def _register_component_in_flows(self) -> None:
        for flow in self.inputs + self.outputs:
            flow.comp = self

    def _register_flows_in_bus(self) -> None:
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

    def transform_to_time_series(self):
        self.excess_penalty_per_flow_hour = _create_time_series(f'{self.label_full}_relative_minimum',
                                                                self.excess_penalty_per_flow_hour, self)

    def add_input(self, flow) -> None:
        flow: Flow
        self.inputs.append(flow)

    def add_output(self, flow) -> None:
        flow: Flow
        self.outputs.append(flow)

    def _plausibility_test(self) -> None:
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

        self._plausibility_test()

    def transform_to_time_series(self):
        self.relative_minimum = _create_time_series(f'{self.label_full}_relative_minimum', self.relative_minimum, self)
        self.relative_maximum = _create_time_series(f'{self.label_full}_relative_maximum', self.relative_maximum, self)
        self.fixed_relative_value = _create_time_series(f'{self.label_full}_fixed_relative_value', self.fixed_relative_value, self)
        self.effects_per_flow_hour = _effect_values_to_ts(f'{self.label_full}_effects_per_flow_hour', self.effects_per_flow_hour, self)
        # TODO: self.on_off_parameters ??

    def _plausibility_test(self) -> None:
        # TODO: Incorporate into Variable? (Lower_bound can not be greater than upper bound
        if np.any(self.relative_minimum > self.relative_maximum):
            raise Exception(self.label_full + ': Take care, that relative_minimum <= relative_maximum!')

        if self.size == Flow._default_size and self.size_is_fixed:  #Default Size --> Most likely by accident
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
