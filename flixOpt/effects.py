# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:40:23 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

from typing import List, Tuple, Dict, Union, Optional, Literal, TYPE_CHECKING
import logging

import numpy as np

from flixOpt.math_modeling import Variable, Equation
from flixOpt.core import TimeSeries, Skalar, Numeric, Numeric_TS, as_effect_dict
from flixOpt.features import ShareAllocationModel
from flixOpt.structure import Element, ElementModel, SystemModel, _create_time_series

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from flixOpt.flow_system import FlowSystem
    from flixOpt.features import ComponentModel, BusModel


logger = logging.getLogger('flixOpt')


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
                 specific_share_to_other_effects_operation: Optional['EffectValues'] = None,
                 specific_share_to_other_effects_invest: Optional['EffectValuesInvest'] = None,
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
        self.specific_share_to_other_effects_operation: Union[EffectValues, EffectTimeSeries] = specific_share_to_other_effects_operation or {}
        self.specific_share_to_other_effects_invest: Union[EffectValuesInvest, EffectDictInvest] = specific_share_to_other_effects_invest or {}
        self.minimum_operation = minimum_operation
        self.maximum_operation = maximum_operation
        self.minimum_operation_per_hour: Numeric_TS = minimum_operation_per_hour
        self.maximum_operation_per_hour: Numeric_TS = maximum_operation_per_hour
        self.minimum_invest = minimum_invest
        self.maximum_invest = maximum_invest
        self.minimum_total = minimum_total
        self.maximum_total = maximum_total

        self._plausibility_checks()

    def _plausibility_checks(self) -> None:
        # Check circular loops in effects: (Effekte fügen sich gegenseitig Shares hinzu):
        #TODO: Improve checks!! Only most basic case covered...

        def error_str(effect_label: str, shareEffect_label: str):
            return (
                f'  {effect_label} -> has share in: {shareEffect_label}\n'
                f'  {shareEffect_label} -> has share in: {effect_label}'
            )

        # Effekt darf nicht selber als Share in seinen ShareEffekten auftauchen:
        # operation:
        for target_effect in self.specific_share_to_other_effects_operation.keys():
            assert self not in target_effect.specific_share_to_other_effects_operation.keys(), \
                f'Error: circular operation-shares \n{error_str(target_effect.label, target_effect.label)}'
        # invest:
        for target_effect in self.specific_share_to_other_effects_invest.keys():
            assert self not in target_effect.specific_share_to_other_effects_invest.keys(), \
                f'Error: circular invest-shares \n{error_str(target_effect.label, target_effect.label)}'

    def transform_data(self):
        self.minimum_operation_per_hour = _create_time_series(
            'minimum_operation_per_hour', self.minimum_operation_per_hour, self)
        self.maximum_operation_per_hour = _create_time_series(
            'maximum_operation_per_hour', self.maximum_operation_per_hour, self)

        self.specific_share_to_other_effects_operation = effect_values_to_time_series(
            f'specific_share_to_other_effects_operation',
            self.specific_share_to_other_effects_operation, self)

    def create_model(self) -> 'EffectModel':
        self.model = EffectModel(self)
        return self.model

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


class EffectModel(ElementModel):
    def __init__(self, element: Effect):
        super().__init__(element)
        self.element: Effect
        self.invest = ShareAllocationModel(self.element, 'invest', False,
                                           total_max=self.element.maximum_invest,
                                           total_min=self.element.minimum_invest)
        self.operation = ShareAllocationModel(
            self.element, 'operation', True, total_max=self.element.maximum_operation,
            total_min=self.element.minimum_operation,
            min_per_hour=self.element.minimum_operation_per_hour.active_data if self.element.minimum_operation_per_hour is not None else None,
            max_per_hour=self.element.maximum_operation_per_hour.active_data if self.element.maximum_operation_per_hour is not None else None
        )
        self.all = ShareAllocationModel(self.element, 'all', False,
                                        total_max=self.element.maximum_total,
                                        total_min=self.element.minimum_total)
        self.sub_models.extend([self.invest, self.operation, self.all])

    def do_modeling(self, system_model: SystemModel):
        for model in self.sub_models:
            model.do_modeling(system_model)

        self.all.add_variable_share(system_model, 'operation', self.element, self.operation.sum)
        self.all.add_variable_share(system_model, 'invest', self.element, self.invest.sum)


EffectDict = Dict[Optional['Effect'], Numeric]
EffectDictInvest = Dict[Optional['Effect'], Skalar]

EffectValues = Optional[Union[Numeric_TS, EffectDict]]  # Datatype for User Input
EffectValuesInvest = Optional[Union[Skalar, EffectDictInvest]]  # Datatype for User Input

EffectTimeSeries = Dict[Optional['Effect'], TimeSeries]  # Final Internal Data Structure
ElementTimeSeries = Dict[Optional[Element], TimeSeries]  # Final Internal Data Structure


def nested_values_to_time_series(nested_values: Dict[Element, Numeric_TS],
                                 label_suffix: str,
                                 parent_element: Element) -> ElementTimeSeries:
    """
    Creates TimeSeries from nested values, which are a Dict of Elements to values.
    The resulting label of the TimeSeries is the label of the parent_element, followed by the label of the element in
    the nested_values and the label_suffix.
    """
    return {element: _create_time_series(f'{element.label}_{label_suffix}', value, parent_element)
            for element, value in nested_values.items() if element is not None}


def effect_values_to_time_series(label_suffix: str,
                                 nested_values: EffectValues,
                                 parent_element: Element) -> Optional[EffectTimeSeries]:
    """
    Creates TimeSeries from EffectValues. The resulting label of the TimeSeries is the label of the parent_element,
    followed by the label of the Effect in the nested_values and the label_suffix.
    If the key in the EffectValues is None, the alias 'Standart_Effect' is used
    """
    nested_values = _as_effect_dict(nested_values)
    if nested_values is None:
        return None
    else:
        standard_value = nested_values.pop(None, None)
        transformed_values = nested_values_to_time_series(nested_values, label_suffix, parent_element)
        if standard_value is not None:
            transformed_values[None] = _create_time_series(f'Standard_Effect_{label_suffix}', standard_value, parent_element)
        return transformed_values


def _as_effect_dict(effect_values: EffectValues) -> Optional[EffectDict]:
    """
    Converts effect values into a dictionary. If a scalar is provided, it is associated with a default effect type.

    Examples
    --------
    costs = 20                        -> {None: 20}
    costs = None                      -> None
    costs = {effect1: 20, effect2: 0.3} -> {effect1: 20, effect2: 0.3}

    Parameters
    ----------
    effect_values : None, int, float, TimeSeries, or dict
        The effect values to convert, either a scalar, TimeSeries, or a dictionary.

    Returns
    -------
    dict or None
        A dictionary with None or Effect as the key, or None if input is None.
    """
    return effect_values if isinstance(effect_values, dict) else {None: effect_values} if effect_values is not None else None


def effect_values_from_effect_time_series(effect_time_series: EffectTimeSeries) -> Dict[Optional[Effect], Numeric]:
    return {effect: time_series.active_data for effect, time_series in effect_time_series.items()}


class EffectCollection:
    """
    Handling all Effects
    """

    def __init__(self, label: str):
        self.label = label
        self.model: Optional[EffectCollectionModel] = None
        self.effects: List[Effect] = []

    def create_model(self, system_model: SystemModel) -> 'EffectCollectionModel':
        self.model = EffectCollectionModel(self, system_model)
        return self.model

    def add_effect(self, effect: 'Effect') -> None:
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

    @property
    def label_full(self):
        return self.label


class EffectCollectionModel(ElementModel):
    # TODO: Maybe all EffectModels should be sub_models of this Model? Including Objective and Penalty?
    def __init__(self, element: EffectCollection, system_model: SystemModel):
        super().__init__(element)
        self.element = element
        self._system_model = system_model
        self._effect_models: Dict[Effect, EffectModel] = {}
        self.penalty: Optional[ShareAllocationModel] = None
        self.objective: Optional[Equation] = None

    def do_modeling(self, system_model: SystemModel):
        self._effect_models = {effect: effect.create_model() for effect in self.element.effects}
        self.penalty = ShareAllocationModel(self.element, 'penalty', False)
        self.sub_models.extend(list(self._effect_models.values()) + [self.penalty])
        for model in self.sub_models:
            model.do_modeling(system_model)

        self.add_share_between_effects()

        self.objective = Equation('OBJECTIVE', 'OBJECTIVE', system_model, 'objective')
        self.add_equations(self.objective)
        self.objective.add_summand(self._objective_effect_model.operation.sum, 1)
        self.objective.add_summand(self._objective_effect_model.invest.sum, 1)
        self.objective.add_summand(self.penalty.sum, 1)

    @property
    def _objective_effect_model(self) -> EffectModel:
        return self._effect_models[self.element.objective_effect]

    def _add_share_to_effects(self,
                              name: str,
                              element: Element,
                              target: Literal['operation', 'invest'],
                              effect_values: EffectDict,
                              factor: Numeric,
                              variable: Optional[Variable] = None) -> None:
        # an alle Effects, die einen Wert haben, anhängen:
        for effect, value in effect_values.items():
            if effect is None:  # Falls None, dann Standard-effekt nutzen:
                effect = self.element.standard_effect
            assert effect in self.element.effects, f'Effect {effect.label} was used but not added to model!'

            if target == 'operation':
                model = self._effect_models[effect].operation
            elif target == 'invest':
                model = self._effect_models[effect].invest
            else:
                raise ValueError(f'Target {target} not supported!')

            name_of_share = f'{element.label_full}__{name}'
            total_factor = np.multiply(value, factor)
            if variable is None:
                model.add_constant_share(self._system_model, name_of_share, effect, total_factor)
            elif isinstance(variable, Variable):
                model.add_variable_share(self._system_model, name_of_share, effect, variable, total_factor)
            else:
                raise TypeError

    def add_share_to_invest(self,
                            name: str,
                            element: Element,
                            effect_values: EffectDictInvest,
                            factor: Numeric,
                            variable: Optional[Variable] = None) -> None:
        #TODO: Add checks
        self._add_share_to_effects(name, element, 'invest', effect_values, factor, variable)

    def add_share_to_operation(self,
                               name: str,
                               element: Element,
                               effect_values: EffectTimeSeries,
                               factor: Numeric,
                               variable: Optional[Variable] = None) -> None:
        # TODO: Add checks
        self._add_share_to_effects(name, element, 'operation', effect_values_from_effect_time_series(effect_values), factor, variable)

    def add_share_to_penalty(self,
                             name: Optional[str],
                             share_holder: Element,
                             variable: Optional[Variable],
                             factor: Numeric,
                             ) -> None:
        assert variable is not None, f'A Varieble must e passed to add a share to penalty! Else its a constant Penalty!'
        self.penalty.add_variable_share(self._system_model, name, share_holder, variable, factor,  True)

    def add_share_between_effects(self):
        for origin_effect in self.element.effects:
            name_of_share = f'Share_from_Effect_{origin_effect.label_full}'  # + effectType.label
            # 1. operation: -> hier sind es Zeitreihen (share_TS)
            for target_effect, time_series in origin_effect.specific_share_to_other_effects_operation.items():
                target_model = self._effect_models[target_effect].operation
                origin_model = self._effect_models[origin_effect].operation
                target_model.add_variable_share(self._system_model, name_of_share, origin_effect, origin_model.sum_TS,
                                                time_series.active_data)
            # 2. invest:    -> hier ist es Skalar (share)
            for target_effect, factor in origin_effect.specific_share_to_other_effects_invest.items():
                target_model = self._effect_models[target_effect].invest
                origin_model = self._effect_models[origin_effect].invest
                target_model.add_variable_share(self._system_model, name_of_share, origin_effect, origin_model.sum,
                                                factor)
