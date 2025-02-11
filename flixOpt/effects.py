"""
This module contains the effects of the flixOpt framework.
Furthermore, it contains the EffectCollection, which is used to collect all effects of a system.
Different Datatypes are used to represent the effects with assigned values by the user,
which are then transformed into the internal data structure.
"""

import logging
from typing import Dict, Literal, Optional, Union

import numpy as np
import linopy

from .core import Numeric, Numeric_TS, Skalar, TimeSeries
from .features import ShareAllocationModel
from .math_modeling import Equation, Variable
from .structure import Element, ElementModel, SystemModel, _create_time_series

logger = logging.getLogger('flixOpt')


class Effect(Element):
    """
    Effect, i.g. costs, CO2 emissions, area, ...
    Components, FLows, and so on can contribute to an Effect. One Effect is chosen as the Objective of the Optimization
    """

    def __init__(
        self,
        label: str,
        unit: str,
        description: str,
        meta_data: Optional[Dict] = None,
        is_standard: bool = False,
        is_objective: bool = False,
        specific_share_to_other_effects_operation: 'EffectValues' = None,
        specific_share_to_other_effects_invest: 'EffectValuesInvest' = None,
        minimum_operation: Optional[Skalar] = None,
        maximum_operation: Optional[Skalar] = None,
        minimum_invest: Optional[Skalar] = None,
        maximum_invest: Optional[Skalar] = None,
        minimum_operation_per_hour: Optional[Numeric_TS] = None,
        maximum_operation_per_hour: Optional[Numeric_TS] = None,
        minimum_total: Optional[Skalar] = None,
        maximum_total: Optional[Skalar] = None,
    ):
        """
        Parameters
        ----------
        label : str
            name
        unit : str
            unit of effect, i.g. €, kg_CO2, kWh_primaryEnergy
        description : str
            long name
        meta_data : Optional[Dict]
            used to store more information about the element. Is not used internally, but saved in the results
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
        super().__init__(label, meta_data=meta_data)
        self.label = label
        self.unit = unit
        self.description = description
        self.is_standard = is_standard
        self.is_objective = is_objective
        self.specific_share_to_other_effects_operation: Union[EffectValues, EffectTimeSeries] = (
            specific_share_to_other_effects_operation or {}
        )
        self.specific_share_to_other_effects_invest: Union[EffectValuesInvest, EffectDictInvest] = (
            specific_share_to_other_effects_invest or {}
        )
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
        # TODO: Improve checks!! Only most basic case covered...

        def error_str(effect_label: str, share_ffect_label: str):
            return (
                f'  {effect_label} -> has share in: {share_ffect_label}\n'
                f'  {share_ffect_label} -> has share in: {effect_label}'
            )

        # Effekt darf nicht selber als Share in seinen ShareEffekten auftauchen:
        # operation:
        for target_effect in self.specific_share_to_other_effects_operation.keys():
            assert self not in target_effect.specific_share_to_other_effects_operation.keys(), (
                f'Error: circular operation-shares \n{error_str(target_effect.label, target_effect.label)}'
            )
        # invest:
        for target_effect in self.specific_share_to_other_effects_invest.keys():
            assert self not in target_effect.specific_share_to_other_effects_invest.keys(), (
                f'Error: circular invest-shares \n{error_str(target_effect.label, target_effect.label)}'
            )

    def transform_data(self):
        self.minimum_operation_per_hour = _create_time_series(
            'minimum_operation_per_hour', self.minimum_operation_per_hour, self
        )
        self.maximum_operation_per_hour = _create_time_series(
            'maximum_operation_per_hour', self.maximum_operation_per_hour, self
        )

        self.specific_share_to_other_effects_operation = effect_values_to_time_series(
            'specific_share_to_other_effects_operation', self.specific_share_to_other_effects_operation, self
        )

    def create_model(self) -> 'EffectModel':
        self.model = EffectModel(self)
        return self.model


class EffectModel(ElementModel):
    def __init__(self, element: Effect):
        super().__init__(element)
        self.element: Effect = element
        self.invest = ShareAllocationModel(
            self.element, 'invest', False, total_max=self.element.maximum_invest, total_min=self.element.minimum_invest
        )
        self.operation = ShareAllocationModel(
            self.element,
            'operation',
            True,
            total_max=self.element.maximum_operation,
            total_min=self.element.minimum_operation,
            min_per_hour=self.element.minimum_operation_per_hour.active_data
            if self.element.minimum_operation_per_hour is not None
            else None,
            max_per_hour=self.element.maximum_operation_per_hour.active_data
            if self.element.maximum_operation_per_hour is not None
            else None,
        )
        self.total = ShareAllocationModel(
            self.element, 'total', False, total_max=self.element.maximum_total, total_min=self.element.minimum_total
        )
        self.sub_models.extend([self.invest, self.operation, self.total])

    def do_modeling(self, system_model: SystemModel):
        for model in self.sub_models:
            model.do_modeling(system_model)

        self.total.add_share(system_model, 'operation', self.operation.total*1)
        self.total.add_share(system_model, 'invest', self.invest.total*1)

EffectValues = Dict[Optional[Union[str, Effect]], linopy.LinearExpression]  # This is new


class EffectCollection(ElementModel):
    """
    Handling all Effects
    """

    def __init__(self):
        super().__init__(Element('Effects'))
        self.effects: Dict[str, Effect] = {}
        self.penalty: Optional[ShareAllocationModel] = None
        self.objective: Optional[Equation] = None

        self._standard_effect: Optional[Effect] = None
        self._objective_effect: Optional[Effect] = None

    def add_share_to_effects(
        self,
        system_model: SystemModel,
        name: str,
        expressions: EffectValues,
        target: Literal['operation', 'invest'],
    ) -> None:
        for effect, expression in expressions.items():
            if target == 'operation':
                self[effect].model.operation.add_share(system_model, name, expression)
            elif target =='invest':
                self[effect].model.invest.add_share(system_model, name, expression)
            else:
                raise ValueError(f'Target {target} not supported!')

    def add_share_to_penalty(self, system_model: SystemModel, name: str, expression: linopy.LinearExpression) -> None:
        if expression.ndim != 0:
            raise Exception(f'Penalty shares must be scalar expressions! ({expression.ndim=})')
        self.penalty.add_share(system_model, name, expression)

    def add_effect(self, effect: 'Effect') -> None:
        if effect.is_standard:
            self.standard_effect = effect
        if effect.is_objective:
            self.objective_effect = effect
        if effect in self.effects.values():
            raise Exception(f'Effect already added! ({effect.label=})')
        if effect.label in self.effects:
            raise Exception(f'Effect with label "{effect.label=}" already added!')
        self.effects[effect.label] = effect

    def do_modeling(self, system_model: SystemModel):
        for effect in self.effects.values():
            effect.create_model()
        self.penalty = ShareAllocationModel(Element('Penalty'), 'penalty', False)
        for model in [effect.model for effect in self.effects.values()] + [self.penalty]:
            model.do_modeling(system_model)

        self._add_share_between_effects(system_model)

        # TODO: Move this to the SystemModel!
        self.objective = Equation('OBJECTIVE', 'OBJECTIVE', is_objective=True)
        self.objective.add_summand(self.objective_effect.model.operation.sum, 1)
        self.objective.add_summand(self.objective_effect.model.invest.sum, 1)
        self.objective.add_summand(self.penalty.sum, 1)

    def _add_share_between_effects(self, system_model: SystemModel):
        for origin_effect in self.effects.values():
            # 1. operation: -> hier sind es Zeitreihen (share_TS)
            for target_effect, time_series in origin_effect.specific_share_to_other_effects_operation.items():
                target_effect.model.operation.add_share(
                    system_model,
                    f'{origin_effect.label_full}_operation',
                    origin_effect.model.operation.sum_TS,
                    time_series.active_data,
                )
            # 2. invest:    -> hier ist es Skalar (share)
            for target_effect, factor in origin_effect.specific_share_to_other_effects_invest.items():
                target_effect.model.invest.add_share(
                    system_model,
                    f'{origin_effect.label_full}_invest',
                    origin_effect.model.invest.sum,
                    factor
                )

    def __getitem__(self, label: str) -> 'Effect':
        """
        Get an effect by label, or return the standard effect if None is passed

        Raises:
            KeyError: If no effect with the given label is found.
            KeyError: If no standard effect is specified.
        """
        if label is None:
            try:
                return self.standard_effect
            except:
                raise KeyError(f'No Standard-effect specified!')
        try:
            return self.effects[label]
        except:
            raise KeyError(f'No effect with label {label} found!')

    def __contains__(self, item: Union[str, 'Effect']) -> bool:
        """Check if the effect exists. Checks for label or object"""
        if isinstance(item, str):
            return item in self.effects  # Check if the label exists
        elif isinstance(item, Effect):
            return item in self.effects.values()  # Check if the object exists
        return False

    @property
    def standard_effect(self) -> Effect:
        if self._standard_effect is None:
            raise KeyError(f'No standard-effect specified!')
        return self._standard_effect

    @standard_effect.setter
    def standard_effect(self, value: Effect) -> None:
        if self._standard_effect is not None:
            raise ValueError(f'A standard-effect already exists! ({self._standard_effect.label=})')
        self._standard_effect = value

    @property
    def objective_effect(self) -> Effect:
        if self._objective_effect is None:
            raise KeyError(f'No objective-effect specified!')
        return self._objective_effect

    @objective_effect.setter
    def objective_effect(self, value: Effect) -> None:
        if self._objective_effect is not None:
            raise ValueError(f'An objective-effect already exists! ({self._objective_effect.label=})')
        self._objective_effect = value
