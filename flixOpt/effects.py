"""
This module contains the effects of the flixOpt framework.
Furthermore, it contains the EffectCollection, which is used to collect all effects of a system.
Different Datatypes are used to represent the effects with assigned values by the user,
which are then transformed into the internal data structure.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union, Iterator

import linopy
import numpy as np
import pandas as pd

from .core import NumericData, NumericDataTS, Scalar, TimeSeries, TimeSeriesCollection
from .features import ShareAllocationModel
from .structure import Element, ElementModel, Model, SystemModel, Interface

if TYPE_CHECKING:
    from .flow_system import FlowSystem

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
        specific_share_to_other_effects_operation: Optional['EffectValuesUser'] = None,
        specific_share_to_other_effects_invest: Optional['EffectValuesUser'] = None,
        minimum_operation: Optional[Scalar] = None,
        maximum_operation: Optional[Scalar] = None,
        minimum_invest: Optional[Scalar] = None,
        maximum_invest: Optional[Scalar] = None,
        minimum_operation_per_hour: Optional[NumericDataTS] = None,
        maximum_operation_per_hour: Optional[NumericDataTS] = None,
        minimum_total: Optional[Scalar] = None,
        maximum_total: Optional[Scalar] = None,
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
        self.specific_share_to_other_effects_operation: EffectValuesUser = (
            specific_share_to_other_effects_operation or {}
        )
        self.specific_share_to_other_effects_invest: EffectValuesUser = (
            specific_share_to_other_effects_invest or {}
        )
        self.minimum_operation = minimum_operation
        self.maximum_operation = maximum_operation
        self.minimum_operation_per_hour: NumericDataTS = minimum_operation_per_hour
        self.maximum_operation_per_hour: NumericDataTS = maximum_operation_per_hour
        self.minimum_invest = minimum_invest
        self.maximum_invest = maximum_invest
        self.minimum_total = minimum_total
        self.maximum_total = maximum_total

    def transform_data(self, time_series_collection: TimeSeriesCollection):
        self.minimum_operation_per_hour = self._create_time_series(
            'minimum_operation_per_hour', self.minimum_operation_per_hour, time_series_collection
        )
        self.maximum_operation_per_hour = self._create_time_series(
            'maximum_operation_per_hour', self.maximum_operation_per_hour, time_series_collection
        )

        self.specific_share_to_other_effects_operation = effect_values_to_time_series(
            'operation_to',
            self.specific_share_to_other_effects_operation,
            self,
            time_series_collection
        )

    def create_model(self, model: SystemModel) -> 'EffectModel':
        self.model = EffectModel(model, self)
        return self.model


class EffectModel(ElementModel):
    def __init__(self, model: SystemModel, element: Effect):
        super().__init__(model, element)
        self.element: Effect = element
        self.total: Optional[linopy.Variable] = None
        self.invest: ShareAllocationModel = self.add(
            ShareAllocationModel(
                self._model,
                False,
                self.label_of_element,
                'invest',
                total_max=self.element.maximum_invest,
                total_min=self.element.minimum_invest
            )
        )

        self.operation: ShareAllocationModel = self.add(
            ShareAllocationModel(
                self._model,
                True,
                self.label_of_element,
                'operation',
                total_max=self.element.maximum_operation,
                total_min=self.element.minimum_operation,
                min_per_hour=self.element.minimum_operation_per_hour.active_data
                if self.element.minimum_operation_per_hour is not None
                else None,
                max_per_hour=self.element.maximum_operation_per_hour.active_data
                if self.element.maximum_operation_per_hour is not None
                else None,
            )
        )

    def do_modeling(self):
        for model in self.sub_models:
            model.do_modeling()

        self.total = self.add(
            self._model.add_variables(
                lower=self.element.minimum_total if self.element.minimum_total is not None else -np.inf,
                upper=self.element.maximum_total if self.element.maximum_total is not None else np.inf,
                coords=None,
                name=f'{self.label_full}|total'
            ),
            'total'
        )

        self.add(
            self._model.add_constraints(
                self.total == self.operation.total.sum() + self.invest.total.sum(),
                name=f'{self.label_full}|total'
            ),
            'total'
        )


EffectKey = Optional[Union[str, Effect]]  # Common key type for effect-related dicts

EffectValuesExpr = Dict[EffectKey, linopy.LinearExpression]  # Used to create Shares
EffectValuesTS = Dict[EffectKey, TimeSeries]  # Used internally to index values
EffectValuesDict = Dict[EffectKey, NumericDataTS]  # How effect values are stored
EffectValuesUser = Union[NumericDataTS, Dict[EffectKey, NumericDataTS]]  # User-specified Shares to Effects
EffectValuesUserScalar = Union[Scalar, Dict[EffectKey, Scalar]]  # User-specified Shares to Effects

def effect_values_to_time_series(label_suffix: str,
                                 effect_values: EffectValuesUser,
                                 parent_element: Element,
                                 time_series_collection: TimeSeriesCollection) -> Optional[EffectValuesTS]:
    """
    Transform EffectValues to EffectValuesTS.
    Creates a TimeSeries for each key in the nested_values dictionary, using the value as the data.

    The resulting label of the TimeSeries is the label of the parent_element,
    followed by the label of the Effect in the nested_values and the label_suffix.
    If the key in the EffectValues is None, the alias 'Standard_Effect' is used
    """
    effect_values: Optional[EffectValuesDict] = effect_values_to_dict(effect_values)
    if effect_values is None:
        return None

    effect_values_ts: EffectValuesTS = {
        effect: parent_element._create_time_series(
            f'{effect.label_full if effect is not None else "Standard_Effect"}|{label_suffix}',
            value,
            time_series_collection
        )
        for effect, value in effect_values.items()
    }

    return effect_values_ts


def effect_values_to_dict(effect_values_user: EffectValuesUser) -> Optional[EffectValuesDict]:
    """
    Converts effect values into a dictionary. If a scalar is provided, it is associated with a default effect type.

    Examples
    --------
    effect_values_user = 20                             -> {None: 20}
    effect_values_user = None                           -> None
    effect_values_user = {effect1: 20, effect2: 0.3}    -> {effect1: 20, effect2: 0.3}

    Returns
    -------
    dict or None
        A dictionary with None or Effect as the key, or None if input is None.
    """
    return effect_values_user if isinstance(effect_values_user, dict) else {
        None: effect_values_user} if effect_values_user is not None else None


class EffectCollection:
    """
    Handling all Effects
    """

    def __init__(self, effects: List[Effect]):
        self._effects = {}
        self._standard_effect: Optional[Effect] = None
        self._objective_effect: Optional[Effect] = None

        self.effects = effects  # Performs some validation
        self.model: Optional[EffectCollectionModel] = None

        self._plausibility_checks()

    def create_model(self, model: SystemModel) -> 'EffectCollectionModel':
        self.model = EffectCollectionModel(model, self)
        return self.model

    def _plausibility_checks(self) -> None:
        # Check circular loops in effects:
        # TODO: Improve checks!! Only most basic case covered...

        def error_str(effect_label: str, share_ffect_label: str):
            return (
                f'  {effect_label} -> has share in: {share_ffect_label}\n'
                f'  {share_ffect_label} -> has share in: {effect_label}'
            )
        for effect in self.effects.values():
            # Effekt darf nicht selber als Share in seinen ShareEffekten auftauchen:
            # operation:
            for target_effect in effect.specific_share_to_other_effects_operation.keys():
                assert self not in target_effect.specific_share_to_other_effects_operation.keys(), (
                    f'Error: circular operation-shares \n{error_str(target_effect.label, target_effect.label)}'
                )
            # invest:
            for target_effect in effect.specific_share_to_other_effects_invest.keys():
                assert self not in target_effect.specific_share_to_other_effects_invest.keys(), (
                    f'Error: circular invest-shares \n{error_str(target_effect.label, target_effect.label)}'
                )

    def __getitem__(self, effect: Union[str, Effect]) -> 'Effect':
        """
        Get an effect by label, or return the standard effect if None is passed

        Raises:
            KeyError: If no effect with the given label is found.
            KeyError: If no standard effect is specified.
        """
        if effect is None:
            return self.standard_effect
        if isinstance(effect, Effect):
            if effect in self:
                return effect
            else:
                raise KeyError(f'Effect {effect} not found!')
        try:
            return self.effects[effect]
        except KeyError as e:
            raise KeyError(f'No effect with label {effect} found!') from e

    def __iter__(self) -> Iterator[Effect]:
        return iter(self._effects.values())

    def __contains__(self, item: Union[str, 'Effect']) -> bool:
        """Check if the effect exists. Checks for label or object"""
        if isinstance(item, str):
            return item in self.effects  # Check if the label exists
        elif isinstance(item, Effect):
            return item in self.effects.values()  # Check if the object exists
        return False

    @property
    def effects(self) -> Dict[str, Effect]:
        return self._effects

    @effects.setter
    def effects(self, value: List[Effect]):
        for effect in value:
            if effect.is_standard:
                self.standard_effect = effect
            if effect.is_objective:
                self.objective_effect = effect
            if effect in self:
                raise Exception(f'Effect with label "{effect.label=}" already added!')
            self._effects[effect.label] = effect

    @property
    def standard_effect(self) -> Effect:
        if self._standard_effect is None:
            raise KeyError('No standard-effect specified!')
        return self._standard_effect

    @standard_effect.setter
    def standard_effect(self, value: Effect) -> None:
        if self._standard_effect is not None:
            raise ValueError(f'A standard-effect already exists! ({self._standard_effect.label=})')
        self._standard_effect = value

    @property
    def objective_effect(self) -> Effect:
        if self._objective_effect is None:
            raise KeyError('No objective-effect specified!')
        return self._objective_effect

    @objective_effect.setter
    def objective_effect(self, value: Effect) -> None:
        if self._objective_effect is not None:
            raise ValueError(f'An objective-effect already exists! ({self._objective_effect.label=})')
        self._objective_effect = value


class EffectCollectionModel(Model):
    """
    Handling all Effects
    """

    def __init__(self, model: SystemModel, effects: EffectCollection):
        super().__init__(model, label_of_element='Effects')
        self.effects = effects
        self.penalty: Optional[ShareAllocationModel] = None

    def add_share_to_effects(
        self,
        name: str,
        expressions: EffectValuesExpr,
        target: Literal['operation', 'invest'],
    ) -> None:
        for effect, expression in expressions.items():
            if target == 'operation':
                self.effects[effect].model.operation.add_share(name, expression)
            elif target =='invest':
                self.effects[effect].model.invest.add_share(name, expression)
            else:
                raise ValueError(f'Target {target} not supported!')

    def add_share_to_penalty(self, name: str, expression: linopy.LinearExpression) -> None:
        if expression.ndim != 0:
            raise Exception(f'Penalty shares must be scalar expressions! ({expression.ndim=})')
        self.penalty.add_share(name, expression)

    def do_modeling(self):
        for effect in self.effects:
            effect.create_model(self._model)
        self.penalty = self.add(ShareAllocationModel(self._model, shares_are_time_series=False, label_of_element='Penalty'))
        for model in [effect.model for effect in self.effects] + [self.penalty]:
            model.do_modeling()

        self._add_share_between_effects()

    def _add_share_between_effects(self):
        for origin_effect in self.effects:
            # 1. operation: -> hier sind es Zeitreihen (share_TS)
            for target_effect, time_series in origin_effect.specific_share_to_other_effects_operation.items():
                target_effect.model.operation.add_share(
                    origin_effect.label_full,
                    origin_effect.model.operation.total_per_timestep * time_series.active_data,
                )
            # 2. invest:    -> hier ist es Scalar (share)
            for target_effect, factor in origin_effect.specific_share_to_other_effects_invest.items():
                target_effect.model.invest.add_share(
                    origin_effect.label_full,
                    origin_effect.model.invest.total * factor,
                )
