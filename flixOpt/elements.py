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

from flixOpt.math_modeling import Variable, VariableTS, Equation
from flixOpt.core import TimeSeries, Numeric, Numeric_TS, Skalar, as_effect_dict, as_effect_dict_with_ts
from flixOpt.flixBasicsPublic import InvestParameters
from flixOpt.structure import Element, SystemModel
from flixOpt import flixOptHelperFcts as helpers

log = logging.getLogger(__name__)

class Effect(Element):
    '''
    Effect, i.g. costs, CO2 emissions, area, ...
    can be used later afterwards for allocating effects to compontents and flows.
    '''

    # is_standard -> Standard-Effekt (bei Eingabe eines skalars oder TS (statt dict) wird dieser automatisch angewendet)
    def __init__(self,
                 label: str,
                 unit: str,
                 description: str,
                 is_standard: bool = False,
                 is_objective: bool = False,
                 specific_share_to_other_effects_operation: Optional[Dict] = None,  # TODO: EffectTypeDict can not be used as type hint
                 specific_share_to_other_effects_invest: Optional[Dict] = None,  # TODO: EffectTypeDict can not be used as type hint
                 minimum_operation: Optional[Skalar] = None,
                 maximum_operation: Optional[Skalar] = None,
                 minimum_invest: Optional[Skalar] = None,
                 maximum_invest: Optional[Skalar] = None,
                 minimum_operation_per_hour: Optional[Numeric_TS] = None,
                 maximum_operation_per_hour: Optional[Numeric_TS] = None,
                 minimum_total: Optional[Skalar] = None,
                 maximum_total: Optional[Skalar] = None,
                 **kwargs):
        '''
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

        '''
        super().__init__(label, **kwargs)
        self.label = label
        self.unit = unit
        self.description = description
        self.is_standard = is_standard
        self.is_objective = is_objective
        self.specific_share_to_other_effects_operation = specific_share_to_other_effects_operation or {}
        self.specific_share_to_other_effects_invest = specific_share_to_other_effects_invest or {}
        self.minimum_operation = minimum_operation
        self.maximum_operation = maximum_operation
        self.minimum_operation_per_hour = minimum_operation_per_hour
        self.maximum_operation_per_hour = maximum_operation_per_hour
        self.minimum_invest = minimum_invest
        self.maximum_invest = maximum_invest
        self.minimum_total = minimum_total
        self.maximum_total = maximum_total

        #  operation-Effect-shares umwandeln in TS (invest bleibt skalar ):
        for effect, share in self.specific_share_to_other_effects_operation.items():
            name_of_ts = 'specificShareToOtherEffect' + '_' + effect.label
            self.specific_share_to_other_effects_operation[effect] = TimeSeries(
                name_of_ts, specific_share_to_other_effects_operation[effect], self)

        # ShareSums:
        # TODO: Why as attributes, and not only in sub_elements?
        from flixOpt.features import Feature_ShareSum
        self.operation = Feature_ShareSum(
            label='operation', owner=self, shares_are_time_series=True,
            total_min=self.minimum_operation, total_max=self.maximum_operation,
            min_per_hour=self.minimum_operation_per_hour, max_per_hour=self.maximum_operation_per_hour)
        self.invest = Feature_ShareSum(label='invest', owner=self, shares_are_time_series=False,
                                       total_min=self.minimum_invest, total_max=self.maximum_invest)
        self.all = Feature_ShareSum(label='all', owner=self, shares_are_time_series=False,
                                    total_min=self.minimum_total, total_max=self.maximum_total)

    def declare_vars_and_eqs(self, system_model) -> None:
        super().declare_vars_and_eqs(system_model)
        self.operation.declare_vars_and_eqs(system_model)
        self.invest.declare_vars_and_eqs(system_model)
        self.all.declare_vars_and_eqs(system_model)

    def do_modeling(self, system_model, time_indices: Union[List[int], range]) -> None:
        print('modeling ' + self.label)
        super().declare_vars_and_eqs(system_model)
        self.operation.do_modeling(system_model, time_indices)
        self.invest.do_modeling(system_model, time_indices)

        # Gleichung für Summe Operation und Invest:
        # eq: shareSum = effect.operation_sum + effect.operation_invest
        self.all.add_variable_share('operation', self, self.operation.model.variables['sum'], 1, 1)
        self.all.add_variable_share('invest', self, self.invest.model.variables['sum'], 1, 1)
        self.all.do_modeling(system_model, time_indices)

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


# ModelingElement (Element) Klasse zum Summieren einzelner Shares
# geht für skalar und TS
# z.B. invest.costs


# Liste mit zusätzlicher Methode für Rückgabe Standard-Element:

EffectTypeDict = Dict[Effect, Numeric_TS]  # Datatype


class EffectCollection(Element):
    """
    Handling Effects and penalties
    """

    def __init__(self, label: str, **kwargs):
        super().__init__(label, **kwargs)
        self.effects = []
        self.penalty = None

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
        self.sub_elements.append(effect)

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

    def finalize(self) -> None:
        for effect in self.effects:
            effect.finalize()
        from flixOpt.features import Feature_ShareSum
        self.penalty = Feature_ShareSum('penalty', self, shares_are_time_series=True)

    # Beiträge registrieren:
    # effectValues kann sein
    #   1. {effecttype1 : TS, effectType2: : TS} oder
    #   2. TS oder skalar
    #     -> Zuweisung zu Standard-EffektType

    def add_share_to_operation(self,
                               name_of_share: str,
                               owner: Element,
                               variable: Variable,
                               effect_values: Dict[Optional[Effect], TimeSeries],
                               factor: Numeric) -> None:
        if variable is None: raise Exception(
            'add_share_to_operation() needs variable or use add_constant_share instead')
        self._add_share('operation', name_of_share, owner, effect_values, factor, variable)

    def add_constant_share_to_operation(self,
                                        name_of_share: str,
                                        owner: Element,
                                        effect_values: Dict[Optional[Effect], TimeSeries],
                                        factor: Numeric) -> None:
        self._add_share('operation', name_of_share, owner, effect_values, factor)

    def add_share_to_invest(self,
                            name_of_share: str,
                            owner: Element,
                            variable: Variable,
                            effect_values: Dict[Optional[Effect], TimeSeries],
                            factor: Numeric) -> None:
        if variable is None:
            raise Exception('add_share_to_invest() needs variable or use add_constant_share instead')
        self._add_share('invest', name_of_share, owner, effect_values, factor, variable)

    def add_constant_share_to_invest(self,
                                     name_of_share: str,
                                     owner: Element,
                                     effect_values: Dict[Optional[Effect], TimeSeries],
                                     factor: Numeric) -> None:
        self._add_share('invest', name_of_share, owner, effect_values, factor)

        # wenn aVariable = None, dann constanter Share

    def _add_share(self,
                   operation_or_invest: Literal['operation', 'invest'],
                   name_of_share: str,
                   owner: Element,
                   effect_values: Union[Numeric, Dict[Optional[Effect], TimeSeries]],
                   factor: Numeric,
                   variable: Optional[Variable] = None) -> None:
        effect_values_dict = as_effect_dict(effect_values)

        # an alle Effekttypen, die einen Wert haben, anhängen:
        for effect, value in effect_values_dict.items():
            # Falls None, dann Standard-effekt nutzen:
            if effect is None:
                effect = self.standard_effect
            elif effect not in self.effects:
                raise Exception('Effect \'' + effect.label + '\' was used but not added to model!')

            if operation_or_invest == 'operation':
                effect.operation.add_share(name_of_share, owner, variable, value,
                                           factor)  # hier darf aVariable auch None sein!
            elif operation_or_invest == 'invest':
                effect.invest.add_share(name_of_share, owner, variable, value,
                                        factor)  # hier darf aVariable auch None sein!
            else:
                raise Exception('operationOrInvest=' + str(operation_or_invest) + ' ist kein zulässiger Wert')

    def declare_vars_and_eqs(self, system_model: SystemModel) -> None:
        self.penalty.declare_vars_and_eqs(system_model)
        # TODO: ggf. Unterscheidung, ob Summen überhaupt als Zeitreihen-Variablen abgebildet werden sollen, oder nicht, wg. Performance.
        for effect in self.effects:
            effect.declare_vars_and_eqs(system_model)

    def do_modeling(self, system_model: SystemModel, time_indices: Union[list[int], range]) -> None:
        self.penalty.do_modeling(system_model, time_indices)
        for effect in self.effects:
            effect.do_modeling(system_model, time_indices)

        ## Beiträge von Effekt zu anderen Effekten, Beispiel 180 €/t_CO2: ##
        for effectType in self.effects:
            # Beitrag/Share ergänzen:
            # 1. operation: -> hier sind es Zeitreihen (share_TS)
            # alle specificSharesToOtherEffects durchgehen:
            nameOfShare = 'specific_share_to_other_effects_operation'  # + effectType.label
            for effectTypeOfShare, specShare_TS in effectType.specific_share_to_other_effects_operation.items():
                # Share anhängen (an jeweiligen Effekt):
                shareSum_op = effectTypeOfShare.operation
                shareSum_op: flixOpt.flixFeatures.Feature_ShareSum
                shareHolder = effectType
                shareSum_op.add_variable_share(nameOfShare, shareHolder, effectType.operation.model.variables['sum_TS'],
                                               specShare_TS, 1)
            # 2. invest:    -> hier ist es Skalar (share)
            # alle specificSharesToOtherEffects durchgehen:
            nameOfShare = 'specificShareToOtherEffects_invest_'  # + effectType.label
            for effectTypeOfShare, specShare in effectType.specific_share_to_other_effects_invest.items():
                # Share anhängen (an jeweiligen Effekt):
                shareSum_inv = effectTypeOfShare.invest
                from flixOpt.features import Feature_ShareSum
                shareSum_inv: Feature_ShareSum
                shareHolder = effectType
                shareSum_inv.add_variable_share(nameOfShare, shareHolder, effectType.invest.model.var_sum, specShare, 1)


class Objective(Element):
    """
    Storing the Objective
    """

    def __init__(self, label: str, **kwargs):
        super().__init__(label, **kwargs)
        self.objective = None

    def declare_vars_and_eqs(self, system_model: SystemModel) -> None:
        # TODO: ggf. Unterscheidung, ob Summen überhaupt als Zeitreihen-Variablen abgebildet werden sollen, oder nicht, wg. Performance.
        self.objective = Equation('objective', self, system_model, 'objective')
        self.model.add_equation(self.objective)
        system_model.objective = self.objective

    def add_objective_effect_and_penalty(self, effect_collection: EffectCollection) -> None:
        if effect_collection.objective_effect is None:
            raise Exception('Kein Effekt als Zielfunktion gewählt!')
        self.objective.add_summand(effect_collection.objective_effect.operation.model.variables['sum'], 1)
        self.objective.add_summand(effect_collection.objective_effect.invest.model.variables['sum'], 1)
        self.objective.add_summand(effect_collection.penalty.model.variables['sum'], 1)


# Beliebige Komponente (:= Element mit Ein- und Ausgängen)
class Component(Element):
    """
    basic component class for all components
    """
    system_model: SystemModel
    new_init_args = ['label', 'on_values_before_begin', 'effects_per_switch_on', 'switch_on_total_max',
                     'on_hours_total_min',
                     'on_hours_total_max', 'effects_per_running_hour', 'exists']
    not_used_args = ['label']

    def __init__(self,
                 label: str,
                 on_values_before_begin: Optional[List[Skalar]] = None,
                 effects_per_switch_on: Optional[Union[EffectTypeDict, Numeric_TS]] = None,
                 switch_on_total_max: Optional[Skalar] = None,
                 on_hours_total_min: Optional[Skalar] = None,
                 on_hours_total_max: Optional[Skalar] = None,
                 effects_per_running_hour: Optional[Union[EffectTypeDict, Numeric_TS]] = None,
                 exists: Numeric = 1,
                 **kwargs):
        '''


        Parameters
        ----------
        label : str
            name.

        Parameters of on/off-feature
        ----------------------------
        (component is off, if all flows are zero!)

        on_values_before_begin :  array (TODO: why not scalar?)
            Ein(1)/Aus(0)-Wert vor Zeitreihe
        effects_per_switch_on : look in Flow for description
        switch_on_total_max : look in Flow for description
        on_hours_total_min : look in Flow for description
        on_hours_total_max : look in Flow for description
        effects_per_running_hour : look in Flow for description
        exists : array, int, None
            indicates when a component is present. Used for timing of Investments. Only contains blocks of 0 and 1.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if on_hours_total_min is not None:
            raise NotImplementedError(
                "'on_hours_total_min' is not implemented yet for Components. Use Flow directly instead")
        if on_hours_total_max is not None:
            raise NotImplementedError(
                "'on_hours_total_max' is not implemented yet for Components. Use Flow directly instead")
        label = helpers.check_name_for_conformity(label)  # todo: indexierbar / eindeutig machen!
        super().__init__(label, **kwargs)
        self.on_values_before_begin = on_values_before_begin if on_values_before_begin else [0, 0]
        self.effects_per_switch_on = as_effect_dict_with_ts('effects_per_switch_on', effects_per_switch_on, self)
        self.switch_on_max = switch_on_total_max
        self.on_hours_total_min = on_hours_total_min
        self.on_hours_total_max = on_hours_total_max
        self.effects_per_running_hour = as_effect_dict_with_ts('effects_per_running_hour', effects_per_running_hour, self)
        self.exists = TimeSeries('exists', helpers.check_exists(exists), self)

        ## TODO: theoretisch müsste man auch zusätzlich checken, ob ein flow Werte beforeBegin hat!
        # % On Werte vorher durch Flow-values bestimmen:
        # self.on_valuesBefore = 1 * (self.featureOwner.values_before_begin >= np.maximum(model.epsilon,self.flowMin)) für alle Flows!

        # TODO: Dict instead of list?
        self.inputs = []  # list of flows
        self.outputs = []  # list of flows
        self.is_storage = False  # TODO: Use isinstance instead?

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
        for aFlow in self.inputs + self.outputs:
            aFlow.comp = self

    def register_flows_in_bus(self) -> None:  # todo: macht aber bei Kindklasse Bus keinen Sinn!
        #
        # ############## register in Bus: ##############
        #
        # input ist output von Bus:
        for aFlow in self.inputs:
            aFlow.bus.register_output_flow(aFlow)  # ist das schön programmiert?
        # output ist input von Bus:
        for aFlow in self.outputs:
            aFlow.bus.register_input_flow(aFlow)

    def declare_vars_and_eqs_of_flows(self,
                                      system_model: SystemModel) -> None:  # todo: macht aber bei Kindklasse Bus keinen Sinn!
        # Flows modellieren:
        for aFlow in self.inputs + self.outputs:
            aFlow.declare_vars_and_eqs(system_model)

    def do_modeling_of_flows(self, system_model: SystemModel, time_indices: Union[
        list[int], range]) -> None:  # todo: macht aber bei Kindklasse Bus keinen Sinn!
        # Flows modellieren:
        for aFlow in self.inputs + self.outputs:
            aFlow.do_modeling(system_model, time_indices)

    def get_results(self) -> Tuple[Dict, Dict]:
        # Variablen der Komponente:
        (results, results_var) = super().get_results()

        # Variablen der In-/Out-Puts ergänzen:
        for aFlow in self.inputs + self.outputs:
            # z.B. results['Q_th'] = {'val':..., 'on': ..., ...}
            if isinstance(self, Bus):
                flow_label = aFlow.label_full  # Kessel_Q_th
            else:
                flow_label = aFlow.label  # Q_th
            (results[flow_label], results_var[flow_label]) = aFlow.get_results()
        return results, results_var

    def finalize(self) -> None:
        super().finalize()
        from flixOpt.features import FeatureOn

        # feature for: On and SwitchOn Vars
        # (kann erst hier gebaut werden wg. weil input/output Flows erst hier vorhanden)
        flows_defining_on = self.inputs + self.outputs  # Sobald ein input oder  output > 0 ist, dann soll On =1 sein!
        self.featureOn = FeatureOn(self, flows_defining_on, self.on_values_before_begin, self.effects_per_switch_on,
                                   self.effects_per_running_hour, on_hours_total_min=self.on_hours_total_min,
                                   on_hours_total_max=self.on_hours_total_max, switch_on_total_max=self.switch_on_max)

    def declare_vars_and_eqs(self, system_model) -> None:
        super().declare_vars_and_eqs(system_model)

        self.featureOn.declare_vars_and_eqs(system_model)

        # Binärvariablen holen (wenn vorh., sonst None):
        #   (hier und nicht erst bei do_modeling, da linearSegments die Variable zum Modellieren benötigt!)
        self.model.var_on = self.featureOn.getVar_on()  # mit None belegt, falls nicht notwendig
        self.model.var_switchOn, self.model.var_switchOff = self.featureOn.getVars_switchOnOff()  # mit None belegt, falls nicht notwendig

    def do_modeling(self, system_model, time_indices: Union[list[int], range]) -> None:
        log.debug(str(self.label) + 'do_modeling()')
        self.featureOn.do_modeling(system_model, time_indices)

    def add_share_to_globals_of_flows(self, effect_collection: EffectCollection, system_model: SystemModel) -> None:
        for aFlow in self.inputs + self.outputs:
            aFlow.add_share_to_globals(effect_collection, system_model)

    # wird von Kindklassen überschrieben:
    def add_share_to_globals(self,
                             effect_collection: EffectCollection,
                             system_model: SystemModel) -> None:
        # Anfahrkosten, Betriebskosten, ... etc ergänzen:
        self.featureOn.add_share_to_globals(effect_collection, system_model)

    def description(self) -> Dict:

        descr = {}
        inhalt = {'In-Flows': [], 'Out-Flows': []}
        aFlow: Flow

        descr[self.label] = inhalt

        if isinstance(self, Bus):
            descrType = 'for bus-list'
        else:
            descrType = 'for comp-list'

        for aFlow in self.inputs:
            inhalt['In-Flows'].append(aFlow.description(type=descrType))  # '  -> Flow: '))
        for aFlow in self.outputs:
            inhalt['Out-Flows'].append(aFlow.description(type=descrType))  # '  <- Flow: '))

        if self.is_storage:
            inhalt['is_storage'] = self.is_storage
        inhalt['class'] = type(self).__name__

        if hasattr(self, 'group'):
            if self.group is not None:
                inhalt['group'] = self.group

        if hasattr(self, 'color'):
            if self.color is not None:
                inhalt['color'] = str(self.color)

        return descr


class Bus(Component):  # sollte das wirklich geerbt werden oder eher nur Element???
    '''
    realizing balance of all linked flows
    (penalty flow is excess can be activated)
    '''

    # --> excess_effects_per_flow_hour
    #        none/ 0 -> kein Exzess berücksichtigt
    #        > 0 berücksichtigt

    new_init_args = ['media', 'label', 'excess_effects_per_flow_hour']
    not_used_args = ['label']

    def __init__(self,
                 media: Optional[str],
                 label: str,
                 excess_effects_per_flow_hour: Optional[Numeric_TS] = 1e5,
                 **kwargs):
        """
        Parameters
        ----------
        media : None, str or set of str
            media or set of allowed media of the coupled flows,
            if None, then any flow is allowed
            example 1: media = None -> every media is allowed
            example 1: media = 'gas' -> flows with medium 'gas' are allowed
            example 2: media = {'gas','biogas','H2'} -> flows of these media are allowed
        label : str
            name.
        excess_effects_per_flow_hour : none or scalar, array or TimeSeriesRaw
            excess costs / penalty costs (bus balance compensation)
            (none/ 0 -> no penalty). The default is 1e5.
            (Take care: if you use a timeseries (no scalar), timeseries is aggregated if calculation_type = aggregated!)
        exists : not implemented yet for Bus!
        **kwargs : TYPE
            DESCRIPTION.
        """

        super().__init__(label, **kwargs)
        if media is None:
            self.media = media  # alle erlaubt
        elif isinstance(media, str):
            self.media = {media}  # convert to set
        elif isinstance(media, set):
            self.media = media
        else:
            raise Exception('no valid input for argument media!')

        self.excess_effects_per_flow_hour = None
        if (excess_effects_per_flow_hour is not None) and (excess_effects_per_flow_hour > 0):
            self.excess_effects_per_flow_hour = TimeSeries('excess_effects_per_flow_hour', excess_effects_per_flow_hour,
                                                           self)

    @property
    def with_excess(self) -> bool:
        return False if self.excess_effects_per_flow_hour is None else True

    def register_input_flow(self, flow) -> None:
        flow: Flow
        self.inputs.append(flow)
        self.check_medium(flow)

    def register_output_flow(self, flow) -> None:
        flow: Flow
        self.outputs.append(flow)
        self.check_medium(flow)

    def check_medium(self, flow) -> None:
        flow: Flow
        if self.media is not None and flow.medium is not None and flow.medium not in self.media:
            raise Exception(
                f"In Bus {self.label}: register_flows_in_bus(): Medium '{flow.medium}' of {flow.label_full} "
                f"and media {self.media} of bus {self.label_full} have no common medium! "
                f"Check if the flow is connected correctly OR append flow-medium to the allowed bus-media in bus-definition! "
                f"OR generally deactivate media-check by setting media in bus-definition to None."
            )

    def declare_vars_and_eqs(self, system_model: SystemModel) -> None:
        super().declare_vars_and_eqs(system_model)
        # Fehlerplus/-minus:
        if self.with_excess:
            # Fehlerplus und -minus definieren
            self.model.add_variable(VariableTS('excess_input', len(system_model.time_series), self.label_full, system_model,
                                           lower_bound=0))
            self.model.add_variable(VariableTS('excess_output', len(system_model.time_series), self.label_full, system_model,
                                            lower_bound=0))

    def do_modeling(self, system_model: SystemModel, time_indices: Union[list[int], range]) -> None:
        super().do_modeling(system_model, time_indices)

        # inputs = outputs
        bus_balance = Equation('busBalance', self, system_model)
        self.model.add_equation(bus_balance)
        for aFlow in self.inputs:
            bus_balance.add_summand(aFlow.model.variables['val'], 1)
        for aFlow in self.outputs:
            bus_balance.add_summand(aFlow.model.variables['val'], -1)

        if self.with_excess:  # Fehlerplus/-minus hinzufügen zur Bilanz:
            bus_balance.add_summand(self.model.variables['excess_output'], -1)
            bus_balance.add_summand(self.model.variables['excess_input'], 1)

    def add_share_to_globals(self, effect_collection: EffectCollection, system_model: SystemModel) -> None:
        super().add_share_to_globals(effect_collection, system_model)
        if self.with_excess:  # Strafkosten hinzufügen:
            effect_collection.penalty.add_variable_share('excess_effects_per_flow_hour_in', self, self.model.variables['excess_input'],
                                                   self.excess_effects_per_flow_hour, system_model.dt_in_hours)
            effect_collection.penalty.add_variable_share('excess_effects_per_flow_hour_out', self, self.model.variables['excess_output'],
                                                   self.excess_effects_per_flow_hour, system_model.dt_in_hours)


# Medien definieren:
class MediumCollection:
    '''
    attributes are defined possible media for flow (not tested!) TODO!
    you can use them, i.g. MediumCollection.heat or you can explicitly work with strings (i.g. 'heat')
    '''
    # predefined medium: (just the string is used for comparison)
    heat = 'heat'  # set(['heat'])
    el = 'el'  # set(['el'])
    fuel = 'fuel'  # gas | lignite | biomass

    # neues Medium hinzufügen:
    def addMedium(attrName, strOfMedium):
        '''
        add new medium to predefined media

        Parameters
        ----------
        attrName : str
        strOfMedium : str
        '''
        MediumCollection.setattr(attrName, strOfMedium)

    # checkifFits(medium1,medium2,...)
    def checkIfFits(*args):
        aCommonMedium = helpers.InfiniteFullSet()
        for aMedium in args:
            if aMedium is not None: aCommonMedium = aCommonMedium & aMedium
        if aCommonMedium:
            return True
        else:
            return False


class Connection:
    # input/output-dock (TODO:
    # -> wäre cool, damit Komponenten auch auch ohne Knoten verbindbar
    # input wären wie Flow,aber statt bus : connectsTo -> hier andere Connection oder aber Bus (dort keine Connection, weil nicht notwendig)

    def __init__(self):
        raise NotImplementedError()


# todo: könnte Flow nicht auch von Basecomponent erben. Hat zumindest auch Variablen und Eqs
# Fluss/Strippe
class Flow(Element):
    '''
    flows are inputs and outputs of components
    '''

    # static var:
    _default_size = 1e9  # Großer Gültigkeitsbereich als Standard

    @property
    def label_full(self) -> str:
        # Wenn im Erstellungsprozess comp noch nicht bekannt:
        comp_label = 'unknownComp' if self.comp is None else self.comp.label
        return f'{comp_label}__{self.label}'  # z.B. für results_struct (deswegen auch _  statt . dazwischen)

    @property  # Richtung
    def is_input_in_comp(self) -> bool:
        comp: Component
        return True if self in self.comp.inputs else False

    @property
    def size_is_fixed(self) -> bool:
        # Wenn kein InvestParameters existiert --> True; Wenn Investparameter, den Wert davon nehmen
        return True if self.invest_parameters is None else self.invest_parameters.fixed_size

    @property
    def invest_is_optional(self) -> bool:
        # Wenn kein InvestParameters existiert: # Investment ist nicht optional -> Keine Variable --> False
        return False if self.invest_parameters is None else self.invest_parameters.optional

    @property
    def on_variable_is_forced(self) -> bool:
        # Wenn Min-Wert > 0 wird binäre On-Variable benötigt (nur bei flow!):
        return self.can_switch_off & np.any(self.relative_minimum.data > 0)

    def __init__(self, label,
                 bus: Bus = None,  # TODO: Is this for sure Optional?
                 size: Optional[Skalar] = _default_size,
                 relative_minimum: Numeric_TS = 0,
                 relative_maximum: Numeric_TS = 1,
                 fixed_relative_value: Optional[Numeric_TS] = None,  # TODO: Rename?
                 flow_hours_total_max: Optional[Skalar] = None,
                 flow_hours_total_min: Optional[Skalar] = None,
                 load_factor_min: Optional[Skalar] = None,
                 load_factor_max: Optional[Skalar] = None,
                 values_before_begin: Optional[List[Skalar]] = None,
                 effects_per_flow_hour: Optional[Union[Numeric_TS, EffectTypeDict]] = None,
                 effects_per_running_hour: Optional[Union[Numeric_TS, EffectTypeDict]] = None,
                 can_switch_off: bool = True,
                 on_hours_total_min: Optional[Skalar] = None,
                 on_hours_total_max: Optional[Skalar] = None,
                 consecutive_on_hours_min: Optional[Skalar] = None,
                 consecutive_on_hours_max: Optional[Skalar] = None,
                 consecutive_off_hours_min: Optional[Skalar] = None,
                 consecutive_off_hours_max: Optional[Skalar] = None,
                 effects_per_switch_on: Optional[Union[Numeric_TS, EffectTypeDict]] = None,
                 switch_on_total_max: Optional[Skalar] = None,
                 invest_parameters: Optional[InvestParameters] = None,
                 medium: Optional[str] = None,
                 exists: Numeric_TS = 1,
                 group: Optional[str] = None,
                 # positive_gradient=None,
                 **kwargs):
        '''
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
        positive_gradient : TYPE, optional
           not implemented yet
        effects_per_flow_hour : scalar, array, TimeSeriesRaw, optional
            operational costs, costs per flow-"work"
        can_switch_off : boolean, optional
            flow can be "off", i.e. be zero (only relevant if relative_minimum > 0)
            Then a binary var "on" is used.
            If any on/off-forcing parameters like "effects_per_switch_on", "consecutive_on_hours_min" etc. are used, then
            this is automatically forced.
        on_hours_total_min : scalar, optional
            min. overall sum of operating hours.
        on_hours_total_max : scalar, optional
            max. overall sum of operating hours.
        consecutive_on_hours_min : scalar, optional
            min sum of operating hours in one piece
            (last on-time period of timeseries is not checked and can be shorter)
        consecutive_on_hours_max : scalar, optional
            max sum of operating hours in one piece
        consecutive_off_hours_min : scalar, optional
            min sum of non-operating hours in one piece
            (last off-time period of timeseries is not checked and can be shorter)
        consecutive_off_hours_max : scalar, optional
            max sum of non-operating hours in one piece
        effects_per_switch_on : scalar, array, TimeSeriesRaw, optional
            cost of one switch from off (var_on=0) to on (var_on=1),
            unit i.g. in Euro
        switch_on_total_max : integer, optional
            max nr of switchOn operations
        effects_per_running_hour : scalar or TS, optional
            costs for operating, i.g. in € per hour
        flow_hours_total_max : TYPE, optional
            maximum flow-hours ("flow-work")
            (if size is not const, maybe load_factor_max fits better for you!)
        flow_hours_total_min : TYPE, optional
            minimum flow-hours ("flow-work")
            (if size is not const, maybe load_factor_min fits better for you!)
        values_before_begin : list (TODO: why not scalar?), optional
            Flow-value before begin (for calculation of i.g. switchOn for first time step, gradient for first time step ,...)'),
            # TODO: integration of option for 'first is last'
        fixed_relative_value : scalar, array, TimeSeriesRaw, optional
            fixed relative values for flow (if given).
            val(t) := fixed_relative_value(t) * size(t)
            With this value, the flow-value is no opt-variable anymore;
            (relative_minimum u. relative_maximum are making sense anymore)
            used for fixed load profiles, i.g. heat demand, wind-power, solarthermal
            If the load-profile is just an upper limit, use relative_maximum instead.
        medium: string, None
            medium is relevant, if the linked bus only allows a special defined set of media.
            If None, any bus can be used.
        invest_parameters : None or InvestParameters, optional
            used for investment costs or/and investment-optimization!
        exists : int, array, None
            indicates when a flow is present. Used for timing of Investments. Only contains blocks of 0 and 1.
            relative_maximum is multiplied with this value before the solve
        group: str, None
            group name to assign flows to groups. Used for later analysis of the results
        '''

        super().__init__(label, **kwargs)
        # args to attributes:
        self.bus = bus
        self.size = size  # skalar!
        self.relative_minimum = TimeSeries('relative_minimum', relative_minimum, self)
        self.relative_maximum = TimeSeries('relative_maximum', relative_maximum, self)

        self.load_factor_min = load_factor_min
        self.load_factor_max = load_factor_max
        # self.positive_gradient = TimeSeries('positive_gradient', positive_gradient, self)
        self.effects_per_flow_hour = as_effect_dict_with_ts('effects_per_flow_hour', effects_per_flow_hour, self)
        self.can_switch_off = can_switch_off
        self.on_hours_total_min = on_hours_total_min
        self.on_hours_total_max = on_hours_total_max
        self.consecutive_on_hours_min = None if (consecutive_on_hours_min is None) else TimeSeries('consecutive_on_hours_min', consecutive_on_hours_min, self)
        self.consecutive_on_hours_max = None if (consecutive_on_hours_max is None) else TimeSeries('consecutive_on_hours_max', consecutive_on_hours_max, self)
        self.consecutive_off_hours_min = None if (consecutive_off_hours_min is None) else TimeSeries('consecutive_off_hours_min', consecutive_off_hours_min, self)
        self.consecutive_off_hours_max = None if (consecutive_off_hours_max is None) else TimeSeries('consecutive_off_hours_max', consecutive_off_hours_max, self)
        self.effects_per_switch_on = as_effect_dict_with_ts('effects_per_switch_on', effects_per_switch_on, self)
        self.switch_on_total_max = switch_on_total_max
        self.effects_per_running_hour = as_effect_dict_with_ts('effects_per_running_hour', effects_per_running_hour, self)
        self.flow_hours_total_max = flow_hours_total_max
        self.flow_hours_total_min = flow_hours_total_min

        self.exists = TimeSeries('exists', helpers.check_exists(exists), self)
        self.group = group  # TODO: wird überschrieben von Component!
        self.values_before_begin = np.array(values_before_begin) if values_before_begin else np.array(
            [0, 0])  # list -> np-array

        self.invest_parameters = invest_parameters  # Info: Plausi-Checks erst, wenn Flow self.comp kennt.
        self.comp = None  # zugehörige Komponente (wird später von Komponente gefüllt)

        self.fixed_relative_value = None
        if fixed_relative_value is not None:
            # Wenn noch size noch Default, aber investment_size nicht optimiert werden soll:
            size_is_default = self.size == Flow._default_size
            if size_is_default and self.size_is_fixed:
                raise Exception(
                    'Achtung: Wenn fixed_relative_value genutzt wird, muss zugehöriges size definiert werden, da: value = fixed_relative_value * size!')
            self.fixed_relative_value = TimeSeries('fixed_relative_value', fixed_relative_value, self)

        self.medium = medium
        if (self.medium is not None) and (not isinstance(self.medium, str)):
            raise Exception('medium must be a string or None')

        # Liste. Ich selbst bin der definierende Flow! (Bei Komponente sind es hingegen alle in/out-flows)
        flows_defining_on = [self]
        # TODO: besser wäre model.epsilon, aber hier noch nicht bekannt!)
        on_values_before_begin = 1 * (self.values_before_begin >= 0.0001)
        # TODO: Wenn can_switch_off = False und min > 0, dann könnte man var_on fest auf 1 setzen um Rechenzeit zu sparen

        # TODO: Why not in sub_elements?
        from flixOpt.features import FeatureOn, FeatureInvest
        self.featureOn = FeatureOn(self, flows_defining_on,
                                   on_values_before_begin,
                                   self.effects_per_switch_on,
                                   self.effects_per_running_hour,
                                   on_hours_total_min=self.on_hours_total_min,
                                   on_hours_total_max=self.on_hours_total_max,
                                   consecutive_on_hours_min=self.consecutive_on_hours_min,
                                   consecutive_on_hours_max=self.consecutive_on_hours_max,
                                   consecutive_off_hours_min=self.consecutive_off_hours_min,
                                   consecutive_off_hours_max=self.consecutive_off_hours_max,
                                   switch_on_total_max=self.switch_on_total_max,
                                   force_on=self.on_variable_is_forced)

        self.featureInvest: Optional[FeatureInvest] = None  # Is defined in finalize()

    def __str__(self):
        details = [
            f"bus={self.bus.label if self.bus else 'None'}",
            f"size={self.size}",
            f"min/relative_maximum={self.relative_minimum}-{self.relative_maximum}",
            f"medium={self.medium}",
            f"invest_parameters={self.invest_parameters.__str__()}" if self.invest_parameters else "",
            f"fixed_relative_value={self.fixed_relative_value}" if self.fixed_relative_value else "",
            f"effects_per_flow_hour={self.effects_per_flow_hour}" if self.effects_per_flow_hour else "",
            f"effects_per_running_hour={self.effects_per_running_hour}" if self.effects_per_running_hour else "",
        ]

        all_relevant_parts = [part for part in details if part != ""]

        full_str = f"{', '.join(all_relevant_parts)}"

        return f"<{self.__class__.__name__}> {self.label}: {full_str}"

    # Plausitest der Eingangsparameter (sollte erst aufgerufen werden, wenn self.comp bekannt ist)
    def plausibility_test(self) -> None:
        # TODO: Incorporate into Variable? (Lower_bound can not be greater than upper bound
        if np.any(self.relative_minimum.data > self.relative_maximum.data):
            raise Exception(self.label_full + ': Take care, that relative_minimum <= relative_maximum!')

    # bei Bedarf kann von außen Existenz von Binärvariable erzwungen werden:
    def force_on_variable(self) -> None:
        self.featureOn.force_on = True

    def finalize(self) -> None:
        self.plausibility_test()  # hier Input-Daten auf Plausibilität testen (erst hier, weil bei __init__ self.comp noch nicht bekannt)

        # exist-merge aus Flow.exist und Comp.exist
        exists_global = np.multiply(self.exists.data, self.comp.exists.data)  # array of 0 and 1
        self.exists_with_comp = TimeSeries('exists_with_comp', helpers.check_exists(exists_global), self)
        # combine relative_maximum with and exist from the flow and the comp it belongs to
        self.max_rel_with_exists = TimeSeries('max_rel_with_exists',
                                              np.multiply(self.relative_maximum.data, self.exists_with_comp.data), self)
        self.relative_minimum_with_exists = TimeSeries('relative_minimum_with_exists',
                                              np.multiply(self.relative_minimum.data, self.exists_with_comp.data), self)

        # prepare invest Feature:
        if self.invest_parameters is not None:
            from flixOpt.features import FeatureInvest
            self.featureInvest = FeatureInvest('size', self, self.invest_parameters,
                                               relative_minimum=self.relative_minimum_with_exists,
                                               relative_maximum=self.max_rel_with_exists,
                                               fixed_relative_value=self.fixed_relative_value,
                                               investment_size=self.size,
                                               featureOn=self.featureOn)

        super().finalize()

    def declare_vars_and_eqs(self, system_model: SystemModel) -> None:
        print('declare_vars_and_eqs ' + self.label)
        super().declare_vars_and_eqs(system_model)

        self.featureOn.declare_vars_and_eqs(system_model)  # TODO: rekursiv aufrufen für sub_elements

        self.system_model = system_model

        def bounds_of_defining_variable() -> Tuple[Optional[Numeric], Optional[Numeric], Optional[Numeric]]:
            """
            Returns the lower and upper bound and the fixed value of the defining variable.
            Returns: (lower_bound, upper_bound, fixed_value)
            """
            # Wenn fixer Lastgang:
            if self.fixed_relative_value is not None:
                # min = max = val !
                lower_bound = None
                upper_bound = None
                fix_value = self.fixed_relative_value.active_data * self.size
            else:
                lower_bound = 0 if self.featureOn.use_on else self.relative_minimum_with_exists.active_data * self.size
                upper_bound = self.max_rel_with_exists.active_data * self.size
                fix_value = None
            return lower_bound, upper_bound, fix_value

        # wenn keine Investrechnung:
        if self.featureInvest is None:
            (lower_bound, upper_bound, fix_value) = bounds_of_defining_variable()
        else:
            (lower_bound, upper_bound, fix_value) = self.featureInvest.bounds_of_defining_variable()

        # TODO --> wird trotzdem modelliert auch wenn value = konst -> Sinnvoll?
        self.model.add_variable(VariableTS('val', system_model.nrOfTimeSteps, self.label_full, system_model,
                                           lower_bound=lower_bound, upper_bound=upper_bound, value=fix_value))
        self.model.add_variable(Variable('sumFlowHours', 1, self.label_full, system_model,
                                         lower_bound=self.flow_hours_total_min, upper_bound=self.flow_hours_total_max))
        # ! Die folgenden Variablen müssen erst von featureOn erstellt worden sein:
        self.model.var_on = self.featureOn.getVar_on()  # mit None belegt, falls nicht notwendig
        self.model.var_switchOn, self.model.var_switchOff = self.featureOn.getVars_switchOnOff()  # mit None belegt, falls nicht notwendig

        # erst hier, da defining_variable vorher nicht belegt!
        if self.featureInvest is not None:
            self.featureInvest.set_defining_variables(self.model.variables['val'], self.model.variables.get('on'))
            self.featureInvest.declare_vars_and_eqs(system_model)

    def do_modeling(self, system_model: SystemModel, time_indices: Union[list[int], range]) -> None:
        # super().do_modeling(model,time_indices)

        # for aFeature in self.features:
        #   aFeature.do_modeling(model,time_indices)

        #
        # ############## Variablen aktivieren: ##############
        #

        # todo -> für pyomo: fix()

        #
        # ############## on_hours_total_max: ##############
        #

        # ineq: sum(var_on(t)) <= on_hours_total_max

        if self.on_hours_total_max is not None:
            eq_on_hours_total_max = Equation('on_hours_total_max', self, system_model, 'ineq')
            self.model.add_equation(eq_on_hours_total_max)
            eq_on_hours_total_max.add_summand(self.model.var_on, 1, as_sum=True)
            eq_on_hours_total_max.add_constant(self.on_hours_total_max / system_model.dt_in_hours)

        #
        # ############## on_hours_total_max: ##############
        #

        # ineq: sum(var_on(t)) >= on_hours_total_min

        if self.on_hours_total_min is not None:
            eq_on_hours_total_min = Equation('on_hours_total_min', self, system_model, 'ineq')
            self.model.add_equation(eq_on_hours_total_min)
            eq_on_hours_total_min.add_summand(self.model.var_on, -1, as_sum=True)
            eq_on_hours_total_min.add_constant(-1 * self.on_hours_total_min / system_model.dt_in_hours)

        #
        # ############## sumFlowHours: ##############
        #

        # eq: var_sumFlowHours - sum(var_val(t)* dt(t) = 0

        eq_sumFlowHours = Equation('sumFlowHours', self, system_model, 'eq')  # general mean
        self.model.add_equation(eq_sumFlowHours)
        eq_sumFlowHours.add_summand(self.model.variables["val"], system_model.dt_in_hours, as_sum=True)
        eq_sumFlowHours.add_summand(self.model.variables['sumFlowHours'], -1)

        #
        # ############## Constraints für Binärvariablen : ##############
        #

        self.featureOn.do_modeling(system_model, time_indices)  # TODO: rekursiv aufrufen für sub_elements

        #
        # ############## Glg. für Investition : ##############
        #

        if self.featureInvest is not None:
            self.featureInvest.do_modeling(system_model, time_indices)

        ## ############## full load fraction bzw. load factor ##############

        ## max load factor:
        #  eq: var_sumFlowHours <= size * dt_tot * load_factor_max

        if self.load_factor_max is not None:
            flowHoursPerInvestsize_max = system_model.dt_in_hours_total * self.load_factor_max  # = fullLoadHours if investsize in [kW]
            eq_flowHoursPerInvestsize_Max = Equation('load_factor_max', self, system_model, 'ineq')  # general mean
            self.model.add_equation(eq_flowHoursPerInvestsize_Max)
            eq_flowHoursPerInvestsize_Max.add_summand(self.model.variables["sumFlowHours"], 1)
            if self.featureInvest is not None:
                eq_flowHoursPerInvestsize_Max.add_summand(self.featureInvest.model.variables[self.featureInvest.name_of_investment_size],
                                                          -1 * flowHoursPerInvestsize_max)
            else:
                eq_flowHoursPerInvestsize_Max.add_constant(self.size * flowHoursPerInvestsize_max)

                ## min load factor:
        #  eq: size * sum(dt)* load_factor_min <= var_sumFlowHours

        if self.load_factor_min is not None:
            flowHoursPerInvestsize_min = system_model.dt_in_hours_total * self.load_factor_min  # = fullLoadHours if investsize in [kW]
            eq_flowHoursPerInvestsize_Min = Equation('load_factor_min', self, system_model, 'ineq')
            self.model.add_equation(eq_flowHoursPerInvestsize_Min)
            eq_flowHoursPerInvestsize_Min.add_summand(self.model.variables["sumFlowHours"], -1)
            if self.featureInvest is not None:
                eq_flowHoursPerInvestsize_Min.add_summand(self.featureInvest.model.variables[self.featureInvest.name_of_investment_size],
                                                          flowHoursPerInvestsize_min)
            else:
                eq_flowHoursPerInvestsize_Min.add_constant(-1 * self.size * flowHoursPerInvestsize_min)

        # ############## positiver Gradient #########

        '''        
        if self.positive_gradient == None :                    
          if model.modeling_language == 'pyomo':
            def positive_gradient_rule(t):
              if t == 0:
                return (self.model.var_val[t] - self.val_initial) / model.dt_in_hours[t] <= self.positive_gradient[t] #             
              else: 
                return (self.model.var_val[t] - self.model.var_val[t-1])    / model.dt_in_hours[t] <= self.positive_gradient[t] #

            # Erster Zeitschritt beachten:          
            if (self.val_initial == None) & (start == 0):
              self.positive_gradient_constr =  Constraint([start+1:end]        ,rule = positive_gradient_rule)          
            else:
              self.positive_gradient_constr =  Constraint(model.timestepsOfRun,rule = positive_gradient_rule)   # timestepsOfRun = [start:end]
              # raise error();
            system_model.registerPyComp(self.positive_gradient_constr, self.label + '_positive_gradient_constr')
          elif model.modeling_language == 'vcxpy':
            raise Exception('not defined for modtype ' + model.modeling_language)
          else:
            raise Exception('not defined for modtype ' + model.modeling_language)'''

        # ############# Beiträge zu globalen constraints ############

        # z.B. max_PEF, max_CO2, ...

    def add_share_to_globals(self, effect_collection: EffectCollection, system_model: SystemModel) -> None:

        # Arbeitskosten:
        if self.effects_per_flow_hour is not None:
            owner = self
            effect_collection.add_share_to_operation(
                'effects_per_flow_hour', owner, self.model.variables['val'],
                self.effects_per_flow_hour, system_model.dt_in_hours)

        # Anfahrkosten, Betriebskosten, ... etc ergänzen:
        self.featureOn.add_share_to_globals(effect_collection, system_model)

        if self.featureInvest is not None:
            self.featureInvest.add_share_to_globals(effect_collection, system_model)

        """
        in oemof gibt es noch 
            if m.flows[i, o].positive_gradient['ub'][0] is not None:
                for t in m.TIMESTEPS:
                    gradient_costs += (self.positive_gradient[i, o, t] *
                                       m.flows[i, o].positive_gradient[
                                           'costs'])

            if m.flows[i, o].negative_gradient['ub'][0] is not None:
                for t in m.TIMESTEPS:
                    gradient_costs += (self.negative_gradient[i, o, t] *
                                       m.flows[i, o].negative_gradient[
                                           'costs'])
        """

    def description(self, type: str = 'full') -> Dict:
        aDescr = {}
        if type == 'for bus-list':
            # aDescr = str(self.comp.label) + '.'
            aDescr['comp'] = self.comp.label
            aDescr = {str(self.label): aDescr}  # label in front of
        elif type == 'for comp-list':
            # aDescr += ' @Bus ' + str(self.bus.label)
            aDescr['bus'] = self.bus.label
            aDescr = {str(self.label): aDescr}  # label in front of
        elif type == 'full':
            aDescr['label'] = self.label
            aDescr['comp'] = self.comp.label
            aDescr['bus'] = self.bus.label
            aDescr['is_input_in_comp'] = self.is_input_in_comp
            if hasattr(self, 'group'):
                if self.group is not None:
                    aDescr["group"] = self.group

            if hasattr(self, 'color'):
                if self.color is not None:
                    aDescr['color'] = str(self.color)

        else:
            raise Exception('type = \'' + str(type) + '\' is not defined')

        return aDescr

    # Preset medium (only if not setted explicitly by user)
    def set_medium_if_not_set(self, medium) -> None:
        if self.medium is None:  # nicht überschreiben, nur wenn leer:
            self.medium = medium
