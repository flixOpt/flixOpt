# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:40:23 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

from typing import List, Tuple, Dict, Union, Optional, TYPE_CHECKING
import logging

import numpy as np

from flixOpt.math_modeling import Variable, VariableTS, Equation
from flixOpt.core import TimeSeries, Skalar, Numeric, Numeric_TS
from flixOpt.interface import InvestParameters, OnOffParameters
from flixOpt.structure import ElementModel, SystemModel, Element

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from flixOpt.effects import Effect
    from flixOpt.elements import Flow
    from flixOpt.components import Storage


logger = logging.getLogger('flixOpt')


class InvestmentModel(ElementModel):
    """Class for modeling an investment"""
    def __init__(self, element: Union['Flow', 'Storage'],
                 invest_parameters: InvestParameters,
                 defining_variable: [VariableTS],
                 relative_bounds_of_defining_variable: Tuple[Numeric, Numeric],
                 on_variable: Optional[VariableTS] = None):
        """
        if relative_bounds are both equal, then its like a fixed relative value
        """
        super().__init__(element)
        self.element: Union['Flow', 'Storage'] = element
        self.size: Optional[Union[Skalar, Variable]] = None
        self.is_invested: Optional[Variable] = None

        self._segments: Optional[SegmentedSharesModel] = None

        self._on_variable = on_variable
        self._defining_variable = defining_variable
        self._relative_bounds_of_defining_variable = relative_bounds_of_defining_variable
        self._invest_parameters = invest_parameters

    def do_modeling(self, system_model: SystemModel):
        invest_parameters = self._invest_parameters
        if invest_parameters.fixed_size:
            self.size = Variable('size', 1, self.element.label_full, system_model,
                                 value=invest_parameters.fixed_size)
        else:
            lower_bound = 0 if invest_parameters.optional else invest_parameters.minimum_size
            self.size = Variable('size', 1, self.element.label_full, system_model,
                                 lower_bound=lower_bound,
                                 upper_bound=invest_parameters.maximum_size)
        self.add_variables(self.size)
        # Optional
        if invest_parameters.optional:
            self.is_invested = Variable('isInvested', 1, self.element.label_full, system_model,
                                        is_binary=True)
            self.add_variables(self.is_invested)
            self._create_bounds_for_optional_investment(system_model)

        # Bounds for defining variable
        self._create_bounds_for_defining_variable(system_model)

        self._create_shares(system_model)

    def _create_shares(self, system_model: SystemModel):
        effect_collection = system_model.flow_system.effect_collection
        invest_parameters = self._invest_parameters

        # fix_effects:
        fix_effects = invest_parameters.fix_effects
        if fix_effects is not None and fix_effects != 0:
            if invest_parameters.optional:  # share: + isInvested * fix_effects
                effect_collection.add_share_to_invest('fix_effects', self.element,
                                                      self.is_invested, fix_effects, 1)
            else:  # share: + fix_effects
                effect_collection.add_constant_share_to_invest('fix_effects', self.element,
                                                               fix_effects ,1)
        # divest_effects:
        divest_effects = invest_parameters.divest_effects
        if divest_effects is not None and divest_effects != 0:
            if invest_parameters.optional:  # share: [divest_effects - isInvested * divest_effects]
                # 1. part of share [+ divest_effects]:
                effect_collection.add_constant_share_to_invest('divest_effects', self.element,
                                                               divest_effects, 1)
                # 2. part of share [- isInvested * divest_effects]:
                effect_collection.add_share_to_invest('divest_cancellation_effects', self.element,
                                                      self.is_invested, divest_effects, -1)
                # TODO : these 2 parts should be one share! -> SingleShareModel...?

        # # specific_effects:
        specific_effects = invest_parameters.specific_effects
        if specific_effects is not None:
            # share: + investment_size (=var)   * specific_effects
            effect_collection.add_share_to_invest('specific_effects', self.element,
                                                  self.size, specific_effects, 1)
        # segmented Effects
        invest_segments = invest_parameters.effects_in_segments
        if invest_segments:
            self._segments = SegmentedSharesModel(self.element,
                                                  (self.size, invest_segments[0]),
                                                  invest_segments[1], self.is_invested)
            self.sub_models.append(self._segments)
            self._segments.do_modeling(system_model)

    def _create_bounds_for_optional_investment(self, system_model: SystemModel):
        if self._invest_parameters.fixed_size:
            # eq: investment_size = isInvested * fixed_size
            eq_is_invested = Equation('is_invested', self.element, system_model, 'eq')
            eq_is_invested.add_summand(self.size, -1)
            eq_is_invested.add_summand(self.is_invested, self._invest_parameters.fixed_size)
            self.add_equations(eq_is_invested)
        else:
            # eq1: P_invest <= isInvested * investSize_max
            eq_is_invested_ub = Equation('is_invested_ub', self.element, system_model, 'ineq')
            eq_is_invested_ub.add_summand(self.size, 1)
            eq_is_invested_ub.add_summand(self.is_invested, np.multiply(-1, self._invest_parameters.maximum_size))

            # eq2: P_invest >= isInvested * max(epsilon, investSize_min)
            eq_is_invested_lb = Equation('is_invested_lb', self.element, system_model, 'ineq')
            eq_is_invested_lb.add_summand(self.size, -1)
            eq_is_invested_lb.add_summand(self.is_invested, np.max(system_model.epsilon, self._invest_parameters.minimum_size))
            self.add_equations(eq_is_invested_ub, eq_is_invested_lb)

    def _create_bounds_for_defining_variable(self, system_model: SystemModel):
        label = self._defining_variable.label
        relative_minimum, relative_maximum = self._relative_bounds_of_defining_variable
        # fixed relative value
        if np.array_equal(relative_minimum, relative_maximum):
            # TODO: Allow Off? Currently not...
            eq_fixed = Equation(f'fixed_{label}', self.element, system_model)
            eq_fixed.add_summand(self._defining_variable, 1)
            eq_fixed.add_summand(self.size, np.multiply(-1, relative_maximum))
            self.add_equations(eq_fixed)
        else:
            eq_upper = Equation(f'ub_{label}', self.element, system_model, 'ineq')
            # eq: defining_variable(t)  <= size * upper_bound(t)
            eq_upper.add_summand(self._defining_variable, 1)
            eq_upper.add_summand(self.size, np.multiply(-1, relative_maximum))

            ## 2. Gleichung: Minimum durch Investmentgröße ##
            eq_lower = Equation(f'lb_{label}', self, system_model, 'ineq')
            if self._on_variable is None:
                # eq: defining_variable(t) >= investment_size * relative_minimum(t)
                eq_lower.add_summand(self._defining_variable, -1)
                eq_lower.add_summand(self.size, relative_minimum)
            else:
                ## 2. Gleichung: Minimum durch Investmentgröße und On
                # eq: defining_variable(t) >= mega * (On(t)-1) + size * relative_minimum(t)
                #     ... mit mega = relative_maximum * maximum_size
                # äquivalent zu:.
                # eq: - defining_variable(t) + mega * On(t) + size * relative_minimum(t) <= + mega
                mega = relative_maximum * self._invest_parameters.maximum_size
                eq_lower.add_summand(self._defining_variable, -1)
                eq_lower.add_summand(self._on_variable, mega)
                eq_lower.add_summand(self.size, relative_minimum)
                eq_lower.add_constant(mega)
                # Anmerkung: Glg bei Spezialfall relative_minimum = 0 redundant zu OnOff ??
            self.add_equations(eq_lower, eq_upper)


class OnOffModel(ElementModel):
    """
    Class for modeling the on and off state of a variable
    If defining_bounds are given, creates sufficient lower bounds
    """
    def __init__(self, element: Element,
                 on_off_parameters: OnOffParameters,
                 defining_variables: List[VariableTS],
                 defining_bounds: List[Tuple[Numeric, Numeric]]):
        """
        defining_bounds: a list of Numeric, that can be  used to create the bound for On/Off more efficiently
        """
        super().__init__(element)
        self.element = element
        self.on: Optional[VariableTS] = None
        self.total_on_hours: Optional[Variable] = None

        self.consecutive_on_hours: Optional[VariableTS] = None
        self.consecutive_off_hours: Optional[VariableTS] = None

        self.off: Optional[VariableTS] = None

        self.switch_on: Optional[VariableTS] = None
        self.switch_off: Optional[VariableTS] = None
        self.nr_switch_on: Optional[VariableTS] = None

        self._on_off_parameters = on_off_parameters
        self._defining_variables = defining_variables
        self._defining_bounds = defining_bounds
        assert len(defining_variables) == len(defining_bounds), f'Every defining Variable needs bounds to Model OnOff'

    def do_modeling(self, system_model: SystemModel):
        if self._on_off_parameters.use_on:
            self.on = VariableTS('on', system_model.nr_of_time_steps, self.element.label_full, system_model,
                                 is_binary=True, before_value=self._on_off_parameters.on_values_before_begin[0])
            self.total_on_hours = Variable('onHoursSum', 1, self.element.label_full, system_model,
                                           lower_bound=self._on_off_parameters.on_hours_total_min,
                                           upper_bound=self._on_off_parameters.on_hours_total_max)
            self.add_variables(self.on, self.total_on_hours)

            self._add_on_constraints(system_model, system_model.time_indices)

        if self._on_off_parameters.use_off:
            self.off = VariableTS('off', system_model.nr_of_time_steps, self.element.label_full, system_model,
                                  is_binary=True)
            self.add_variables(self.off)

            self._add_off_constraints(system_model, system_model.time_indices)

        if self._on_off_parameters.use_on_hours:
            self.consecutive_on_hours = VariableTS('onHours', system_model.nr_of_time_steps,
                                                   self.element.label_full, system_model,
                                                   lower_bound=0,
                                                   upper_bound=self._on_off_parameters.consecutive_on_hours_max)
            self.add_variables(self.consecutive_on_hours)
            self._add_duration_constraints(self.consecutive_on_hours, self.on,
                                           self._on_off_parameters.consecutive_on_hours_min,
                                           system_model, system_model.time_indices)
        # offHours:
        if self._on_off_parameters.use_off_hours:
            self.consecutive_off_hours = VariableTS('offHours', system_model.nr_of_time_steps,
                                                    self.element.label_full, system_model,
                                                    lower_bound=0,
                                                    upper_bound=self._on_off_parameters.consecutive_off_hours_max)
            self.add_variables(self.consecutive_off_hours)
            self._add_duration_constraints(self.consecutive_off_hours, self.off,
                                           self._on_off_parameters.consecutive_off_hours_min,
                                           system_model, system_model.time_indices)
        # Var SwitchOn
        if self._on_off_parameters.use_switch_on:
            self.switch_on = VariableTS('switchOn', system_model.nr_of_time_steps, self.element.label_full, system_model, is_binary=True)
            self.switch_off = VariableTS('switchOff', system_model.nr_of_time_steps, self.element.label_full, system_model, is_binary=True)
            self.nr_switch_on = Variable('nrSwitchOn', 1, self.element.label_full, system_model,
                                         upper_bound=self._on_off_parameters.switch_on_total_max)
            self.add_variables(self.switch_on, self.switch_off, self.nr_switch_on)
            self._add_switch_constraints(system_model, system_model.time_indices)

        self._create_shares(system_model)

    def _add_on_constraints(self, system_model: SystemModel, time_indices: Union[list[int], range]):
        assert self.on is not None, f'On variable of {self.element} must be defined to add constraints'
        # % Bedingungen 1) und 2) müssen erfüllt sein:

        # % Anmerkung: Falls "abschnittsweise linear" gewählt, dann ist eigentlich nur Bedingung 1) noch notwendig
        # %            (und dann auch nur wenn erstes Segment bei Q_th=0 beginnt. Dann soll bei Q_th=0 (d.h. die Maschine ist Aus) On = 0 und segment1.onSeg = 0):)
        # %            Fazit: Wenn kein Performance-Verlust durch mehr Gleichungen, dann egal!

        nr_of_defining_variables = len(self._defining_variables)
        assert nr_of_defining_variables > 0, 'Achtung: mindestens 1 Flow notwendig'

        eq_on_1 = Equation('On_Constraint_1', self.element, system_model, eqType='ineq')
        eq_on_2 = Equation('On_Constraint_2', self.element, system_model, eqType='ineq')
        self.add_equations(eq_on_1, eq_on_2)
        if nr_of_defining_variables == 1:
            variable = self._defining_variables[0]
            lower_bound, upper_bound = self._defining_bounds
            #### Bedingung 1) ####
            # eq: On(t) * max(epsilon, lower_bound) <= -1 * Q_th(t)
            eq_on_1.add_summand(variable, -1, time_indices)
            eq_on_1.add_summand(self.on, np.maximum(system_model.epsilon, lower_bound), time_indices)

            #### Bedingung 2) ####
            # eq: Q_th(t) <= Q_th_max * On(t)
            eq_on_2.add_summand(variable, 1, time_indices)
            eq_on_2.add_summand(self.on, upper_bound, time_indices)

        else:  # Bei mehreren Leistungsvariablen:
            #### Bedingung 1) ####
            # When all defining variables are 0, On is 0
            # eq: - sum(alle Leistungen(t)) + Epsilon * On(t) <= 0
            for variable in self._defining_variables:
                eq_on_1.add_summand(variable, -1, time_indices)
            eq_on_1.add_summand(self.on, system_model.epsilon, time_indices)

            #### Bedingung 2) ####
            ## sum(alle Leistung) >0 -> On = 1 | On=0 -> sum(Leistung)=0
            #  eq: sum( Leistung(t,i))              - sum(Leistung_max(i))             * On(t) <= 0
            #  --> damit Gleichungswerte nicht zu groß werden, noch durch nr_of_flows geteilt:
            #  eq: sum( Leistung(t,i) / nr_of_flows ) - sum(Leistung_max(i)) / nr_of_flows * On(t) <= 0
            absolute_maximum: Numeric = 0
            for variable, bounds in zip(self._defining_variables, self._defining_bounds):
                eq_on_2.add_summand(variable, 1 / nr_of_defining_variables, time_indices)
                absolute_maximum += bounds[1]  # der maximale Nennwert reicht als Obergrenze hier aus. (immer noch math. günster als BigM)

            upper_bound = absolute_maximum / nr_of_defining_variables
            eq_on_2.add_summand(self.on, -1 * upper_bound, time_indices)

        if np.max(upper_bound) > 1000:
            logger.warning(f'!!! ACHTUNG in {self.element.label_full}  Binärdefinition mit großem Max-Wert ('
                           f'{np.max(upper_bound)}). Ggf. falsche Ergebnisse !!!')

    def _add_off_constraints(self, system_model: SystemModel, time_indices: Union[list[int], range]):
        assert self.off is not None, f'Off variable of {self.element} must be defined to add constraints'
        # Definition var_off:
        # eq: var_on(t) + var_off(t) = 1
        eq_off = Equation('var_off', self, system_model, eqType='eq')
        self.add_equations(eq_off)
        eq_off.add_summand(self.off, 1, time_indices)
        eq_off.add_summand(self.on, 1, time_indices)
        eq_off.add_constant(1)

    def _add_duration_constraints(self,
                                  duration_variable: VariableTS,
                                  binary_variable: VariableTS,
                                  minimum_duration: Optional[Numeric_TS],
                                  system_model: SystemModel,
                                  time_indices: Union[list[int], range]):
        """
        i.g.
        binary_variable        = [0 0 1 1 1 1 0 1 1 1 0 ...]
        duration_variable =      [0 0 1 2 3 4 0 1 2 3 0 ...] (bei dt=1)
                                            |-> min_onHours = 3!

        if you want to count zeros, define var_bin_off: = 1-binary_variable before!
        """
        assert duration_variable is not None, f'Duration Variable of {self.element} must be defined to add constraints'
        assert binary_variable is not None, f'Duration Variable of {self.element} must be defined to add constraints'
        # TODO: Einfachere Variante von Peter umsetzen!

        # 1) eq: onHours(t) <= On(t)*Big | On(t)=0 -> onHours(t) = 0
        # mit Big = dt_in_hours_total
        label_prefix = duration_variable.label
        mega = system_model.dt_in_hours_total
        constraint_1 = Equation(f'{label_prefix}_constraint_1', self.element, system_model, eqType='ineq')
        self.add_equations(constraint_1)
        constraint_1.add_summand(duration_variable, 1)
        constraint_1.add_summand(binary_variable, -1 * mega)

        # 2a) eq: onHours(t) - onHours(t-1) <= dt(t)
        #    on(t)=1 -> ...<= dt(t)
        #    on(t)=0 -> onHours(t-1)>=
        constraint_2a = Equation(f'{label_prefix}_constraint_2a', self.element, system_model, eqType='ineq')
        self.add_equations(constraint_2a)
        constraint_2a.add_summand(duration_variable, 1, time_indices[1:])  # onHours(t)
        constraint_2a.add_summand(duration_variable, -1, time_indices[0:-1])  # onHours(t-1)
        constraint_2a.add_constant(system_model.dt_in_hours[1:])  # dt(t)

        # 2b) eq:  onHours(t) - onHours(t-1)             >=  dt(t) - Big*(1-On(t)))
        #    eq: -onHours(t) + onHours(t-1) + On(t)*Big <= -dt(t) + Big
        # with Big = dt_in_hours_total # (Big = maxOnHours, should be usable, too!)
        constraint_2b = Equation(f'{label_prefix}_constraint_2b', self.element, system_model, eqType='ineq')
        self.add_equations(constraint_2b)
        constraint_2b.add_summand(duration_variable, -1, time_indices[1:])  # onHours(t)
        constraint_2b.add_summand(duration_variable, 1, time_indices[0:-1])  # onHours(t-1)
        constraint_2b.add_summand(binary_variable, mega, time_indices[1:])  # on(t)
        constraint_2b.add_constant(-1 + system_model.dt_in_hours[1:] + mega)  # dt(t)

        # 3) check minimum_duration before switchOff-step
        # (last on-time period of timeseries is not checked and can be shorter)
        if minimum_duration is not None:
            # Note: switchOff-step is when: On(t)-On(t-1) == -1
            # eq:  onHours(t-1) >= minOnHours * -1 * [On(t)-On(t-1)]
            # eq: -onHours(t-1) - minimum_duration * On(t) + minimum_duration*On(t-1) <= 0
            eq_min_duration = Equation(f'{label_prefix}_minimum_duration', self.element, system_model, eqType='ineq')
            self.add_equations(eq_min_duration)
            eq_min_duration.add_summand(duration_variable, -1, time_indices[0:-1])  # onHours(t-1)
            eq_min_duration.add_summand(binary_variable, -1 * minimum_duration, time_indices[1:])  # on(t)
            eq_min_duration.add_summand(binary_variable, minimum_duration, time_indices[0:-1])  # on(t-1)

        # TODO: Maximum Duration?? Is this not modeled yet?!!

        # 4) first index:
        #    eq: onHours(t=0)= dt(0) * On(0)
        first_index = time_indices[0]  # only first element
        eq_first = Equation(f'{label_prefix}_firstTimeStep', self.element, system_model)
        self.add_equations(eq_first)
        eq_first.add_summand(duration_variable, 1, first_index)
        eq_first.add_summand(binary_variable, -1 * system_model.dt_in_hours[first_index], first_index)

    def _add_switch_constraints(self, system_model: SystemModel, time_indices: Union[list[int], range]):
        assert self.switch_on is not None, f'Switch On Variable of {self.element} must be defined to add constraints'
        assert self.switch_off is not None, f'Switch Off Variable of {self.element} must be defined to add constraints'
        assert self.nr_switch_on is not None, f'Nr of Switch On Variable of {self.element} must be defined to add constraints'
        assert self.on is not None, f'On Variable of {self.element} must be defined to add constraints'
        # % Schaltänderung aus On-Variable
        # % SwitchOn(t)-SwitchOff(t) = On(t)-On(t-1)
        eq_switch = Equation('Switch', self.element, system_model)
        self.add_equations(eq_switch)
        eq_switch.add_summand(self.switch_on, 1, time_indices[1:])  # SwitchOn(t)
        eq_switch.add_summand(self.switch_off, -1, time_indices[1:])  # SwitchOff(t)
        eq_switch.add_summand(self.on, -1, time_indices[1:])  # On(t)
        eq_switch.add_summand(self.on, +1, time_indices[0:-1])  # On(t-1)

        ## Ersten Wert SwitchOn(t=1) bzw. SwitchOff(t=1) festlegen
        # eq: SwitchOn(t=1)-SwitchOff(t=1) = On(t=1)- ValueBeforeBeginOfTimeSeries;
        eq_initial_switch = Equation('Initial_Switch', self.element, system_model)
        first_index = time_indices[0]
        eq_initial_switch.add_summand(self.switch_on, 1, first_index)
        eq_initial_switch.add_summand(self.switch_off, -1, first_index)
        eq_initial_switch.add_summand(self.on, -1, first_index)
        eq_initial_switch.add_constant \
            (-1 * self.on.before_value)  # letztes Element der Before-Werte nutzen,  Anmerkung: wäre besser auf lhs aufgehoben

        ## Entweder SwitchOff oder SwitchOn
        # eq: SwitchOn(t) + SwitchOff(t) <= 1
        eq_switch_on_or_off = Equation('Switch_On_or_Off', self.element, system_model, eqType='ineq')
        eq_switch_on_or_off.add_summand(self.switch_on, 1)
        eq_switch_on_or_off.add_summand(self.switch_off, 1)
        eq_switch_on_or_off.add_constant(1)

        ## Anzahl Starts:
        # eq: nrSwitchOn = sum(SwitchOn(t))
        eq_nr_switch_on = Equation('NrSwitchOn', self.element, system_model)
        eq_nr_switch_on.add_summand(self.nr_switch_on, 1)
        eq_nr_switch_on.add_summand(self.switch_on, -1, as_sum=True)

    def _create_shares(self, system_model: SystemModel):
        # Anfahrkosten:
        effect_collection = system_model.effect_collection_model
        effects_per_switch_on = self._on_off_parameters.effects_per_switch_on
        if effects_per_switch_on is not None:
            effect_collection.add_share_to_effects('switch_on_effects', 'operation',
                                                   effects_per_switch_on, 1, self.switch_on)

        # Betriebskosten:
        effects_per_running_hour = self._on_off_parameters.effects_per_running_hour
        if effects_per_running_hour is not None:
            effect_collection.add_share_to_effects('running_hour_effects', 'operation',
                                                   effects_per_running_hour, system_model.dt_in_hours, self.on)


class SegmentModel(ElementModel):
    """Class for modeling a linear segment of one or more variables in parallel"""
    def __init__(self, element: Element, segment_index: Union[int, str],
                 sample_points: Dict[Variable, Tuple[Union[Numeric, TimeSeries], Union[Numeric, TimeSeries]]]):
        super().__init__(element)
        self.element = element
        self.in_segment: Optional[VariableTS] = None
        self.lambda0: Optional[VariableTS] = None
        self.lambda1: Optional[VariableTS] = None

        self._segment_index = segment_index
        self._sample_points = sample_points

    def do_modeling(self, system_model: SystemModel):
        length = system_model.nr_of_time_steps
        self.in_segment = VariableTS(f'onSeg_{self._segment_index}', length, self.element.label_full, system_model,
                                     is_binary=True)  # Binär-Variable
        self.lambda0 = VariableTS(f'lambda0_{self._segment_index}', length, self.element.label_full, system_model,
                                  lower_bound=0, upper_bound=1)  # Wertebereich 0..1
        self.lambda1 = VariableTS(f'lambda1_{self._segment_index}', length, self.element.label_full, system_model,
                                  lower_bound=0, upper_bound=1)  # Wertebereich 0..1
        self.add_variables(self.in_segment)
        self.add_variables(self.lambda0)
        self.add_variables(self.lambda1)

        # eq: -aSegment.onSeg(t) + aSegment.lambda1(t) + aSegment.lambda2(t)  = 0
        equation = Equation(f'Lambda_onSeg_{self._segment_index}', self, system_model)
        self.add_equations(equation)

        equation.add_summand(self.in_segment, -1)
        equation.add_summand(self.lambda0, 1)
        equation.add_summand(self.lambda1, 1)

        #  eq: - v(t) + (v_0 * lambda_0 + v_1 * lambda_1) = 0       -> v_0, v_1 = Stützstellen des Segments
        for variable, sample_points in self._sample_points.items():
            sample_0, sample_1 = sample_points
            if isinstance(sample_0, TimeSeries):
                sample_0 = sample_0.active_data
                sample_1 = sample_1.active_data
            else:
                sample_0 = sample_0
                sample_1 = sample_1

            lambda_eq = Equation(f'{variable.label_full}_lambda', self, system_model)
            lambda_eq.add_summand(variable, -1)
            lambda_eq.add_summand(self.lambda0, sample_0)
            lambda_eq.add_summand(self.lambda1, sample_1)
            self.add_equations(lambda_eq)


class MultipleSegmentsModel(ElementModel):
    def __init__(self, element: Element,
                 sample_points: Dict[Variable, List[Tuple[Union[Numeric, TimeSeries], Union[Numeric, TimeSeries]]]],
                 outside_segments: Optional[Variable] = None):
        super().__init__(element)
        self.element = element

        self.outside_segments: Optional[VariableTS] = outside_segments  # Variable to allow being outside segments = 0

        self._sample_points = sample_points
        self._segment_models: List[SegmentModel] = []

    def do_modeling(self, system_model: SystemModel):
        restructured_variables_with_segments: List[Dict[Variable, Tuple[Numeric, Numeric]]] = [
            {key: values[i] for key, values in self._sample_points.items()}
            for i in range(self._nr_of_segments)
        ]

        for i, sample_points in enumerate(restructured_variables_with_segments):
            self._segment_models.append(SegmentModel(self.element, i, sample_points))

        for segment_model in self._segment_models:
            segment_model.do_modeling(system_model)

        # Outside of Segments
        if self.outside_segments is None:  # TODO: Make optional
            self.outside_segments = VariableTS(f'outside_segments', system_model.nr_of_time_steps, self.element.label_full,
                                               system_model, is_binary=True)
            self.add_variables(self.outside_segments)

        # a) eq: Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 1                Aufenthalt nur in Segmenten erlaubt
        # b) eq: -On(t) + Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 0       zusätzlich kann alles auch Null sein
        in_single_segment = Equation('in_single_Segment', self, system_model)
        for segment_model in self._segment_models:
            in_single_segment.add_summand(segment_model.in_segment, 1)
        if self.outside_segments is None:
            in_single_segment.add_constant(1)
        else:
            in_single_segment.add_summand(self.outside_segments, -1)

    @property
    def _nr_of_segments(self):
        return len(next(iter(self._sample_points.values())))


class ShareAllocationModel(ElementModel):
    def __init__(self,
                 element: Element,
                 shares_are_time_series: bool,
                 total_max: Optional[Skalar] = None,
                 total_min: Optional[Skalar] = None,
                 max_per_hour: Optional[TimeSeries] = None,
                 min_per_hour: Optional[TimeSeries] = None):
        super().__init__(element)
        if not shares_are_time_series:  # If the condition is True
            assert max_per_hour is None and min_per_hour is None, \
                "Both max_per_hour and min_per_hour cannot be used when shares_are_time_series is False"
        self.element = element
        self.sum_TS: Optional[VariableTS] = None
        self.sum: Optional[Variable] = None

        self._eq_time_series: Optional[Equation] = None
        self._eq_sum: Optional[Equation] = None

        # Parameters
        self._shares_are_time_series = shares_are_time_series
        self._total_max = total_max
        self._total_min = total_min
        self._max_per_hour = max_per_hour
        self._min_per_hour = min_per_hour

    def do_modeling(self, system_model: SystemModel):
        self.sum = Variable('sum', 1, self.element.label_full, system_model,
                            lower_bound=self._total_min, upper_bound=self._total_max)
        self.add_variables(self.sum)
        # eq: sum = sum(share_i) # skalar
        self._eq_sum = Equation('sum', self, system_model)
        self._eq_sum.add_summand(self.sum, -1)
        self.add_equations(self._eq_sum)

        if self._shares_are_time_series:
            lb_TS = None if (self._min_per_hour is None) else np.multiply(self._min_per_hour.active_data, system_model.dt_in_hours)
            ub_TS = None if (self._max_per_hour is None) else np.multiply(self._max_per_hour.active_data, system_model.dt_in_hours)
            self.sum_TS = VariableTS('sum_TS', system_model.nr_of_time_steps, self.element.label_full, system_model,
                                     lower_bound=lb_TS, upper_bound=ub_TS)
            self.add_variables(self.sum_TS)

            # eq: sum_TS = sum(share_TS_i) # TS
            self._eq_time_series = Equation('time_series', self, system_model)
            self._eq_time_series.add_summand(self.sum_TS, -1)
            self.add_equations(self._eq_time_series)

            # eq: sum = sum(sum_TS(t)) # additionaly to self.sum
            self._eq_sum.add_summand(self.sum_TS, 1, as_sum=True)

    def add_variable_share(self,
                           system_model: SystemModel,
                           name_of_share: Optional[str],
                           share_holder: Element,
                           variable: Variable,
                           factor1: Numeric_TS,
                           factor2: Numeric_TS):  # if variable = None, then fix Share
        if variable is None:
            raise Exception('add_variable_share() needs variable as input. Use add_constant_share() instead')
        self._add_share(system_model, name_of_share, share_holder, variable, factor1, factor2)

    def add_constant_share(self,
                           system_model: SystemModel,
                           name_of_share: Optional[str],
                           share_holder: Element,
                           factor1: Numeric_TS,
                           factor2: Numeric_TS):
        variable = None
        self._add_share(system_model, name_of_share, share_holder, variable, factor1, factor2)

    def _add_share(self,
                   system_model: SystemModel,
                   name_of_share: Optional[str],
                   share_holder: Element,
                   variable: Optional[Variable],
                   factor1: Numeric_TS,
                   factor2: Numeric_TS):
        # TODO: accept only one factor or accept unlimited factors -> *factors

        # Falls TimeSeries, Daten auslesen:
        if isinstance(factor1, TimeSeries):
            factor1 = factor1.active_data
        if isinstance(factor2, TimeSeries):
            factor2 = factor2.active_data
        total_factor = np.multiply(factor1, factor2)

        # var and eq for publishing share-values in results:
        if name_of_share is not None:  # TODO: is this check necessary?
            new_share = SingleShareModel(share_holder, self._shares_are_time_series, name_of_share)
            new_share.do_modeling(system_model)
            new_share.add_summand_to_share(variable, total_factor)

        # Check to which equation the share should be added
        if self._shares_are_time_series:
            target_eq = self._eq_time_series
        else:
            # checking for single value
            assert any([np.issubdtype(type(total_factor), np.integer),
                        np.issubdtype(type(total_factor), np.floating),
                        isinstance(total_factor, Skalar)]) or total_factor.shape[0] == 1, \
                f'factor1 und factor2 müssen Skalare sein, da shareSum {self.element.label} skalar ist'
            target_eq = self._eq_sum

        if variable is None:  # constant share
            target_eq.add_constant(-1 * total_factor)
        else:  # variable share
            target_eq.add_summand(variable, total_factor)
        # TODO: Instead use new_share.single_share: Variable ?


class SingleShareModel(ElementModel):
    def __init__(self, element: Element, shares_are_time_series: bool, name_of_share: str):
        super().__init__(element)
        self.single_share: Optional[Variable] = None
        self._equation: Optional[Equation] = None
        self._full_name_of_share = f'{element.label_full}_{name_of_share}'
        self._shares_are_time_series = shares_are_time_series
        self._name_of_share = name_of_share

    def do_modeling(self, system_model: SystemModel):
        self.single_share = Variable(self._full_name_of_share, 1, self.element.label_full, system_model)
        self.add_variables(self.single_share)

        self._equation = Equation(self._full_name_of_share, self, system_model)
        self._equation.add_summand(self.single_share, -1)
        self.add_equations(self._equation)

    def add_summand_to_share(self, variable: Optional[Variable], factor: Numeric):
        """share to a sum"""
        if variable is None:  # if constant share:
            constant_value = np.sum(factor) if self._shares_are_time_series else factor
            self._equation.add_constant(-1 * constant_value)
        else:  # if variable share - always as a skalar -> as_sum=True if shares are timeseries
            self._equation.add_summand(variable, factor, as_sum=self._shares_are_time_series)


class SegmentedSharesModel(ElementModel):
    def __init__(self,
                 element: Element,
                 variable_segments: Tuple[Variable, List[Tuple[Skalar, Skalar]]],
                 share_segments: Dict['Effect', List[Tuple[Skalar, Skalar]]],
                 outside_segments: Optional[Variable]):
        super().__init__(element)
        assert len(variable_segments[1]) == len(list(share_segments.values())[0]), \
            f'Segment length of variable_segments and share_segments must be equal'
        self.element: Element
        self._outside_segments = outside_segments
        self._variable_segments = variable_segments
        self._share_segments = share_segments
        self._shares: Optional[Dict['Effect', SingleShareModel]] = None
        self._segments_model: Optional[MultipleSegmentsModel] = None

    def do_modeling(self, system_model: SystemModel):
        self._shares = {effect: SingleShareModel(self.element, False, f'segmented')
                        for effect in self._share_segments}
        for single_share in self._shares.values():
            single_share.do_modeling(system_model)

        segments: Dict[Variable, List[Tuple[Skalar, Skalar]]] = {
            self._shares[effect].single_share: segment for effect, segment in self._share_segments.values()}
        self._segments_model = MultipleSegmentsModel(self.element, self._outside_segments, segments)
        self._segments_model.do_modeling(system_model)

        # Shares
        effect_collection = system_model.flow_system.effect_collection
        for effect, single_share_model in self._shares.items():
            effect_collection.add_share_to_invest(
                name_of_share='segmented_effects',
                owner=self.element, variable=single_share_model.single_share,
                effect_values={effect: 1},
                factor=1
            )


class PreventSimultaneousUsageModel(ElementModel):
    """
    Prevents multiple Multiple Binary variables from being 1 at the same time

    Only 'classic type is modeled for now (# "classic" -> alle Flows brauchen Binärvariable:)
    In 'new', the binary Variables need to be forced beforehand, which is not that straight forward... --> TODO maybe


    # "new":
    # eq: flow_1.on(t) + flow_2.on(t) + .. + flow_i.val(t)/flow_i.max <= 1 (1 Flow ohne Binärvariable!)

    # Anmerkung: Patrick Schönfeld (oemof, custom/link.py) macht bei 2 Flows ohne Binärvariable dies:
    # 1)	bin + flow1/flow1_max <= 1
    # 2)	bin - flow2/flow2_max >= 0
    # 3)    geht nur, wenn alle flow.min >= 0
    # --> könnte man auch umsetzen (statt force_on_variable() für die Flows, aber sollte aufs selbe wie "new" kommen)
    """
    def __init__(self,
                 element: Element,
                 variables: List[VariableTS]):
        super().__init__(element)
        self._variables = variables
        assert len(self._variables) >= 2, f'Model {self.__class__.__name__} must get at least two variables'
        for variable in self._variables:  # classic
            assert variable.is_binary, f'Variable {variable} must be binary for use in {self.__class__.__name__}'

    def do_modeling(self, system_model: SystemModel):
        # eq: sum(flow_i.on(t)) <= 1.1 (1 wird etwas größer gewählt wg. Binärvariablengenauigkeit)
        eq = Equation('prevent_simultaneous_use', self, system_model, eqType='ineq')
        self.add_equations(eq)
        for variable in self._variables:
            eq.add_summand(variable, 1)
        eq.add_constant(1.1)
