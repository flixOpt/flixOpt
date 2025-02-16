"""
This module contains the features of the flixOpt framework.
Features extend the functionality of Elements.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import linopy
import numpy as np

from .config import CONFIG
from .core import Numeric, Skalar, TimeSeries
from .interface import InvestParameters, OnOffParameters
from .math_modeling import Equation, Variable, VariableTS
from .structure import (
    Element,
    ElementModel,
    InterfaceModel,
    Interface,
    Model,
    SystemModel,
    create_equation,
    create_variable,
)

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from .components import Storage
    from .effects import Effect
    from .elements import Flow


logger = logging.getLogger('flixOpt')


class InvestmentModel(Model):
    """Class for modeling an investment"""

    def __init__(
        self,
        model: SystemModel,
        label_of_parent: str,
        parameters: InvestParameters,
        defining_variable: [linopy.Variable],
        relative_bounds_of_defining_variable: Tuple[Numeric, Numeric],
        fixed_relative_profile: Optional[Numeric] = None,
        label: str = 'Investment',
        on_variable: Optional[linopy.Variable] = None,
    ):
        """
        If fixed relative profile is used, the relative bounds are ignored
        """
        super().__init__(model, label_of_parent, label)
        self.size: Optional[Union[Skalar, Variable]] = None
        self.is_invested: Optional[Variable] = None

        self._segments: Optional[SegmentedSharesModel] = None

        self._on_variable = on_variable
        self._defining_variable = defining_variable
        self._relative_bounds_of_defining_variable = relative_bounds_of_defining_variable
        self._fixed_relative_profile = fixed_relative_profile
        self.parameters = parameters

    def do_modeling(self, system_model: SystemModel):
        if self.parameters.fixed_size and not self.parameters.optional:
            self.size = self.add(self._model.add_variables(
                lower=self.parameters.fixed_size,
                upper=self.parameters.fixed_size,
                name=f'{self.label_full}__size'),
                'size')
        else:
            self.size = self.add(self._model.add_variables(
                lower=0 if self.parameters.optional else self.parameters.minimum_size,
                upper=self.parameters.maximum_size,
                name=f'{self.label_full}__size'),
                'size')

        # Optional
        if self.parameters.optional:
            self.is_invested = self.add(self._model.add_variables(
                binary=True,
                name=f'{self.label_full}__is_invested'),
                'is_invested')

            self._create_bounds_for_optional_investment()

        # Bounds for defining variable
        self._create_bounds_for_defining_variable()

        self._create_shares(system_model)

    def _create_shares(self, system_model: SystemModel):

        # fix_effects:
        fix_effects = self.parameters.fix_effects
        if fix_effects != {}:
            self._model.effects.add_share_to_effects(
                system_model=self._model,
                name=self._label_of_parent,
                expressions={effect: self.is_invested * factor if self.is_invested is not None else factor
                             for effect, factor in fix_effects.items()},
                target='invest',
            )

        if self.parameters.divest_effects != {} and self.parameters.optional:
            # share: divest_effects - isInvested * divest_effects
            self._model.effects.add_share_to_effects(
                system_model=self._model,
                name=self._label_of_parent,
                expressions={effect: -self.is_invested * factor + factor for effect, factor in fix_effects.items()},
                target='invest',
            )

        if self.parameters.specific_effects != {}:
            self._model.effects.add_share_to_effects(
                system_model=self._model,
                name=self._label_of_parent,
                expressions={effect: self.size * factor for effect, factor in self.parameters.specific_effects.items()},
                target='invest',
            )

        if self.parameters.effects_in_segments:
            self._segments = self.add(
                SegmentedSharesModel(
                    model=self._model,
                    label_of_parent=self._label_of_parent,
                    variable_segments=(self.size, self.parameters.effects_in_segments[0]),
                    share_segments=self.parameters.effects_in_segments[1],
                    can_be_outside_segments=self.is_invested),
                'segments'
            )
            self._segments.do_modeling(self._model)

    def _create_bounds_for_optional_investment(self):
        if self.parameters.fixed_size:
            # eq: investment_size = isInvested * fixed_size
            self.add(self._model.add_constraints(
                self.size == self.is_invested * self.parameters.fixed_size,
                name=f'{self.label_full}__is_invested'),
                'is_invested')

        else:
            # eq1: P_invest <= isInvested * investSize_max
            self.add(self._model.add_constraints(
                self.size == self.is_invested * self.parameters.maximum_size,
                name=f'{self.label_full}__is_invested_ub'),
                'is_invested_ub')

            # eq2: P_invest >= isInvested * max(epsilon, investSize_min)
            self.add(self._model.add_constraints(
                self.size >= self.is_invested * np.maximum(CONFIG.modeling.EPSILON,self.parameters.minimum_size),
                name=f'{self.label_full}__is_invested_lb'),
                'is_invested_lb')

    def _create_bounds_for_defining_variable(self):
        variable = self._defining_variable
        # fixed relative value
        if self._fixed_relative_profile is not None:
            # TODO: Allow Off? Currently not..
            self.add(self._model.add_constraints(
                variable == self.size * self._fixed_relative_profile,
                name=f'{self.label_full}__fixed_{variable.name}'),
                f'fixed_{variable.name}')

        else:
            lb_relative, ub_relative = self._relative_bounds_of_defining_variable
            # eq: defining_variable(t)  <= size * upper_bound(t)
            self.add(self._model.add_constraints(
                variable <= self.size * ub_relative,
                name=f'{self.label_full}__ub_{variable.name}'),
                f'ub_{variable.name}')

            if self._on_variable is None:
                # eq: defining_variable(t) >= investment_size * relative_minimum(t)
                self.add(self._model.add_constraints(
                    variable >= self.size * lb_relative,
                    name=f'{self.label_full}__lb_{variable.name}'),
                    f'lb_{variable.name}')
            else:
                ## 2. Gleichung: Minimum durch Investmentgröße und On
                # eq: defining_variable(t) >= mega * (On(t)-1) + size * relative_minimum(t)
                #     ... mit mega = relative_maximum * maximum_size
                # äquivalent zu:.
                # eq: - defining_variable(t) + mega * On(t) + size * relative_minimum(t) <= + mega
                mega = lb_relative * self.parameters.maximum_size
                on = self._on_variable
                self.add(self._model.add_constraints(
                    variable >= mega * (on - 1) + self.size * lb_relative,
                    name=f'{self.label_full}__lb_{variable.name}'),
                    f'lb_{variable.name}')
                # anmerkung: Glg bei Spezialfall relative_minimum = 0 redundant zu OnOff ??


class OnOffModel(Model):
    """
    Class for modeling the on and off state of a variable
    If defining_bounds are given, creates sufficient lower bounds
    """

    def __init__(
        self,
        model: SystemModel,
        on_off_parameters: OnOffParameters,
        label_of_parent: str,
        defining_variables: List[linopy.Variable],
        defining_bounds: List[Tuple[Numeric, Numeric]],
        label: str = 'OnOffModel',
    ):
        """
        defining_bounds: a list of Numeric, that can be  used to create the bound for On/Off more efficiently
        """
        super().__init__(model, label_of_parent, label)
        assert len(defining_variables) == len(defining_bounds), 'Every defining Variable needs bounds to Model OnOff'
        self.parameters = on_off_parameters
        self.on: Optional[linopy.Variable] = None
        self.total_on_hours: Optional[Variable] = None

        self.consecutive_on_hours: Optional[linopy.Variable] = None
        self.consecutive_off_hours: Optional[linopy.Variable] = None

        self.off: Optional[linopy.Variable] = None

        self.switch_on: Optional[linopy.Variable] = None
        self.switch_off: Optional[linopy.Variable] = None
        self.switch_on_nr: Optional[linopy.Variable] = None

        self._defining_variables = defining_variables
        self._defining_bounds = defining_bounds

    def do_modeling(self, system_model: SystemModel):
        self.on = self.add(
            self._model.add_variables(
                name=f'{self.label_full}__on',
                binary=True,
                coords=system_model.coords,
                #TODO: previous_values=self._previous_on_values(CONFIG.modeling.EPSILON)
            ),
            'on',
        )

        self.total_on_hours = self.add(
            self._model.add_variables(
                lower=self.parameters.on_hours_total_min if self.parameters.on_hours_total_min is not None else 0,
                upper=self.parameters.on_hours_total_max if self.parameters.on_hours_total_max is not None else np.inf,
                name=f'{self.label_full}__on_hours_total'
            ),
            'on_hours_total'
        )

        self.add(
            self._model.add_constraints(
                self.total_on_hours == (self.on * self._model.hours_per_step).sum(),
                name=f'{self.label_full}__on_hours_total'
            ),
            'on_hours_total'
        )

        self._add_on_constraints()

        if self.parameters.use_off:
            self.off = self.add(
                self._model.add_variables(
                    name=f'{self.label_full}__off',
                    binary=True,
                    coords=system_model.coords,
                    # TODO: previous_values=1 - self._previous_on_values(CONFIG.modeling.EPSILON),
                ),
                'off'
            )

            # eq: var_on(t) + var_off(t) = 1
            self.add(self._model.add_constraints(self.on + self.off == 1, name=f'{self.label_full}__off'), 'off')

        if self.parameters.use_consecutive_on_hours:
            # TODO: Implement consecutive_on_hours
            if False:
                self.consecutive_on_hours = self._get_duration_in_hours(
                    'consecutiveOnHours',
                    self.on,
                    self.parameters.consecutive_on_hours_min,
                    self.parameters.consecutive_on_hours_max,
                    system_model,
                    system_model.indices,
                )

        if self.parameters.use_consecutive_off_hours:
            # TODO: Implement consecutive_on_hours
            if False:
                self.consecutive_off_hours = self._get_duration_in_hours(
                    'consecutiveOffHours',
                    self.off,
                    self.parameters.consecutive_off_hours_min,
                    self.parameters.consecutive_off_hours_max,
                    system_model,
                    system_model.indices,
                )

        if self.parameters.use_switch_on:
            self.switch_on = self.add(self._model.add_variables(
                binary=True, name=f'{self.label_full}__switch_on', coords=system_model.coords),'switch_on')

            self.switch_off = self.add(self._model.add_variables(
                binary=True, name=f'{self.label_full}__switch_off', coords=system_model.coords), 'switch_off')

            self.switch_on_nr = self.add(self._model.add_variables(
                upper=self.parameters.switch_on_total_max if self.parameters.switch_on_total_max is not None else np.inf,
                name=f'{self.label_full}__switch_on_nr',
                coords=system_model.coords),
                'switch_on_nr')

            self._add_switch_constraints(system_model)

        self._create_shares(system_model)

    def _add_on_constraints(self):
        assert self.on is not None, f'On variable of {self.label_full} must be defined to add constraints'
        # % Bedingungen 1) und 2) müssen erfüllt sein:

        # % Anmerkung: Falls "abschnittsweise linear" gewählt, dann ist eigentlich nur Bedingung 1) noch notwendig
        # %            (und dann auch nur wenn erstes Segment bei Q_th=0 beginnt. Dann soll bei Q_th=0 (d.h. die Maschine ist Aus) On = 0 und segment1.onSeg = 0):)
        # %            Fazit: Wenn kein Performance-Verlust durch mehr Gleichungen, dann egal!

        nr_of_def_vars = len(self._defining_variables)
        assert nr_of_def_vars > 0, 'Achtung: mindestens 1 Flow notwendig'
        EPSILON = CONFIG.modeling.EPSILON

        if nr_of_def_vars == 1:
            def_var = self._defining_variables[0]
            lb, ub = self._defining_bounds[0]

            # eq: On(t) * max(epsilon, lower_bound) <= Q_th(t)
            self.add(
                self._model.add_constraints(
                    self.on * np.maximum(EPSILON, lb) <= def_var,
                    name=f'{self.label_full}__on_con1'
                ),
                'on_con1'
            )

            # eq: Q_th(t) <= Q_th_max * On(t)
            self.add(
                self._model.add_constraints(
                    self.on * np.maximum(EPSILON, ub) >= def_var,
                    name=f'{self.label_full}__on_con2'
                ),
                'on_con2'
            )

        else:  # Bei mehreren Leistungsvariablen:
            ub = sum(bound[1] for bound in self._defining_bounds)
            lb = EPSILON

            # When all defining variables are 0, On is 0
            # eq: On(t) * Epsilon <= sum(alle Leistungen(t))
            self.add(
                self._model.add_constraints(
                    self.on * lb <= sum(self._defining_variables),
                    name=f'{self.label_full}__on_con1'
                ),
                'on_con1'
            )

            ## sum(alle Leistung) >0 -> On = 1 | On=0 -> sum(Leistung)=0
            #  eq: sum( Leistung(t,i))              - sum(Leistung_max(i))             * On(t) <= 0
            #  --> damit Gleichungswerte nicht zu groß werden, noch durch nr_of_flows geteilt:
            #  eq: sum( Leistung(t,i) / nr_of_flows ) - sum(Leistung_max(i)) / nr_of_flows * On(t) <= 0
            self.add(
                self._model.add_constraints(
                    self.on * ub >= sum([def_var / nr_of_def_vars for def_var in self._defining_variables]),
                    name=f'{self.label_full}__on_con2'
                ),
                'on_con2'
            )

        if np.max(ub) > CONFIG.modeling.BIG_BINARY_BOUND:
            logger.warning(
                f'In "{self.label_full}", a binary definition was created with a big upper bound '
                f'({np.max(ub)}). This can lead to wrong results regarding the on and off variables. '
                f'Avoid this warning by reducing the size of {self.label_full} '
                f'(or the maximum_size of the corresponding InvestParameters). '
                f'If its a Component, you might need to adjust the sizes of all of its flows.'
            )

    def _get_duration_in_hours(
        self,
        variable_label: str,
        binary_variable: linopy.Variable,
        minimum_duration: Optional[TimeSeries],
        maximum_duration: Optional[TimeSeries],
        system_model: SystemModel,
        time_indices: Union[list[int], range],
    ) -> linopy.Variable:
        """
        creates duration variable and adds constraints to a time-series variable to enforce duration limits based on
        binary activity.
        The minimum duration in the last time step is not restricted.
        Previous values before t=0 are not recognised!

        Parameters:
            variable_label (str):
                Label for the duration variable to be created.
            binary_variable (linopy.Variable):
                Time-series binary variable (e.g., [0, 0, 1, 1, 1, 0, ...]) representing activity states.
            minimum_duration (Optional[TimeSeries]):
                Minimum duration the activity must remain active once started.
                If None, no minimum duration constraint is applied.
            maximum_duration (Optional[TimeSeries]):
                Maximum duration the activity can remain active.
                If None, the maximum duration is set to the total available time.
            system_model (SystemModel):
                The system model containing time step information.
            time_indices (Union[list[int], range]):
                List or range of indices to which to apply the constraints.

        Returns:
            linopy.Variable: The created duration variable representing consecutive active durations.

        Example:
            binary_variable: [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, ...]
            duration_in_hours: [0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0, ...] (only if dt_in_hours=1)

            Here, duration_in_hours increments while binary_variable is 1. Minimum and maximum durations
            can be enforced to constrain how long the activity remains active.

        Notes:
            - To count consecutive zeros instead of ones, use a transformed binary variable
              (e.g., `1 - binary_variable`).
            - Constraints ensure the duration variable properly resets or increments based on activity.

        Raises:
            AssertionError: If the binary_variable is None, indicating the duration constraints cannot be applied.

        """
        try:
            previous_duration: Skalar = self.get_consecutive_duration(
                binary_variable.previous_values, system_model.previous_dt_in_hours
            )
        except TypeError as e:
            raise TypeError(f'The consecutive_duration of "{variable_label}" could not be calculated. {e}') from e
        mega = system_model.dt_in_hours_total + previous_duration

        if maximum_duration is not None:
            first_step_max: Skalar = (
                maximum_duration.active_data[0] if maximum_duration.is_array else maximum_duration.active_data
            )
            if previous_duration + system_model.dt_in_hours[0] > first_step_max:
                logger.warning(
                    f'The maximum duration of "{variable_label}" is set to {maximum_duration.active_data}h, '
                    f'but the consecutive_duration previous to this model is {previous_duration}h. '
                    f'This forces "{binary_variable.label} = 0" in the first time step '
                    f'(dt={system_model.dt_in_hours[0]}h)!'
                )

        duration_in_hours = create_variable(
            variable_label,
            self,
            system_model.nr_of_time_steps,
            lower_bound=0,
            upper_bound=maximum_duration.active_data if maximum_duration is not None else mega,
            previous_values=previous_duration,
        )
        label_prefix = duration_in_hours.label

        assert binary_variable is not None, f'Duration Variable of {self.element} must be defined to add constraints'
        # TODO: Einfachere Variante von Peter umsetzen!

        # 1) eq: duration(t) - On(t) * BIG <= 0
        constraint_1 = create_equation(f'{label_prefix}_constraint_1', self, eq_type='ineq')
        constraint_1.add_summand(duration_in_hours, 1)
        constraint_1.add_summand(binary_variable, -1 * mega)

        # 2a) eq: duration(t) - duration(t-1) <= dt(t)
        #    on(t)=1 -> duration(t) - duration(t-1) <= dt(t)
        #    on(t)=0 -> duration(t-1) >= negat. value
        constraint_2a = create_equation(f'{label_prefix}_constraint_2a', self, eq_type='ineq')
        constraint_2a.add_summand(duration_in_hours, 1, time_indices[1:])  # duration(t)
        constraint_2a.add_summand(duration_in_hours, -1, time_indices[0:-1])  # duration(t-1)
        constraint_2a.add_constant(system_model.dt_in_hours[1:])  # dt(t)

        # 2b) eq: dt(t) - BIG * ( 1-On(t) ) <= duration(t) - duration(t-1)
        # eq: -duration(t) + duration(t-1) + On(t) * BIG <= -dt(t) + BIG
        # with BIG = dt_in_hours_total.
        #   on(t)=1 -> duration(t)- duration(t-1) >= dt(t)
        #   on(t)=0 -> duration(t)- duration(t-1) >= negat. value

        constraint_2b = create_equation(f'{label_prefix}_constraint_2b', self, eq_type='ineq')
        constraint_2b.add_summand(duration_in_hours, -1, time_indices[1:])  # duration(t)
        constraint_2b.add_summand(duration_in_hours, 1, time_indices[0:-1])  # duration(t-1)
        constraint_2b.add_summand(binary_variable, mega, time_indices[1:])  # on(t)
        constraint_2b.add_constant(-1 * system_model.dt_in_hours[1:] + mega)  # dt(t)

        # 3) check minimum_duration before switchOff-step

        if minimum_duration is not None:
            # Note: switchOff-step is when: On(t) - On(t+1) == 1
            # Note: (last on-time period (with last timestep of period t=n) is not checked and can be shorter)
            # Note: (previous values before t=1 are not recognised!)
            # eq: duration(t) >= minimum_duration(t) * [On(t) - On(t+1)] for t=1..(n-1)
            # eq: -duration(t) + minimum_duration(t) * On(t) - minimum_duration(t) * On(t+1) <= 0
            if minimum_duration.is_scalar:
                minimum_duration_used = minimum_duration.active_data
            else:
                minimum_duration_used = minimum_duration.active_data[0:-1]  # only checked for t=1...(n-1)
            eq_min_duration = create_equation(f'{label_prefix}_minimum_duration', self, eq_type='ineq')
            eq_min_duration.add_summand(duration_in_hours, -1, time_indices[0:-1])  # -duration(t)
            eq_min_duration.add_summand(
                binary_variable, -1 * minimum_duration_used, time_indices[1:]
            )  # - minimum_duration (t) * On(t+1)
            eq_min_duration.add_summand(
                binary_variable, minimum_duration_used, time_indices[0:-1]
            )  # minimum_duration * On(t)

            first_step_min: Skalar = (
                minimum_duration.active_data[0] if minimum_duration.is_array else minimum_duration.active_data
            )
            if duration_in_hours.previous_values < first_step_min:
                # Force the first step to be = 1, if the minimum_duration is not reached in previous_values
                # Note: Only if the previous consecutive_duration is smaller than the minimum duration
                # eq: On(t=0) = 1
                eq_min_duration_inital = create_equation(f'{label_prefix}_minimum_duration_inital', self, eq_type='eq')
                eq_min_duration_inital.add_summand(binary_variable, 1, time_indices[0])
                eq_min_duration_inital.add_constant(1)

        # 4) first index:
        # eq: duration(t=0)= dt(0) * On(0)
        first_index = time_indices[0]  # only first element
        eq_first = create_equation(f'{label_prefix}_initial', self)
        eq_first.add_summand(duration_in_hours, 1, first_index)
        eq_first.add_summand(
            binary_variable,
            -1 * (system_model.dt_in_hours[first_index] + duration_in_hours.previous_values),
            first_index,
        )

        return duration_in_hours

    def _add_switch_constraints(self, system_model: SystemModel):
        assert self.switch_on is not None, f'Switch On Variable of {self.label_full} must be defined to add constraints'
        assert self.switch_off is not None, f'Switch Off Variable of {self.label_full} must be defined to add constraints'
        assert self.switch_on_nr is not None, (
            f'Nr of Switch On Variable of {self.label_full} must be defined to add constraints'
        )
        assert self.on is not None, f'On Variable of {self.label_full} must be defined to add constraints'
        # % Schaltänderung aus On-Variable
        # % SwitchOn(t)-SwitchOff(t) = On(t)-On(t-1)
        self.add(
            self._model.add_constraints(
                self.switch_on.isel(time=slice(1, None)) - self.switch_off.isel(time=slice(1, None))
                ==
                self.on.isel(time=slice(1,None)) - self.on.isel(time=slice(None,-1)),
                name=f'{self.label_full}__switch_con'
            ),
            'switch_con'
        )
        # Initital switch on
        # eq: SwitchOn(t=0)-SwitchOff(t=0) = On(t=0) - On(t=-1)
        self.add(
            self._model.add_constraints(
                self.switch_on.isel(time=0) - self.switch_off.isel(time=0)
                ==
                self.on.isel(time=0), #TODO:  - self.on.previous_values[-1]
                name=f'{self.label_full}__initial_switch_con'
            ),
            'initial_switch_con'
        )
        ## Entweder SwitchOff oder SwitchOn
        # eq: SwitchOn(t) + SwitchOff(t) <= 1.1
        self.add(
            self._model.add_constraints(
                self.switch_on + self.switch_off <= 1.1,
                name=f'{self.label_full}__switch_on_or_off'
            ),
            'switch_on_or_off'
        )

        ## Anzahl Starts:
        # eq: nrSwitchOn = sum(SwitchOn(t))
        self.add(
            self._model.add_constraints(
                self.switch_on_nr == self.switch_on.sum(),
                name=f'{self.label_full}__switch_on_nr'
            ),
            'switch_on_nr'
        )

    def _create_shares(self, system_model: SystemModel):
        # Anfahrkosten:
        effects_per_switch_on = self.parameters.effects_per_switch_on
        if effects_per_switch_on != {}:
            self._model.effects.add_share_to_effects(
                system_model=self._model,
                name=self._label_of_parent,
                expressions={effect: self.switch_on * factor for effect, factor in effects_per_switch_on.items()},
                target='operation',
            )

        # Betriebskosten:
        effects_per_running_hour = self.parameters.effects_per_running_hour
        if effects_per_running_hour != {}:
            self._model.effects.add_share_to_effects(
                system_model=self._model,
                name=self._label_of_parent,
                expressions={effect: self.on * factor * self._model.hours_per_step
                             for effect, factor in effects_per_running_hour.items()},
                target='operation',
            )

    def _previous_on_values(self, epsilon: float = 1e-5) -> np.ndarray:
        """
        Returns the previous 'on' states of defining variables as a binary array.

        Parameters:
        ----------
        epsilon : float, optional
            Tolerance for equality to determine "off" state, default is 1e-5.

        Returns:
        -------
        np.ndarray
            A binary array (0 and 1) indicating the previous on/off states of the variables.
            Returns `array([0])` if no previous values are available.
        """
        previous_values = [var.previous_values for var in self._defining_variables if var.previous_values is not None]

        if not previous_values:
            return np.array([0])
        else:  # Convert to 2D-array and compute binary on/off states
            previous_values = np.array(previous_values)
            if previous_values.ndim > 1:
                return np.any(~np.isclose(previous_values, 0, atol=epsilon), axis=0).astype(int)
            else:
                return (~np.isclose(previous_values, 0, atol=epsilon)).astype(int)

    @classmethod
    def get_consecutive_duration(
        cls, binary_values: Union[int, np.ndarray], dt_in_hours: Union[int, float, np.ndarray]
    ) -> Skalar:
        """
        Returns the current consecutive duration in hours, computed from binary values.
        If only one binary value is availlable, the last dt_in_hours is used.
        Of both binary_values and dt_in_hours are arrays, checks that the length of dt_in_hours has at least as
        many elements as the last  consecutive duration in binary_values.

        Parameters
        ----------
        binary_values : int, np.ndarray
            An int or 1D binary array containing only `0`s and `1`s.
        dt_in_hours : int, float, np.ndarray
            The duration of each time step in hours.

        Returns
        -------
        np.ndarray
            The duration of the binary variable in hours.

        Raises
        ------
        TypeError
            If the length of binary_values and dt_in_hours is not equal, but None is a scalar.
        """
        if np.isscalar(binary_values) and np.isscalar(dt_in_hours):
            return binary_values * dt_in_hours
        elif np.isscalar(binary_values) and not np.isscalar(dt_in_hours):
            return binary_values * dt_in_hours[-1]

        # Find the indexes where value=`0` in a 1D-array
        zero_indices = np.where(np.isclose(binary_values, 0, atol=CONFIG.modeling.EPSILON))[0]
        length_of_last_duration = zero_indices[-1] + 1 if zero_indices.size > 0 else len(binary_values)

        if not np.isscalar(binary_values) and np.isscalar(dt_in_hours):
            return np.sum(binary_values[-length_of_last_duration:] * dt_in_hours)

        elif not np.isscalar(binary_values) and not np.isscalar(dt_in_hours):
            if length_of_last_duration > len(dt_in_hours):  # check that lengths are compatible
                raise TypeError(
                    f'When trying to calculate the consecutive duration, the length of the last duration '
                    f'({len(length_of_last_duration)}) is longer than the dt_in_hours ({len(dt_in_hours)}), '
                    f'as {binary_values=}'
                )
            return np.sum(binary_values[-length_of_last_duration:] * dt_in_hours[-length_of_last_duration:])

        else:
            raise Exception(
                f'Unexpected state reached in function get_consecutive_duration(). binary_values={binary_values}; '
                f'dt_in_hours={dt_in_hours}'
            )


class SegmentModel(Model):
    """Class for modeling a linear segment of one or more variables in parallel"""

    def __init__(
        self,
        model: SystemModel,
        label_of_parent: str,
        segment_index: Union[int, str],
        sample_points: Dict[str, Tuple[Union[Numeric, TimeSeries], Union[Numeric, TimeSeries]]],
        as_time_series: bool = True,
    ):
        super().__init__(model, label_of_parent, f'Segment{segment_index}')
        self.in_segment: Optional[VariableTS] = None
        self.lambda0: Optional[VariableTS] = None
        self.lambda1: Optional[VariableTS] = None

        self._segment_index = segment_index
        self._as_time_series = as_time_series
        self.sample_points = sample_points

    def do_modeling(self, system_model: SystemModel):
        self.in_segment = self.add(self._model.add_variables(
            binary=True,
            name=f'{self.label_full}__in_segment',
            coords=system_model.coords if self._as_time_series else None),
            'in_segment'
        )

        self.lambda0 = self.add(self._model.add_variables(
            lower=0, upper=1,
            name=f'{self.label_full}__lambda0',
            coords=system_model.coords if self._as_time_series else None),
            'lambda0'
        )

        self.lambda1 = self.add(self._model.add_variables(
            lower=0, upper=1,
            name=f'{self.label_full}__lambda1',
            coords=system_model.coords if self._as_time_series else None),
            'lambda1'
        )

        # eq:  lambda0(t) + lambda1(t) = in_segment(t)
        self.add(self._model.add_constraints(
            self.in_segment == self.lambda0 + self.lambda1,
            name=f'{self.label_full}__in_segment'),
            'in_segment'
        )


class MultipleSegmentsModel(Model):
    # TODO: Length...
    def __init__(
        self,
        model: SystemModel,
        label_of_parent: str,
        sample_points: Dict[str, List[Tuple[Numeric, Numeric]]],
        can_be_outside_segments: Optional[Union[bool, Variable]],
        as_time_series: bool = True,
        label: str = 'MultipleSegments',
    ):
        """
        Parameters
        ----------
        model : linopy.Model
            Model to which the segmented variable belongs.
        label_of_parent : str
            Name of the parent variable.
        sample_points : dict[str, list[tuple[float, float]]]
            Dictionary mapping variables (names) to their sample points for each segment.
            The sample points are tuples of the form (start, end).
        can_be_outside_segments : bool or linopy.Variable, optional
            Whether the variable can be outside the segments. If True, a variable is created.
            If False or None, no variable is created. If a Variable is passed, it is used.
        as_time_series : bool, optional
        """
        super().__init__(model, label_of_parent, label)
        self.outside_segments: Optional[linopy.Variable] = None

        self._as_time_series = as_time_series
        self._can_be_outside_segments = can_be_outside_segments
        self._sample_points = sample_points
        self._segment_models: List[SegmentModel] = []

    def do_modeling(self, system_model: SystemModel):
        restructured_variables_with_segments: List[Dict[str, Tuple[Numeric, Numeric]]] = [
            {key: values[i] for key, values in self._sample_points.items()} for i in range(self._nr_of_segments)
        ]

        self._segment_models = [
            self.add(
                SegmentModel(
                    self._model,
                    label_of_parent=self._label_of_parent,
                    segment_index=i,
                    sample_points=sample_points,
                    as_time_series=self._as_time_series),
                f'Segment_{i}')
            for i, sample_points in enumerate(restructured_variables_with_segments)
        ]

        for segment_model in self._segment_models:
            segment_model.do_modeling(system_model)

        #  eq: - v(t) + (v_0_0 * lambda_0_0 + v_0_1 * lambda_0_1) + (v_1_0 * lambda_1_0 + v_1_1 * lambda_1_1) ... = 0
        #  -> v_0_0, v_0_1 = Stützstellen des Segments 0
        for var_name in self._sample_points.keys():
            variable = self._model.variables[var_name]
            self.add(self._model.add_constraints(
                variable == sum([segment.lambda0 * segment.sample_points[var_name][0]
                                 + segment.lambda1 * segment.sample_points[var_name][1]
                                 for segment in self._segment_models]),
                name=f'{self.label_full}__{var_name}_lambda'),
                f'{var_name}_lambda'
            )

            # a) eq: Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 1                Aufenthalt nur in Segmenten erlaubt
            # b) eq: -On(t) + Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 0       zusätzlich kann alles auch Null sein
            if isinstance(self._can_be_outside_segments, linopy.Variable):
                self.outside_segments = self._can_be_outside_segments
                rhs = self.outside_segments
            elif self._can_be_outside_segments is True:
                self.outside_segments = self.add(self._model.add_variables(
                    coords=self._model.coords,
                    binary=True,
                    name=f'{self.label_full}__outside_segments'),
                    'outside_segments'
                )
                rhs = self.outside_segments
            else:
                rhs = 1

            self.add(self._model.add_constraints(
                sum([segment.in_segment for segment in self._segment_models]) <= rhs,
                name=f'{self.label_full}__{variable.name}_single_segment'),
                f'single_segment'
            )

    @property
    def _nr_of_segments(self):
        return len(next(iter(self._sample_points.values())))


class ShareAllocationModel(Model):
    def __init__(
        self,
        model: SystemModel,
        shares_are_time_series: bool,
        label_of_parent: Optional[str] = None,
        label: Optional[str] = None,
        total_max: Optional[Skalar] = None,
        total_min: Optional[Skalar] = None,
        max_per_hour: Optional[Numeric] = None,
        min_per_hour: Optional[Numeric] = None,
    ):
        super().__init__(model, label_of_parent=label_of_parent, label=label)
        if not shares_are_time_series:  # If the condition is True
            assert max_per_hour is None and min_per_hour is None, (
                'Both max_per_hour and min_per_hour cannot be used when shares_are_time_series is False'
            )
        self.total_per_timestep: Optional[linopy.Variable] = None
        self.total: Optional[linopy.Variable] = None
        self.shares: Dict[str, linopy.Variable] = {}
        self.share_constraints: Dict[str, linopy.Constraint] = {}

        self._eq_total_per_timestep: Optional[linopy.Constraint] = None
        self._eq_total: Optional[linopy.Constraint] = None

        # Parameters
        self._shares_are_time_series = shares_are_time_series
        self._total_max = total_max if total_min is not None else np.inf
        self._total_min = total_min if total_min is not None else -np.inf
        self._max_per_hour = max_per_hour if max_per_hour is not None else np.inf
        self._min_per_hour = min_per_hour if min_per_hour is not None else -np.inf

    def do_modeling(self, system_model: SystemModel):
        self.total = self.add(
            system_model.add_variables(
                lower=self._total_min, upper=self._total_max, coords=None, name=f'{self.label_full}__total'
            ),
            'total'
        )
        # eq: sum = sum(share_i) # skalar
        self._eq_total = self.add(system_model.add_constraints(self.total == 0, name=f'{self.label_full}__total'), 'total')

        if self._shares_are_time_series:
            self.total_per_timestep = self.add(
                    system_model.add_variables(
                    lower=-np.inf if (self._min_per_hour is None) else np.multiply(self._min_per_hour, system_model.hours_per_step),
                    upper=np.inf if (self._max_per_hour is None) else np.multiply(self._max_per_hour, system_model.hours_per_step),
                    coords=system_model.coords,
                    name=f'{self.label_full}_total_per_timestep'
                ),
                'total_per_timestep'
            )

            self._eq_total_per_timestep = self.add(
                system_model.add_constraints(self.total_per_timestep == 0, name=f'{self.label_full}__total_per_timestep'),
                'total_per_timestep'
            )

            # Add it to the total
            self._eq_total.lhs -= self.total_per_timestep.sum()

    def add_share(
        self,
        system_model: SystemModel,
        name: str,
        expression: linopy.LinearExpression,
    ):
        """
        Add a share to the share allocation model. If the share already exists, the expression is added to the existing share.
        The expression is added to the right hand side (rhs) of the constraint.
        The variable representing the total share is on the left hand side (lhs) of the constraint.
        var_total = sum(expressions)

        Parameters
        ----------
        system_model : SystemModel
            The system model.
        name : str
            The name of the share.
        expression : linopy.LinearExpression
            The expression of the share. Added to the right hand side of the constraint.
        """
        if name in self.shares:
            self.share_constraints[name].lhs -= expression
        else:
            self.shares[name] = self.add(
                system_model.add_variables(
                    coords=None if isinstance(expression, linopy.LinearExpression) and expression.ndim == 0 or not isinstance(expression, linopy.LinearExpression) else system_model.coords,
                    name=f'{name}__{self.label_full}'
                ),
                name
            )
            self.share_constraints[name] = self.add(
                system_model.add_constraints(
                    self.shares[name] == expression, name=f'{name}__{self.label_full}'
                ),
                name
            )
            if self.shares[name].ndim == 0:
                self._eq_total.lhs -= self.shares[name]
            else:
                self._eq_total_per_timestep.lhs -= self.shares[name]

    def solution_structured(
        self,
        use_numpy: bool = True,
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        shares_var_names = [var.name for var in self.shares.values()]
        results = {
            self._variables_short[var_name]: var.values
            for var_name, var in self.variables.solution.data_vars.items() if var_name not in shares_var_names
        }
        results['Shares'] = {
            self._variables_short[var_name]: var.values
            for var_name, var in self.variables.solution.data_vars.items() if var_name in shares_var_names
        }
        return {
            **results,
            **{sub_model.label: sub_model.solution_structured(use_numpy) for sub_model in self.sub_models}
        }


class SegmentedSharesModel(Model):
    # TODO: Length...
    def __init__(
        self,
        model: SystemModel,
        label_of_parent: str,
        variable_segments: Tuple[Variable, List[Tuple[Skalar, Skalar]]],
        share_segments: Dict['Effect', List[Tuple[Skalar, Skalar]]],
        can_be_outside_segments: Optional[Union[bool, Variable]],
        label: str = 'SegmentedShares',
    ):
        super().__init__(model, label_of_parent, label)
        assert len(variable_segments[1]) == len(list(share_segments.values())[0]), (
            'Segment length of variable_segments and share_segments must be equal'
        )
        self._can_be_outside_segments = can_be_outside_segments
        self._variable_segments = variable_segments
        self._share_segments = share_segments
        self._shares: Dict['Effect', linopy.Variable] = {}
        self._segments_model: Optional[MultipleSegmentsModel] = None
        self._as_tme_series: bool = isinstance(self._variable_segments[0], VariableTS)

    def do_modeling(self, system_model: SystemModel):
        self._shares = {
            effect: self.add(self._model.add_variables(
                coords=self._model.coords if self._as_tme_series else None,
                name=f'{self.label_full}__{effect.label}'),
                f'{effect.label}'
            ) for effect in self._share_segments
        }

        # Mapping variable names to segments
        segments: Dict[str, List[Tuple[Skalar, Skalar]]] = {
            **{self._shares[effect].name: segment for effect, segment in self._share_segments.items()},
            **{self._variable_segments[0].name: self._variable_segments[1]},
        }

        self._segments_model = self.add(
            MultipleSegmentsModel(
                model=self._model,
                label_of_parent=self._label_of_parent,
                sample_points=segments,
                can_be_outside_segments=self._can_be_outside_segments,
                as_time_series=self._as_tme_series),
            'segments'
        )
        self._segments_model.do_modeling(system_model)

        # Shares
        self._model.effects.add_share_to_effects(
            system_model=self._model,
            name=self._label_of_parent,
            expressions={effect: variable*1 for effect, variable in self._shares.items()},
            target='operation' if self._as_tme_series else 'invest',
        )


class PreventSimultaneousUsageModel(Model):
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

    def __init__(self, model: SystemModel, variables: List[linopy.Variable], label_of_parent: str, label: str = 'PreventSimultaneousUsage'):
        super().__init__(model, label_of_parent, label)
        self._simultanious_use_variables = variables
        assert len(self._simultanious_use_variables) >= 2, f'Model {self.__class__.__name__} must get at least two variables'
        for variable in self._simultanious_use_variables:  # classic
            assert variable.attrs['binary'], f'Variable {variable} must be binary for use in {self.__class__.__name__}'

    def do_modeling(self, system_model: SystemModel):
        # eq: sum(flow_i.on(t)) <= 1.1 (1 wird etwas größer gewählt wg. Binärvariablengenauigkeit)
        self.add(self._model.add_constraints(sum(self._simultanious_use_variables) <= 1.1,
                                             name=f'{self.label_full}__prevent_simultaneous_use'),
                 'prevent_simultaneous_use')
