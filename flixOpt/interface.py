# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:25:43 2022
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""
import logging
from typing import Union, Optional, Dict, List, Tuple, TYPE_CHECKING

import numpy as np

from flixOpt.core import Numeric, Skalar
if TYPE_CHECKING:
    from flixOpt.structure import Effect

logger = logging.getLogger('flixOpt')


class InvestParameters:
    '''
    collects arguments for invest-stuff
    '''

    def __init__(self,
                 fixed_size: Optional[Union[int, float]] = None,
                 minimum_size: Union[int, float] = 0,  # nur wenn size_is_fixed = False
                 maximum_size: Union[int, float] = 1e9,  # nur wenn size_is_fixed = False
                 optional: bool = True,  # Investition ist weglassbar
                 fix_effects: Optional[Union[Dict, int, float]] = None,
                 specific_effects: Union[Dict, int, float] = 0,  # costs per Flow-Unit/Storage-Size/...
                 effects_in_segments: Optional[Tuple[List[Tuple[Skalar, Skalar]], Dict['Effect', List[Tuple[Skalar, Skalar]]]]] = None,
                 divest_effects: Optional[Union[Dict, int, float]] = None):
        """
        Parameters
        ----------
        fix_effects : None or scalar, optional
            Fixed investment costs if invested.
            (Attention: Annualize costs to chosen period!)
        divest_effects : None or scalar, optional
            Fixed divestment costs (if not invested, e.g., demolition costs or contractual penalty).
        fixed_size : int, float, optional
            Determines if the investment size is fixed.
        optional : bool, optional
            If True, investment is not forced.
        specific_effects : scalar or Dict[Effect: Union[int, float, np.ndarray], optional
            Specific costs, e.g., in €/kW_nominal or €/m²_nominal.
            Example: {costs: 3, CO2: 0.3} with costs and CO2 representing an Object of class Effect
            (Attention: Annualize costs to chosen period!)
        effects_in_segments : list or List[ List[Union[int,float]], Dict[cEffecType: Union[List[Union[int,float]], optional
            Linear relation in segments [invest_segments, cost_segments].
            Example 1:
                [           [5, 25, 25, 100],       # size in kW
                 {costs:    [50,250,250,800],       # €
                  PE:       [5, 25, 25, 100]        # kWh_PrimaryEnergy
                  }
                ]
            Example 2 (if only standard-effect):
                [   [5, 25, 25, 100],  # kW # size in kW
                    [50,250,250,800]        # value for standart effect, typically €
                 ]  # €
            (Attention: Annualize costs to chosen period!)
            (Args 'specific_effects' and 'fix_effects' can be used in parallel to InvestsizeSegments)
        minimum_size : scalar
            Min nominal value (only if: size_is_fixed = False).
        maximum_size : scalar
            Max nominal value (only if: size_is_fixed = False).
        """

        self.fix_effects = fix_effects
        self.divest_effects = divest_effects
        self.fixed_size = fixed_size
        self.optional = optional
        self.specific_effects = specific_effects
        self.effects_in_segments = effects_in_segments
        self._minimum_size = minimum_size
        self._maximum_size = maximum_size

    @property
    def minimum_size(self):
        return self.fixed_size or self._minimum_size

    @property
    def maximum_size(self):
        return self.fixed_size or self._maximum_size

    def __repr__(self):
        return f"<{self.__class__.__name__}>: {self.__dict__}"

    def __str__(self):
        details = [
            f"size={self.fixed_size}" if self.fixed_size else f"size='{self.minimum_size}-{self.maximum_size}'",
            f"optional" if self.optional else "",
            f"fix_effects={self.fix_effects}" if self.fix_effects else "",
            f"specific_effects={self.specific_effects}" if self.specific_effects else "",
            f"effects_in_segments={self.effects_in_segments}, " if self.effects_in_segments else "",
            f"divest_effects={self.divest_effects}" if self.divest_effects else ""
        ]
        all_relevant_parts = [part for part in details if part != ""]
        full_str = f"{', '.join(all_relevant_parts)}"
        return f"<{self.__class__.__name__}>: {full_str}"


class OnOffParameters:
    def __init__(self,
                 effects_per_switch_on: Optional[Union[Dict, Numeric]] = None,
                 effects_per_running_hour: Optional[Union[Dict, Numeric]] = None,
                 on_hours_total_min: Optional[int] = None,
                 on_hours_total_max: Optional[int] = None,
                 consecutive_on_hours_min: Optional[Numeric] = None,
                 consecutive_on_hours_max: Optional[Numeric] = None,
                 consecutive_off_hours_min: Optional[Numeric] = None,
                 consecutive_off_hours_max: Optional[Numeric] = None,
                 switch_on_total_max: Optional[int] = None,
                 force_on: bool = False,
                 force_switch_on: bool = False):
        """
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
        """
        # self.flows_defining_on = flows_defining_on
        # self.on_values_before_begin = on_values_before_begin
        self.effects_per_switch_on = effects_per_switch_on
        self.effects_per_running_hour = effects_per_running_hour
        self.on_hours_total_min = on_hours_total_min  # scalar
        self.on_hours_total_max = on_hours_total_max  # scalar
        self.consecutive_on_hours_min = consecutive_on_hours_min  # TimeSeries
        self.consecutive_on_hours_max = consecutive_on_hours_max  # TimeSeries
        self.consecutive_off_hours_min = consecutive_off_hours_min  # TimeSeries
        self.consecutive_off_hours_max = consecutive_off_hours_max  # TimeSeries
        self.switch_on_total_max = switch_on_total_max
        self.force_on = force_on  # Can be set to True if needed, even after creation
        self.force_switch_on = force_switch_on

    '''
    def to_time_series(self, owner: Element):
        """Transforms all Numeric Values to TimeSeries"""
        self.consecutive_on_hours_min = self.consecutive_on_hours_min or TimeSeries('consecutive_on_hours_min', self.consecutive_on_hours_min, owner)
        self.consecutive_on_hours_max = self.consecutive_on_hours_max or TimeSeries('consecutive_on_hours_max', self.consecutive_on_hours_max, owner)
        self.consecutive_off_hours_min = self.consecutive_off_hours_min or TimeSeries('consecutive_off_hours_min', self.consecutive_off_hours_min, owner)
        self.consecutive_off_hours_max = self.consecutive_off_hours_max or TimeSeries('consecutive_off_hours_max',  self.consecutive_off_hours_max, owner)
        self.effects_per_switch_on = as_effect_dict_with_ts('effects_per_switch_on', self.effects_per_switch_on, owner)
        self.effects_per_running_hour = as_effect_dict_with_ts('effects_per_running_hour', self.effects_per_running_hour, owner)
    '''


    @property
    def use_on(self) -> bool:
        """Determines wether the ON Variable is needed or not"""
        return (any(param is not None for param in [self.effects_per_running_hour,
                                                    self.on_hours_total_min,
                                                    self.on_hours_total_max])
                or self.force_on or self.use_switch_on or self.use_on_hours or self.use_off_hours or self.use_off)

    @property
    def use_off(self) -> bool:
        """Determines wether the OFF Variable is needed or not"""
        return self.use_off_hours

    @property
    def use_on_hours(self) -> bool:
        """Determines wether a Variable for consecutive off hours is needed or not"""
        return any(param is not None for param in [self.consecutive_on_hours_min, self.consecutive_on_hours_max])

    @property
    def use_off_hours(self) -> bool:
        """Determines wether a Variable for consecutive off hours is needed or not"""
        return any(param is not None for param in [self.consecutive_off_hours_min, self.consecutive_off_hours_max])

    @property
    def use_switch_on(self) -> bool:
        """Determines wether a Variable for SWITCH-ON is needed or not"""
        return (any(param is not None for param in [self.effects_per_switch_on,
                                                    self.switch_on_total_max,
                                                    self.on_hours_total_min,
                                                    self.on_hours_total_max])
                or self.force_switch_on)
