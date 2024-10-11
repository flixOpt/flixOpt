# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:45:12 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import numpy as np
import textwrap
import logging
from typing import Union, Optional, Literal, List, Dict, Tuple

from flixOpt import utils
from flixOpt.elements import Flow, Component, EffectCollection, _create_time_series
from flixOpt.core import Skalar, Numeric_TS, TimeSeries
from flixOpt.math_modeling import VariableTS, Equation
from flixOpt.structure import SystemModel, ComponentModel, StorageModel
from flixOpt.features import FeatureLinearSegmentSet, FeatureInvest, FeatureAvoidFlowsAtOnce
from flixOpt.interface import InvestParameters, TimeSeriesRaw, OnOffParameters

logger = logging.getLogger('flixOpt')


class LinearConverter(Component):
    """
    Converts one FLow into another via linear conversion factors
    """

    def __init__(self,
                 label: str,
                 inputs: List[Flow],
                 outputs: List[Flow],
                 on_off_parameters: OnOffParameters = None,
                 conversion_factors: Optional[List[Dict[Flow, Numeric_TS]]] = None,
                 segmented_conversion_factors: Optional[Dict[Flow, List[Tuple[Numeric_TS, Numeric_TS]]]] = None):
        """
        Parameters
        ----------
        label : str
            name.
        inputs : list of flows
            input flows.
        outputs : list of flows
            output flows.
        group: str, None
            group name to assign components to groups. Used for later analysis of the results
        conversion_factors : list
            linear relation between flows
            eq: sum (factor * flow_in) = sum (factor * flow_out)
            factor can be TimeSeries, scalar or list.
            Either 'conversion_factors' or 'segmented_conversion_factors' can be used!

            example heat pump:  
                
            >>> conversion_factors= [{Q_th: COP_th , Q_0 : 1}
                              {P_el: COP_el , Q_0 : 1},              # COP_th
                              {Q_th: 1 , P_el: 1, Q_0 : 1, Q_ab: 1}] # Energiebilanz
                
        segmented_conversion_factors : dict
            Segmented linear correlation. begin and end of segment has to be given/defined.
            factors can be scalar or lists (i.e.timeseries)!
            if Begin of segment n+1 is not end of segment n, then "gap", i.e. not allowed area
            Either 'segmented_conversion_factors' or 'conversion_factors' can be used!
            example with two segments:
            
            >>> #           flow    begin, end, begin, end
                segments = {Q_fu: [ 5    , 10,  10,    22], # Abschnitte von 5 bis 10 und 10 bis 22
                            P_el: [ 2    , 5,   5,     8 ],
                            Q_fu: [ 2.5  , 4,   4,     12]}
            
            --> "points" can expressed as segment with same begin and end, i.g. [5, 5]

        Returns
        -------
        None.

        """
        super().__init__(label, inputs, outputs, on_off_parameters)
        self.conversion_factors = conversion_factors
        self.segmented_conversion_factors = segmented_conversion_factors
        self._plausibility_checks()

    def _plausibility_checks(self) -> None:
        if self.conversion_factors is None and self.segmented_conversion_factors is None:
            raise Exception('Either conversion_factors or segmented_conversion_factors must be defined!')
        if self.conversion_factors is not None and self.segmented_conversion_factors is not None:
            raise Exception('Only one of conversion_factors or segmented_conversion_factors can be defined, not both!')

        if self.conversion_factors is not None:
            if self.degrees_of_freedom <= 0:
                raise Exception(
                    f"Too Many conversion_factors_specified. Care that you use less conversion_factors "
                    f"then inputs + outputs!! With {len(self.inputs+self.outputs)} inputs and outputs, "
                    f"use not more than {len(self.inputs+self.outputs)-1} conversion_factors!")

            for conversion_factor in self.conversion_factors:
                for flow in conversion_factor:
                    if flow not in (self.inputs + self.outputs):
                        raise Exception(f'{self.label}: Flow {flow.label} in conversion_factors '
                                        f'is not in inputs/outputs')
        if self.segmented_conversion_factors is not None:
            for flow in (self.inputs + self.outputs):
                if isinstance(flow.size, InvestParameters) and flow.size.fixed_size is None:
                    raise Exception(f"segmented_conversion_factors (in {self.label_full}) and variable size "
                                    f"(in flow {flow.label_full}) do not make sense together!")

    def transform_to_time_series(self):
        if self.conversion_factors is not None:
            self.conversion_factors = self._transform_conversion_factors()
        else:
            segmented_conversion_factors = {}
            for flow, segments in self.segmented_conversion_factors.items():
                segmented_conversion_factors[flow] = [
                    (_create_time_series('Stuetzstelle', segment[0], self),
                     _create_time_series('Stuetzstelle', segment[1], self)
                     ) for segment in segments
                ]

    def _transform_conversion_factors(self) -> List[Dict[Flow, TimeSeries]]:
        """macht alle Faktoren, die nicht TimeSeries sind, zu TimeSeries"""
        list_of_conversion_factors = []
        for conversion_factor in self.conversion_factors:
            transformed_dict = {}
            for flow, values in conversion_factor.items():
                if not isinstance(values, TimeSeries):
                    transformed_dict[flow] = _create_time_series(f"{flow.label}_factor", values, self)
            list_of_conversion_factors.append(transformed_dict)
        return list_of_conversion_factors

    def __str__(self):
        # Creating a representation for conversion_factors with flow labels and their corresponding values
        if self.conversion_factors:
            conversion_factors_rep = []
            for conversion_factor in self.conversion_factors:
                conversion_factors_rep.append({flow.__repr__(): value for flow, value in conversion_factor.items()})
        else:
            conversion_factors_rep = "None"

        # Representing inputs and outputs by their labels
        inputs_str = ",\n".join([flow.__str__() for flow in self.inputs])
        outputs_str = ",\n".join([flow.__str__() for flow in self.outputs])
        inputs_str = f"inputs=\n{textwrap.indent(inputs_str, ' ' * 3)}" if self.inputs else "inputs=[]"
        outputs_str = f"outputs=\n{textwrap.indent(outputs_str, ' ' * 3)}" if self.inputs else "outputs=[]"

        other_relevant_data = (f"conversion_factors={conversion_factors_rep},\n"
                               f"segmented_conversion_factors={self.segmented_conversion_factors}")

        remaining_data = {
            key: value for key, value in self.__dict__.items()
            if value and
               not isinstance(value, Flow) and
               key not in ["label", "TS_list", "segmented_conversion_factors", "conversion_factors", "inputs", "outputs"]
        }

        remaining_data_str = ""
        for key, value in remaining_data.items():
            if hasattr(value, '__str__'):
                remaining_data_str += f"{key}: {value}\n"
            elif hasattr(value, '__repr__'):
                remaining_data_str += f"{key}: {repr(value)}\n"
            else:
                remaining_data_str += f"{key}: {value}\n"

        str_desc = (f"<{self.__class__.__name__}> {self.label}:\n"
                    f"{textwrap.indent(inputs_str, ' ' * 3)}\n"
                    f"{textwrap.indent(outputs_str, ' ' * 3)}\n"
                    f"{textwrap.indent(other_relevant_data, ' ' * 3)}\n"
                    f"{textwrap.indent(remaining_data_str, ' ' * 3)}"
                    )

        return str_desc

    @property
    def degrees_of_freedom(self):
        return len(self.inputs+self.outputs) - len(self.conversion_factors)


class Storage(Component):
    """
    Klasse Storage
    """

    # TODO: Dabei fällt mir auf. Vielleicht sollte man mal überlegen, ob man für Ladeleistungen bereits in dem
    #  jeweiligen Zeitschritt mit einem Verlust berücksichtigt. Zumindest für große Zeitschritte bzw. große Verluste
    #  eventuell relevant.
    #  -> Sprich: speicherverlust = charge_state(t) * relative_loss_per_hour * dt + 0.5 * Q_lade(t) * dt * relative_loss_per_hour * dt
    #  -> müsste man aber auch für den sich ändernden Ladezustand berücksichtigten

    def __init__(self,
                 label: str,
                 charging: Flow,
                 discharging: Flow,
                 capacity_in_flow_hours: Union[Skalar, InvestParameters],
                 relative_minimum_charge_state: Numeric_TS = 0,
                 relative_maximum_charge_state: Numeric_TS = 1,
                 initial_charge_state: Optional[Union[Skalar, Literal['lastValueOfSim']]] = 0,
                 minimal_final_charge_state: Optional[Skalar] = None,
                 maximal_final_charge_state: Optional[Skalar] = None,
                 eta_charge: Numeric_TS = 1,
                 eta_discharge: Numeric_TS = 1,
                 relative_loss_per_hour: Numeric_TS = 0,
                 prevent_simultaneous_charge_and_discharge: bool = True):
        """
        constructor of storage

        Parameters
        ----------
        label : str
            description.
        charging : Flow
            ingoing flow.
        discharging : Flow
            outgoing flow.
        capacity_in_flow_hours : Skalar or InvestParameter
            nominal capacity of the storage
        relative_minimum_charge_state : float or TS, optional
            minimum relative charge state. The default is 0.
        relative_maximum_charge_state : float or TS, optional
            maximum relative charge state. The default is 1.
        initial_charge_state : None, float (0...1), 'lastValueOfSim',  optional
            storage capacity in Flowhours at the beginning. The default is 0.
            float: defined capacity at start of first timestep
            None: free to choose by optimizer
            'lastValueOfSim': chargeState0 is equal to chargestate of last timestep ("closed simulation")
        minimal_final_charge_state : float or None, optional
            minimal value of chargeState at the end of timeseries. 
        maximal_final_charge_state : float or None, optional
            maximal value of chargeState at the end of timeseries. 
        eta_charge : float, optional
            efficiency factor of charging/loading. The default is 1.
        eta_discharge : TYPE, optional
            efficiency factor of uncharging/unloading. The default is 1.
        relative_loss_per_hour : float or TS. optional
            loss per chargeState-Unit per hour. The default is 0.
        prevent_simultaneous_charge_and_discharge : boolean, optional
            should simultaneously Loading and Unloading be avoided? (Attention, Performance maybe becomes worse with avoidInAndOutAtOnce=True). The default is True.
        """
        # TODO: fixed_relative_chargeState implementieren
        super().__init__(label, inputs=[charging], outputs=[discharging])

        self.charging = charging
        self.discharging = discharging
        self.capacity_inFlowHours = capacity_in_flow_hours
        self.relative_minimum_charge_state = relative_minimum_charge_state
        self.relative_maximum_charge_state = relative_maximum_charge_state

        self.initial_charge_state = initial_charge_state
        self.minimal_final_charge_state = minimal_final_charge_state
        self.maximal_final_charge_state = maximal_final_charge_state

        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        self.relative_loss_per_hour = relative_loss_per_hour
        self.prevent_simultaneous_charge_and_discharge = prevent_simultaneous_charge_and_discharge

    def create_model(self) -> StorageModel:
        self.model = StorageModel(self)
        return self.model


class SourceAndSink(Component):
    """
    class for source (output-flow) and sink (input-flow) in one commponent
    """
    # source : Flow
    # sink   : Flow

    new_init_args = ['label', 'source', 'sink', 'prevent_simultaneous_charge_and_discharge']

    not_used_args = ['label']

    def __init__(self, label: str, source: Flow, sink: Flow, group: str = None,
                 avoidInAndOutAtOnce: bool = True, **kwargs):
        '''
        Parameters
        ----------
        label : str
            name of sourceAndSink
        source : Flow
            output-flow of this component
        sink : Flow
            input-flow of this component
        group: str, None
            group name to assign components to groups. Used for later analysis of the results
        avoidInAndOutAtOnce: boolean. Default ist True.
            True: inflow and outflow are not allowed to be both non-zero at same timestep.
            False: inflow and outflow are working independently.
            
        **kwargs : TYPE
            DESCRIPTION.


        '''
        super().__init__(label, **kwargs)
        self.source = source
        self.sink = sink
        self.avoidInAndOutAtOnce = avoidInAndOutAtOnce
        self.outputs.append(source)  # ein Output-Flow
        self.inputs.append(sink)

        self.group = group

        # copy information of group to in-flows and out-flows
        for flow in self.inputs + self.outputs:
            flow.group = self.group

        # Erzwinge die Erstellung der On-Variablen, da notwendig für gleichung
        self.source.force_on = True
        self.sink.force_on = True

        if self.avoidInAndOutAtOnce:
            self.featureAvoidInAndOutAtOnce = FeatureAvoidFlowsAtOnce('sinkOrSource', self, [self.source, self.sink])
        else:
            self.featureAvoidInAndOutAtOnce = None

    def declare_vars_and_eqs(self, system_model):
        """
        Deklarieren von Variablen und Gleichungen

        :param system_model:
        :return:
        """
        super().declare_vars_and_eqs(system_model)

    def do_modeling(self, system_model):
        super().do_modeling(system_model)
        # Entweder Sink-Flow oder Source-Flow aktiv. Nicht beide Zeitgleich!
        if self.featureAvoidInAndOutAtOnce is not None:
            self.featureAvoidInAndOutAtOnce.do_modeling(system_model)


class Source(Component):
    """
    class of a source
    """
    new_init_args = ['label', 'source']
    not_used_args = ['label']

    def __init__(self, label: str, source: Flow, group: str = None, **kwargs):
        '''       
        Parameters
        ----------
        label : str
            name of source
        source : Flow
            output-flow of source
        group: str, None
            group name to assign components to groups. Used for later analysis of the results
        **kwargs : TYPE

        Returns
        -------
        None.

        '''

        """
        Konstruktor für Instanzen der Klasse Source

        :param str label: Bezeichnung
        :param Flow source: flow-output Quelle
        :param kwargs:
        """
        super().__init__(label, **kwargs)
        self.source = source
        self.outputs.append(source)  # ein Output-Flow

        self.group = group

        # copy information of group to in-flows and out-flows
        for flow in self.inputs + self.outputs:
            flow.group = self.group


class Sink(Component):
    """
    Klasse Sink
    """
    new_init_args = ['label', 'source']
    not_used_args = ['label']

    def __init__(self, label: str, sink: Flow, group: str = None, **kwargs):
        '''
        constructor of sink

        Parameters
        ----------
        label : str
            name of sink.
        sink : Flow
            input-flow of sink
        group: str, None
            group name to assign components to groups. Used for later analysis of the results
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''

        super().__init__(label, **kwargs)
        self.sink = sink
        self.inputs.append(sink)  # ein Input-Flow

        self.group = group

        # copy information of group to in-flows and out-flows
        for flow in self.inputs + self.outputs:
            flow.group = self.group


class Transportation(Component):
    # TODO: automatic on-Value in Flows if loss_abs
    # TODO: loss_abs must be: investment_size * loss_abs_rel!!!
    # TODO: investmentsize only on 1 flow
    # TODO: automatic invest_parameters for both in-flows (or alternatively both out-flows!)
    # TODO: optional: capacities should be recognised for losses

    def __init__(self,
                 label: str,
                 in1: Flow,
                 out1: Flow,
                 in2: Optional[Flow] = None,
                 out2: Optional[Flow] = None,
                 loss_rel: Numeric_TS = 0,
                 loss_abs: Numeric_TS = 0,
                 isAlwaysOn: bool = True,
                 avoidFlowInBothDirectionsAtOnce: bool = True,
                 **kwargs):
        '''
        pipe/cable/connector between side A and side B
        losses can be modelled
        investmentsize is recognised
        for investment_size use invest_parameters of in1 and in2-flows.
        (The investment_size of the both directions (in-flows) is equated)
        
        (when no flow through it, then loss is still there and has to be
        covered by one in-flow (gedanklicher Überströmer)
                         side A ... side B
        first  direction: in1   -> out1
        second direction: out2  <- in2
        
        Parameters
        ----------
        label : str
            name of Transportation.
        in1 : Flow
            inflow of input at side A
        out1 : Flow
            outflow (of in1) at side B
        in2 : cFlow, optional
            optional inflow of side B
        out2 : cFlow, optional
            outflow (of in2) at side A            
        loss_rel : float, TS
            relative loss between in and out, i.g. 0.02 i.e. 2 % loss
        loss_abs : float, TS
            absolut loss. is active until on=0 for in-flows
            example: loss_abs=2 -> 2 kW fix loss on transportation
        ... featureOnVars for Active Transportation:
        effects_per_switch_on : 
            #costs of switch rohr on
        exists : Warning! Functionality not implemented!

        Returns
        -------
        None.

        '''

        super().__init__(label, **kwargs)

        self.in1 = in1
        self.out1 = out1
        self.in2 = in2
        self.out2 = out2

        self.inputs.append(in1)
        self.outputs.append(out1)
        if in2 is not None:
            self.inputs.append(in2)
            self.outputs.append(out2)
            # check buses:
            assert in2.bus == out1.bus, 'in2.bus is not equal out1.bus!'
            assert out2.bus == in1.bus, 'out2.bus is not equal in1.bus!'

        self.loss_rel = TimeSeries('loss_rel', loss_rel, self)  #
        self.loss_abs = TimeSeries('loss_abs', loss_abs, self)  #
        self.isAlwaysOn = isAlwaysOn
        self.avoidFlowInBothDirectionsAtOnce = avoidFlowInBothDirectionsAtOnce

        if self.avoidFlowInBothDirectionsAtOnce and (in2 is not None):
            self.featureAvoidBothDirectionsAtOnce = FeatureAvoidFlowsAtOnce('feature_avoidBothDirectionsAtOnce', self,
                                                                            [self.in1, self.in2])

    def declare_vars_and_eqs(self, system_model: SystemModel):
        """
        Deklarieren von Variablen und Gleichungen
        
        :param system_model:
        :return:
        """
        super().declare_vars_and_eqs(system_model)

    def do_modeling(self, system_model):
        super().do_modeling(system_model)

        # not both directions at once:
        if self.avoidFlowInBothDirectionsAtOnce and (self.in2 is not None):
            self.featureAvoidBothDirectionsAtOnce.do_modeling(system_model)

        # first direction
        # eq: in(t)*(1-loss_rel(t)) = out(t) + on(t)*loss_abs(t)
        self.model.add_equations(Equation('transport_dir1', self, system_model, eqType='eq'))
        self.model.eqs['transport_dir1'].add_summand(self.in1.model.var_val, (1 - self.loss_rel.active_data))
        self.model.eqs['transport_dir1'].add_summand(self.out1.model.var_val, -1)
        if (self.loss_abs.active_data is not None) and np.any(self.loss_abs.active_data != 0):
            assert self.in1.model.variables['on'] is not None, 'Var on wird benötigt für in1! Set relative_minimum!'
            self.model.eqs['transport_dir1'].add_summand(self.in1.model.var_on, -1 * self.loss_abs.active_data)

        # second direction:        
        if self.in2 is not None:
            # eq: in(t)*(1-loss_rel(t)) = out(t) + on(t)*loss_abs(t)
            self.model.add_equations(Equation('transport_dir2', self, system_model, eqType='eq'))
            self.model.eqs['transport_dir2'].add_summand(self.in2.model.var_val, 1 - self.loss_rel.active_data)
            self.model.eqs['transport_dir2'].add_summand(self.out2.model.var_val, -1)
            if (self.loss_abs.active_data is not None) and np.any(self.loss_abs.active_data != 0):
                assert self.in2.model.variables['on'] is not None, 'Var on wird benötigt für in2! Set relative_minimum!'
                self.model.eqs['transport_dir2'].add_summand(self.in2.model.var_on, -1 * self.loss_abs.active_data)

        # always On (in at least one direction)
        # eq: in1.on(t) +in2.on(t) >= 1 # TODO: this is some redundant to avoidFlowInBothDirections
        if self.isAlwaysOn:
            self.model.add_equations(Equation('alwaysOn', self, system_model, eqType='ineq'))
            self.model.eqs['alwaysOn'].add_summand(self.in1.model.var_on, -1)
            if self.in2 is not None:
                self.model.eqs['alwaysOn'].add_summand(self.in2.model.var_on, -1)
            self.model.eqs['alwaysOn'].add_constant(-.5)  # wg binärungenauigkeit 0.5 statt 1

        # equate nominal value of second direction
        if (self.in2 is not None):
            oneInFlowHasFeatureInvest = (self.in1.featureInvest is not None) or (self.in1.featureInvest is not None)
            bothInFlowsHaveFeatureInvest = (self.in1.featureInvest is not None) and (self.in1.featureInvest is not None)
            if oneInFlowHasFeatureInvest:
                if bothInFlowsHaveFeatureInvest:
                    # eq: in1.nom_value = in2.nom_value
                    self.model.add_equations(Equation('equalSizeInBothDirections', self, system_model, eqType='eq'))
                    self.model.eqs['equalSizeInBothDirections'].add_summand(self.in1.featureInvest.model.var_investmentSize, 1)
                    self.model.eqs['equalSizeInBothDirections'].add_summand(self.in2.featureInvest.model.var_investmentSize, -1)
                else:
                    raise Exception(
                        'define invest_parameters also for second In-Flow (values can be empty!)')  # TODO: anders lösen (automatisiert)!
