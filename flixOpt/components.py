# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:45:12 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import numpy as np
import textwrap
import logging
from typing import Union, Optional, Literal

from flixOpt import utils
from flixOpt.elements import Flow, Component, MediumCollection, EffectCollection, Objective
from flixOpt.core import Skalar, Numeric, Numeric_TS, TimeSeries, effect_values_to_ts
from flixOpt.math_modeling import VariableTS, Equation
from flixOpt.structure import SystemModel
from flixOpt.features import FeatureLinearSegmentSet, FeatureInvest, FeatureAvoidFlowsAtOnce
from flixOpt.interface import InvestParameters, TimeSeriesRaw

logger = logging.getLogger('flixOpt')

class LinearConverter(Component):
    """
    Klasse LinearConverter: Grundgerüst lineare Übertragungskomponente
    """
    new_init_args = ['label', 'inputs', 'outputs', 'conversion_factors', 'segmented_conversion_factors']
    not_used_args = ['label']

    def __init__(self, label: str, inputs: list, outputs: list, group: str = None, conversion_factors=None,
                 segmented_conversion_factors=None, **kwargs):
        '''
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


        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''

        super().__init__(label, **kwargs)
        # invest_parameters to attributes:
        self.inputs = inputs
        self.outputs = outputs
        self.conversion_factors = conversion_factors
        self.segmented_conversion_factors = segmented_conversion_factors
        if (conversion_factors is None) and (segmented_conversion_factors is None):
            raise Exception('conversion_factors or segmented_conversion_factors must be defined!')
        elif (conversion_factors is not None) and (segmented_conversion_factors is not None):
            raise Exception('Either conversion_factors or segmented_conversion_factors must \
                            be defined! Not Both!')

        self.group = group

        # copy information of group to in-flows and out-flows
        for flow in self.inputs + self.outputs:
            flow.group = self.group

        # copy information about exists into segments of flows
        if self.segmented_conversion_factors is not None:
            if isinstance(self.exists.active_data, (np.ndarray, list)):
                for key, item in self.segmented_conversion_factors.items():
                    self.segmented_conversion_factors[key] = [list(np.array(item) * factor) for factor in self.exists.active_data]
            elif isinstance(self.exists.active_data, (int, float)):
                for key, item in self.segmented_conversion_factors.items():
                    self.segmented_conversion_factors[key] = list(np.array(item) * self.exists.active_data)

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

    def transformFactorsToTS(self, conversion_factors):
        """
        macht alle Faktoren, die nicht TimeSeries sind, zu TimeSeries

        :param conversion_factors:
        :return:
        """
        # Einzelne Faktoren zu Vektoren:
        conversion_factors_TS = []
        # für jedes Dict -> Values (=Faktoren) zu Vektoren umwandeln:
        for aFactor_Dict in conversion_factors:  # Liste of dicts
            # Transform to TS:
            aFactor_Dict_TS = effect_values_to_ts('Faktor', aFactor_Dict, self)
            conversion_factors_TS.append(aFactor_Dict_TS)
            # check flows:
            for flow in aFactor_Dict_TS:
                if not (flow in self.inputs + self.outputs):
                    raise Exception(self.label + ': Flow ' + flow.label + ' in conversion_factors ist nicht in inputs/outputs')
        return conversion_factors_TS

    def finalize(self):
        """

        :return:
        """
        super().finalize()

        # factor-sets:
        if self.segmented_conversion_factors is None:

            # TODO: mathematisch für jeden Zeitschritt checken!!!!
            #  Anzahl Freiheitsgrade checken: =  Anz. Variablen - Anz. Gleichungen

            # alle Faktoren, die noch nicht TS_vector sind, umwandeln:
            self.conversion_factors = self.transformFactorsToTS(self.conversion_factors)

            self.degreesOfFreedom = (len(self.inputs) + len(self.outputs)) - len(self.conversion_factors)
            if self.degreesOfFreedom <= 0:
                raise Exception(self.label + ': ' + str(len(self.conversion_factors)) + ' Gleichungen VERSUS '
                                + str(len(self.inputs + self.outputs)) + ' Variablen!!!')

        # linear segments:
        else:
            # check if investsize is variable for any flow:            
            for flow in (self.inputs + self.outputs):
                if isinstance(flow.size, InvestParameters) and flow.size.fixed_size is None:
                    raise Exception(
                        f"Linear segments of flows (in {self.label_full}) and variable size "
                        f"(invest_size) (in flow {flow.label_full}) do not make sense together!")

            # Flow als Keys rauspicken und alle Stützstellen als TimeSeries:
            self.segmented_conversion_factors_TS = self.segmented_conversion_factors
            for aFlow in self.segmented_conversion_factors.keys():
                # 2. Stützstellen zu TimeSeries machen, wenn noch nicht TimeSeries!:
                for i in range(len(self.segmented_conversion_factors[aFlow])):
                    stuetzstelle = self.segmented_conversion_factors[aFlow][i]
                    self.segmented_conversion_factors_TS[aFlow][i] = TimeSeries('Stuetzstelle', stuetzstelle, self)

            def get_var_on():
                return self.model.var_on

            self.feature_linSegments = FeatureLinearSegmentSet('linearSegments', self, self.segmented_conversion_factors_TS,
                                                               get_var_on=get_var_on,
                                                               flows=self.inputs + self.outputs)  # erst hier, damit auch nach __init__() noch Übergabe möglich.

    def declare_vars_and_eqs(self, system_model: SystemModel):
        """
        Deklarieren von Variablen und Gleichungen

        :param system_model:
        :return:
        """
        super().declare_vars_and_eqs(system_model)  # (ab hier sollte auch self.model.var_on dann vorhanden sein)

        # factor-sets:
        if self.segmented_conversion_factors is None:
            pass
        # linear segments:
        else:
            self.feature_linSegments.declare_vars_and_eqs(system_model)

    def do_modeling(self, system_model: SystemModel):
        super().do_modeling(system_model)
        # conversion_factors:
        if self.segmented_conversion_factors is None:
            # Transformer-Constraints:

            inputs_set = set(self.inputs)
            outputs_set = set(self.outputs)

            # für alle linearen Gleichungen:
            for i in range(len(self.conversion_factors)):
                # erstelle Gleichung für jedes t:
                # sum(inputs * factor) = sum(outputs * factor)
                # in1.val[t] * factor_in1[t] + in2.val[t] * factor_in2[t] + ... = out1.val[t] * factor_out1[t] + out2.val[t] * factor_out2[t] + ...

                aFactorVec_Dict = self.conversion_factors[i]

                leftSideFlows = inputs_set & aFactorVec_Dict.keys()  # davon nur die input-flows, die in Glg sind.
                rightSideFlows = outputs_set & aFactorVec_Dict.keys()  # davon nur die output-flows, die in Glg. sind.

                eq_linearFlowRelation_i = Equation('linearFlowRelation_' + str(i), self, system_model)
                self.model.add_equation(eq_linearFlowRelation_i)
                for inFlow in leftSideFlows:
                    aFactor = aFactorVec_Dict[inFlow].active_data
                    eq_linearFlowRelation_i.add_summand(inFlow.model.variables['val'], aFactor)  # input1.val[t]      * factor[t]
                for outFlow in rightSideFlows:
                    aFactor = aFactorVec_Dict[outFlow].active_data
                    eq_linearFlowRelation_i.add_summand(outFlow.model.variables['val'], -aFactor)  # output.val[t] * -1 * factor[t]

                eq_linearFlowRelation_i.add_constant(0)  # nur zur Komplettisierung der Gleichung

        # (linear) segments:
        # Zusammenhänge zw. inputs & outputs können auch vollständig über Segmente beschrieben werden:
        else:
            self.feature_linSegments.do_modeling(system_model)

    # todo: checkbounds!
    # def initializeParameter(self,aStr,aBounds):
    # private Variable:
    #     self._eta          = aBounds['eta'][0]
    # exec('self.__' + aStr + ' = aBounds[0] ')
    # property dazu:
    #    self.eta            = property(lambda s: s.__get_param('eta'), lambda s,v: s.__set_param(v,'eta')')
    # exec('self.'   + aStr + ' = property(lambda s: s.__get_param(aStr) , lambda s,v: s.__set_param(v,aStr ))')


class Storage(Component):
    """
    Klasse Storage
    """

    # TODO: Dabei fällt mir auf. Vielleicht sollte man mal überlegen, ob man für Ladeleistungen bereits in dem
    #  jeweiligen Zeitschritt mit einem Verlust berücksichtigt. Zumindest für große Zeitschritte bzw. große Verluste
    #  eventuell relevant.
    #  -> Sprich: speicherverlust = charge_state(t) * relative_loss_per_hour * dt + 0.5 * Q_lade(t) * dt * relative_loss_per_hour * dt
    #  -> müsste man aber auch für den sich ändernden Ladezustand berücksichtigten

    # costs_default = property(get_costs())
    # param_defalt  = property(get_params())

    new_init_args = ['label', 'exists', 'charging', 'discharging', 'capacity_in_flow_hours', 'relative_minimum_charge_state', 'relative_maximum_charge_state',
                     'initial_charge_state', 'minimal_final_charge_state', 'maximal_final_charge_state', 'eta_load',
                     'eta_unload', 'relative_loss_per_hour', 'prevent_simultaneous_charge_and_discharge']

    not_used_args = ['label', 'exists']

    # capacity_in_flow_hours: float, 'lastValueOfSim', None
    def __init__(self,
                 label: str,
                 charging: Flow,
                 discharging: Flow,
                 capacity_in_flow_hours: Union[Skalar, InvestParameters],
                 group: Optional[str] = None,
                 relative_minimum_charge_state: Numeric_TS = 0,
                 relative_maximum_charge_state: Numeric_TS = 1,
                 initial_charge_state: Optional[Union[Skalar, Literal['lastValueOfSim']]] = 0,
                 minimal_final_charge_state: Optional[Skalar] = None,
                 maximal_final_charge_state: Optional[Skalar] = None,
                 eta_load: Numeric_TS = 1, eta_unload: Numeric_TS = 1,
                 relative_loss_per_hour: Numeric_TS = 0,
                 prevent_simultaneous_charge_and_discharge: bool = True,
                 **kwargs):
        '''
        constructor of storage

        Parameters
        ----------
        label : str
            description.
        charging : Flow
            ingoing flow.
        discharging : Flow
            outgoing flow.
        group: str, None
            group name to assign components to groups. Used for later analysis of the results
        exists: Numeric_TS
            Limits the availlable capacity, and the in and out flow. DOes not affect other parameters yet
            (like frac_loss_per_hour, starting value, ...)
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
        eta_load : float, optional
            efficiency factor of charging/loading. The default is 1.
        eta_unload : TYPE, optional
            efficiency factor of uncharging/unloading. The default is 1.
        relative_loss_per_hour : float or TS. optional
            loss per chargeState-Unit per hour. The default is 0.
        prevent_simultaneous_charge_and_discharge : boolean, optional
            should simultaneously Loading and Unloading be avoided? (Attention, Performance maybe becomes worse with avoidInAndOutAtOnce=True). The default is True.
        
        **kwargs : TYPE # TODO welche kwargs werden hier genutzt???
            DESCRIPTION.
        '''
        # TODO: neben relative_minimum_charge_state, relative_maximum_charge_state ggf. noch "fixed_relative_value_chargeState" implementieren damit konsistent zu flow (relative_maximum, relative_minimum, val_re)

        # minimal_final_charge_state (absolute Werte, aber relative wären ggf. auch manchmal hilfreich)
        super().__init__(label, **kwargs)

        # invest_parameters to attributes:
        self.inputs = [charging]
        self.outputs = [discharging]
        self.charging = charging
        self.discharging = discharging
        self.capacity_inFlowHours = capacity_in_flow_hours
        self.maximum_relative_chargeState = TimeSeries('relative_maximum_charge_state', relative_maximum_charge_state, self)
        self.minimum_relative_chargeState = TimeSeries('relative_minimum_charge_state', relative_minimum_charge_state, self)

        self.group = group

        # add last time step (if not scalar):
        existsWithEndTimestep = self.exists.active_data if np.isscalar(self.exists.active_data) else np.append(self.exists.active_data, self.exists.active_data[-1])
        self.maximum_relative_chargeState = TimeSeries('relative_maximum_charge_state',
                                              self.maximum_relative_chargeState.active_data * existsWithEndTimestep, self)
        self.minimum_relative_chargeState = TimeSeries('relative_minimum_charge_state',
                                              self.minimum_relative_chargeState.active_data * existsWithEndTimestep, self)

        # copy information of "group" to in-flows and out-flows
        for flow in self.inputs + self.outputs:
            flow.group = self.group

        self.chargeState0_inFlowHours = initial_charge_state
        self.charge_state_end_min = minimal_final_charge_state
        if maximal_final_charge_state:
            self.charge_state_end_max = maximal_final_charge_state
        elif isinstance(self.capacity_inFlowHours, InvestParameters):
            self.charge_state_end_max = self.capacity_inFlowHours.fixed_size
        else:
            self.charge_state_end_max = minimal_final_charge_state

        self.eta_load = TimeSeries('eta_load', eta_load, self)
        self.eta_unload = TimeSeries('eta_unload', eta_unload, self)
        self.fracLossPerHour = TimeSeries('relative_loss_per_hour', relative_loss_per_hour, self)
        self.avoidInAndOutAtOnce = prevent_simultaneous_charge_and_discharge

        self.featureInvest = None

        if self.avoidInAndOutAtOnce:
            self.featureAvoidInAndOut = FeatureAvoidFlowsAtOnce('feature_avoidInAndOutAtOnce', self,
                                                                [self.charging, self.discharging])

        if isinstance(self.capacity_inFlowHours, InvestParameters):
            self.featureInvest = FeatureInvest('used_capacity_inFlowHours', self, self.capacity_inFlowHours,
                                               relative_minimum=self.minimum_relative_chargeState,
                                               relative_maximum=self.maximum_relative_chargeState,
                                               fixed_relative_value=None,  # kein vorgegebenes Profil
                                               featureOn=None)  # hier gibt es kein On-Wert

        # Medium-Check:
        if not (MediumCollection.checkIfFits(charging.medium, discharging.medium)):
            raise Exception('in Storage ' + self.label + ': input.medium = ' + str(charging.medium) +
                            ' and output.medium = ' + str(discharging.medium) + ' don`t fit!')
        # TODO: chargeState0 darf nicht größer max usw. abfangen!

        self.isStorage = True  # for postprocessing

    def declare_vars_and_eqs(self, system_model: SystemModel):
        """
        Deklarieren von Variablen und Gleichungen

        :param system_model:
        :return:
        """
        super().declare_vars_and_eqs(system_model)

        # Variablen:

        if self.featureInvest is None:
            lb = self.minimum_relative_chargeState.active_data * self.capacity_inFlowHours
            ub = self.maximum_relative_chargeState.active_data * self.capacity_inFlowHours
            fix_value = None

            if np.isscalar(lb):
                pass
            else:
                lb = np.append(lb, 0)  # self.minimal_final_charge_state)
            if np.isscalar(ub):
                pass
            else:
                ub = np.append(ub, self.capacity_inFlowHours)  # maximal_final_charge_state)

        else:
            (lb, ub, fix_value) = self.featureInvest.bounds_of_defining_variable()

            if np.isscalar(lb):
                pass
            else:
                lb = np.append(lb, lb[-1])  # self.minimal_final_charge_state)
            if np.isscalar(ub):
                pass
            else:
                ub = np.append(ub, ub[-1])  # maximal_final_charge_state)

        self.model.add_variable(
            VariableTS('charge_state', system_model.nrOfTimeSteps + 1, self.label_full,
                       system_model, lower_bound=lb, upper_bound=ub, value=fix_value,
                       before_value=self.chargeState0_inFlowHours, before_value_is_start_value=True))  # Eins mehr am Ende!
        self.model.add_variable(VariableTS('nettoFlow', system_model.nrOfTimeSteps, self.label_full, system_model,
                                              lower_bound=-np.inf))  # negative Werte zulässig!

        # erst hier, da defining_variable vorher nicht belegt!
        if self.featureInvest is not None:
            self.featureInvest.set_defining_variables(self.model.variables['charge_state'], None)  # None, da kein On-Wert
            self.featureInvest.declare_vars_and_eqs(system_model)

        # obj.vars.Q_Ladezustand   .setBoundaries(0, obj.inputData.Q_Ladezustand_Max);
        # obj.vars.Q_th_Lade       .setBoundaries(0, inf);
        # obj.vars.Q_th_Entlade    .setBoundaries(0, inf);

        # ############ Variablen ###############

        # obj.addVariable('Q_th'             ,obj.lengthOfTS  , 0);
        # obj.addVariable('Q_th_Lade'        ,obj.lengthOfTS  , 0);
        # obj.addVariable('Q_th_Entlade'     ,obj.lengthOfTS  , 0);
        # obj.addVariable('Q_Ladezustand'    ,obj.lengthOfTS+1, 0);  % Eins mehr am Ende!
        # obj.addVariable('IchLadeMich'      ,obj.lengthOfTS  , 1);  % binäre Variable um zu verhindern, dass gleichzeitig Be- und Entladen wird (bei KWK durchaus ein Kostenoptimum)
        # obj.addVariable('IchEntladeMich'   ,obj.lengthOfTS  , 1);  % binäre Variable um zu verhindern, dass gleichzeitig Be- und Entladen wird (bei KWK durchaus ein Kostenoptimum)

        # ############### verknüpfung mit anderen Variablen ##################
        # % Pumpstromaufwand Beladen/Entladen
        # refToStromLastEq.add_summand(obj.vars.Q_th_Lade   ,-1*obj.inputData.spezifPumpstromAufwandBeladeEntlade); % für diese Komponenten Stromverbrauch!
        # refToStromLastEq.add_summand(obj.vars.Q_th_Entlade,-1*obj.inputData.spezifPumpstromAufwandBeladeEntlade); % für diese Komponenten Stromverbrauch!

    def do_modeling(self, system_model):
        super().do_modeling(system_model)

        # Gleichzeitiges Be-/Entladen verhindern:
        if self.avoidInAndOutAtOnce: self.featureAvoidInAndOut.do_modeling(system_model)

        # % Speicherladezustand am Start
        if self.chargeState0_inFlowHours is None:
            # Startzustand bleibt Freiheitsgrad
            pass
        elif utils.is_number(self.chargeState0_inFlowHours):
            # eq: Q_Ladezustand(1) = Q_Ladezustand_Start;
            self.model.add_equation(Equation('charge_state_start', self, system_model, eqType='eq'))
            self.model.eqs['charge_state_start'].add_constant(self.model.variables['charge_state'].before_value)  # chargeState_0 !
            self.model.eqs['charge_state_start'].add_summand(self.model.variables['charge_state'], 1,
                                                             system_model.time_indices[0])
        elif self.chargeState0_inFlowHours == 'lastValueOfSim':
            # eq: Q_Ladezustand(1) - Q_Ladezustand(end) = 0;
            self.model.add_equation(Equation('charge_state_start', self, system_model, eqType='eq'))
            self.model.eqs['charge_state_start'].add_summand(self.model.variables['charge_state'], 1,
                                                             system_model.time_indices[0])
            self.model.eqs['charge_state_start'].add_summand(self.model.variables['charge_state'], -1,
                                                             system_model.time_indices[-1])
        else:
            raise Exception('initial_charge_state has undefined value = ' + str(self.chargeState0_inFlowHours))

        # Speicherleistung / Speicherladezustand / Speicherverlust
        #                                                                          | Speicher-Beladung       |   |Speicher-Entladung                |
        # Q_Ladezustand(n+1) + (-1+VerlustanteilProStunde*dt(n)) *Q_Ladezustand(n) -  dt(n)*eta_Lade*Q_th_Lade(n) +  dt(n)* 1/eta_Entlade*Q_th_Entlade(n)  = 0

        # charge_state hat ein Index mehr:
        time_indicesChargeState = range(system_model.time_indices.start, system_model.time_indices.stop + 1)
        self.model.add_equation(Equation('charge_state', self, system_model, eqType='eq'))
        self.model.eqs['charge_state'].add_summand(self.model.variables['charge_state'],
                                         -1 * (1 - self.fracLossPerHour.active_data * system_model.dt_in_hours),
                                        time_indicesChargeState[
                                        :-1])  # sprich 0 .. end-1 % nach letztem Zeitschritt gibt es noch einen weiteren Ladezustand!
        self.model.eqs['charge_state'].add_summand(self.model.variables['charge_state'], 1, time_indicesChargeState[1:])  # 1:end
        self.model.eqs['charge_state'].add_summand(self.charging.model.variables['val'], -1 * self.eta_load.active_data * system_model.dt_in_hours)
        self.model.eqs['charge_state'].add_summand(self.discharging.model.variables['val'],
                                         1 / self.eta_unload.active_data * system_model.dt_in_hours)  # Achtung hier 1/eta!

        # Speicherladezustand am Ende
        # -> eigentlich min/max-Wert für variable, aber da nur für ein Element hier als Glg:
        # 1: eq:  Q_charge_state(end) <= Q_max
        if self.charge_state_end_max is not None:
            self.model.add_equation(Equation('eq_charge_state_end_max', self, system_model, eqType='ineq'))
            self.model.eqs['eq_charge_state_end_max'].add_summand(self.model.variables['charge_state'], 1, time_indicesChargeState[-1])
            self.model.eqs['eq_charge_state_end_max'].add_constant(self.charge_state_end_max)

        # 2: eq: - Q_charge_state(end) <= - Q_min
        if self.charge_state_end_min is not None:
            self.model.add_equation(Equation('eq_charge_state_end_min', self, system_model, eqType='ineq'))
            self.model.eqs['eq_charge_state_end_min'].add_summand(self.model.variables['charge_state'], -1, time_indicesChargeState[-1])
            self.model.eqs['eq_charge_state_end_min'].add_constant(- self.charge_state_end_min)

        # nettoflow:
        # eq: nettoFlow(t) - discharging(t) + charging(t) = 0
        self.model.add_equation(Equation('nettoFlow', self, system_model, eqType='eq'))
        self.model.eqs['nettoFlow'].add_summand(self.model.variables['nettoFlow'], 1)
        self.model.eqs['nettoFlow'].add_summand(self.charging.model.variables['val'], 1)
        self.model.eqs['nettoFlow'].add_summand(self.discharging.model.variables['val'], -1)

        if self.featureInvest is not None:
            self.featureInvest.do_modeling(system_model)

        # ############# Gleichungen ##########################
        # % Speicherleistung an Bilanzgrenze / Speicher-Ladung / Speicher-Entladung
        # % Q_th(n) + Q_th_Lade(n) - Q_th_Entlade(n) = 0;
        # obj.eqs.Leistungen = Equation('Leistungen');
        # obj.eqs.Leistungen.add_summand(obj.vars.Q_th        , 1);
        # obj.eqs.Leistungen.add_summand(obj.vars.Q_th_Lade   , 1);
        # obj.eqs.Leistungen.add_summand(obj.vars.Q_th_Entlade,-1);

        # % Bedingungen der binären Variable "IchLadeMich"
        # Q_th_Lade_Max   = obj.inputData.Q_Ladezustand_Max / obj.inputData.eta_Lade /obj.dt; % maximale Entladeleistung, wenn in einem Zeitschritt alles ausgeschoben wird
        # Q_th_Lade_Min   = 0; % könnte eigtl auch größer Null sein.
        # obj.addConstraintsOfVariableOn(obj.vars.IchLadeMich   ,obj.vars.Q_th_Lade   ,Q_th_Lade_Max   ,Q_th_Lade_Min); % korrelierende Leistungsvariable und ihr Maximum!

        # % Bedingungen der binären Variable "IchEntladeMich"
        # Q_th_Entlade_Max = obj.inputData.Q_Ladezustand_Max * obj.inputData.eta_Entlade /obj.dt; % maximale Entladeleistung, wenn in einem Zeitschritt alles ausgeschoben wird
        # Q_th_Entlade_min = 0; % könnte eigtl auch größer Null sein.
        # obj.addConstraintsOfVariableOn(obj.vars.IchEntladeMich,obj.vars.Q_th_Entlade,Q_th_Entlade_Max,Q_th_Entlade_min);  % korrelierende Leistungsvariable und ihr Maximum!

        # % Bedingung "Laden ODER Entladen ODER nix von beiden" (insbesondere für KWK-Anlagen wichtig, da gleichzeitiges Entladen und Beladen sonst Kostenoptimum sein kann
        # % eq: IchLadeMich(n) + IchEntladeMich(n) <= 1;
        # obj.ineqs.EntwederLadenOderEntladen = Equation('EntwederLadenOderEntladen');
        # obj.ineqs.EntwederLadenOderEntladen.add_summand(obj.vars.IchLadeMich   ,1);
        # obj.ineqs.EntwederLadenOderEntladen.add_summand(obj.vars.IchEntladeMich,1);
        # obj.ineqs.EntwederLadenOderEntladen.add_constant(1);

    def add_share_to_globals(self, effect_collection: EffectCollection, system_model: SystemModel):
        """
        :param effect_collection:
        :param system_model:
        :return:
        """
        super().add_share_to_globals(effect_collection, system_model)

        if self.featureInvest is not None:
            self.featureInvest.add_share_to_globals(effect_collection, system_model)


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
        self.model.add_equation(Equation('transport_dir1', self, system_model, eqType='eq'))
        self.model.eqs['transport_dir1'].add_summand(self.in1.model.var_val, (1 - self.loss_rel.active_data))
        self.model.eqs['transport_dir1'].add_summand(self.out1.model.var_val, -1)
        if (self.loss_abs.active_data is not None) and np.any(self.loss_abs.active_data != 0):
            assert self.in1.model.variables['on'] is not None, 'Var on wird benötigt für in1! Set relative_minimum!'
            self.model.eqs['transport_dir1'].add_summand(self.in1.model.var_on, -1 * self.loss_abs.active_data)

        # second direction:        
        if self.in2 is not None:
            # eq: in(t)*(1-loss_rel(t)) = out(t) + on(t)*loss_abs(t)
            self.model.add_equation(Equation('transport_dir2', self, system_model, eqType='eq'))
            self.model.eqs['transport_dir2'].add_summand(self.in2.model.var_val, 1 - self.loss_rel.active_data)
            self.model.eqs['transport_dir2'].add_summand(self.out2.model.var_val, -1)
            if (self.loss_abs.active_data is not None) and np.any(self.loss_abs.active_data != 0):
                assert self.in2.model.variables['on'] is not None, 'Var on wird benötigt für in2! Set relative_minimum!'
                self.model.eqs['transport_dir2'].add_summand(self.in2.model.var_on, -1 * self.loss_abs.active_data)

        # always On (in at least one direction)
        # eq: in1.on(t) +in2.on(t) >= 1 # TODO: this is some redundant to avoidFlowInBothDirections
        if self.isAlwaysOn:
            self.model.add_equation(Equation('alwaysOn', self, system_model, eqType='ineq'))
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
                    self.model.add_equation(Equation('equalSizeInBothDirections', self, system_model, eqType='eq'))
                    self.model.eqs['equalSizeInBothDirections'].add_summand(self.in1.featureInvest.model.var_investmentSize, 1)
                    self.model.eqs['equalSizeInBothDirections'].add_summand(self.in2.featureInvest.model.var_investmentSize, -1)
                else:
                    raise Exception(
                        'define invest_parameters also for second In-Flow (values can be empty!)')  # TODO: anders lösen (automatisiert)!
