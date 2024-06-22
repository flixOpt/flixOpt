# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:45:12 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import numpy as np
import textwrap

from . import flixOptHelperFcts as helpers
from .basicModeling import *
from .flixStructure import *
from .flixFeatures import *


class LinearTransformer(cBaseComponent):
    """
    Klasse LinearTransformer: Grundgerüst lineare Übertragungskomponente
    """
    new_init_args = ['label', 'inputs', 'outputs', 'factor_Sets', 'segmentsOfFlows']
    not_used_args = ['label']

    def __init__(self, label: str, inputs: list, outputs: list, group: str = None, factor_Sets=None,
                 segmentsOfFlows=None, **kwargs):
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
        factor_Sets : list
            linear relation between flows
            eq: sum (factor * flow_in) = sum (factor * flow_out)
            factor can be TimeSeries, scalar or list.
            Either 'factor_Sets' or 'segmentsOfFlows' can be used!

            example heat pump:  
                
            >>> factor_Sets= [{Q_th: COP_th , Q_0 : 1}
                              {P_el: COP_el , Q_0 : 1},              # COP_th
                              {Q_th: 1 , P_el: 1, Q_0 : 1, Q_ab: 1}] # Energiebilanz
                
        segmentsOfFlows : dict
            Segmented linear correlation. begin and end of segment has to be given/defined.
            factors can be scalar or lists (i.e.timeseries)!
            if Begin of segment n+1 is not end of segment n, then "gap", i.e. not allowed area
            Either 'segmentsOfFlows' or 'factor_Sets' can be used!
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
        # args to attributes:
        self.inputs = inputs
        self.outputs = outputs
        self.factor_Sets = factor_Sets
        self.segmentsOfFlows = segmentsOfFlows
        if (factor_Sets is None) and (segmentsOfFlows is None):
            raise Exception('factor_Sets or segmentsOfFlows must be defined!')
        elif (factor_Sets is not None) and (segmentsOfFlows is not None):
            raise Exception('Either factor_Sets or segmentsOfFlows must \
                            be defined! Not Both!')

        self.group = group

        # copy information of group to in-flows and out-flows
        for flow in self.inputs + self.outputs:
            flow.group = self.group

        # copy information about exists into segments of flows
        if self.segmentsOfFlows is not None:
            if isinstance(self.exists.active_data, (np.ndarray, list)):
                for key, item in self.segmentsOfFlows.items():
                    self.segmentsOfFlows[key] = [list(np.array(item) * factor) for factor in self.exists.active_data]
            elif isinstance(self.exists.active_data, (int, float)):
                for key, item in self.segmentsOfFlows.items():
                    self.segmentsOfFlows[key] = list(np.array(item) * self.exists.active_data)

    def __str__(self):
        # Creating a representation for factor_Sets with flow labels and their corresponding values
        if self.factor_Sets:
            factor_sets_rep = []
            for factor_set in self.factor_Sets:
                factor_sets_rep.append({flow.__repr__(): value for flow, value in factor_set.items()})
        else:
            factor_sets_rep = "None"

        # Representing inputs and outputs by their labels
        inputs_str = ",\n".join([flow.__str__() for flow in self.inputs])
        outputs_str = ",\n".join([flow.__str__() for flow in self.outputs])
        inputs_str = f"inputs=\n{textwrap.indent(inputs_str, ' ' * 3)}" if self.inputs else "inputs=[]"
        outputs_str = f"outputs=\n{textwrap.indent(outputs_str, ' ' * 3)}" if self.inputs else "outputs=[]"

        other_relevant_data = (f"factor_Sets={factor_sets_rep},\n"
                               f"segmentsOfFlows={self.segmentsOfFlows}")

        remaining_data = {
            key: value for key, value in self.__dict__.items()
            if value and
               not isinstance(value, cFlow) and
               key not in ["label", "TS_list", "segmentsOfFlows", "factor_Sets", "inputs", "outputs"]
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

    def transformFactorsToTS(self, factor_Sets):
        """
        macht alle Faktoren, die nicht TimeSeries sind, zu TimeSeries

        :param factor_Sets:
        :return:
        """
        # Einzelne Faktoren zu Vektoren:
        factor_Sets_TS = []
        # für jedes Dict -> Values (=Faktoren) zu Vektoren umwandeln:
        for aFactor_Dict in factor_Sets:  # Liste of dicts
            # Transform to TS:
            aFactor_Dict_TS = effect_values_to_ts('Faktor', aFactor_Dict, self)
            factor_Sets_TS.append(aFactor_Dict_TS)
            # check flows:
            for flow in aFactor_Dict_TS:
                if not (flow in self.inputs + self.outputs):
                    raise Exception(self.label + ': Flow ' + flow.label + ' in Factor_Set ist nicht in inputs/outputs')
        return factor_Sets_TS

    def finalize(self):
        """

        :return:
        """
        super().finalize()

        # factor-sets:
        if self.segmentsOfFlows is None:

            # TODO: mathematisch für jeden Zeitschritt checken!!!!
            #  Anzahl Freiheitsgrade checken: =  Anz. Variablen - Anz. Gleichungen

            # alle Faktoren, die noch nicht TS_vector sind, umwandeln:
            self.factor_Sets = self.transformFactorsToTS(self.factor_Sets)

            self.degreesOfFreedom = (len(self.inputs) + len(self.outputs)) - len(self.factor_Sets)
            if self.degreesOfFreedom <= 0:
                raise Exception(self.label + ': ' + str(len(self.factor_Sets)) + ' Gleichungen VERSUS '
                                + str(len(self.inputs + self.outputs)) + ' Variablen!!!')

        # linear segments:
        else:

            # check if investsize is variable for any flow:            
            for flow in (self.inputs + self.outputs):
                if (flow.invest_parameters is not None) and \
                        not (flow.invest_parameters.investmentSize_is_fixed):
                    raise Exception('linearSegmentsOfFlows (in ' +
                                    self.label_full +
                                    ') and variable nominal_value' +
                                    '(invest_size) (in flow ' +
                                    flow.label_full +
                                    ') , does not make sense together!')

            # Flow als Keys rauspicken und alle Stützstellen als TimeSeries:
            self.segmentsOfFlows_TS = self.segmentsOfFlows
            for aFlow in self.segmentsOfFlows.keys():
                # 2. Stützstellen zu TimeSeries machen, wenn noch nicht TimeSeries!:
                for i in range(len(self.segmentsOfFlows[aFlow])):
                    stuetzstelle = self.segmentsOfFlows[aFlow][i]
                    self.segmentsOfFlows_TS[aFlow][i] = TimeSeries('Stuetzstelle', stuetzstelle, self)

            def get_var_on():
                return self.mod.var_on

            self.feature_linSegments = cFeatureLinearSegmentSet('linearSegments', self, self.segmentsOfFlows_TS,
                                                                get_var_on=get_var_on,
                                                                checkListOfFlows=self.inputs + self.outputs)  # erst hier, damit auch nach __init__() noch Übergabe möglich.

    def declareVarsAndEqs(self, modBox: cModelBoxOfES):
        """
        Deklarieren von Variablen und Gleichungen

        :param modBox:
        :return:
        """
        super().declareVarsAndEqs(modBox)  # (ab hier sollte auch self.mod.var_on dann vorhanden sein)

        # factor-sets:
        if self.segmentsOfFlows is None:
            pass
        # linear segments:
        else:
            self.feature_linSegments.declareVarsAndEqs(modBox)

    def doModeling(self, modBox: cModelBoxOfES, timeIndexe):
        """
        Durchführen der Modellierung?

        :param modBox:
        :param timeIndexe:
        :return:
        """
        super().doModeling(modBox, timeIndexe)
        # factor_Sets:
        if self.segmentsOfFlows is None:
            # Transformer-Constraints:

            inputs_set = set(self.inputs)
            outputs_set = set(self.outputs)

            # für alle linearen Gleichungen:
            for i in range(len(self.factor_Sets)):
                # erstelle Gleichung für jedes t:
                # sum(inputs * factor) = sum(outputs * factor)
                # in1.val[t] * factor_in1[t] + in2.val[t] * factor_in2[t] + ... = out1.val[t] * factor_out1[t] + out2.val[t] * factor_out2[t] + ...

                aFactorVec_Dict = self.factor_Sets[i]

                leftSideFlows = inputs_set & aFactorVec_Dict.keys()  # davon nur die input-flows, die in Glg sind.
                rightSideFlows = outputs_set & aFactorVec_Dict.keys()  # davon nur die output-flows, die in Glg. sind.

                eq_linearFlowRelation_i = Equation('linearFlowRelation_' + str(i), self, modBox)
                for inFlow in leftSideFlows:
                    aFactor = aFactorVec_Dict[inFlow].active_data
                    eq_linearFlowRelation_i.addSummand(inFlow.mod.var_val, aFactor)  # input1.val[t]      * factor[t]
                for outFlow in rightSideFlows:
                    aFactor = aFactorVec_Dict[outFlow].active_data
                    eq_linearFlowRelation_i.addSummand(outFlow.mod.var_val, -aFactor)  # output.val[t] * -1 * factor[t]

                eq_linearFlowRelation_i.addRightSide(0)  # nur zur Komplettisierung der Gleichung

        # (linear) segments:
        # Zusammenhänge zw. inputs & outputs können auch vollständig über Segmente beschrieben werden:
        else:
            self.feature_linSegments.doModeling(modBox, timeIndexe)

    def print(self, shiftChars):
        """
        Ausgabe von irgendwas?

        :param shiftChars:
        :return:
        """
        super().print(shiftChars)
        # attribut hat es nur bei factor_sets:
        if hasattr(self, 'degreesOfFreedom'):
            print(shiftChars + '  ' + 'Degr. of Freedom: ' + str(self.degreesOfFreedom))

    # todo: checkbounds!
    # def initializeParameter(self,aStr,aBounds):
    # private Variable:
    #     self._eta          = aBounds['eta'][0]
    # exec('self.__' + aStr + ' = aBounds[0] ')
    # property dazu:
    #    self.eta            = property(lambda s: s.__get_param('eta'), lambda s,v: s.__set_param(v,'eta')')
    # exec('self.'   + aStr + ' = property(lambda s: s.__get_param(aStr) , lambda s,v: s.__set_param(v,aStr ))')

    def setLinearSegments(self, segmentsOfFlows):
        """
        alternative input of segments -> advantage: flows are already integrated
        
        segmentsOfFlow: dict
            description see in arguments of flow
        :return:
        """
        print('#################')
        print('warning: function setLinearSegments() will be replaced! Use init argument segmentsOfFlows instead!')
        self.segmentsOfFlows = segmentsOfFlows  # attribute of mother-class


class Boiler(LinearTransformer):
    """
    class Boiler
    """
    new_init_args = ['label', 'eta', 'Q_fu', 'Q_th', ]
    not_used_args = ['label', 'inputs', 'outputs', 'factor_Sets']

    def __init__(self, label:str, eta:Numeric_TS, Q_fu:cFlow, Q_th:cFlow, **kwargs):
        '''
        constructor for boiler

        Parameters
        ----------
        label : str
            name of bolier.
        eta : float or TS
            thermal efficiency.
        Q_fu : cFlow
            fuel input-flow
        Q_th : cFlow
            thermal output-flow.
        **kwargs : see mother classes!
        

        '''
        # super:
        kessel_bilanz = {Q_fu: eta,
                         Q_th: 1}  # eq: eta * Q_fu = 1 * Q_th # TODO: Achtung eta ist hier noch nicht TS-vector!!!

        super().__init__(label, inputs=[Q_fu], outputs=[Q_th], factor_Sets=[kessel_bilanz], **kwargs)

        # args to attributes:
        self.eta = TimeSeries('eta', eta, self)  # thermischer Wirkungsgrad
        self.Q_fu = Q_fu
        self.Q_th = Q_th

        # allowed medium:
        Q_fu.setMediumIfNotSet(cMediumCollection.fuel)
        Q_th.setMediumIfNotSet(cMediumCollection.heat)

        # Plausibilität eta:
        self.eta_bounds = [0 + 1e-10, 1 - 1e-10]  # 0 < eta_th < 1
        helpers.checkBoundsOfParameter(eta, 'eta', self.eta_bounds, self)

        # # generische property für jeden Koeffizienten
        # self.eta = property(lambda s: s.__get_coeff('eta'), lambda s,v: s.__set_coeff(v,'eta'))


class Power2Heat(LinearTransformer):
    """
    class Power2Heat
    """
    new_init_args = ['label', 'eta', 'P_el', 'Q_th', ]
    not_used_args = ['label', 'inputs', 'outputs', 'factor_Sets']

    def __init__(self, label:str, eta:Numeric_TS, P_el:cFlow, Q_th:cFlow, **kwargs):
        '''
        constructor for boiler

        Parameters
        ----------
        label : str
            name of bolier.
        eta : float or TS
            thermal efficiency.
        P_el : cFlow
            electric input-flow
        Q_th : cFlow
            thermal output-flow.
        **kwargs : see mother classes!


        '''
        # super:
        kessel_bilanz = {P_el: eta,
                         Q_th: 1}  # eq: eta * Q_fu = 1 * Q_th # TODO: Achtung eta ist hier noch nicht TS-vector!!!

        super().__init__(label, inputs=[P_el], outputs=[Q_th], factor_Sets=[kessel_bilanz], **kwargs)

        # args to attributes:
        self.eta = TimeSeries('eta', eta, self)  # thermischer Wirkungsgrad
        self.P_el = P_el
        self.Q_th = Q_th

        # allowed medium:
        P_el.setMediumIfNotSet(cMediumCollection.el)
        Q_th.setMediumIfNotSet(cMediumCollection.heat)

        # Plausibilität eta:
        self.eta_bounds = [0 + 1e-10, 1 - 1e-10]  # 0 < eta_th < 1
        helpers.checkBoundsOfParameter(eta, 'eta', self.eta_bounds, self)

        # # generische property für jeden Koeffizienten
        # self.eta = property(lambda s: s.__get_coeff('eta'), lambda s,v: s.__set_coeff(v,'eta'))


class HeatPump(LinearTransformer):
    """
    class HeatPump
    """
    new_init_args = ['label', 'COP', 'P_el', 'Q_th', ]
    not_used_args = ['label', 'inputs', 'outputs', 'factor_Sets']

    def __init__(self, label:str, COP:Numeric_TS, P_el:cFlow, Q_th:cFlow, **kwargs):
        '''
        Parameters
        ----------
        label : str
            name of heatpump.
        COP : float or TS
            Coefficient of performance.
        P_el : cFlow
            electricity input-flow.
        Q_th : cFlow
            thermal output-flow.
        **kwargs : see motherclasses
        '''

        # super:
        heatPump_bilanz = {P_el: COP, Q_th: 1}  # TODO: Achtung eta ist hier noch nicht TS-vector!!!
        super().__init__(label, inputs=[P_el], outputs=[Q_th], factor_Sets=[heatPump_bilanz], **kwargs)

        # args to attributes:
        self.COP = TimeSeries('COP', COP, self)  # thermischer Wirkungsgrad
        self.P_el = P_el
        self.Q_th = Q_th

        # allowed medium:
        P_el.setMediumIfNotSet(cMediumCollection.el)
        Q_th.setMediumIfNotSet(cMediumCollection.heat)

        # Plausibilität eta:
        self.eta_bounds = [0 + 1e-10, 20 - 1e-10]  # 0 < COP < 1
        helpers.checkBoundsOfParameter(COP, 'COP', self.eta_bounds, self)


class CoolingTower(LinearTransformer):
    """
    Klasse CoolingTower
    """
    new_init_args = ['label', 'specificElectricityDemand', 'P_el', 'Q_th', ]
    not_used_args = ['label', 'inputs', 'outputs', 'factor_Sets']

    def __init__(self, label:str, specificElectricityDemand:Numeric_TS, P_el:cFlow, Q_th:cFlow, **kwargs):
        '''
        Parameters
        ----------
        label : str
            name of cooling tower.
        specificElectricityDemand : float or TS
            auxiliary electricty demand per cooling power, i.g. 0.02 (2 %).
        P_el : cFlow
            electricity input-flow.
        Q_th : cFlow
            thermal input-flow.
        **kwargs : see getKwargs() and their description in motherclasses
            
        '''
        # super:         
        auxElectricity_eq = {P_el: 1,
                             Q_th: -specificElectricityDemand}  # eq: 1 * P_el - specificElectricityDemand * Q_th = 0  # TODO: Achtung eta ist hier noch nicht TS-vector!!!
        super().__init__(label, inputs=[P_el, Q_th], outputs=[], factor_Sets=[auxElectricity_eq], **kwargs)

        # args to attributes:
        self.specificElectricityDemand = TimeSeries('specificElectricityDemand', specificElectricityDemand,
                                                    self)  # thermischer Wirkungsgrad
        self.P_el = P_el
        self.Q_th = Q_th

        # allowed medium:
        P_el.setMediumIfNotSet(cMediumCollection.el)
        Q_th.setMediumIfNotSet(cMediumCollection.heat)

        # Plausibilität eta:
        self.specificElectricityDemand_bounds = [0, 1]  # 0 < eta_th < 1
        helpers.checkBoundsOfParameter(specificElectricityDemand, 'specificElectricityDemand',
                                       self.specificElectricityDemand_bounds, self)


class cKWK(LinearTransformer):
    """
    class of combined heat and power unit (CHP)
    """
    new_init_args = ['label', 'eta_th', 'eta_el', 'Q_fu', 'P_el', 'Q_th']
    not_used_args = ['label', 'inputs', 'outputs', 'factor_Sets']

    # eta = 1 # Thermischer Wirkungsgrad
    # __eta_bound = [0,1]

    def __init__(self, label:str, eta_th:Numeric_TS, eta_el:Numeric_TS, Q_fu:cFlow, P_el:cFlow, Q_th:cFlow, **kwargs):
        '''
        constructor of cCHP

        Parameters
        ----------
        label : str
            name of CHP-unit.
        eta_th : float or TS
            thermal efficiency.
        eta_el : float or TS
            electrical efficiency.
        Q_fu : cFlow
            fuel input-flow.
        P_el : cFlow
            electricity output-flow.
        Q_th : cFlow
            heat output-flow.
        **kwargs : 
        
        '''

        # super:
        waerme_glg = {Q_fu: eta_th, Q_th: 1}
        strom_glg = {Q_fu: eta_el, P_el: 1}
        #                      inputs         outputs               lineare Gleichungen
        super().__init__(label, inputs=[Q_fu], outputs=[P_el, Q_th], factor_Sets=[waerme_glg, strom_glg], **kwargs)

        # args to attributes:
        self.eta_th = TimeSeries('eta_th', eta_th, self)
        self.eta_el = TimeSeries('eta_el', eta_el, self)
        self.Q_fu = Q_fu
        self.P_el = P_el
        self.Q_th = Q_th

        # allowed medium:
        Q_fu.setMediumIfNotSet(cMediumCollection.fuel)
        Q_th.setMediumIfNotSet(cMediumCollection.heat)
        P_el.setMediumIfNotSet(cMediumCollection.el)

        # Plausibilität eta:
        self.eta_th_bounds = [0 + 1e-10, 1 - 1e-10]  # 0 < eta_th < 1
        self.eta_el_bounds = [0 + 1e-10, 1 - 1e-10]  # 0 < eta_el < 1

        helpers.checkBoundsOfParameter(eta_th, 'eta_th', self.eta_th_bounds, self)
        helpers.checkBoundsOfParameter(eta_el, 'eta_el', self.eta_el_bounds, self)
        helpers.checkBoundsOfParameter(eta_th + eta_el, 'eta_th+eta_el', [0 + 1e-10, 1 - 1e-10], self)


class cAbwaermeHP(LinearTransformer):
    """
    class cAbwaermeHP
    """
    new_init_args = ['label', 'COP', 'Q_ab', 'P_el', 'Q_th', ]
    not_used_args = ['label', 'inputs', 'outputs', 'factor_Sets']

    def __init__(self, label:str, COP:Numeric_TS, P_el:cFlow, Q_ab:cFlow, Q_th:cFlow, **kwargs):
        '''
        Parameters
        ----------
        label : str
            name of heatpump.
        COP : float, TS
            Coefficient of performance.
        Q_ab : cFlow
            Heatsource input-flow.
        P_el : cFlow
            electricity input-flow.
        Q_th : cFlow
            thermal output-flow.
        **kwargs : see motherclasses
        '''

        # super:
        heatPump_bilanzEl = {P_el: COP, Q_th: 1}
        if isinstance(COP, TimeSeriesRaw):
            COP = COP.value
            heatPump_bilanzAb = {Q_ab: COP / (COP - 1), Q_th: 1}
        else:
            heatPump_bilanzAb = {Q_ab: COP / (COP - 1), Q_th: 1}
        super().__init__(label, inputs=[P_el, Q_ab], outputs=[Q_th],
                         factor_Sets=[heatPump_bilanzEl, heatPump_bilanzAb], **kwargs)

        # args to attributes:
        self.COP = TimeSeries('COP', COP, self)  # thermischer Wirkungsgrad
        self.P_el = P_el
        self.Q_ab = Q_ab
        self.Q_th = Q_th

        # allowed medium:
        P_el.setMediumIfNotSet(cMediumCollection.el)
        Q_th.setMediumIfNotSet(cMediumCollection.heat)
        Q_ab.setMediumIfNotSet(cMediumCollection.heat)

        # Plausibilität eta:
        self.eta_bounds = [0 + 1e-10, 20 - 1e-10]  # 0 < COP < 1
        helpers.checkBoundsOfParameter(COP, 'COP', self.eta_bounds, self)


class cStorage(cBaseComponent):
    """
    Klasse cStorage
    """

    # TODO: Dabei fällt mir auf. Vielleicht sollte man mal überlegen, ob man für Ladeleistungen bereits in dem
    #  jeweiligen Zeitschritt mit einem Verlust berücksichtigt. Zumindest für große Zeitschritte bzw. große Verluste
    #  eventuell relevant.
    #  -> Sprich: speicherverlust = charge_state(t) * fracLossPerHour * dt + 0.5 * Q_lade(t) * dt * fracLossPerHour * dt
    #  -> müsste man aber auch für den sich ändernden Ladezustand berücksichtigten

    # costs_default = property(get_costs())
    # param_defalt  = property(get_params())

    new_init_args = ['label', 'exists', 'inFlow', 'outFlow', 'capacity_inFlowHours', 'min_rel_chargeState', 'max_rel_chargeState',
                     'chargeState0_inFlowHours', 'charge_state_end_min', 'charge_state_end_max', 'eta_load',
                     'eta_unload', 'fracLossPerHour', 'avoidInAndOutAtOnce', 'invest_parameters']

    not_used_args = ['label', 'exists']

    # capacity_inFlowHours: float, 'lastValueOfSim', None
    def __init__(self,
                 label: str,
                 inFlow: cFlow,
                 outFlow: cFlow,
                 capacity_inFlowHours: Optional[Union[Skalar, Literal['lastValueOfSim']]],
                 group: Optional[str] = None,
                 min_rel_chargeState: Numeric_TS = 0,
                 max_rel_chargeState: Numeric_TS = 1,
                 chargeState0_inFlowHours: Skalar = 0,
                 charge_state_end_min: Optional[Skalar] = None,
                 charge_state_end_max: Optional[Skalar] = None,
                 eta_load: Numeric_TS = 1, eta_unload: Numeric_TS = 1,
                 fracLossPerHour: Numeric_TS = 0,
                 avoidInAndOutAtOnce: bool = True,
                 invest_parameters: Optional[InvestParameters] = None,
                 **kwargs):
        '''
        constructor of storage

        Parameters
        ----------
        label : str
            description.
        inFlow : cFlow
            ingoing flow.
        outFlow : cFlow
            outgoing flow.
        group: str, None
            group name to assign components to groups. Used for later analysis of the results
        exists: Numeric_TS
            Limits the availlable capacity, and the in and out flow. DOes not affect other parameters yet
            (like frac_loss_per_hour, starting value, ...)
        capacity_inFlowHours : float or None
            nominal capacity of the storage 
            float: capacity in FlowHours
            None:  if invest_parameters.investmentSize_is_fixed = False
        min_rel_chargeState : float or TS, optional
            minimum relative charge state. The default is 0.
        max_rel_chargeState : float or TS, optional
            maximum relative charge state. The default is 1.
        chargeState0_inFlowHours : None, float (0...1), 'lastValueOfSim',  optional
            storage capacity in Flowhours at the beginning. The default is 0.
            float: defined capacity at start of first timestep
            None: free to choose by optimizer
            'lastValueOfSim': chargeState0 is equal to chargestate of last timestep ("closed simulation")
        charge_state_end_min : float or None, optional
            minimal value of chargeState at the end of timeseries. 
        charge_state_end_max : float or None, optional
            maximal value of chargeState at the end of timeseries. 
        eta_load : float, optional
            efficiency factor of charging/loading. The default is 1.
        eta_unload : TYPE, optional
            efficiency factor of uncharging/unloading. The default is 1.
        fracLossPerHour : float or TS. optional
            loss per chargeState-Unit per hour. The default is 0.
        avoidInAndOutAtOnce : boolean, optional
            should simultaneously Loading and Unloading be avoided? (Attention, Performance maybe becomes worse with avoidInAndOutAtOnce=True). The default is True.
        invest_parameters : InvestParameters, optional
            invest arguments. The default is None.
        
        **kwargs : TYPE # TODO welche kwargs werden hier genutzt???
            DESCRIPTION.
        '''
        # TODO: neben min_rel_chargeState, max_rel_chargeState ggf. noch "val_rel_chargeState" implementieren damit konsistent zu flow (max_rel, min_rel, val_re)

        # charge_state_end_min (absolute Werte, aber relative wären ggf. auch manchmal hilfreich)
        super().__init__(label, **kwargs)

        # args to attributes:
        self.inputs = [inFlow]
        self.outputs = [outFlow]
        self.inFlow = inFlow
        self.outFlow = outFlow
        self.capacity_inFlowHours = capacity_inFlowHours
        self.max_rel_chargeState = TimeSeries('max_rel_chargeState', max_rel_chargeState, self)
        self.min_rel_chargeState = TimeSeries('min_rel_chargeState', min_rel_chargeState, self)

        self.group = group

        # add last time step (if not scalar):
        existsWithEndTimestep = self.exists.active_data if np.isscalar(self.exists.active_data) else np.append(self.exists.active_data, self.exists.active_data[-1])
        self.max_rel_chargeState = TimeSeries('max_rel_chargeState',
                                              self.max_rel_chargeState.active_data * existsWithEndTimestep, self)
        self.min_rel_chargeState = TimeSeries('min_rel_chargeState',
                                              self.min_rel_chargeState.active_data * existsWithEndTimestep, self)

        # copy information of "group" to in-flows and out-flows
        for flow in self.inputs + self.outputs:
            flow.group = self.group

        self.chargeState0_inFlowHours = chargeState0_inFlowHours
        self.charge_state_end_min = charge_state_end_min

        if charge_state_end_max is None:
            # Verwende Lösungen bis zum vollen Speicher
            self.charge_state_end_max = self.capacity_inFlowHours
        else:
            self.charge_state_end_max = charge_state_end_max
        self.eta_load = TimeSeries('eta_load', eta_load, self)
        self.eta_unload = TimeSeries('eta_unload', eta_unload, self)
        self.fracLossPerHour = TimeSeries('fracLossPerHour', fracLossPerHour, self)
        self.avoidInAndOutAtOnce = avoidInAndOutAtOnce

        self.invest_parameters = invest_parameters
        self.featureInvest = None

        if self.avoidInAndOutAtOnce:
            self.featureAvoidInAndOut = cFeatureAvoidFlowsAtOnce('feature_avoidInAndOutAtOnce', self,
                                                                 [self.inFlow, self.outFlow])

        if self.invest_parameters is not None:
            self.featureInvest = cFeatureInvest('used_capacity_inFlowHours', self, self.invest_parameters,
                                                min_rel=self.min_rel_chargeState,
                                                max_rel=self.max_rel_chargeState,
                                                val_rel=None,  # kein vorgegebenes Profil
                                                investmentSize=self.capacity_inFlowHours,
                                                featureOn=None)  # hier gibt es kein On-Wert

        # Medium-Check:
        if not (cMediumCollection.checkIfFits(inFlow.medium, outFlow.medium)):
            raise Exception('in cStorage ' + self.label + ': input.medium = ' + str(inFlow.medium) +
                            ' and output.medium = ' + str(outFlow.medium) + ' don`t fit!')
        # TODO: chargeState0 darf nicht größer max usw. abfangen!

        self.isStorage = True  # for postprocessing

    def declareVarsAndEqs(self, modBox: cModelBoxOfES):
        """
        Deklarieren von Variablen und Gleichungen

        :param modBox:
        :return:
        """
        super().declareVarsAndEqs(modBox)

        # Variablen:

        if self.featureInvest is None:
            lb = self.min_rel_chargeState.active_data * self.capacity_inFlowHours
            ub = self.max_rel_chargeState.active_data * self.capacity_inFlowHours
            fix_value = None

            if np.isscalar(lb):
                pass
            else:
                lb = np.append(lb, 0)  # self.charge_state_end_min)
            if np.isscalar(ub):
                pass
            else:
                ub = np.append(ub, self.capacity_inFlowHours)  # charge_state_end_max)

        else:
            (lb, ub, fix_value) = self.featureInvest.getMinMaxOfDefiningVar()

            if np.isscalar(lb):
                pass
            else:
                lb = np.append(lb, lb[-1])  # self.charge_state_end_min)
            if np.isscalar(ub):
                pass
            else:
                ub = np.append(ub, ub[-1])  # charge_state_end_max)

        self.mod.var_charge_state = VariableTS('charge_state', modBox.nrOfTimeSteps + 1, self, modBox, min=lb, max=ub,
                                               value=fix_value)  # Eins mehr am Ende!
        self.mod.var_charge_state.activateBeforeValues(self.chargeState0_inFlowHours, True)
        self.mod.var_nettoFlow = VariableTS('nettoFlow', modBox.nrOfTimeSteps, self, modBox,
                                            min=-np.inf)  # negative Werte zulässig!

        # erst hier, da definingVar vorher nicht belegt!
        if self.featureInvest is not None:
            self.featureInvest.setDefiningVar(self.mod.var_charge_state, None)  # None, da kein On-Wert
            self.featureInvest.declareVarsAndEqs(modBox)

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
        # refToStromLastEq.addSummand(obj.vars.Q_th_Lade   ,-1*obj.inputData.spezifPumpstromAufwandBeladeEntlade); % für diese Komponenten Stromverbrauch!
        # refToStromLastEq.addSummand(obj.vars.Q_th_Entlade,-1*obj.inputData.spezifPumpstromAufwandBeladeEntlade); % für diese Komponenten Stromverbrauch!

    def getInitialStatesOfNextSection(timeIndex):
        """
        TODO: was passiert hier? Zuweisungen noch nicht richtig?

        :return:
        """
        initialStates['chargeState0_inFlowHours'] = charge_state[timeIndexe[0]]
        return initialStates

    def doModeling(self, modBox, timeIndexe):
        """
        Durchführen der Modellierung?

        :param modBox:
        :param timeIndexe:
        :return:
        """
        super().doModeling(modBox, timeIndexe)

        # Gleichzeitiges Be-/Entladen verhindern:
        if self.avoidInAndOutAtOnce: self.featureAvoidInAndOut.doModeling(modBox, timeIndexe)

        # % Speicherladezustand am Start
        if self.chargeState0_inFlowHours is None:
            # Startzustand bleibt Freiheitsgrad
            pass
        elif helpers.is_number(self.chargeState0_inFlowHours):
            # eq: Q_Ladezustand(1) = Q_Ladezustand_Start;
            self.eq_charge_state_start = Equation('charge_state_start', self, modBox, eqType='eq')
            self.eq_charge_state_start.addRightSide(self.mod.var_charge_state.beforeVal())  # chargeState_0 !
            self.eq_charge_state_start.addSummand(self.mod.var_charge_state, 1, timeIndexe[0])
        elif self.chargeState0_inFlowHours == 'lastValueOfSim':
            # eq: Q_Ladezustand(1) - Q_Ladezustand(end) = 0;
            self.eq_charge_state_start = Equation('charge_state_start', self, modBox, eqType='eq')
            self.eq_charge_state_start.addSummand(self.mod.var_charge_state, 1, timeIndexe[0])
            self.eq_charge_state_start.addSummand(self.mod.var_charge_state, -1, timeIndexe[-1])
        else:
            raise Exception('chargeState0_inFlowHours has undefined value = ' + str(self.chargeState0_inFlowHours))

        # Speicherleistung / Speicherladezustand / Speicherverlust
        #                                                                          | Speicher-Beladung       |   |Speicher-Entladung                |
        # Q_Ladezustand(n+1) + (-1+VerlustanteilProStunde*dt(n)) *Q_Ladezustand(n) -  dt(n)*eta_Lade*Q_th_Lade(n) +  dt(n)* 1/eta_Entlade*Q_th_Entlade(n)  = 0

        # charge_state hat ein Index mehr:
        timeIndexeChargeState = range(timeIndexe.start, timeIndexe.stop + 1)
        self.eq_charge_state = Equation('charge_state', self, modBox, eqType='eq')
        self.eq_charge_state.addSummand(self.mod.var_charge_state,
                                        -1 * (1 - self.fracLossPerHour.active_data * modBox.dtInHours),
                                        timeIndexeChargeState[
                                        :-1])  # sprich 0 .. end-1 % nach letztem Zeitschritt gibt es noch einen weiteren Ladezustand!
        self.eq_charge_state.addSummand(self.mod.var_charge_state, 1, timeIndexeChargeState[1:])  # 1:end
        self.eq_charge_state.addSummand(self.inFlow.mod.var_val, -1 * self.eta_load.active_data * modBox.dtInHours)
        self.eq_charge_state.addSummand(self.outFlow.mod.var_val,
                                        1 / self.eta_unload.active_data * modBox.dtInHours)  # Achtung hier 1/eta!

        # Speicherladezustand am Ende
        # -> eigentlich min/max-Wert für variable, aber da nur für ein Element hier als Glg:
        # 1: eq:  Q_charge_state(end) <= Q_max
        if self.charge_state_end_max is not None:
            self.eq_charge_state_end_max = Equation('eq_charge_state_end_max', self, modBox, eqType='ineq')
            self.eq_charge_state_end_max.addSummand(self.mod.var_charge_state, 1, timeIndexeChargeState[-1])
            self.eq_charge_state_end_max.addRightSide(self.charge_state_end_max)

        # 2: eq: - Q_charge_state(end) <= - Q_min
        if self.charge_state_end_min is not None:
            self.eq_charge_state_end_min = Equation('eq_charge_state_end_min', self, modBox, eqType='ineq')
            self.eq_charge_state_end_min.addSummand(self.mod.var_charge_state, -1, timeIndexeChargeState[-1])
            self.eq_charge_state_end_min.addRightSide(- self.charge_state_end_min)

        # nettoflow:
        # eq: nettoFlow(t) - outFlow(t) + inFlow(t) = 0
        self.eq_nettoFlow = Equation('nettoFlow', self, modBox, eqType='eq')
        self.eq_nettoFlow.addSummand(self.mod.var_nettoFlow, 1)
        self.eq_nettoFlow.addSummand(self.inFlow.mod.var_val, 1)
        self.eq_nettoFlow.addSummand(self.outFlow.mod.var_val, -1)

        if self.featureInvest is not None:
            self.featureInvest.doModeling(modBox, timeIndexe)

        # ############# Gleichungen ##########################
        # % Speicherleistung an Bilanzgrenze / Speicher-Ladung / Speicher-Entladung
        # % Q_th(n) + Q_th_Lade(n) - Q_th_Entlade(n) = 0;
        # obj.eqs.Leistungen = Equation('Leistungen');
        # obj.eqs.Leistungen.addSummand(obj.vars.Q_th        , 1);
        # obj.eqs.Leistungen.addSummand(obj.vars.Q_th_Lade   , 1);
        # obj.eqs.Leistungen.addSummand(obj.vars.Q_th_Entlade,-1);

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
        # obj.ineqs.EntwederLadenOderEntladen.addSummand(obj.vars.IchLadeMich   ,1);
        # obj.ineqs.EntwederLadenOderEntladen.addSummand(obj.vars.IchEntladeMich,1);
        # obj.ineqs.EntwederLadenOderEntladen.addRightSide(1);

    def addShareToGlobals(self, globalComp: cGlobal, modBox):
        """

        :param globalComp:
        :param modBox:
        :return:
        """
        super().addShareToGlobals(globalComp, modBox)

        if self.featureInvest is not None:
            self.featureInvest.addShareToGlobals(globalComp, modBox)


class cSourceAndSink(cBaseComponent):
    """
    class for source (output-flow) and sink (input-flow) in one commponent
    """
    # source : cFlow
    # sink   : cFlow

    new_init_args = ['label', 'source', 'sink', 'avoidInAndOutAtOnce']

    not_used_args = ['label']

    def __init__(self, label: str, source: cFlow, sink: cFlow, group: str = None,
                 avoidInAndOutAtOnce: bool = True, **kwargs):
        '''
        Parameters
        ----------
        label : str
            name of sourceAndSink
        source : cFlow
            output-flow of this component
        sink : cFlow
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
        self.source.activateOnValue()
        self.sink.activateOnValue()

        if self.avoidInAndOutAtOnce:
            self.featureAvoidInAndOutAtOnce = cFeatureAvoidFlowsAtOnce('sinkOrSource', self, [self.source, self.sink])
        else:
            self.featureAvoidInAndOutAtOnce = None

    def declareVarsAndEqs(self, modBox):
        """
        Deklarieren von Variablen und Gleichungen

        :param modBox:
        :return:
        """
        super().declareVarsAndEqs(modBox)

    def doModeling(self, modBox, timeIndexe):
        """
        Durchführen der Modellierung?

        :param modBox:
        :param timeIndexe:
        :return:
        """
        super().doModeling(modBox, timeIndexe)
        # Entweder Sink-Flow oder Source-Flow aktiv. Nicht beide Zeitgleich!
        if self.featureAvoidInAndOutAtOnce is not None:
            self.featureAvoidInAndOutAtOnce.doModeling(modBox, timeIndexe)


class cSource(cBaseComponent):
    """
    class of a source
    """
    new_init_args = ['label', 'source']
    not_used_args = ['label']

    def __init__(self, label: str, source: cFlow, group: str = None, **kwargs):
        '''       
        Parameters
        ----------
        label : str
            name of source
        source : cFlow
            output-flow of source
        group: str, None
            group name to assign components to groups. Used for later analysis of the results
        **kwargs : TYPE

        Returns
        -------
        None.

        '''

        """
        Konstruktor für Instanzen der Klasse cSource

        :param str label: Bezeichnung
        :param cFlow source: flow-output Quelle
        :param kwargs:
        """
        super().__init__(label, **kwargs)
        self.source = source
        self.outputs.append(source)  # ein Output-Flow

        self.group = group

        # copy information of group to in-flows and out-flows
        for flow in self.inputs + self.outputs:
            flow.group = self.group


class cSink(cBaseComponent):
    """
    Klasse cSink
    """
    new_init_args = ['label', 'source']
    not_used_args = ['label']

    def __init__(self, label: str, sink: cFlow, group: str = None, **kwargs):
        '''
        constructor of sink 

        Parameters
        ----------
        label : str
            name of sink.
        sink : cFlow
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



class cTransportation(cBaseComponent):
    # TODO: automatic on-Value in Flows if loss_abs
    # TODO: loss_abs must be: investment_size * loss_abs_rel!!!
    # TODO: investmentsize only on 1 flow
    # TODO: automatic invest_parameters for both in-flows (or alternatively both out-flows!)
    # TODO: optional: capacities should be recognised for losses

    def __init__(self,
                 label: str,
                 in1: cFlow,
                 out1: cFlow,
                 in2: Optional[cFlow] = None,
                 out2: Optional[cFlow] = None,
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
            name of cTransportation.
        in1 : cFlow
            inflow of input at side A
        out1 : cFlow
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
        switchOnCosts : 
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
            self.featureAvoidBothDirectionsAtOnce = cFeatureAvoidFlowsAtOnce('feature_avoidBothDirectionsAtOnce', self,
                                                                             [self.in1, self.in2])

    def declareVarsAndEqs(self, modBox: cModelBoxOfES):
        """
        Deklarieren von Variablen und Gleichungen
        
        :param modBox:
        :return:
        """
        super().declareVarsAndEqs(modBox)

    def doModeling(self, modBox, timeIndexe):
        super().doModeling(modBox, timeIndexe)

        # not both directions at once:
        if self.avoidFlowInBothDirectionsAtOnce and (
                self.in2 is not None): self.featureAvoidBothDirectionsAtOnce.doModeling(modBox, timeIndexe)

        # first direction
        # eq: in(t)*(1-loss_rel(t)) = out(t) + on(t)*loss_abs(t)
        self.eq_dir1 = Equation('transport_dir1', self, modBox, eqType='eq')
        self.eq_dir1.addSummand(self.in1.mod.var_val, (1 - self.loss_rel.active_data))
        self.eq_dir1.addSummand(self.out1.mod.var_val, -1)
        if (self.loss_abs.active_data is not None) and np.any(self.loss_abs.active_data != 0):
            assert self.in1.mod.var_on is not None, 'Var on wird benötigt für in1! Set min_rel!'
            self.eq_dir1.addSummand(self.in1.mod.var_on, -1 * self.loss_abs.active_data)

        # second direction:        
        if self.in2 is not None:
            # eq: in(t)*(1-loss_rel(t)) = out(t) + on(t)*loss_abs(t)
            self.eq_dir2 = Equation('transport_dir2', self, modBox, eqType='eq')
            self.eq_dir2.addSummand(self.in2.mod.var_val, 1 - self.loss_rel.active_data)
            self.eq_dir2.addSummand(self.out2.mod.var_val, -1)
            if (self.loss_abs.active_data is not None) and np.any(self.loss_abs.active_data != 0):
                assert self.in2.mod.var_on is not None, 'Var on wird benötigt für in2! Set min_rel!'
                self.eq_dir2.addSummand(self.in2.mod.var_on, -1 * self.loss_abs.active_data)

        # always On (in at least one direction)
        # eq: in1.on(t) +in2.on(t) >= 1 # TODO: this is some redundant to avoidFlowInBothDirections
        if self.isAlwaysOn:
            self.eq_alwaysOn = Equation('alwaysOn', self, modBox, eqType='ineq')
            self.eq_alwaysOn.addSummand(self.in1.mod.var_on, -1)
            if (self.in2 is not None): self.eq_alwaysOn.addSummand(self.in2.mod.var_on, -1)
            self.eq_alwaysOn.addRightSide(-.5)  # wg binärungenauigkeit 0.5 statt 1

        # equate nominal value of second direction
        if (self.in2 is not None):
            oneInFlowHasFeatureInvest = (self.in1.featureInvest is not None) or (self.in1.featureInvest is not None)
            bothInFlowsHaveFeatureInvest = (self.in1.featureInvest is not None) and (self.in1.featureInvest is not None)
            if oneInFlowHasFeatureInvest:
                if bothInFlowsHaveFeatureInvest:
                    # eq: in1.nom_value = in2.nom_value
                    self.eq_nom_value = Equation('equalSizeInBothDirections', self, modBox, eqType='eq')
                    self.eq_nom_value.addSummand(self.in1.featureInvest.mod.var_investmentSize, 1)
                    self.eq_nom_value.addSummand(self.in2.featureInvest.mod.var_investmentSize, -1)
                else:
                    raise Exception(
                        'define invest_parameters also for second In-Flow (values can be empty!)')  # TODO: anders lösen (automatisiert)!
