# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:45:12 2020

@author: Panitz

# references: 
#   from flixoptmat 1.0  : structure / features / math, constraints, ...
#   from oemof        : some name-definition/ some structure
"""

# TODO:
#  optionale Flows einführen -> z.b. Asche: --> Bei denen kommt keine Fehlermeldung, wenn nicht verknüpft und nicht aktiviert/genutzt
#  cVariable(,...,self,modBox,...) -> modBox entfernen und lieber über self.modBox aufrufen -> kürzer!
#  Hilfsenergiebedarf als Feature einführen?
#  self.xy = cVariable('xy'), -> kürzer möglich, in cVariabe() über aComponent.addAttribute()
#  Variablen vielleicht doch bei allen Komponenten lieber unter var_struct abspeichern ?

import numpy as np

import flixOptHelperFcts as helpers
from basicModeling import *  # Modelliersprache
from flixStructure import *  # Grundstruktur
from flixFeatures import *


class cBaseLinearTransformer(cBaseComponent):
    """
    Klasse cBaseLinearTransformer: Grundgerüst lineare Übertragungskomponente
    """
    new_init_args = ['label', 'inputs', 'outputs', 'factor_Sets', 'segmentsOfFlows']
    not_used_args = ['label']
    def __init__(self, label, inputs, outputs, factor_Sets=None, segmentsOfFlows=None, **kwargs):
        '''
        Parameters
        ----------
        label : str
            name.
        inputs : list of flows
            input flows.
        outputs : list of flows
            output flows.
        factor_Sets : list
            linear relation between flows
            eq: sum (factor * flow_in) = sum (factor * flow_out)
            factor can be cTS_vector, scalar or list.
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
            

    def transformFactorsToTS(self, factor_Sets):
        """
        macht alle Faktoren, die nicht cTS_vector sind, zu cTS_vector

        :param factor_Sets:
        :return:
        """
        # Einzelne Faktoren zu Vektoren:
        factor_Sets_TS = []
        # für jedes Dict -> Values (=Faktoren) zu Vektoren umwandeln:
        for aFactor_Dict in factor_Sets:  # Liste of dicts
            # Transform to TS:
            aFactor_Dict_TS = transformDictValuesToTS('Faktor', aFactor_Dict, self)
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
                                + str(len(self.inputs+self.outputs)) + ' Variablen!!!')

        # linear segments:
        else:
            
            # check if investsize is variable for any flow:            
            for flow in (self.inputs+self.outputs):
                if (flow.investArgs is not None) and \
                    not (flow.investArgs.investmentSize_is_fixed):                
                        raise Exception('linearSegmentsOfFlows (in '+
                                        self.label_full +
                                        ') and variable nominal_value'+
                                        '(invest_size) (in flow ' + 
                                        flow.label_full + 
                                        ') , does not make sense together!')
            
            # Flow als Keys rauspicken und alle Stützstellen als cTS_Vector:
            self.segmentsOfFlows_TS = self.segmentsOfFlows
            for aFlow in self.segmentsOfFlows.keys():
                # 2. Stützstellen zu cTS_vector machen, wenn noch nicht cTS_vector!:
                for i in range(len(self.segmentsOfFlows[aFlow])):
                    stuetzstelle = self.segmentsOfFlows[aFlow][i]
                    self.segmentsOfFlows_TS[aFlow][i] = cTS_vector('Stuetzstelle', stuetzstelle, self)

            def get_var_on():
                return self.mod.var_on

            self.feature_linSegments = cFeatureLinearSegmentSet('linearSegments', self, self.segmentsOfFlows_TS,
                                                                get_var_on=get_var_on,
                                                                checkListOfFlows=self.inputs + self.outputs)  # erst hier, damit auch nach __init__() noch Übergabe möglich.

    def declareVarsAndEqs(self,modBox:cModelBoxOfES):
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

    def doModeling(self,modBox:cModelBoxOfES,timeIndexe):
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

                eq_linearFlowRelation_i = cEquation('linearFlowRelation_' + str(i), self, modBox)
                for inFlow in leftSideFlows:
                    aFactor = aFactorVec_Dict[inFlow].d_i
                    eq_linearFlowRelation_i.addSummand(inFlow.mod.var_val, aFactor)  # input1.val[t]      * factor[t]
                for outFlow in rightSideFlows:
                    aFactor = aFactorVec_Dict[outFlow].d_i
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


            

class cKessel(cBaseLinearTransformer):
    """
    class cKessel
    """
    new_init_args = ['label', 'eta', 'Q_fu', 'Q_th',]
    not_used_args = ['label', 'inputs', 'outputs', 'factor_Sets']
    
    def __init__(self, label, eta, Q_fu, Q_th, **kwargs):
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
        self.eta = cTS_vector('eta', eta, self)  # thermischer Wirkungsgrad
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


class cHeatPump(cBaseLinearTransformer):
    """
    class cHeatPump
    """
    new_init_args = ['label', 'COP', 'P_el', 'Q_th',]
    not_used_args = ['label', 'inputs', 'outputs', 'factor_Sets']
    
    def __init__(self, label, COP, P_el, Q_th, **kwargs):
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
        self.COP = cTS_vector('COP', COP, self)  # thermischer Wirkungsgrad
        self.P_el = P_el
        self.Q_th = Q_th

        # allowed medium:
        P_el.setMediumIfNotSet(cMediumCollection.el)
        Q_th.setMediumIfNotSet(cMediumCollection.heat)

        # Plausibilität eta:
        self.eta_bounds = [0 + 1e-10, 20 - 1e-10]  # 0 < COP < 1
        helpers.checkBoundsOfParameter(COP, 'COP', self.eta_bounds, self)


class cCoolingTower(cBaseLinearTransformer):
    """
    Klasse cCoolingTower
    """
    new_init_args = ['label', 'specificElectricityDemand', 'P_el', 'Q_th',]
    not_used_args = ['label', 'inputs', 'outputs', 'factor_Sets']

    def __init__(self, label, specificElectricityDemand, P_el, Q_th, **kwargs):
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
        self.specificElectricityDemand = cTS_vector('specificElectricityDemand', specificElectricityDemand,
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


class cKWK(cBaseLinearTransformer):
    """
    class of combined heat and power unit (CHP)
    """
    new_init_args = ['label', 'eta_th', 'eta_el', 'Q_fu', 'P_el', 'Q_th']
    not_used_args = ['label', 'inputs', 'outputs', 'factor_Sets']

    # eta = 1 # Thermischer Wirkungsgrad
    # __eta_bound = [0,1]

    def __init__(self, label, eta_th, eta_el, Q_fu, P_el, Q_th, **kwargs):
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
        self.eta_th = cTS_vector('eta_th', eta_th, self)
        self.eta_el = cTS_vector('eta_el', eta_el, self)
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
        if eta_th + eta_el > 1:
            raise Exception('Fehler in ' + self.label + ': eta_th + eta_el > 1 !')


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

    new_init_args = ['label', 'inFlow', 'outFlow', 'capacity_inFlowHours', 'min_rel_chargeState', 'max_rel_chargeState',
                 'chargeState0_inFlowHours', 'charge_state_end_min', 'charge_state_end_max', 'eta_load',
                 'eta_unload', 'fracLossPerHour', 'avoidInAndOutAtOnce' 'investArgs']

    not_used_args = ['label']

    # capacity_inFlowHours: float, 'lastValueOfSim', None
    def __init__(self, label, inFlow, outFlow, capacity_inFlowHours, min_rel_chargeState=0, max_rel_chargeState=1,
                 chargeState0_inFlowHours=0, charge_state_end_min=0, charge_state_end_max=None, eta_load=1,
                 eta_unload=1, fracLossPerHour=0, avoidInAndOutAtOnce=True, investArgs=None, **kwargs):
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
        capacity_inFlowHours : float or None
            float: Speicherkapazität in kWh. 
            None:  if investArgs.investmentSize_is_fixed = False
        min_rel_chargeState : float or TS, optional
            minimum relative charge state. The default is 0.
        max_rel_chargeState : float or TS, optional
            maximum relative charge state. The default is 1.
        chargeState0_inFlowHours : float (0...1), optional
            Speicherkapazität in kWh zu Beginn des Betrachtungszeitraums. The default is 0.
        charge_state_end_min : float, optional
            minimaler relativer (?) Speicherstand zum Ende des Betrachtungszeitraums (0...1). The default is 0.
        charge_state_end_max : float, optional
            maximaler relativer (?) Speicherstand zum Ende des Betrachtungszeitraums (0...1). The default is None.
        eta_load : float, optional
            Wirkungsgrad beim Laden (0...1). The default is 1.
        eta_unload : TYPE, optional
            Wirkungsgrad beim Entladen (0...1). The default is 1.
        fracLossPerHour : float or TS. optional
            Verlust pro Speichereinheit und Stunde TODO: pro Stunde oder pro Zeitschritt?. The default is 0.
        avoidInAndOutAtOnce : boolean, optional
            should simultaneously Loading and Unloading be avoided? (Attention, Performance maybe becomes worse with avoidInAndOutAtOnce=True). The default is True.
        investArgs : cInvestArgs, optional
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
        self.max_rel_chargeState = cTS_vector('max_rel_chargeState', max_rel_chargeState, self)
        self.min_rel_chargeState = cTS_vector('min_rel_chargeState', min_rel_chargeState, self)
        self.chargeState0_inFlowHours = chargeState0_inFlowHours
        self.charge_state_end_min = charge_state_end_min

        if charge_state_end_max is None:
            # Verwende Lösungen bis zum vollen Speicher
            self.charge_state_end_max = self.capacity_inFlowHours
        else:
            self.charge_state_end_max = charge_state_end_max
        self.eta_load = cTS_vector('eta_load', eta_load, self)
        self.eta_unload = cTS_vector('eta_unload', eta_unload, self)
        self.fracLossPerHour = cTS_vector('fracLossPerHour', fracLossPerHour, self)
        self.avoidInAndOutAtOnce = avoidInAndOutAtOnce

        self.investArgs = investArgs
        self.featureInvest = None

        if self.avoidInAndOutAtOnce:
            self.featureAvoidInAndOut = cFeatureAvoidFlowsAtOnce('feature_avoidInAndOutAtOnce', self,
                                                                 [self.inFlow, self.outFlow])

        if self.investArgs is not None:
            self.featureInvest = cFeatureInvest('used_capacity_inFlowHours', self, self.investArgs,
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

    def declareVarsAndEqs(self, modBox:cModelBoxOfES):
        """
        Deklarieren von Variablen und Gleichungen

        :param modBox:
        :return:
        """
        super().declareVarsAndEqs(modBox)

        # Variablen:

        if self.featureInvest is None:
            lb = self.min_rel_chargeState.d_i * self.capacity_inFlowHours
            ub = self.max_rel_chargeState.d_i * self.capacity_inFlowHours
            fix_value = None
        else:
            (lb, ub, fix_value) = self.featureInvest.getMinMaxOfDefiningVar()
        # todo: lb und ub muss noch um ein Element (chargeStateEnd_max, chargeStateEnd_min oder aber jeweils None) ergänzt werden!

        self.mod.var_charge_state = cVariable_TS('charge_state', modBox.nrOfTimeSteps + 1, self, modBox, min=lb, max=ub,
                                               value=fix_value)  # Eins mehr am Ende!
        self.mod.var_charge_state.activateBeforeValues(self.chargeState0_inFlowHours, True)
        self.mod.var_nettoFlow = cVariable_TS('nettoFlow', modBox.nrOfTimeSteps, self, modBox,
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
            self.eq_charge_state_start = cEquation('charge_state_start', self, modBox, eqType='eq')
            self.eq_charge_state_start.addRightSide(self.mod.var_charge_state.beforeVal())  # chargeState_0 !
            self.eq_charge_state_start.addSummand(self.mod.var_charge_state, 1, timeIndexe[0])
        elif self.chargeState0_inFlowHours == 'lastValueOfSim':
            # eq: Q_Ladezustand(1) - Q_Ladezustand(end) = 0;
            self.eq_charge_state_start = cEquation('charge_state_start', self, modBox, eqType='eq')
            self.eq_charge_state_start.addSummand(self.mod.var_charge_state, 1, timeIndexe[0])
            self.eq_charge_state_start.addSummand(self.mod.var_charge_state, -1, timeIndexe[-1])
        else:
            raise Exception('chargeState0_inFlowHours has undefined value = ' + str(chargeState0_inFlowHours))

        # Speicherleistung / Speicherladezustand / Speicherverlust
        #                                                                          | Speicher-Beladung       |   |Speicher-Entladung                |
        # Q_Ladezustand(n+1) + (-1+VerlustanteilProStunde*dt(n)) *Q_Ladezustand(n) -  dt(n)*eta_Lade*Q_th_Lade(n) +  dt(n)* 1/eta_Entlade*Q_th_Entlade(n)  = 0

        # charge_state hat ein Index mehr:
        timeIndexeChargeState = range(timeIndexe.start, timeIndexe.stop + 1)
        self.eq_charge_state = cEquation('charge_state', self, modBox, eqType='eq')
        self.eq_charge_state.addSummand(self.mod.var_charge_state,
                                        -1 * (1 - self.fracLossPerHour.d_i * modBox.dtInHours),
                                        timeIndexeChargeState[:-1])  # sprich 0 .. end-1 % nach letztem Zeitschritt gibt es noch einen weiteren Ladezustand!
        self.eq_charge_state.addSummand(self.mod.var_charge_state, 1, timeIndexeChargeState[1:])  # 1:end
        self.eq_charge_state.addSummand(self.inFlow.mod.var_val, -1 * self.eta_load.d_i * modBox.dtInHours)
        self.eq_charge_state.addSummand(self.outFlow.mod.var_val,
                                        1 / self.eta_unload.d_i * modBox.dtInHours)  # Achtung hier 1/eta!

        # Speicherladezustand am Ende
        # -> eigentlich min/max-Wert für variable, aber da nur für ein Element hier als Glg:
        # 1: eq:  Q_charge_state(end) <= Q_max
        self.eq_charge_state_end_max = cEquation('eq_charge_state_end_max', self, modBox, eqType='ineq')
        self.eq_charge_state_end_max.addSummand(self.mod.var_charge_state, 1, timeIndexeChargeState[-1])
        self.eq_charge_state_end_max.addRightSide(self.charge_state_end_max)

        # 2: eq: - Q_charge_state(end) <= - Q_min
        self.eq_charge_state_end_min = cEquation('eq_charge_state_end_min', self, modBox, eqType='ineq')
        self.eq_charge_state_end_min.addSummand(self.mod.var_charge_state, -1, timeIndexeChargeState[-1])
        self.eq_charge_state_end_min.addRightSide(- self.charge_state_end_min)

        # nettoflow:
        # eq: nettoFlow(t) - outFlow(t) + inFlow(t) = 0
        self.eq_nettoFlow = cEquation('nettoFlow', self, modBox, eqType='eq')
        self.eq_nettoFlow.addSummand(self.mod.var_nettoFlow, 1)
        self.eq_nettoFlow.addSummand(self.inFlow.mod.var_val, 1)
        self.eq_nettoFlow.addSummand(self.outFlow.mod.var_val, -1)

        if self.featureInvest is not None:
            self.featureInvest.doModeling(modBox, timeIndexe)

        # ############# Gleichungen ##########################
        # % Speicherleistung an Bilanzgrenze / Speicher-Ladung / Speicher-Entladung
        # % Q_th(n) + Q_th_Lade(n) - Q_th_Entlade(n) = 0;
        # obj.eqs.Leistungen = cEquation('Leistungen');
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
        # obj.ineqs.EntwederLadenOderEntladen = cEquation('EntwederLadenOderEntladen');
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

    new_init_args = ['label', 'source', 'sink','avoidInAndOutAtOnce']

    not_used_args = ['label']

    def __init__(self, label, source, sink, avoidInAndOutAtOnce = True, **kwargs):
        '''
        Parameters
        ----------
        label : str
            name of sourceAndSink
        source : cFlow
            output-flow of this component
        sink : cFlow
            input-flow of this component
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

    def __init__(self, label, source, **kwargs):
        '''       
        Parameters
        ----------
        label : str
            name of source
        source : cFlow
            output-flow of source
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


class cSink(cBaseComponent):
    """
    Klasse cSink
    """
    new_init_args = ['label', 'source']
    not_used_args = ['label']

    def __init__(self, label, sink, **kwargs):
        '''
        constructor of sink 

        Parameters
        ----------
        label : str
            name of sink.
        sink : cFlow
            input-flow of sink
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        super().__init__(label)
        self.sink = sink
        self.inputs.append(sink)  # ein Input-Flow

class cTransportation(cBaseComponent):
    # TODO: automatic on-Value in Flows if loss_abs
    # TODO: loss_abs must be: investment_size * loss_abs_rel!!!
    # TODO: automatic investArgs for both in-flows (or alternatively both out-flows!)
    # TODO: loss should be realized from 
    
    def __init__(self, label, in1, out1, in2=None, out2=None, loss_rel=0, loss_abs=0, isAlwaysOn=True, avoidFlowInBothDirectionsAtOnce = True, **kwargs):
        '''
        pipe with loss (when no flow, then loss is still there and has to be
        covered by one in-flow (gedanklicher Überströmer)

        Parameters
        ----------
        label : TYPE
            DESCRIPTION.
        in1 : TYPE
            DESCRIPTION.
        out1 : TYPE
            DESCRIPTION.
        in2 : TYPE
            DESCRIPTION.
        out2 : TYPE
            DESCRIPTION.
        loss_rel : TYPE
            DESCRIPTION.
        loss_abs : TYPE
            absolut loss. is active until on=0 for in-flows

        ... featureOnVars for Active Transportation:
        switchOnCosts : 
            #costs of switch rohr on
        Returns
        -------
        None.

        '''
        
        super().__init__(label)
        
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
            
            
        self.loss_rel = cTS_vector('loss_rel', loss_rel, self)#
        self.loss_abs = cTS_vector('loss_abs', loss_abs, self)#
        self.isAlwaysOn = isAlwaysOn
        self.avoidFlowInBothDirectionsAtOnce = avoidFlowInBothDirectionsAtOnce
        
        if self.avoidFlowInBothDirectionsAtOnce and (in2 is not None):
            self.featureAvoidBothDirectionsAtOnce = cFeatureAvoidFlowsAtOnce('feature_avoidBothDirectionsAtOnce', self,
                                                                 [self.in1, self.in2])



    def declareVarsAndEqs(self, modBox:cModelBoxOfES):
        """
        Deklarieren von Variablen und Gleichungen
        
        :param modBox:
        :return:
        """
        super().declareVarsAndEqs(modBox)        

    def doModeling(self, modBox, timeIndexe):
        super().doModeling(modBox, timeIndexe)


            
        # not both directions at once:
        if self.avoidFlowInBothDirectionsAtOnce and (self.in2 is not None): self.featureAvoidBothDirectionsAtOnce.doModeling(modBox, timeIndexe)


        # first direction
        # eq: in(t)*(1-loss_rel(t)) = out(t) + on(t)*loss_abs(t)
        self.eq_dir1 = cEquation('transport_dir1', self, modBox, eqType='eq')
        self.eq_dir1.addSummand(self.in1.mod.var_val, (1-self.loss_rel.d_i))
        self.eq_dir1.addSummand(self.out1.mod.var_val, -1)
        if (self.loss_abs.d_i is not None) and np.any(self.loss_abs.d_i!=0) :
            assert self.in1.mod.var_on is not None, 'Var on wird benötigt für in1! Set min_rel!'
            self.eq_dir1.addSummand(self.in1.mod.var_on, -1* self.loss_abs.d_i)

        # second direction:        
        if self.in2 is not None:
            # eq: in(t)*(1-loss_rel(t)) = out(t) + on(t)*loss_abs(t)
            self.eq_dir2 = cEquation('transport_dir2', self, modBox, eqType='eq')
            self.eq_dir2.addSummand(self.in2.mod.var_val, 1-self.loss_rel.d_i)
            self.eq_dir2.addSummand(self.out2.mod.var_val, -1)
            if (self.loss_abs.d_i is not None) and np.any(self.loss_abs.d_i!=0):            
                
                assert self.in2.mod.var_on is not None, 'Var on wird benötigt für in2! Set min_rel!'
                self.eq_dir2.addSummand(self.in2.mod.var_on, -1* self.loss_abs.d_i)
        
        # always On (in at least one direction)
        # eq: in1.on(t) +in2.on(t) >= 1 # TODO: this is some redundant to avoidFlowInBothDirections
        if self.isAlwaysOn:
            self.eq_alwaysOn = cEquation('alwaysOn',self, modBox,eqType='ineq')
            self.eq_alwaysOn.addSummand(self.in1.mod.var_on, -1)
            if (self.in2 is not None) : self.eq_alwaysOn.addSummand(self.in2.mod.var_on, -1)
            self.eq_alwaysOn.addRightSide(-.5)# wg binärungenauigkeit 0.5 statt 1
            


        # equate nominal value of second direction
        if (self.in2 is not None):
            oneInFlowHasFeatureInvest = (self.in1.featureInvest is not None) or (self.in1.featureInvest is not None)
            bothInFlowsHaveFeatureInvest = (self.in1.featureInvest is not None) and (self.in1.featureInvest is not None)
            if oneInFlowHasFeatureInvest:
                if bothInFlowsHaveFeatureInvest:
                    # eq: in1.nom_value = in2.nom_value
                    self.eq_nom_value = cEquation('equalSizeInBothDirections', self, modBox, eqType='eq')
                    self.eq_nom_value.addSummand(self.in1.featureInvest.mod.var_investmentSize, 1)            
                    self.eq_nom_value.addSummand(self.in2.featureInvest.mod.var_investmentSize, -1)
                else:
                    raise Exception('define investArgs also for second In-Flow (values can be empty!)') # TODO: anders lösen (automatisiert)!
            