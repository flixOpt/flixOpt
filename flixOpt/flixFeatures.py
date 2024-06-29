# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 18:43:55 2021
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""
## TODO:
# featureAvoidFlowsAtOnce:
# neue Variante (typ="new") austesten

from .flixStructure import *  # Grundstruktur
from .flixBasicsPublic import *
import numpy as np


##############################################################
## Funktionalität/Features zum Anhängen an die Komponenten: ##  

class cFeature(Element):

    def __init__(self, label, owner, **kwargs):
        self.owner = owner
        if not self in self.owner.sub_elements:
            self.owner.sub_elements.append(self)  # register in owner
        super().__init__(label, **kwargs)

    def __str__(self):
        return f"<{self.__class__.__name__}> {self.label}"

    @property
    def label_full(self):
        return self.owner.label_full + '_' + self.label

    def finalize(self):  # TODO: evtl. besser bei Element aufgehoben
        super().finalize()


# Abschnittsweise linear:
class cFeatureLinearSegmentVars(cFeature):
    # TODO: beser wäre hier schon Übergabe segmentsOfVars, aber schwierig, weil diese hier noch nicht vorhanden sind!
    def __init__(self, label, owner):
        super().__init__(label, owner)

    # segements separat erst jetzt definieren, damit Variablen schon erstellt sind.
    # todo: wenn Variable-Dummys existieren, dann kann das alles in __init__
    def defineSegments(self, segmentsOfVars, var_on, checkListOfVars):
        # segementsData - Elemente sind Listen!.
        # segmentsOfVars = {var_Q_fu: [ 5  , 10,  10, 22], # je zwei Werte bilden ein Segment. Indexspezfika (z.B. für Zeitreihenabbildung) über arrays oder TS!!
        #                   var_P_el: [ 2  , 5,    5, 8 ],
        #                   var_Q_th: [ 2.5, 4,    4, 12]}

        # -> onVariable ist optional
        # -> auch einzelne zulässige Punkte können über Segment ausgedrückt werden, z.B. [5, 5]

        self.segmentsOfVars = segmentsOfVars
        self.var_on = var_on

        # Anzahl Segmente bestimmen:      
        segmentDataOfFirstVariable = next(iter(segmentsOfVars.values()))
        nrOfColumns = len(segmentDataOfFirstVariable)  # Anzahl Spalten
        self.nrOfSegments = nrOfColumns / 2  # Anzahl der Spalten der Matrix / 2 = Anzahl der Segmente

        # Check ob gerade Anzahl an Werten:
        if not self.nrOfSegments.is_integer():
            raise Exception('Nr of Values should be even, because pairs (start,end of every section)')

        # Check, ob alle Variables vorhanden:
        if checkListOfVars is not None:
            setOfVars = set(segmentsOfVars.keys())
            toMuchSet = setOfVars - set(checkListOfVars)
            missingSet = set(checkListOfVars) - setOfVars

            # Wenn Unterschiede vorhanden:
            def getStrOfSet(aSet):
                aStr = []
                for aVar in aSet:
                    aStr += ',' + aVar.label_full
                return aStr

            # überflüssige Flows:
            if toMuchSet:
                raise Exception('segmentsOfVars-Definition has not necessary vars: ' + getStrOfSet(toMuchSet))
            # fehlende Flows:
            if missingSet:
                raise Exception('segmentsOfVars miss following vars: ' + getStrOfSet(missingSet))
        # Aufteilen der Daten in die Segmente:
        self.listOfSegments = []
        for aSecNr in range(int(self.nrOfSegments)):
            # samplePoints für das Segment extrahieren:
            # z.B.   {var1:[TS1.1, TS1.2]
            #         var2:[TS2.1, TS2.2]}            
            samplePointsOfSegment = cFeatureLinearSegmentVars.__extractSamplePoints4Segment(segmentsOfVars, aSecNr)
            # Segment erstellen und in Liste::
            newSegment = cSegment('seg_' + str(aSecNr), self, samplePointsOfSegment, aSecNr)
            # todo: hier muss activate() selbst gesetzt werden, weil bereits gesetzt 
            # todo: alle Elemente sollten eigentlich hier schon längst instanziert sein und werden dann auch activated!!!
            newSegment.create_new_model_and_activate_system_model(self.system_model)
            self.listOfSegments.append(newSegment)

    def declare_vars_and_eqs(self, system_model: SystemModel):
        for aSegment in self.listOfSegments:
            # Segmentvariablen erstellen:
            aSegment.declare_vars_and_eqs(system_model)

    def do_modeling(self, system_model: SystemModel, timeIndexe):
        #########################################
        ## 1. Gleichungen für: Nur ein Segment kann aktiv sein! ##
        # eq: -On(t) + Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 0 
        # -> Wenn Variable On(t) nicht existiert, dann nur 
        # eq:          Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 1                                         

        self.eq_IcanOnlyBeInOneSegment = Equation('ICanOnlyBeInOneSegment', self, system_model)

        # a) zusätzlich zu Aufenthalt in Segmenten kann alles auch Null sein:
        if (self.var_on is not None) and (
                self.var_on is not None):  # Eigentlich wird die On-Variable durch linearSegment-equations bereits vollständig definiert.
            self.eq_IcanOnlyBeInOneSegment.add_summand(self.var_on, -1);
        # b) Aufenthalt nur in Segmenten erlaubt:
        else:
            self.eq_IcanOnlyBeInOneSegment.add_constant(1);  #

        for aSegment in self.listOfSegments:
            self.eq_IcanOnlyBeInOneSegment.add_summand(aSegment.model.var_onSeg, 1);

            #################################
        ## 2. Gleichungen der Segmente ##
        # eq: -aSegment.onSeg(t) + aSegment.lambda1(t) + aSegment.lambda2(t)  = 0    
        for aSegment in self.listOfSegments:
            aNameOfEq = 'Lambda_onSeg_' + str(aSegment.index)

            eq_Lambda_onSeg = Equation(aNameOfEq, self, system_model)
            eq_Lambda_onSeg.add_summand(aSegment.model.var_onSeg, -1);
            eq_Lambda_onSeg.add_summand(aSegment.model.var_lambda1, 1);
            eq_Lambda_onSeg.add_summand(aSegment.model.var_lambda2, 1);

            ##################################################
        ## 3. Gleichungen für die Variablen mit lambda: ##
        #   z.B. Gleichungen für Q_th mit lambda
        #  eq: - Q_th(t) + sum(Q_th_1_j * lambda_1_j + Q_th_2_j * lambda_2_j) = 0
        #  mit -> j                   = Segmentnummer 
        #      -> Q_th_1_j, Q_th_2_j  = Stützstellen des Segments (können auch Vektor sein)

        for aVar in self.segmentsOfVars.keys():
            # aVar = aFlow.model.var_val
            eqLambda = Equation(aVar.label + '_lambda', self, system_model)  # z.B. Q_th(t)
            eqLambda.add_summand(aVar, -1)
            for aSegment in self.listOfSegments:
                #  Stützstellen einfügen:
                stuetz1 = aSegment.samplePoints[aVar][0]
                stuetz2 = aSegment.samplePoints[aVar][1]
                # wenn Stützstellen TS_vector:
                if isinstance(stuetz1, TimeSeries):
                    samplePoint1 = stuetz1.active_data
                    samplePoint2 = stuetz2.active_data
                # wenn Stützstellen Skalar oder array
                else:
                    samplePoint1 = stuetz1
                    samplePoint2 = stuetz2

                eqLambda.add_summand(aSegment.model.var_lambda1,
                                     samplePoint1)  # Spalte 1 (Faktor kann hier Skalar sein oder Vektor)
                eqLambda.add_summand(aSegment.model.var_lambda2,
                                     samplePoint2)  # Spalte 2 (Faktor kann hier Skalar sein oder Vektor)

    # extract the 2 TS_vectors for the segment:
    def __extractSamplePoints4Segment(samplePointsOfAllSegments, nrOfSegment):
        samplePoints4Segment = {}
        # für alle Variablen Segment-Stützstellen holen:
        aSpalteOfSecStart = (nrOfSegment) * 2  # 0, 2, 4
        for aVar in samplePointsOfAllSegments.keys():
            # 1. und 2. Stützstellen des Segments auswählen:
            samplePoints4Segment[aVar] = samplePointsOfAllSegments[aVar][aSpalteOfSecStart: aSpalteOfSecStart + 2]
        return samplePoints4Segment


class cFeatureLinearSegmentSet(cFeatureLinearSegmentVars):
    # TODO: beser wäre segmentsOfVars, aber schwierig, weil diese hier noch nicht vorhanden sind!
    def __init__(self, label, owner, segmentsOfFlows_TS, get_var_on=None, checkListOfFlows=None):
        # segementsData - Elemente sind Listen!.
        # segmentsOfFlows = {Q_fu: [ 5  , 10,  10, 22], # je zwei Werte bilden ein Segment. Zeitreihenabbildung über arrays!!!
        #                    P_el: [ 2  , 5,    5, 8 ],
        #                    Q_th: [ 2.5, 4,    4, 12]}
        # -> auch einzelne zulässige Punkte können über Segment ausgedrückt werden, z.B. [5, 5]

        self.segmentsOfFlows_TS = segmentsOfFlows_TS
        self.get_var_on = get_var_on
        self.checkListOfFlows = checkListOfFlows
        super().__init__(label, owner)

    def declare_vars_and_eqs(self, system_model):
        # 1. Variablen-Segmente definieren:
        segmentsOfVars = {}
        for flow in self.segmentsOfFlows_TS:
            segmentsOfVars[flow.model.var_val] = self.segmentsOfFlows_TS[flow]

        checkListOfVars = []
        for flow in self.checkListOfFlows:
            checkListOfVars.append(flow.model.var_val)

        # hier erst Variablen vorhanden un damit segmentsOfVars definierbar!
        super().defineSegments(segmentsOfVars, var_on=self.get_var_on(),
                               checkListOfVars=checkListOfVars)  # todo: das ist nur hier, damit schon variablen Bekannt

        # 2. declare vars:      
        super().declare_vars_and_eqs(system_model)


# Abschnittsweise linear, 1 Abschnitt:
class cSegment(cFeature):
    def __init__(self, label, owner, samplePoints, index):
        super().__init__(label, owner)

        self.label = label
        self.samplePoints = samplePoints
        self.index = index

    def declare_vars_and_eqs(self, system_model):
        aLen = system_model.nrOfTimeSteps
        self.model.var_onSeg = VariableTS('onSeg_' + str(self.index), aLen, self, system_model,
                                          is_binary=True)  # Binär-Variable
        self.model.var_lambda1 = VariableTS('lambda1_' + str(self.index), aLen, self, system_model, lower_bound=0,
                                            upper_bound=1)  # Wertebereich 0..1
        self.model.var_lambda2 = VariableTS('lambda2_' + str(self.index), aLen, self, system_model, lower_bound=0,
                                            upper_bound=1)  # Wertebereich 0..1


# Verhindern gleichzeitig mehrere Flows > 0 
class cFeatureAvoidFlowsAtOnce(cFeature):

    def __init__(self, label, owner, flows, typ='classic'):
        super().__init__(label, owner)
        self.flows = flows
        self.typ = typ
        assert len(self.flows) >= 2, 'Beachte für Feature AvoidFlowsAtOnce: Mindestens 2 Flows notwendig'

    def finalize(self):
        super().finalize
        # Beachte: Hiervor muss featureOn in den Flows existieren!
        aFlow: Flow

        if self.typ == 'classic':
            # "classic" -> alle Flows brauchen Binärvariable:
            for aFlow in self.flows:
                aFlow.activateOnValue()

        elif self.typ == 'new':
            # "new" -> n-1 Flows brauchen Binärvariable: (eine wird eingespart)

            # 1. Get nr of existing on_vars in Flows
            self.nrOfExistingOn_vars = 0
            for aFlow in self.flows:
                self.nrOfExistingOn_vars += aFlow.featureOn.useOn

            # 2. Add necessary further flow binaries:
            # Anzahl on_vars solange erhöhen bis mindestens n-1 vorhanden:
            i = 0
            while self.nrOfExistingOn_vars < (len(self.flows) - 1):
                aFlow = flows[i]
                # Falls noch nicht on-Var für flow existiert, dann erzwingen:
                if not aFlow.featureOn.useOn:
                    aFlow.activateOnValue()
                    self.nrOfExistingOn_vars += 1
                i += 1

    def do_modeling(self, system_model, timeIndexe):
        # Nur 1 Flow aktiv! Nicht mehrere Zeitgleich!    
        # "classic":
        # eq: sum(flow_i.on(t)) <= 1.1 (1 wird etwas größer gewählt wg. Binärvariablengenauigkeit)
        # "new": 
        # eq: flow_1.on(t) + flow_2.on(t) + .. + flow_i.val(t)/flow_i.max <= 1 (1 Flow ohne Binärvariable!)

        # Anmerkung: Patrick Schönfeld (oemof, custom/link.py) macht bei 2 Flows ohne Binärvariable dies:
        # 1)	bin + flow1/flow1_max <= 1
        # 2)	bin - flow2/flow2_max >= 0
        # 3)    geht nur, wenn alle flow.min >= 0
        # --> könnte man auch umsetzen (statt activateOnValue() für die Flows, aber sollte aufs selbe wie "new" kommen)

        self.eq_flowLock = Equation('flowLock', self, system_model, eqType='ineq')
        # Summanden hinzufügen:
        for aFlow in self.flows:
            # + flow_i.on(t):
            if aFlow.model.var_on is not None:
                self.eq_flowLock.add_summand(aFlow.model.var_on, 1)
                # + flow_i.val(t)/flow_i.max
            else:  # nur bei "new"
                assert aFlow.min >= 0, 'cFeatureAvoidFlowsAtOnce(): typ "new" geht nur für Flows mit min >= 0!'
                self.eq_flowLock.add_summand(aFlow.model.var_val, 1 / aFlow.max)

        if self.typ == 'classic':
            self.eq_flowLock.add_constant(
                1.1)  # sicherheitshalber etwas mehr, damit auch leicht größer Binärvariablen 1.00001 funktionieren.
        elif typ == 'new':
            self.eq_flowLock.add_constant(1)  # TODO: hier ggf. Problem bei großen Binärungenauigkeit!!!!


## Klasse, die in Komponenten UND Flows benötigt wird: ##
class cFeatureOn(cFeature):
    # def __init__(self, featureOwner, nameOfVariable, useOn, useSwitchOn):  
    # #   # on definierende Variablen:
    # #   self.featureOwner = featureOwner
    # #   self.nameOfVariable = nameOfVariable
    # #   self.flows  = flows
    # #   self.model.var_on = None
    def __init__(self, owner, flowsDefiningOn, on_valuesBeforeBegin,
                 switchOnCosts, costsPerRunningHour,
                 onHoursSum_min=None, onHoursSum_max=None,
                 onHours_min=None, onHours_max=None,
                 offHours_min=None, offHours_max=None,
                 switchOn_maxNr=None,
                 useOn_explicit=False,
                 useSwitchOn_explicit=False):
        super().__init__('featureOn', owner)
        self.flowsDefiningOn = flowsDefiningOn
        self.on_valuesBeforeBegin = on_valuesBeforeBegin
        self.switchOnCosts = switchOnCosts
        self.costsPerRunningHour = costsPerRunningHour
        self.onHoursSum_min = onHoursSum_min  # scalar
        self.onHoursSum_max = onHoursSum_max  # scalar
        self.onHours_min = onHours_min  # TimeSeries
        self.onHours_max = onHours_max  # TimeSeries
        self.offHours_min = offHours_min  # TimeSeries
        self.offHours_max = offHours_max  # TimeSeries
        self.switchOn_maxNr = switchOn_maxNr
        # default:
        self.useOn = False
        self.useOff = False
        self.useSwitchOn = False
        self.useOnHours = False
        self.useOffHours = False

        # Notwendige Variablen entsprechend der übergebenen Parameter:        
        paramsForcingOn = [costsPerRunningHour, onHoursSum_min, onHoursSum_max,
                           onHours_min, onHours_max, offHours_min, offHours_max]
        if any(param is not None for param in paramsForcingOn):
            self.useOn = True

        paramsForcingSwitchOn = [switchOnCosts, switchOn_maxNr, onHoursSum_max, onHoursSum_min]
        if any(param is not None for param in paramsForcingSwitchOn):
            self.useOn = True
            self.useSwitchOn = True

        paramsForcingOnHours = [self.onHours_min, self.onHours_max]  # onHoursSum alway realized
        if any(param is not None for param in paramsForcingOnHours):
            self.useOnHours = True

        paramsForcingOffHours = [self.offHours_min, self.offHours_max]  # offHoursSum alway realized
        if any(param is not None for param in paramsForcingOffHours):
            self.useOffHours = True

        self.useOn = self.useOn | useOn_explicit | self.useOnHours | self.useOffHours
        self.useOff = self.useOffHours
        self.useSwitchOn = self.useSwitchOn | useSwitchOn_explicit

    # Befehl von außen zum Erzwingen einer On-Variable:
    def activateOnValueExplicitly(self):
        self.useOn = True

    # varOwner braucht die Variable auch:
    def getVar_on(self):
        return self.model.var_on

    def getVars_switchOnOff(self):
        return self.model.var_switchOn, self.model.var_switchOff

    #   # Variable wird erstellt und auch gleich in featureOwner registiert:
    #      
    #   # ## Variable als Attribut in featureOwner übergeben:
    #   # # TODO: ist das so schick? oder sollte man nicht so versteckt Attribute setzen?
    #   # # Check:
    #   # if (hasattr(self.featureOwner, 'var_on')) and (self.featureOwner.model.var_on == None) :
    #   #   self.featureOwner.model.var_on = self.model.var_on
    #   # else :
    #   #   raise Exception('featureOwner ' + self.featureOwner.label + ' has no attribute var_on or it is already used')

    def declare_vars_and_eqs(self, system_model):
        # Beachte: Variablen gehören nicht diesem Element, sondern varOwner (meist ist das der featureOwner)!!!  
        # Var On:
        if self.useOn:
            # Before-Variable:
            self.model.var_on = VariableTS('on', system_model.nrOfTimeSteps, self.owner, system_model, is_binary=True)
            self.model.var_on.set_before_value(default_before_value=self.on_valuesBeforeBegin[0],
                                               is_start_value=False)
            self.model.var_onHoursSum = Variable('onHoursSum', 1, self.owner, system_model, lower_bound=self.onHoursSum_min,
                                                 upper_bound=self.onHoursSum_max)  # wenn max/min = None, dann bleibt das frei

        else:
            self.model.var_on = None
            self.model.var_onHoursSum = None

        if self.useOff:
            # off-Var is needed:
            self.model.var_off = VariableTS('off', system_model.nrOfTimeSteps, self.owner, system_model, is_binary=True)

        # onHours:
        #   i.g. 
        #   var_on      = [0 0 1 1 1 1 0 0 0 1 1 1 0 ...]
        #   var_onHours = [0 0 1 2 3 4 0 0 0 1 2 3 0 ...] (bei dt=1)
        if self.useOnHours:
            aMax = None if (self.onHours_max is None) else self.onHours_max.active_data
            self.model.var_onHours = VariableTS('onHours', system_model.nrOfTimeSteps, self.owner, system_model,
                                                lower_bound=0, upper_bound=aMax)  # min separat
        # offHours:
        if self.useOffHours:
            aMax = None if (self.offHours_max is None) else self.offHours_max.active_data
            self.model.var_offHours = VariableTS('offHours', system_model.nrOfTimeSteps, self.owner, system_model,
                                                 lower_bound=0, upper_bound=aMax)  # min separat

        # Var SwitchOn
        if self.useSwitchOn:
            self.model.var_switchOn = VariableTS('switchOn', system_model.nrOfTimeSteps, self.owner, system_model, is_binary=True)
            self.model.var_switchOff = VariableTS('switchOff', system_model.nrOfTimeSteps, self.owner, system_model, is_binary=True)
            self.model.var_nrSwitchOn = Variable('nrSwitchOn', 1, self.owner, system_model,
                                                 upper_bound=self.switchOn_maxNr)  # wenn max/min = None, dann bleibt das frei
        else:
            self.model.var_switchOn = None
            self.model.var_switchOff = None
            self.model.var_nrSwitchOn = None

    def do_modeling(self, system_model, timeIndexe):
        eqsOwner = self
        if self.useOn:
            self.__addConstraintsForOn(eqsOwner, self.flowsDefiningOn, system_model, timeIndexe)
        if self.useOff:
            self.__addConstraintsForOff(eqsOwner, system_model, timeIndexe)
        if self.useSwitchOn:
            self.__addConstraintsForSwitchOnSwitchOff(eqsOwner, system_model, timeIndexe)
        if self.useOnHours:
            cFeatureOn.__addConstraintsForOnTimeOfBinary(
                self.model.var_onHours, self.model.var_on, self.onHours_min,
                eqsOwner, system_model, timeIndexe)
        if self.useOffHours:
            cFeatureOn.__addConstraintsForOnTimeOfBinary(
                self.model.var_offHours, self.model.var_off, self.offHours_min,
                eqsOwner, system_model, timeIndexe)

    def __addConstraintsForOn(self, eqsOwner, flowsDefiningOn, system_model, timeIndexe):
        # % Bedingungen 1) und 2) müssen erfüllt sein:

        # % Anmerkung: Falls "abschnittsweise linear" gewählt, dann ist eigentlich nur Bedingung 1) noch notwendig 
        # %            (und dann auch nur wenn erstes Segment bei Q_th=0 beginnt. Dann soll bei Q_th=0 (d.h. die Maschine ist Aus) On = 0 und segment1.onSeg = 0):)
        # %            Fazit: Wenn kein Performance-Verlust durch mehr Gleichungen, dann egal!      

        nrOfFlows = len(flowsDefiningOn)
        assert nrOfFlows > 0, 'Achtung: mindestens 1 Flow notwendig'
        #######################################################################
        #### Bedingung 1) ####

        # Glg. wird erstellt und auch gleich in featureOwner registiert:
        eq1 = Equation('On_Constraint_1', eqsOwner, system_model, eqType='ineq')
        # TODO: eventuell label besser über  nameOfIneq = [aOnVariable.name '_Constraint_1']; % z.B. On_Constraint_1

        # Wenn nur 1 Leistungsvariable (!Unterscheidet sich von >1 Leistungsvariablen wg. Minimum-Beachtung!):
        if nrOfFlows == 1:
            ## Leistung<=MinLeistung -> On = 0 | On=1 -> Leistung>MinLeistung
            # eq: Q_th(t) - max(Epsilon, Q_th_min) * On(t) >= 0  (mit Epsilon = sehr kleine Zahl, wird nur im Falle Q_th_min = 0 gebraucht)
            # gleichbedeutend mit eq: -Q_th(t) + max(Epsilon, Q_th_min)* On(t) <= 0 
            aFlow = flowsDefiningOn[0]
            eq1.add_summand(aFlow.model.var_val, -1, timeIndexe)
            # wenn variabler Nennwert:
            if aFlow.nominal_val is None:
                min_val = aFlow.invest_parameters.minimum_size * aFlow.min_rel.active_data  # kleinst-Möglichen Wert nutzen. (Immer noch math. günstiger als Epsilon)
            # wenn fixer Nennwert
            else:
                min_val = aFlow.nominal_val * aFlow.min_rel.active_data

            eq1.add_summand(self.model.var_on, 1 * np.maximum(system_model.epsilon, min_val),
                            timeIndexe)  # % aLeistungsVariableMin kann hier Skalar oder Zeitreihe sein!

        # Bei mehreren Leistungsvariablen:
        else:

            # Nur wenn alle Flows = 0, dann ist On = 0
            ## 1) sum(alle Leistung)=0 -> On = 0 | On=1 -> sum(alle Leistungen) > 0
            # eq: - sum(alle Leistungen(t)) + Epsilon * On(t) <= 0
            for aFlow in flowsDefiningOn:
                eq1.add_summand(aFlow.model.var_val, -1, timeIndexe)
            eq1.add_summand(self.model.var_on, 1 * system_model.epsilon,
                            timeIndexe)  # % aLeistungsVariableMin kann hier Skalar oder Zeitreihe sein!

        #######################################################################
        #### Bedingung 2) ####

        # Glg. wird erstellt und auch gleich in featureOwner registiert:
        eq2 = Equation('On_Constraint_2', eqsOwner, system_model, eqType='ineq')
        # Wenn nur 1 Leistungsvariable:
        #  eq: Q_th(t) <= Q_th_max * On(t)
        # (Leistung>0 -> On = 1 | On=0 -> Leistung<=0)
        # Bei mehreren Leistungsvariablen:
        ## sum(alle Leistung) >0 -> On = 1 | On=0 -> sum(Leistung)=0
        #  eq: sum( Leistung(t,i))              - sum(Leistung_max(i))             * On(t) <= 0
        #  --> damit Gleichungswerte nicht zu groß werden, noch durch nrOfFlows geteilt:
        #  eq: sum( Leistung(t,i) / nrOfFlows ) - sum(Leistung_max(i)) / nrOfFlows * On(t) <= 0
        sumOfFlowMax = 0
        for aFlow in flowsDefiningOn:
            eq2.add_summand(aFlow.model.var_val, 1 / nrOfFlows, timeIndexe)
            # wenn variabler Nennwert:
            if aFlow.nominal_val is None:
                sumOfFlowMax += aFlow.max_rel.active_data * aFlow.invest_parameters.maximum_size  # der maximale Nennwert reicht als Obergrenze hier aus. (immer noch math. günster als BigM)
            else:
                sumOfFlowMax += aFlow.max_rel.active_data * aFlow.nominal_val

        eq2.add_summand(self.model.var_on, - sumOfFlowMax / nrOfFlows, timeIndexe)  #

        if isinstance(sumOfFlowMax, (np.ndarray, list)):
            if max(sumOfFlowMax) / nrOfFlows > 1000: log.warning(
                '!!! ACHTUNG in ' + self.owner.label_full + ' : Binärdefinition mit großem Max-Wert (' + str(
                    int(max(sumOfFlowMax) / nrOfFlows)) + '). Ggf. falsche Ergebnisse !!!')
        else:
            if sumOfFlowMax / nrOfFlows > 1000: log.warning(
                '!!! ACHTUNG in ' + self.owner.label_full + ' : Binärdefinition mit großem Max-Wert (' + str(
                    int(sumOfFlowMax / nrOfFlows)) + '). Ggf. falsche Ergebnisse !!!')

    def __addConstraintsForOff(self, eqsOwner, system_model, timeIndexe):
        # Definition var_off:
        # eq: var_off(t) = 1-var_on(t)
        eq_var_off = Equation('var_off', self, system_model, eqType='eq')
        eq_var_off.add_summand(self.model.var_off, 1)
        eq_var_off.add_summand(self.model.var_on, 1)
        eq_var_off.add_constant(1)

    @staticmethod  # to be sure not using any self-Variables
    def __addConstraintsForOnTimeOfBinary(var_bin_onTime, var_bin, onHours_min, eqsOwner, system_model, timeIndexe):
        '''
        i.g. 
        var_bin        = [0 0 1 1 1 1 0 1 1 1 0 ...]
        var_bin_onTime = [0 0 1 2 3 4 0 1 2 3 0 ...] (bei dt=1)
                                            |-> min_onHours = 3!
        
        if you wanna count zeros, define var_bin_off: = 1-var_bin before!
        '''
        # TODO: Einfachere Variante von Peter umsetzen!

        # 1) eq: onHours(t) <= On(t)*Big | On(t)=0 -> onHours(t) = 0
        # mit Big = dtInHours_tot
        aLabel = var_bin_onTime.label
        ineq_1 = Equation(aLabel + '_constraint_1', eqsOwner, system_model, eqType='ineq')
        ineq_1.add_summand(var_bin_onTime, 1)
        ineq_1.add_summand(var_bin, -1 * system_model.dtInHours_tot)

        # 2a) eq: onHours(t) - onHours(t-1) <= dt(t)
        #    on(t)=1 -> ...<= dt(t)
        #    on(t)=0 -> onHours(t-1)>=
        ineq_2a = Equation(aLabel + '_constraint_2a', eqsOwner, system_model, eqType='ineq')
        ineq_2a.add_summand(var_bin_onTime, 1, timeIndexe[1:])  # onHours(t)
        ineq_2a.add_summand(var_bin_onTime, -1, timeIndexe[0:-1])  # onHours(t-1)
        ineq_2a.add_constant(system_model.dtInHours[1:])  # dt(t)

        # 2b) eq:  onHours(t) - onHours(t-1)             >=  dt(t) - Big*(1-On(t)))
        #    eq: -onHours(t) + onHours(t-1) + On(t)*Big <= -dt(t) + Big
        # with Big = dtInHours_tot # (Big = maxOnHours, should be usable, too!)
        Big = system_model.dtInHours_tot
        ineq_2b = Equation(aLabel + '_constraint_2b', eqsOwner, system_model, eqType='ineq')
        ineq_2b.add_summand(var_bin_onTime, -1, timeIndexe[1:])  # onHours(t)
        ineq_2b.add_summand(var_bin_onTime, 1, timeIndexe[0:-1])  # onHours(t-1)
        ineq_2b.add_summand(var_bin, Big, timeIndexe[1:])  # on(t)
        ineq_2b.add_constant(-system_model.dtInHours[1:] + Big)  # dt(t)

        # 3) check onHours_min before switchOff-step
        # (last on-time period of timeseries is not checked and can be shorter)
        if onHours_min is not None:
            # Note: switchOff-step is when: On(t)-On(t-1) == -1
            # eq:  onHours(t-1) >= minOnHours * -1 * [On(t)-On(t-1)]
            # eq: -onHours(t-1) - onHours_min * On(t) + onHours_min*On(t-1) <= 0
            ineq_min = Equation(aLabel + '_min', eqsOwner, system_model, eqType='ineq')
            ineq_min.add_summand(var_bin_onTime, -1, timeIndexe[0:-1])  # onHours(t-1)
            ineq_min.add_summand(var_bin, -1 * onHours_min.active_data, timeIndexe[1:])  # on(t)
            ineq_min.add_summand(var_bin, onHours_min.active_data, timeIndexe[0:-1])  # on(t-1)

        # 4) first index:
        #    eq: onHours(t=0)= dt(0) * On(0)
        firstIndex = timeIndexe[0]  # only first element
        eq_first = Equation(aLabel + '_firstTimeStep', eqsOwner, system_model)
        eq_first.add_summand(var_bin_onTime, 1, firstIndex)
        eq_first.add_summand(var_bin, -1 * system_model.dtInHours[firstIndex], firstIndex)

    def __addConstraintsForSwitchOnSwitchOff(self, eqsOwner, system_model, timeIndexe):
        # % Schaltänderung aus On-Variable
        # % SwitchOn(t)-SwitchOff(t) = On(t)-On(t-1) 

        eq_SwitchOnOff_andOn = Equation('SwitchOnOff_andOn', eqsOwner, system_model)
        eq_SwitchOnOff_andOn.add_summand(self.model.var_switchOn, 1, timeIndexe[1:])  # SwitchOn(t)
        eq_SwitchOnOff_andOn.add_summand(self.model.var_switchOff, -1, timeIndexe[1:])  # SwitchOff(t)
        eq_SwitchOnOff_andOn.add_summand(self.model.var_on, -1, timeIndexe[1:])  # On(t)
        eq_SwitchOnOff_andOn.add_summand(self.model.var_on, +1, timeIndexe[0:-1])  # On(t-1)

        ## Ersten Wert SwitchOn(t=1) bzw. SwitchOff(t=1) festlegen
        # eq: SwitchOn(t=1)-SwitchOff(t=1) = On(t=1)- ValueBeforeBeginOfTimeSeries;      

        eq_SwitchOnOffAtFirstTime = Equation('SwitchOnOffAtFirstTime', eqsOwner, system_model)
        firstIndex = timeIndexe[0]  # nur erstes Element!
        eq_SwitchOnOffAtFirstTime.add_summand(self.model.var_switchOn, 1, firstIndex)
        eq_SwitchOnOffAtFirstTime.add_summand(self.model.var_switchOff, -1, firstIndex)
        eq_SwitchOnOffAtFirstTime.add_summand(self.model.var_on, -1, firstIndex)
        # eq_SwitchOnOffAtFirstTime.add_constant(-on_valuesBefore[-1]) # letztes Element der Before-Werte nutzen,  Anmerkung: wäre besser auf lhs aufgehoben
        eq_SwitchOnOffAtFirstTime.add_constant(
            -self.model.var_on.before_value)  # letztes Element der Before-Werte nutzen,  Anmerkung: wäre besser auf lhs aufgehoben

        ## Entweder SwitchOff oder SwitchOn
        # eq: SwitchOn(t) + SwitchOff(t) <= 1 

        ineq = Equation('SwitchOnOrSwitchOff', eqsOwner, system_model, eqType='ineq')
        ineq.add_summand(self.model.var_switchOn, 1)
        ineq.add_summand(self.model.var_switchOff, 1)
        ineq.add_constant(1)

        ## Anzahl Starts:
        # eq: nrSwitchOn = sum(SwitchOn(t))  

        eq_NrSwitchOn = Equation('NrSwitchOn', eqsOwner, system_model)
        eq_NrSwitchOn.add_summand(self.model.var_nrSwitchOn, 1)
        eq_NrSwitchOn.add_summand(self.model.var_switchOn, -1, as_sum=True)

    def add_share_to_globals(self, globalComp, system_model):

        shareHolder = self.owner
        # Anfahrkosten:
        if self.switchOnCosts is not None:  # and any(self.switch_on_effects.active_data != 0):
            globalComp.addShareToOperation('switch_on_effects', shareHolder, self.model.var_switchOn, self.switchOnCosts, 1)
        # Betriebskosten:
        if self.costsPerRunningHour is not None:  # and any(self.effects_per_running_hour):
            globalComp.addShareToOperation('effects_per_running_hour', shareHolder, self.model.var_on,
                                           self.costsPerRunningHour, system_model.dtInHours)
            # globalComp.costsOfOperating_eq.add_summand(self.model.var_on, np.multiply(self.effects_per_running_hour.active_data, model.dtInHours))# np.multiply = elementweise Multiplikation


# TODO: als cFeature_TSShareSum
class cFeature_ShareSum(cFeature):

    def __init__(self, label, owner, sharesAreTS, maxOfSum=None, minOfSum=None, max_per_hour = None, min_per_hour = None):
        '''
        sharesAreTS = True : 
          Output: 
            var_all (TS), var_sum
          variables:
            sum_TS (Zeitreihe)
            sum    (Skalar)
          Equations: 
            eq_sum_TS : sum_TS = sum(share_TS_i) # Zeitserie
            eq_sum    : sum    = sum(sum_TS(t)) # skalar

        # sharesAreTS = False: 
        #   Output: 
        #     var_sum
        #   Equations:
        #     eq_sum   : sum     = sum(share_i) # skalar

        Parameters
        ----------
        label : TYPE
            DESCRIPTION.
        owner : TYPE
            DESCRIPTION.
        sharesAreTS : TYPE
            DESCRIPTION.
        maxOfSum : TYPE, optional
            DESCRIPTION. The default is None.
        minOfSum : TYPE, optional
            DESCRIPTION. The default is None.
        max_per_hour : scalar or list(TS) (if sharesAreTS=True)
            maximum value per hour of shareSum;
            only usable if sharesAreTS=True         
        min_per_hour : scalar or list(TS) (if sharesAreTS=True)
            minimum value per hour of shareSum of each timestep    
            only usable if sharesAreTS=True 

        '''
        super().__init__(label, owner)
        self.sharesAreTS = sharesAreTS
        self.maxOfSum = maxOfSum
        self.minOfSum = minOfSum
        max_min_per_hour_is_not_None = (max_per_hour is not None) or (min_per_hour is not None)
        if (not self.sharesAreTS) and max_min_per_hour_is_not_None:
            raise Exception('max_per_hour or min_per_hour can only be used, if sharesAreTS==True!')
        self.max_per_hour = None if (max_per_hour is None) else TimeSeries('max_per_hour', max_per_hour, self)
        self.min_per_hour = None if (min_per_hour is None) else TimeSeries('min_per_hour', min_per_hour, self)
        

        self.shares = cFeatureShares('shares', self)
        # self.effectType = effectType    

    # def setProperties(self, min = 0, max = nan)

    def declare_vars_and_eqs(self, system_model):
        super().declare_vars_and_eqs(system_model)
        self.shares.declare_vars_and_eqs(system_model)

        # TODO: summe auch über Bus abbildbar!
        #   -> aber Beachte Effekt ist nicht einfach gleichzusetzen mit Flow, da hier eine Menge z.b. in € im Zeitschritt übergeben wird
        # variable für alle TS zusammen (TS-Summe):
        if self.sharesAreTS:
            lb_TS = None if (self.min_per_hour is None) else np.multiply(self.min_per_hour.active_data, system_model.dtInHours)
            ub_TS = None if (self.max_per_hour is None) else np.multiply(self.max_per_hour.active_data, system_model.dtInHours)
            self.model.var_sum_TS = VariableTS('sum_TS', system_model.nrOfTimeSteps, self, system_model, lower_bound= lb_TS, upper_bound= ub_TS)  # TS

        # Variable für Summe (Skalar-Summe):
        self.model.var_sum = Variable('sum', 1, self, system_model, lower_bound=self.minOfSum, upper_bound=self.maxOfSum)  # Skalar

        # Gleichungen schon hier definiert, damit andere Elemente beim modeling Beiträge eintragen können:
        if self.sharesAreTS:
            self.eq_sum_TS = Equation('bilanz', self, system_model)
        self.eq_sum = Equation('sum', self, system_model)

    def do_modeling(self, system_model, timeIndexe):
        self.shares.do_modeling(system_model, timeIndexe)
        if self.sharesAreTS:
            # eq: sum_TS = sum(share_TS_i) # TS
            self.eq_sum_TS.add_summand(self.model.var_sum_TS, -1)
            # eq: sum = sum(sum_TS(t)) # skalar
            self.eq_sum.add_summand(self.model.var_sum_TS, 1, as_sum=True)
            self.eq_sum.add_summand(self.model.var_sum, -1)
        else:
            # eq: sum = sum(share_i) # skalar
            self.eq_sum.add_summand(self.model.var_sum, -1)

    def addConstantShare(self, nameOfShare, shareHolder, factor1, factor2):
        '''
        Beiträge zu Effekt_Sum registrieren

        Parameters
        ----------
        factor1 : TS oder skalar, bei sharesAreTS=False nur skalar
            DESCRIPTION.
        factor2 : TS oder skalar, bei sharesAreTS=False nur skalar
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.addShare(self, nameOfShare, shareHolder, None, factor1, factor2)

    def addVariableShare(self, nameOfShare, shareHolder, variable, factor1,
                         factor2):  # if variable = None, then fix Share
        if variable is None: raise Exception(
            'addVariableShare() needs variable as input or use addConstantShare() instead')
        self.addShare(nameOfShare, shareHolder, variable, factor1, factor2)

    # allgemein variable oder constant (dann variable = None):
    # if variable = None, then fix Share    
    def addShare(self, nameOfShare, shareHolder, variable, factor1, factor2):
        '''
        share to a sum

        Parameters
        ----------
        variable : TYPE
            DESCRIPTION.
        factor1 : TYPE
            DESCRIPTION.
        factor2 : TYPE
            DESCRIPTION.
        nameOfShare : str or None
            None, if it is not a real share (i.g. -1*var_sum )

        Returns
        -------
        None.

        '''
        # var and eq for publishing share-values in results:                    
        if nameOfShare is not None:
            eq_oneShare = self.shares.get_eqOfNewShare(nameOfShare, shareHolder, self.system_model)

        if self.sharesAreTS:

            # Falls TimeSeries, Daten auslesen:
            if isinstance(factor1, TimeSeries): factor1 = factor1.active_data
            if isinstance(factor2, TimeSeries): factor2 = factor2.active_data

            factorOfSummand = np.multiply(factor1, factor2)  # np.multiply = elementweise Multiplikation
            ## Share zu TS-equation hinzufügen:
            # if constant share:      
            if variable is None:
                self.eq_sum_TS.add_constant(-1 * factorOfSummand)  # share in global
                if nameOfShare is not None:
                    eq_oneShare.add_constant(-1 * sum(factorOfSummand))  # share itself
            # if variable share:
            else:
                self.eq_sum_TS.add_summand(variable, factorOfSummand)  # share in global
                if nameOfShare is not None:
                    eq_oneShare.add_summand(variable, factorOfSummand, as_sum=True)  # share itself


        else:
            assert (not (isinstance(factor1, TimeSeries))) & (not (isinstance(factor2,
                                                                              TimeSeries))), 'factor1 und factor2 müssen Skalare sein, da shareSum ' + self.label + 'skalar ist'
            factorOfSummand = factor1 * factor2
            ## Share zu skalar-equation hinzufügen:
            # if constant share:
            if variable is None:
                self.eq_sum.add_constant(-1 * factorOfSummand)  # share in global
                if nameOfShare is not None:
                    eq_oneShare.add_constant(-1 * factorOfSummand)  # share itself
            # if variable share:
            else:
                self.eq_sum.add_summand(variable, factorOfSummand)  # share in global
                if nameOfShare is not None:
                    eq_oneShare.add_summand(variable, factorOfSummand)  # share itself


class cFeatureShares(cFeature):
    ''' 
    used to list all shares
    (owner is cFeature_ShareSum)
    
    '''

    def __init__(self, label, owner):
        super().__init__(label, owner)

    def do_modeling(self, system_model, timeIndexe):
        pass

    def declare_vars_and_eqs(self, system_model):
        super().declare_vars_and_eqs(system_model)

    def get_eqOfNewShare(self, nameOfShare, shareHolder, system_model):
        '''         
        creates variable and equation for every share for 
        publishing share-values in results

        Parameters
        ----------
        nameOfShare : str

        Returns
        -------
        eq_oneShare : Equation

        '''
        try:
            full_name_of_share = shareHolder.label_full + '_' + nameOfShare
        except:
            pass
        var_oneShare = Variable(full_name_of_share, 1, self, system_model)  # Skalar
        eq_oneShare = Equation(full_name_of_share, self, system_model)
        eq_oneShare.add_summand(var_oneShare, -1)

        return eq_oneShare


class cFeatureInvest(cFeature):
    # -> var_name            : z.B. "nominal_val", "capacity_inFlowHours"
    # -> fixedInvestmentSize : nominal_val, capacity_inFlowHours, ...
    # -> definingVar         : z.B. flow.model.var_val
    # -> min_rel,max_rel     : ist relatives Min,Max der definingVar bzgl. investmentSize

    @property
    def _existOn(self):  # existiert On-variable
        if self.featureOn is None:
            existOn = False
        else:
            existOn = self.featureOn.useOn
        return existOn

    def __init__(self, nameOfInvestmentSize, owner, invest_parameters: InvestParameters, min_rel, max_rel, val_rel, investmentSize,
                 featureOn=None):
        '''
        

        Parameters
        ----------
        nameOfInvestmentSize : TYPE
            DESCRIPTION.
        owner : TYPE
            owner of this Element
        invest_parameters : InvestParameters
            arguments for modeling
        min_rel : scalar or TS
            given min_rel of definingVar 
            (min = min_rel * investmentSize)
        max_rel : scalar or TS        
            given max_rel of definingVar
            (max = max_rel * investmentSize)
        val_rel : scalar or TS
            given val_rel of definingVar
            (val = val_rel * investmentSize)
        investmentSize : scalar or None
            value of fixed investmentSize (None if no fixed investmentSize)
            Flow: investmentSize = nominal_val
            Storage: investmentSize =
        featureOn : cFeatureOn
            cFeatureOn of the definingVar (if it has a cFeatureOn)

        Returns
        -------
        None.

        '''
        super().__init__('invest', owner)
        self.nameOfInvestmentSize = nameOfInvestmentSize
        self.owner = owner
        self.args = invest_parameters
        self.definingVar = None
        self.max_rel = max_rel
        self.min_rel = min_rel
        self.val_rel = val_rel
        self.fixedInvestmentSize = investmentSize  # nominalValue
        self.featureOn = featureOn

        self.checkPlausibility()

        # segmented investcosts:
        self.featureLinearSegments = None
        if self.args.effects_in_segments is not None:
            self.featureLinearSegments = cFeatureLinearSegmentVars('segmentedInvestcosts', self)

    def checkPlausibility(self):
        # Check fixedInvestmentSize:
        # todo: vielleicht ist es aber auch ok, wenn der nominal_val belegt ist und einfach nicht genutzt wird....
        if self.args.fixed_size:
            assert ((self.fixedInvestmentSize is not None) and (
                        self.fixedInvestmentSize != 0)), 'fixedInvestmentSize muss gesetzt werden'
        else:
            assert self.fixedInvestmentSize is None, '!' + self.nameOfInvestmentSize + ' of ' + self.owner.label_full + ' must be None if investmentSize is variable'

    def getMinMaxOfDefiningVar(self):

        # Wenn fixer relativer Lastgang:
        if self.val_rel is not None:
            # max_rel = min_rel = val_rel !
            min_rel_eff = self.val_rel.active_data
            max_rel_eff = min_rel_eff
        else:
            min_rel_eff = self.min_rel.active_data
            max_rel_eff = self.max_rel.active_data

        onIsUsed = ((self.featureOn is not None) and (self.featureOn.useOn))
        onIsUsedAndvalIsNotFix = (self.val_rel is None) and onIsUsed

        # min-Wert:
        if self.args.optional:
            lb = 0  # can be zero (if no invest) (than for all timesteps)
        elif onIsUsedAndvalIsNotFix:
            lb = 0  # can switch off and therefore be zero
        else:
            if self.args.fixed_size:
                lb = min_rel_eff * self.fixedInvestmentSize  # immer an
            else:
                lb = min_rel_eff * self.args.minimum_size  # investSize is variabel

        #  max-Wert:
        if self.args.fixed_size:
            ub = max_rel_eff * self.fixedInvestmentSize
        else:
            ub = max_rel_eff * self.args.maximum_size  # investSize is variabel

        # ub und lb gleich, dann fix:
        if np.all(ub == lb):  # np.all -> kann listen oder werte vergleichen
            fix_value = ub
            ub = None
            lb = None
        else:
            fix_value = None

        return (lb, ub, fix_value)

    # Variablenreferenz kann erst später hinzugefügt werden, da erst später erstellt:
    # todo-> abändern durch Variable-Dummies
    def setDefiningVar(self, definingVar, definingVar_On):
        self.definingVar = definingVar
        self.definingVar_On = definingVar_On

    def declare_vars_and_eqs(self, system_model):

        # a) var_investmentSize: (wird immer gebaut, auch wenn fix)           

        # lb..ub of investSize unterscheiden:        
        # min:
        if self.args.optional:
            lb = 0
            # Wenn invest nicht optional:
        else:
            if self.args.fixed_size:
                lb = self.fixedInvestmentSize  # einschränken, damit P_inv = P_nom !
            else:
                lb = self.args.minimum_size  #
        # max:
        if self.args.fixed_size:
            ub = self.fixedInvestmentSize
            # wenn nicht fixed:
        else:
            ub = self.args.maximum_size
            # Definition:

        if lb == ub:
            # fix:
            self.model.var_investmentSize = Variable(self.nameOfInvestmentSize, 1, self, system_model, value=lb)
        else:
            # Bereich:
            self.model.var_investmentSize = Variable(self.nameOfInvestmentSize, 1, self, system_model, lower_bound=lb, upper_bound=ub)

        # b) var_isInvested:
        if self.args.optional:
            self.model.var_isInvested = Variable('isInvested', 1, self, system_model, is_binary=True)

            ## investCosts in Segments: ##
        # wenn vorhanden,
        if self.featureLinearSegments is not None:
            self._defineCostSegments(system_model)
            self.featureLinearSegments.declare_vars_and_eqs(system_model)

    # definingInvestcosts in Segments:
    def _defineCostSegments(self, system_model: SystemModel):
        investSizeSegs = self.args.effects_in_segments[0]  # segments of investSize
        costSegs = self.args.effects_in_segments[1]  # effect-dict with segments as entries
        costSegs = as_effect_dict(costSegs)

        ## 1. create segments for investSize and every effect##
        ## 1.a) add investSize-Variablen-Segmente: ##
        segmentsOfVars = {self.model.var_investmentSize: investSizeSegs}  # i.e. {var_investSize: [0,5, 5,20]}

        ## 1.b) je Effekt -> new Variable und zugehörige Segmente ##
        self.model.var_list_investCosts_segmented = []
        self.investVar_effect_dict = {}  # benötigt
        for aEffect, aSegmentCosts in costSegs.items():
            var_investForEffect = self.__create_var_segmentedInvestCost(aEffect, system_model)
            aSegment = {var_investForEffect: aSegmentCosts}  # i.e. {var_investCosts_segmented_costs : [0,10, 10,30]}
            segmentsOfVars |= aSegment  #
            self.investVar_effect_dict |= {aEffect: var_investForEffect}

        ## 2. on_var: ##
        if self.args.optional:
            var_isInvested = self.model.var_isInvested
        else:
            var_isInvested = None

        ## 3. transfer segmentsOfVars to cFeatureLinearSegmentVars: ##
        self.featureLinearSegments.defineSegments(segmentsOfVars, var_on=var_isInvested,
                                                  checkListOfVars=list(segmentsOfVars.keys()))

    def __create_var_segmentedInvestCost(self, aEffect, system_model):
        # define cost-Variable (=costs through segmented Investsize-costs):
        if isinstance(aEffect, Effect):
            aStr = aEffect.label
        elif aEffect is None:
            aStr = system_model.system.listOfEffectTypes.standard_effect().label  # Standard-Effekt
        else:
            raise Exception('Given effect (' + str(aEffect) + ') is not an effect!')
        # new variable, i.e for costs, CO2,... :
        var_investForEffect = Variable('investCosts_segmented_' + aStr, 1, self, system_model, lower_bound=0)
        self.model.var_list_investCosts_segmented.append(var_investForEffect)
        return var_investForEffect

    def do_modeling(self, system_model, timeIndexe):
        assert self.definingVar is not None, 'setDefiningVar() still not executed!'
        # wenn var_isInvested existiert:    
        if self.args.optional:
            self._add_defining_var_isInvested(system_model)

        # Bereich von definingVar in Abh. von var_investmentSize:

        # Wenn fixer relativer Lastgang:
        if self.val_rel is not None:
            self._add_fixEq_of_definingVar_with_var_investmentSize(system_model)
        # Wenn nicht fix:
        else:
            self._add_max_min_of_definingVar_with_var_investmentSize(system_model)

            # if linear Segments defined:
        if self.featureLinearSegments is not None:
            self.featureLinearSegments.do_modeling(system_model, timeIndexe)

    def _add_fixEq_of_definingVar_with_var_investmentSize(self, system_model):

        ## Gleichung zw. DefiningVar und Investgröße:    
        # eq: definingVar(t) = var_investmentSize * val_rel

        self.eq_fix_via_investmentSize = Equation('fix_via_InvestmentSize', self, system_model, 'eq')
        self.eq_fix_via_investmentSize.add_summand(self.definingVar, 1)
        self.eq_fix_via_investmentSize.add_summand(self.model.var_investmentSize, np.multiply(-1, self.val_rel.active_data))

    def _add_max_min_of_definingVar_with_var_investmentSize(self, system_model):

        ## 1. Gleichung: Maximum durch Investmentgröße ##     
        # eq: definingVar(t) <=                var_investmentSize * max_rel(t)     
        # eq: P(t) <= max_rel(t) * P_inv    
        self.eq_max_via_investmentSize = Equation('max_via_InvestmentSize', self, system_model, 'ineq')
        self.eq_max_via_investmentSize.add_summand(self.definingVar, 1)
        # TODO: Changed by FB
        # self.eq_max_via_investmentSize.add_summand(self.model.var_investmentSize, np.multiply(-1, self.max_rel.active_data))
        self.eq_max_via_investmentSize.add_summand(self.model.var_investmentSize, np.multiply(-1, self.max_rel.data))
        # TODO: Changed by FB

        ## 2. Gleichung: Minimum durch Investmentgröße ##        

        # Glg nur, wenn nicht Kombination On und fixed:
        if not (self._existOn and self.args.fixed_size):
            self.eq_min_via_investmentSize = Equation('min_via_investmentSize', self, system_model, 'ineq')

        if self._existOn:
            # Wenn InvestSize nicht fix, dann weitere Glg notwendig für Minimum (abhängig von var_investSize)
            if not self.args.fixed_size:
                # eq: definingVar(t) >= Big * (On(t)-1) + investmentSize * min_rel(t)
                #     ... mit Big = max(min_rel*P_inv_max, epsilon)
                # (P < min_rel*P_inv -> On=0 | On=1 -> P >= min_rel*P_inv)

                # äquivalent zu:.
                # eq: - definingVar(t) + Big * On(t) + min_rel(t) * investmentSize <= Big

                Big = helpers.max_args(self.min_rel.active_data * self.args.maximum_size, system_model.epsilon)

                self.eq_min_via_investmentSize.add_summand(self.definingVar, -1)
                self.eq_min_via_investmentSize.add_summand(self.definingVar_On, Big)  # übergebene On-Variable
                self.eq_min_via_investmentSize.add_summand(self.model.var_investmentSize, self.min_rel.active_data)
                self.eq_min_via_investmentSize.add_constant(Big)
                # Anmerkung: Glg bei Spezialfall min_rel = 0 redundant zu cFeatureOn-Glg.
            else:
                pass  # Bereits in cFeatureOn mit P>= On(t)*Min ausreichend definiert
        else:
            # eq: definingVar(t) >= investmentSize * min_rel(t)    

            self.eq_min_via_investmentSize.add_summand(self.definingVar, -1)
            self.eq_min_via_investmentSize.add_summand(self.model.var_investmentSize, self.min_rel.active_data)

            #### Defining var_isInvested ####

    def _add_defining_var_isInvested(self, system_model):

        # wenn fixed, dann const:
        if self.args.fixed_size:

            # eq: investmentSize = isInvested * nominalValue            
            self.eq_isInvested_1 = Equation('isInvested_constraint_1', self, system_model, 'eq')
            self.eq_isInvested_1.add_summand(self.model.var_investmentSize, -1)
            self.eq_isInvested_1.add_summand(self.model.var_isInvested, self.fixedInvestmentSize)

            # wenn nicht fix, dann Bereich:
        else:
            ## 1. Gleichung (skalar):            
            # eq1: P_invest <= isInvested * investSize_max
            # (isInvested = 0 -> P_invest=0  |  P_invest>0 -> isInvested = 1 ->  P_invest < investSize_max )   

            self.eq_isInvested_1 = Equation('isInvested_constraint_1', self, system_model, 'ineq')
            self.eq_isInvested_1.add_summand(self.model.var_investmentSize, 1)
            self.eq_isInvested_1.add_summand(self.model.var_isInvested,
                                             np.multiply(-1, self.args.maximum_size))  # Variable ist Skalar!

            ## 2. Gleichung (skalar):                  
            # eq2: P_invest  >= isInvested * max(epsilon, investSize_min)
            # (isInvested = 1 -> P_invest>0  |  P_invest=0 -> isInvested = 0)
            self.eq_isInvested_2 = Equation('isInvested_constraint_2', self, system_model, 'ineq')
            self.eq_isInvested_2.add_summand(self.model.var_investmentSize, -1)
            self.eq_isInvested_2.add_summand(self.model.var_isInvested, max(system_model.epsilon, self.args.minimum_size))

    def add_share_to_globals(self, globalComp, system_model):

        # # fix_effects:
        # wenn fix_effects vorhanden:
        if not (self.args.fix_effects is None) and self.args.fix_effects != 0:
            if self.args.optional:
                # fix Share to InvestCosts: 
                # share: + isInvested * fix_effects
                globalComp.addShareToInvest('fix_effects', self.owner, self.model.var_isInvested, self.args.fix_effects, 1)
            else:
                # share: + fix_effects
                globalComp.addConstantShareToInvest('fix_effects', self.owner, self.args.fix_effects,
                                                    1)  # fester Wert hinufügen

        # # divest_effects:

        if not (self.args.divest_effects is None) and self.args.divestCost != 0:
            if self.args.optional:
                # fix Share to InvestCosts: 
                # share: [(1- isInvested) * divest_effects]
                # share: [divest_effects - isInvested * divest_effects]
                # 1. part of share [+ divest_effects]:
                globalComp.addConstantShareToInvest('divest_effects', self.owner, self.args.divest_effects, 1)
                # 2. part of share [- isInvested * divest_effects]:
                globalComp.addShareToInvest('divestCosts_cancellation', self.owner, self.model.var_isInvested,
                                            self.args.divest_effects, -1)
                # TODO : these 2 parts should be one share!
            else:
                pass  # no divest costs if invest is not optional

        # # specific_effects:
        # wenn specific_effects vorhanden:
        if not (self.args.specific_effects is None):
            # share: + investmentSize (=var)   * specific_effects
            globalComp.addShareToInvest('specific_effects', self.owner, self.model.var_investmentSize,
                                        self.args.specific_effects, 1)

        # # segmentedCosts:                                        
        if self.featureLinearSegments is not None:
            for effect, var_investSegs in self.investVar_effect_dict.items():
                globalComp.addShareToInvest('linearSegments', self.owner, var_investSegs, {effect: 1}, 1)
