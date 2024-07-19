# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 18:43:55 2021
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""
## TODO:
# featureAvoidFlowsAtOnce:
# neue Variante (typ="new") austesten
from typing import Optional, Union, Tuple, Dict, List, Set, Callable, Literal, TYPE_CHECKING
import logging

import numpy as np

from flixOpt.structure import Element, SystemModel
from flixOpt.elements import Flow, EffectTypeDict, Global
from flixOpt.core import TimeSeries, Skalar, Numeric, Numeric_TS, as_effect_dict
from flixOpt.modeling import Variable, VariableTS, Equation
from flixOpt.flixBasicsPublic import InvestParameters
import flixOpt.flixOptHelperFcts as helpers

log = logging.getLogger(__name__)


##############################################################
## Funktionalität/Features zum Anhängen an die Komponenten: ##  

class Feature(Element):

    def __init__(self, label: str, owner: Element, **kwargs):
        self.owner = owner
        if self not in self.owner.sub_elements:
            self.owner.sub_elements.append(self)  # register in owner
        super().__init__(label, **kwargs)

    def __str__(self):
        return f"<{self.__class__.__name__}> {self.label}"

    @property
    def label_full(self):
        return self.owner.label_full + '__' + self.label

    def finalize(self):  # TODO: evtl. besser bei Element aufgehoben
        super().finalize()


class Segment(Feature):
    """
    Für Abschnittsweise Linearisierung ist das ein Abschnitt
    """
    def __init__(self,
                 label: str,
                 owner: Element,
                 sample_points: Dict[Variable, Tuple[Numeric, Numeric]],
                 index):
        super().__init__(label, owner)
        self.label = label
        self.sample_points = sample_points
        self.index = index

    def declare_vars_and_eqs(self, system_model):
        length = system_model.nrOfTimeSteps
        self.model.add_variable(VariableTS(f'onSeg_{self.index}', length, self, system_model,
                                          is_binary=True))  # Binär-Variable
        self.model.add_variable(VariableTS(f'lambda1_{self.index}', length, self, system_model, lower_bound=0,
                                            upper_bound=1))  # Wertebereich 0..1
        self.model.add_variable(VariableTS(f'lambda2_{self.index}', length, self, system_model, lower_bound=0,
                                            upper_bound=1))  # Wertebereich 0..1


# Abschnittsweise linear:
class FeatureLinearSegmentVars(Feature):
    # TODO: beser wäre hier schon Übergabe segments_of_variables, aber schwierig, weil diese hier noch nicht vorhanden sind!
    def __init__(self, label: str, owner: Element):
        super().__init__(label, owner)
        self.eq_canOnlyBeInOneSegment: Equation = None
        self.segments_of_variables: Dict[Variable, List[Skalar]] = None
        self.binary_variable: Optional[Variable] = None

    @property
    def segments(self) -> List[Segment]:
        return self.sub_elements

    @property
    def nr_of_segments(self) -> Skalar:
        # Anzahl Segmente bestimmen:
        nr_of_columns = len(next(iter(self.segments_of_variables.values())))   # Anzahl Spalten
        return nr_of_columns / 2  # Anzahl der Spalten der Matrix / 2 = Anzahl der Segmente

    @property
    def variables(self) -> Set[Variable]:
        return set(self.segments_of_variables.keys())

    # segements separat erst jetzt definieren, damit Variablen schon erstellt sind.
    # todo: wenn Variable-Dummys existieren, dann kann das alles in __init__
    def define_segments(self,
                        segments_of_variables: Dict[Variable, List[Skalar]],
                        binary_variable: Optional[Variable],
                        vars_for_check: List[Variable]):
        # segementsData - Elemente sind Listen!.
        # segments_of_variables = {var_Q_fu: [ 5  , 10,  10, 22], # je zwei Werte bilden ein Segment. Indexspezfika (z.B. für Zeitreihenabbildung) über arrays oder TS!!
        #                   var_P_el: [ 2  , 5,    5, 8 ],
        #                   var_Q_th: [ 2.5, 4,    4, 12]}

        # -> onVariable ist optional
        # -> auch einzelne zulässige Punkte können über Segment ausgedrückt werden, z.B. [5, 5]
        
        self.sub_elements: List[Segment] = []   # Resetting the segments for new calculation
        self.segments_of_variables = segments_of_variables
        self.binary_variable = binary_variable

        if not self.nr_of_segments.is_integer():   # Check ob gerade Anzahl an Werten:
            raise Exception('Nr of Values should be even, because pairs (start,end of every section)')

        if vars_for_check is not None:
            extra_vars = self.variables - set(vars_for_check)
            missing_vars = set(vars_for_check) - self.variables

            def get_str_of_set(a_set):
                return ','.join(var.label_full for var in a_set)

            if extra_vars:
                raise Exception('segments_of_variables-Definition has unnecessary vars: ' + get_str_of_set(extra_vars))

            if missing_vars:
                raise Exception('segments_of_variables is missing the following vars: ' + get_str_of_set(missing_vars))

        # Aufteilen der Daten in die Segmente:
        for section_index in range(int(self.nr_of_segments)):
            # sample_points für das Segment extrahieren:
            # z.B.   {var1:[TS1.1, TS1.2]
            #         var2:[TS2.1, TS2.2]}            
            sample_points_of_segment = (
                FeatureLinearSegmentVars._extract_sample_points_for_segment(self.segments_of_variables, section_index))
            # Segment erstellen und in Liste::
            new_segment = Segment(f'seg_{section_index}', self, sample_points_of_segment, section_index)
            # todo: hier muss activate() selbst gesetzt werden, weil bereits gesetzt 
            # todo: alle Elemente sollten eigentlich hier schon längst instanziert sein und werden dann auch activated!!!
            new_segment.create_new_model_and_activate_system_model(self.system_model)

    def declare_vars_and_eqs(self, system_model: SystemModel):
        for aSegment in self.segments:
            # Segmentvariablen erstellen:
            aSegment.declare_vars_and_eqs(system_model)

    def do_modeling(self, system_model: SystemModel, time_indices: Union[list[int], range]):
        #########################################
        ## 1. Gleichungen für: Nur ein Segment kann aktiv sein! ##
        # eq: -On(t) + Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 0 
        # -> Wenn Variable On(t) nicht existiert, dann nur 
        # eq:          Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 1                                         

        self.model.add_equation(Equation('ICanOnlyBeInOneSegment', self, system_model))

        # a) zusätzlich zu Aufenthalt in Segmenten kann alles auch Null sein:
        if self.binary_variable is not None:
            # TODO: # Eigentlich wird die On-Variable durch linearSegment-equations bereits vollständig definiert.
            self.model.eqs['ICanOnlyBeInOneSegment'].add_summand(self.binary_variable, -1)
        else:
            self.model.eqs['ICanOnlyBeInOneSegment'].add_constant(1)   # b) Aufenthalt nur in Segmenten erlaubt:

        for aSegment in self.segments:
            self.model.eqs['ICanOnlyBeInOneSegment'].add_summand(aSegment.model.variables[f'onSeg_{aSegment.index}'], 1)

            #################################
        ## 2. Gleichungen der Segmente ##
        # eq: -aSegment.onSeg(t) + aSegment.lambda1(t) + aSegment.lambda2(t)  = 0    
        for aSegment in self.segments:
            name_of_equation = f'Lambda_onSeg_{aSegment.index}'

            self.model.add_equation(Equation(name_of_equation, self, system_model))
            self.model.eqs[name_of_equation].add_summand(aSegment.model.variables[f'onSeg_{aSegment.index}'], -1)
            self.model.eqs[name_of_equation].add_summand(aSegment.model.variables[f'lambda1_{aSegment.index}'], 1)
            self.model.eqs[name_of_equation].add_summand(aSegment.model.variables[f'lambda2_{aSegment.index}'], 1)

            ##################################################
        ## 3. Gleichungen für die Variablen mit lambda: ##
        #   z.B. Gleichungen für Q_th mit lambda
        #  eq: - Q_th(t) + sum(Q_th_1_j * lambda_1_j + Q_th_2_j * lambda_2_j) = 0
        #  mit -> j                   = Segmentnummer 
        #      -> Q_th_1_j, Q_th_2_j  = Stützstellen des Segments (können auch Vektor sein)

        for aVar in self.variables:
            # aVar = aFlow.model.var_val
            lambda_eq = Equation(aVar.label_full + '_lambda', self, system_model)  # z.B. Q_th(t)
            self.model.add_equation(lambda_eq)
            lambda_eq.add_summand(aVar, -1)
            for aSegment in self.segments:
                #  Stützstellen einfügen:
                stuetz1 = aSegment.sample_points[aVar][0]
                stuetz2 = aSegment.sample_points[aVar][1]
                # wenn Stützstellen TS_vector:
                if isinstance(stuetz1, TimeSeries):
                    samplePoint1 = stuetz1.active_data
                    samplePoint2 = stuetz2.active_data
                # wenn Stützstellen Skalar oder array
                else:
                    samplePoint1 = stuetz1
                    samplePoint2 = stuetz2

                lambda_eq.add_summand(aSegment.model.variables[f'lambda1_{aSegment.index}'],
                                     samplePoint1)  # Spalte 1 (Faktor kann hier Skalar sein oder Vektor)
                lambda_eq.add_summand(aSegment.model.variables[f'lambda2_{aSegment.index}'],
                                     samplePoint2)  # Spalte 2 (Faktor kann hier Skalar sein oder Vektor)

    # extract the 2 TS_vectors for the segment:
    @staticmethod
    def _extract_sample_points_for_segment(sample_points_of_all_segments, segment_index):
        # sample_points_of_all_segments = {var1:[TS1.1, TS1.2]
        sample_points_for_segments = {}
        # für alle Variablen Segment-Stützstellen holen:
        start_column_of_section = segment_index * 2  # 0, 2, 4
        for aVar in sample_points_of_all_segments.keys():
            # 1. und 2. Stützstellen des Segments auswählen:
            sample_points_for_segments[aVar] = sample_points_of_all_segments[aVar][start_column_of_section: start_column_of_section + 2]
        return sample_points_for_segments


class FeatureLinearSegmentSet(FeatureLinearSegmentVars):
    # TODO: beser wäre segments_of_variables, aber schwierig, weil diese hier noch nicht vorhanden sind!
    def __init__(self,
                 label: str,
                 owner: Element,
                 segments_of_flows: Dict[Flow, List[Union[Numeric]]],
                 get_var_on: Optional[Callable] = None,
                 flows: Optional[List[Flow]] = None):
        # segementsData - Elemente sind Listen!.
        # segmentsOfFlows = {Q_fu: [ 5  , 10,  10, 22], # je zwei Werte bilden ein Segment. Zeitreihenabbildung über arrays!!!
        #                    P_el: [ 2  , 5,    5, 8 ],
        #                    Q_th: [ 2.5, 4,    4, 12]}
        # -> auch einzelne zulässige Punkte können über Segment ausgedrückt werden, z.B. [5, 5]

        self.segments_of_flows = segments_of_flows
        self.get_var_on = get_var_on
        self.flows = flows
        self.eq_flowLock = None
        super().__init__(label, owner)

    def declare_vars_and_eqs(self, system_model):

        segments_of_variables = {flow.model.variables['val']: self.segments_of_flows[flow] for flow in self.segments_of_flows}
        variables = {flow.model.variables['val'] for flow in self.flows}

        # hier erst Variablen vorhanden un damit segments_of_variables definierbar!
        super().define_segments(segments_of_variables,
                                binary_variable=self.get_var_on(),
                                vars_for_check=variables)  # todo: das ist nur hier, damit schon variablen Bekannt

        super().declare_vars_and_eqs(system_model)   # 2. declare vars:


# Verhindern gleichzeitig mehrere Flows > 0 
class FeatureAvoidFlowsAtOnce(Feature):

    def __init__(self, label: str, owner: Element, flows: List[Flow], typ: Literal['classic', 'new'] = 'classic'):
        super().__init__(label, owner)
        self.flows = flows
        self.typ = typ
        assert len(self.flows) >= 2, 'Beachte für Feature AvoidFlowsAtOnce: Mindestens 2 Flows notwendig'

    def finalize(self):
        super().finalize()
        # Beachte: Hiervor muss featureOn in den Flows existieren!
        aFlow: Flow

        if self.typ == 'classic':
            # "classic" -> alle Flows brauchen Binärvariable:
            for aFlow in self.flows:
                aFlow.force_on_variable()

        elif self.typ == 'new':
            # "new" -> n-1 Flows brauchen Binärvariable: (eine wird eingespart)

            # 1. Get nr of existing on_vars in Flows
            existing_on_variables = 0
            for aFlow in self.flows:
                existing_on_variables += aFlow.featureOn.use_on

            # 2. Add necessary further flow binaries:
            # Anzahl on_vars solange erhöhen bis mindestens n-1 vorhanden:
            i = 0
            while existing_on_variables < (len(self.flows) - 1):
                aFlow = self.flows[i]
                # Falls noch nicht on-Var für flow existiert, dann erzwingen:
                if not aFlow.featureOn.use_on:
                    aFlow.force_on_variable()
                    existing_on_variables += 1
                i += 1

    def do_modeling(self, system_model, time_indices: Union[list[int], range]):
        # Nur 1 Flow aktiv! Nicht mehrere Zeitgleich!    
        # "classic":
        # eq: sum(flow_i.on(t)) <= 1.1 (1 wird etwas größer gewählt wg. Binärvariablengenauigkeit)
        # "new": 
        # eq: flow_1.on(t) + flow_2.on(t) + .. + flow_i.val(t)/flow_i.max <= 1 (1 Flow ohne Binärvariable!)

        # Anmerkung: Patrick Schönfeld (oemof, custom/link.py) macht bei 2 Flows ohne Binärvariable dies:
        # 1)	bin + flow1/flow1_max <= 1
        # 2)	bin - flow2/flow2_max >= 0
        # 3)    geht nur, wenn alle flow.min >= 0
        # --> könnte man auch umsetzen (statt force_on_variable() für die Flows, aber sollte aufs selbe wie "new" kommen)

        self.model.add_equation(Equation('flowLock', self, system_model, eqType='ineq'))
        # Summanden hinzufügen:
        for aFlow in self.flows:
            # + flow_i.on(t):
            if aFlow.featureOn.model.variables.get('on') is not None:
                self.model.eqs['flowLock'].add_summand(aFlow.featureOn.model.variables['on'], 1)

        if self.typ == 'classic':
            # TODO: Decrease the value 1.1?
            self.model.eqs['flowLock'].add_constant(
                1.1)  # sicherheitshalber etwas mehr, damit auch leicht größer Binärvariablen 1.00001 funktionieren.
        else:
            raise NotImplementedError(f'FeatureAvoidFlowsAtOnce: "{self.typ=}" not implemented!')


class FeatureOn(Feature):
    """
    Klasse, die in Komponenten UND Flows benötigt wird
    
        def __init__(self,
                 label: str,
                 on_values_before_begin:Optional[List[Skalar]] = None,
                 switch_on_effects: Optional[Union[EffectTypeDict, Numeric_TS]] = None,
                 switch_on_total_max: Optional[Skalar] = None,
                 on_hours_total_min: Optional[Skalar] = None,
                 on_hours_total_max: Optional[Skalar] = None,
                 running_hour_effects: Optional[Union[EffectTypeDict, Numeric_TS]] = None,
    
    """
    def __init__(self,
                 owner: Element,
                 flows_defining_on: List[Flow],
                 on_values_before_begin: List[int],
                 switch_on_effects: Optional[EffectTypeDict] = None,
                 running_hour_effects: Optional[EffectTypeDict] = None,
                 on_hours_total_min: Optional[int] = None,
                 on_hours_total_max: Optional[int] = None,
                 on_hours_min: Optional[Numeric] = None,
                 on_hours_max: Optional[Numeric] = None,
                 off_hours_min: Optional[Numeric] = None,
                 off_hours_max: Optional[Numeric] = None,
                 switch_on_total_max: Optional[int] = None,
                 force_on: bool = False,
                 force_switch_on: bool = False):
        super().__init__('featureOn', owner)
        self.flows_defining_on = flows_defining_on
        self.on_values_before_begin = on_values_before_begin
        self.switch_on_effects = switch_on_effects
        self.running_hour_effects = running_hour_effects
        self.on_hours_total_min = on_hours_total_min  # scalar
        self.on_hours_total_max = on_hours_total_max  # scalar
        self.on_hours_min = on_hours_min  # TimeSeries
        self.on_hours_max = on_hours_max  # TimeSeries
        self.off_hours_min = off_hours_min  # TimeSeries
        self.off_hours_max = off_hours_max  # TimeSeries
        self.switch_on_total_max = switch_on_total_max
        self.force_on = force_on   # Can be set to True if needed, even after creation
        self.force_switch_on = force_switch_on

    @property
    def use_on(self) -> bool:
        return (any(param is not None for param in [self.running_hour_effects,
                                                    self.on_hours_total_min,
                                                    self.on_hours_total_max])
                or self.force_on or self.use_switch_on or self.use_on_hours or self.use_off_hours or self.use_off)

    @property
    def use_off(self) -> bool:
        return self.use_off_hours

    @property
    def use_on_hours(self) -> bool:
        return any(param is not None for param in [self.on_hours_min, self.on_hours_max])

    @property
    def use_off_hours(self) -> bool:
        return any(param is not None for param in [self.off_hours_min, self.off_hours_max])

    @property
    def use_switch_on(self) -> bool:
        return (any(param is not None for param in [self.switch_on_effects,
                                                   self.switch_on_total_max,
                                                   self.on_hours_total_min,
                                                   self.on_hours_total_max])
                or self.force_switch_on)

    # varOwner braucht die Variable auch:
    def getVar_on(self) -> Optional[VariableTS]:
        return self.model.variables.get('on')

    def getVars_switchOnOff(self) -> Tuple[Optional[VariableTS], Optional[VariableTS]]:
        return self.model.variables.get('switchOn'), self.model.variables.get('switchOff')

    #   # Variable wird erstellt und auch gleich in featureOwner registiert:
    #      
    #   # ## Variable als Attribut in featureOwner übergeben:
    #   # # TODO: ist das so schick? oder sollte man nicht so versteckt Attribute setzen?
    #   # # Check:
    #   # if (hasattr(self.featureOwner, 'var_on')) and (self.featureOwner.model.var_on == None) :
    #   #   self.featureOwner.model.var_on = self.model.var_on
    #   # else :
    #   #   raise Exception('featureOwner ' + self.featureOwner.label + ' has no attribute var_on or it is already used')

    def declare_vars_and_eqs(self, system_model: SystemModel):
        # Beachte: Variablen gehören nicht diesem Element, sondern varOwner (meist ist das der featureOwner)!!!  
        # Var On:
        if self.use_on:
            # Before-Variable:
            self.model.add_variable(VariableTS('on', system_model.nrOfTimeSteps, self.owner, system_model, is_binary=True))
            self.model.variables['on'].set_before_value(default_before_value=self.on_values_before_begin[0],
                                               is_start_value=False)
            self.model.add_variable(Variable('onHoursSum', 1, self.owner, system_model, lower_bound=self.on_hours_total_min,
                                                 upper_bound=self.on_hours_total_max))  # wenn max/min = None, dann bleibt das frei

        if self.use_off:
            # off-Var is needed:
            self.model.add_variable(VariableTS('off', system_model.nrOfTimeSteps, self.owner, system_model, is_binary=True))

        # onHours:
        #   i.g. 
        #   var_on      = [0 0 1 1 1 1 0 0 0 1 1 1 0 ...]
        #   var_onHours = [0 0 1 2 3 4 0 0 0 1 2 3 0 ...] (bei dt=1)
        if self.use_on_hours:
            aMax = None if (self.on_hours_max is None) else self.on_hours_max.active_data
            self.model.add_variable(VariableTS('onHours', system_model.nrOfTimeSteps, self.owner, system_model,
                                                lower_bound=0, upper_bound=aMax))  # min separat
        # offHours:
        if self.use_off_hours:
            aMax = None if (self.off_hours_max is None) else self.off_hours_max.active_data
            self.model.add_variable(VariableTS('offHours', system_model.nrOfTimeSteps, self.owner, system_model,
                                                 lower_bound=0, upper_bound=aMax))  # min separat

        # Var SwitchOn
        if self.use_switch_on:
            self.model.add_variable(VariableTS('switchOn', system_model.nrOfTimeSteps, self.owner, system_model, is_binary=True))
            self.model.add_variable(VariableTS('switchOff', system_model.nrOfTimeSteps, self.owner, system_model, is_binary=True))
            self.model.add_variable(Variable('nrSwitchOn', 1, self.owner, system_model,
                                                 upper_bound=self.switch_on_total_max))  # wenn max/min = None, dann bleibt das frei

    def do_modeling(self, system_model: SystemModel, time_indices: Union[list[int], range]):
        if self.use_on:
            self._add_on_constraints(system_model, time_indices)
        if self.use_off:
            self._add_off_constraints(system_model, time_indices)
        if self.use_switch_on:
            self.add_switch_constraints(system_model, time_indices)
        if self.use_on_hours:
            FeatureOn._add_duration_constraints(
                self.model.variables['onHours'], self.model.variables['on'], self.on_hours_min,
                self, system_model, time_indices)
        if self.use_off_hours:
            FeatureOn._add_duration_constraints(
                self.model.variables['offHours'], self.model.variables['off'], self.off_hours_min,
                self, system_model, time_indices)

    def _add_on_constraints(self, system_model: SystemModel, time_indices: Union[list[int], range]):
        # % Bedingungen 1) und 2) müssen erfüllt sein:

        # % Anmerkung: Falls "abschnittsweise linear" gewählt, dann ist eigentlich nur Bedingung 1) noch notwendig 
        # %            (und dann auch nur wenn erstes Segment bei Q_th=0 beginnt. Dann soll bei Q_th=0 (d.h. die Maschine ist Aus) On = 0 und segment1.onSeg = 0):)
        # %            Fazit: Wenn kein Performance-Verlust durch mehr Gleichungen, dann egal!      

        nr_of_flows = len(self.flows_defining_on)
        assert nr_of_flows > 0, 'Achtung: mindestens 1 Flow notwendig'
        #### Bedingung 1) ####

        # Glg. wird erstellt und auch gleich in featureOwner registiert:
        self.model.add_equation(Equation('On_Constraint_1', self, system_model, eqType='ineq'))
        # TODO: eventuell label besser über  nameOfIneq = [aOnVariable.name '_Constraint_1']; % z.B. On_Constraint_1

        # Wenn nur 1 Leistungsvariable (!Unterscheidet sich von >1 Leistungsvariablen wg. Minimum-Beachtung!):
        if len(self.flows_defining_on) == 1:
            ## Leistung<=MinLeistung -> On = 0 | On=1 -> Leistung>MinLeistung
            # eq: Q_th(t) - max(Epsilon, Q_th_min) * On(t) >= 0  (mit Epsilon = sehr kleine Zahl, wird nur im Falle Q_th_min = 0 gebraucht)
            # gleichbedeutend mit eq: -Q_th(t) + max(Epsilon, Q_th_min)* On(t) <= 0 
            aFlow = self.flows_defining_on[0]
            self.model.eqs['On_Constraint_1'].add_summand(aFlow.model.variables['val'], -1, time_indices)
            # wenn variabler Nennwert:
            if aFlow.size is None:
                min_val = aFlow.invest_parameters.minimum_size * aFlow.min_rel.active_data  # kleinst-Möglichen Wert nutzen. (Immer noch math. günstiger als Epsilon)
            # wenn fixer Nennwert
            else:
                min_val = aFlow.size * aFlow.min_rel.active_data

            self.model.eqs['On_Constraint_1'].add_summand(self.model.variables['on'], 1 * np.maximum(system_model.epsilon, min_val),
                            time_indices)  # % aLeistungsVariableMin kann hier Skalar oder Zeitreihe sein!

        # Bei mehreren Leistungsvariablen:
        else:
            # Nur wenn alle Flows = 0, dann ist On = 0
            ## 1) sum(alle Leistung)=0 -> On = 0 | On=1 -> sum(alle Leistungen) > 0
            # eq: - sum(alle Leistungen(t)) + Epsilon * On(t) <= 0
            for aFlow in self.flows_defining_on:
                self.model.eqs['On_Constraint_1'].add_summand(aFlow.model.variables['val'], -1, time_indices)
            self.model.eqs['On_Constraint_1'].add_summand(self.model.variables['on'], 1 * system_model.epsilon,
                            time_indices)  # % aLeistungsVariableMin kann hier Skalar oder Zeitreihe sein!

        #### Bedingung 2) ####

        # Glg. wird erstellt und auch gleich in featureOwner registiert:
        self.model.add_equation(Equation('On_Constraint_2', self, system_model, eqType='ineq'))
        # Wenn nur 1 Leistungsvariable:
        #  eq: Q_th(t) <= Q_th_max * On(t)
        # (Leistung>0 -> On = 1 | On=0 -> Leistung<=0)
        # Bei mehreren Leistungsvariablen:
        ## sum(alle Leistung) >0 -> On = 1 | On=0 -> sum(Leistung)=0
        #  eq: sum( Leistung(t,i))              - sum(Leistung_max(i))             * On(t) <= 0
        #  --> damit Gleichungswerte nicht zu groß werden, noch durch nr_of_flows geteilt:
        #  eq: sum( Leistung(t,i) / nr_of_flows ) - sum(Leistung_max(i)) / nr_of_flows * On(t) <= 0
        sumOfFlowMax = 0
        for aFlow in self.flows_defining_on:
            self.model.eqs['On_Constraint_2'].add_summand(aFlow.model.variables['val'], 1 / nr_of_flows, time_indices)
            # wenn variabler Nennwert:
            if aFlow.size is None:
                sumOfFlowMax += aFlow.max_rel.active_data * aFlow.invest_parameters.maximum_size  # der maximale Nennwert reicht als Obergrenze hier aus. (immer noch math. günster als BigM)
            else:
                sumOfFlowMax += aFlow.max_rel.active_data * aFlow.size

        self.model.eqs['On_Constraint_2'].add_summand(self.model.variables['on'], - sumOfFlowMax / nr_of_flows, time_indices)  #

        if isinstance(sumOfFlowMax, (np.ndarray, list)):
            if max(sumOfFlowMax) / nr_of_flows > 1000: log.warning(
                '!!! ACHTUNG in ' + self.owner.label_full + ' : Binärdefinition mit großem Max-Wert (' + str(
                    int(max(sumOfFlowMax) / nr_of_flows)) + '). Ggf. falsche Ergebnisse !!!')
        else:
            if sumOfFlowMax / nr_of_flows > 1000: log.warning(
                '!!! ACHTUNG in ' + self.owner.label_full + ' : Binärdefinition mit großem Max-Wert (' + str(
                    int(sumOfFlowMax / nr_of_flows)) + '). Ggf. falsche Ergebnisse !!!')

    def _add_off_constraints(self, system_model: SystemModel, time_indices: Union[list[int], range]):
        # Definition var_off:
        # eq: var_off(t) = 1-var_on(t)
        self.model.add_equation(Equation('var_off', self, system_model, eqType='eq'))
        self.model.eqs['var_off'].add_summand(self.model.variables['off'], 1)
        self.model.eqs['var_off'].add_summand(self.model.variables['on'], 1)
        self.model.eqs['var_off'].add_constant(1)

    @staticmethod  # to be sure not using any self-Variables
    def _add_duration_constraints(duration_variable: VariableTS,
                                  binary_variable: VariableTS,
                                  minimum_duration: Optional[TimeSeries],
                                  eqsOwner: Element,
                                  system_model: SystemModel,
                                  time_indices: Union[list[int], range]):
        '''
        i.g. 
        binary_variable        = [0 0 1 1 1 1 0 1 1 1 0 ...]
        duration_variable = [0 0 1 2 3 4 0 1 2 3 0 ...] (bei dt=1)
                                            |-> min_onHours = 3!
        
        if you want to count zeros, define var_bin_off: = 1-binary_variable before!
        '''
        # TODO: Einfachere Variante von Peter umsetzen!

        # 1) eq: onHours(t) <= On(t)*Big | On(t)=0 -> onHours(t) = 0
        # mit Big = dt_in_hours_total
        aLabel = duration_variable.label
        eqsOwner.model.add_equation(Equation(aLabel + '_constraint_1', eqsOwner, system_model, eqType='ineq'))
        eqsOwner.model.eqs[aLabel + '_constraint_1'].add_summand(duration_variable, 1)
        eqsOwner.model.eqs[aLabel + '_constraint_1'].add_summand(binary_variable, -1 * system_model.dt_in_hours_total)

        # 2a) eq: onHours(t) - onHours(t-1) <= dt(t)
        #    on(t)=1 -> ...<= dt(t)
        #    on(t)=0 -> onHours(t-1)>=
        eqsOwner.model.add_equation(Equation(aLabel + '_constraint_2a', eqsOwner, system_model, eqType='ineq'))
        eqsOwner.model.eqs[aLabel + '_constraint_2a'].add_summand(duration_variable, 1, time_indices[1:])  # onHours(t)
        eqsOwner.model.eqs[aLabel + '_constraint_2a'].add_summand(duration_variable, -1, time_indices[0:-1])  # onHours(t-1)
        eqsOwner.model.eqs[aLabel + '_constraint_2a'].add_constant(system_model.dt_in_hours[1:])  # dt(t)

        # 2b) eq:  onHours(t) - onHours(t-1)             >=  dt(t) - Big*(1-On(t)))
        #    eq: -onHours(t) + onHours(t-1) + On(t)*Big <= -dt(t) + Big
        # with Big = dt_in_hours_total # (Big = maxOnHours, should be usable, too!)
        Big = system_model.dt_in_hours_total
        eqsOwner.model.add_equation(Equation(aLabel + '_constraint_2b', eqsOwner, system_model, eqType='ineq'))
        eqsOwner.model.eqs[aLabel + '_constraint_2b'].add_summand(duration_variable, -1, time_indices[1:])  # onHours(t)
        eqsOwner.model.eqs[aLabel + '_constraint_2b'].add_summand(duration_variable, 1, time_indices[0:-1])  # onHours(t-1)
        eqsOwner.model.eqs[aLabel + '_constraint_2b'].add_summand(binary_variable, Big, time_indices[1:])  # on(t)
        eqsOwner.model.eqs[aLabel + '_constraint_2b'].add_constant(-system_model.dt_in_hours[1:] + Big)  # dt(t)

        # 3) check minimum_duration before switchOff-step
        # (last on-time period of timeseries is not checked and can be shorter)
        if minimum_duration is not None:
            # Note: switchOff-step is when: On(t)-On(t-1) == -1
            # eq:  onHours(t-1) >= minOnHours * -1 * [On(t)-On(t-1)]
            # eq: -onHours(t-1) - minimum_duration * On(t) + minimum_duration*On(t-1) <= 0
            eqsOwner.model.add_equation(Equation(aLabel + '_min', eqsOwner, system_model, eqType='ineq'))
            eqsOwner.model.eqs[aLabel + '_min'].add_summand(duration_variable, -1, time_indices[0:-1])  # onHours(t-1)
            eqsOwner.model.eqs[aLabel + '_min'].add_summand(binary_variable, -1 * minimum_duration.active_data, time_indices[1:])  # on(t)
            eqsOwner.model.eqs[aLabel + '_min'].add_summand(binary_variable, minimum_duration.active_data, time_indices[0:-1])  # on(t-1)

        # TODO: Maximum Duration?? Is this not modeled yet?!!

        # 4) first index:
        #    eq: onHours(t=0)= dt(0) * On(0)
        firstIndex = time_indices[0]  # only first element
        eqsOwner.model.add_equation(Equation(aLabel + '_firstTimeStep', eqsOwner, system_model))
        eqsOwner.model.eqs[aLabel + '_firstTimeStep'].add_summand(duration_variable, 1, firstIndex)
        eqsOwner.model.eqs[aLabel + '_firstTimeStep'].add_summand(binary_variable, -1 * system_model.dt_in_hours[firstIndex], firstIndex)

    def add_switch_constraints(self, system_model: SystemModel, time_indices: Union[list[int], range]):
        # % Schaltänderung aus On-Variable
        # % SwitchOn(t)-SwitchOff(t) = On(t)-On(t-1) 

        self.model.add_equation(Equation('SwitchOnOff_andOn', self, system_model))
        self.model.eqs['SwitchOnOff_andOn'].add_summand(self.model.variables['switchOn'], 1, time_indices[1:])  # SwitchOn(t)
        self.model.eqs['SwitchOnOff_andOn'].add_summand(self.model.variables['switchOff'], -1, time_indices[1:])  # SwitchOff(t)
        self.model.eqs['SwitchOnOff_andOn'].add_summand(self.model.variables['on'], -1, time_indices[1:])  # On(t)
        self.model.eqs['SwitchOnOff_andOn'].add_summand(self.model.variables['on'], +1, time_indices[0:-1])  # On(t-1)

        ## Ersten Wert SwitchOn(t=1) bzw. SwitchOff(t=1) festlegen
        # eq: SwitchOn(t=1)-SwitchOff(t=1) = On(t=1)- ValueBeforeBeginOfTimeSeries;      

        self.model.add_equation(Equation('SwitchOnOffAtFirstTime', self, system_model))
        firstIndex = time_indices[0]  # nur erstes Element!
        self.model.eqs['SwitchOnOffAtFirstTime'].add_summand(self.model.variables['switchOn'], 1, firstIndex)
        self.model.eqs['SwitchOnOffAtFirstTime'].add_summand(self.model.variables['switchOff'], -1, firstIndex)
        self.model.eqs['SwitchOnOffAtFirstTime'].add_summand(self.model.variables['on'], -1, firstIndex)
        # eq_SwitchOnOffAtFirstTime.add_constant(-on_valuesBefore[-1]) # letztes Element der Before-Werte nutzen,  Anmerkung: wäre besser auf lhs aufgehoben
        self.model.eqs['SwitchOnOffAtFirstTime'].add_constant(
            -self.model.variables['on'].before_value)  # letztes Element der Before-Werte nutzen,  Anmerkung: wäre besser auf lhs aufgehoben

        ## Entweder SwitchOff oder SwitchOn
        # eq: SwitchOn(t) + SwitchOff(t) <= 1 

        self.model.add_equation(Equation('SwitchOnOrSwitchOff', self, system_model, eqType='ineq'))
        self.model.eqs['SwitchOnOrSwitchOff'].add_summand(self.model.variables['switchOn'], 1)
        self.model.eqs['SwitchOnOrSwitchOff'].add_summand(self.model.variables['switchOff'], 1)
        self.model.eqs['SwitchOnOrSwitchOff'].add_constant(1)

        ## Anzahl Starts:
        # eq: nrSwitchOn = sum(SwitchOn(t))  

        self.model.add_equation(Equation('NrSwitchOn', self, system_model))
        self.model.eqs['NrSwitchOn'].add_summand(self.model.variables['nrSwitchOn'], 1)
        self.model.eqs['NrSwitchOn'].add_summand(self.model.variables['nrSwitchOn'], -1, as_sum=True)

    def add_share_to_globals(self, global_comp: Global, system_model: SystemModel):

        shareHolder = self.owner
        # Anfahrkosten:
        if self.switch_on_effects is not None:  # and any(self.switch_on_effects.active_data != 0):
            global_comp.add_share_to_operation('switch_on_effects', shareHolder, self.model.variables['switchOn'], self.switch_on_effects, 1)
        # Betriebskosten:
        if self.running_hour_effects is not None:  # and any(self.running_hour_effects):
            global_comp.add_share_to_operation('running_hour_effects', shareHolder, self.model.variables['on'],
                                               self.running_hour_effects, system_model.dt_in_hours)
            # global_comp.costsOfOperating_eq.add_summand(self.model.var_on, np.multiply(self.running_hour_effects.active_data, model.dt_in_hours))# np.multiply = elementweise Multiplikation


# TODO: als Feature_TSShareSum
class Feature_ShareSum(Feature):

    def __init__(self,
                 label: str,
                 owner: Element,
                 shares_are_time_series: bool,
                 total_max: Optional[Skalar] = None,
                 total_min: Optional[Skalar] = None,
                 max_per_hour: Optional[Numeric] = None,
                 min_per_hour: Optional[Numeric] = None):
        '''
        shares_are_time_series = True :
          Output: 
            var_all (TS), var_sum
          variables:
            sum_TS (Zeitreihe)
            sum    (Skalar)
          Equations: 
            eq_sum_TS : sum_TS = sum(share_TS_i) # Zeitserie
            eq_sum    : sum    = sum(sum_TS(t)) # skalar

        # shares_are_time_series = False:
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
        shares_are_time_series : TYPE
            DESCRIPTION.
        total_max : TYPE, optional
            DESCRIPTION. The default is None.
        total_min : TYPE, optional
            DESCRIPTION. The default is None.
        max_per_hour : scalar or list(TS) (if shares_are_time_series=True)
            maximum value per hour of shareSum;
            only usable if shares_are_time_series=True
        min_per_hour : scalar or list(TS) (if shares_are_time_series=True)
            minimum value per hour of shareSum of each timestep    
            only usable if shares_are_time_series=True

        '''
        super().__init__(label, owner)
        self.shares_are_time_series = shares_are_time_series
        self.total_max = total_max
        self.total_min = total_min
        max_min_per_hour_is_used = (max_per_hour is not None) or (min_per_hour is not None)
        if max_min_per_hour_is_used and not self.shares_are_time_series:
            raise Exception('max_per_hour or min_per_hour can only be used, if shares_are_time_series==True!')
        self.max_per_hour = None if (max_per_hour is None) else TimeSeries('max_per_hour', max_per_hour, self)
        self.min_per_hour = None if (min_per_hour is None) else TimeSeries('min_per_hour', min_per_hour, self)
        self.shares = FeatureShares('shares', self)

    def declare_vars_and_eqs(self, system_model: SystemModel):
        super().declare_vars_and_eqs(system_model)
        self.shares.declare_vars_and_eqs(system_model)

        # TODO: summe auch über Bus abbildbar!
        #   -> aber Beachte Effekt ist nicht einfach gleichzusetzen mit Flow, da hier eine Menge z.b. in € im Zeitschritt übergeben wird
        # variable für alle TS zusammen (TS-Summe):
        if self.shares_are_time_series:
            lb_TS = None if (self.min_per_hour is None) else np.multiply(self.min_per_hour.active_data, system_model.dt_in_hours)
            ub_TS = None if (self.max_per_hour is None) else np.multiply(self.max_per_hour.active_data, system_model.dt_in_hours)
            self.model.add_variable(VariableTS('sum_TS', system_model.nrOfTimeSteps, self, system_model, lower_bound= lb_TS, upper_bound= ub_TS))  # TS

        # Variable für Summe (Skalar-Summe):
        self.model.add_variable(Variable('sum', 1, self, system_model, lower_bound=self.total_min, upper_bound=self.total_max))  # Skalar

        # Gleichungen schon hier definiert, damit andere Elemente beim modeling Beiträge eintragen können:
        if self.shares_are_time_series:
            self.model.add_equation(Equation('bilanz', self, system_model))
        self.model.add_equation(Equation('sum', self, system_model))

    def do_modeling(self, system_model: SystemModel, time_indices: Union[list[int], range]):
        self.shares.do_modeling(system_model, time_indices)
        if self.shares_are_time_series:
            # eq: sum_TS = sum(share_TS_i) # TS
            self.model.eqs['bilanz'].add_summand(self.model.variables['sum_TS'], -1)
            # eq: sum = sum(sum_TS(t)) # skalar
            self.model.eqs['sum'].add_summand(self.model.variables['sum_TS'], 1, as_sum=True)
            self.model.eqs['sum'].add_summand(self.model.variables['sum'], -1)
        else:
            # eq: sum = sum(share_i) # skalar
            self.model.eqs['sum'].add_summand(self.model.variables['sum'], -1)

    def add_constant_share(self,
                           name_of_share: Optional[str],
                           share_holder: Element,
                           factor1: Numeric_TS,
                           factor2: Numeric_TS):
        """
        Beiträge zu Effekt_Sum registrieren

        Parameters
        ----------
        factor1 : TS oder skalar, bei shares_are_time_series=False nur skalar
            DESCRIPTION.
        factor2 : TS oder skalar, bei shares_are_time_series=False nur skalar
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.add_share(self, name_of_share, share_holder, None, factor1, factor2)

    def add_variable_share(self,
                           name_of_share: Optional[str],
                           share_holder: Element,
                           variable: Variable,
                           factor1: Numeric_TS,
                           factor2: Numeric_TS):  # if variable = None, then fix Share
        if variable is None:
            raise Exception('add_variable_share() needs variable as input or use add_constant_share() instead')
        self.add_share(name_of_share, share_holder, variable, factor1, factor2)

    # allgemein variable oder constant (dann variable = None):
    # if variable = None, then fix Share    
    def add_share(self,
                  name_of_share: Optional[str],
                  share_holder: Element,
                  variable: Optional[Variable],
                  factor1: Numeric_TS,
                  factor2: Numeric_TS):
        """
        share to a sum

        Parameters
        ----------
        variable : TYPE
            DESCRIPTION.
        factor1 : TYPE
            DESCRIPTION.
        factor2 : TYPE
            DESCRIPTION.
        name_of_share : str or None
            None, if it is not a real share (i.g. -1*var_sum )

        Returns
        -------
        None.
        """
        # var and eq for publishing share-values in results:                    
        if name_of_share is not None:
            eq_oneShare = self.shares.get_equation_of_new_share(name_of_share, share_holder, self.system_model)

        if self.shares_are_time_series:

            # Falls TimeSeries, Daten auslesen:
            if isinstance(factor1, TimeSeries): factor1 = factor1.active_data
            if isinstance(factor2, TimeSeries): factor2 = factor2.active_data

            factorOfSummand = np.multiply(factor1, factor2)  # np.multiply = elementweise Multiplikation
            ## Share zu TS-equation hinzufügen:
            # if constant share:      
            if variable is None:
                self.model.eqs['bilanz'].add_constant(-1 * factorOfSummand)  # share in global
                if name_of_share is not None:
                    eq_oneShare.add_constant(-1 * sum(factorOfSummand))  # share itself
            # if variable share:
            else:
                self.model.eqs['bilanz'].add_summand(variable, factorOfSummand)  # share in global
                if name_of_share is not None:
                    eq_oneShare.add_summand(variable, factorOfSummand, as_sum=True)  # share itself


        else:
            assert (not (isinstance(factor1, TimeSeries))) & (not (isinstance(factor2,
                                                                              TimeSeries))), 'factor1 und factor2 müssen Skalare sein, da shareSum ' + self.label + 'skalar ist'
            factorOfSummand = factor1 * factor2
            ## Share zu skalar-equation hinzufügen:
            # if constant share:
            if variable is None:
                self.model.eqs['sum'].add_constant(-1 * factorOfSummand)  # share in global
                if name_of_share is not None:
                    eq_oneShare.add_constant(-1 * factorOfSummand)  # share itself
            # if variable share:
            else:
                self.model.eqs['sum'].add_summand(variable, factorOfSummand)  # share in global
                if name_of_share is not None:
                    eq_oneShare.add_summand(variable, factorOfSummand)  # share itself


class FeatureShares(Feature):
    """
    used to list all shares
    (owner is Feature_ShareSum)
    """

    def __init__(self, label: str, owner: Feature_ShareSum):
        super().__init__(label, owner)

    def do_modeling(self, system_model: SystemModel, time_indices: Union[list[int], range]):
        pass

    def declare_vars_and_eqs(self, system_model: SystemModel):
        super().declare_vars_and_eqs(system_model)

    def get_equation_of_new_share(self,
                                  name_of_share: str,
                                  share_holder: Element,
                                  system_model: SystemModel) -> Equation:
        """
        creates variable and equation for every share for 
        publishing share-values in results

        Parameters
        ----------
        nameOfShare : str

        Returns
        -------
        eq_oneShare : Equation
        """
        full_name_of_share = share_holder.label_full + '_' + name_of_share
        self.model.add_variable(Variable(full_name_of_share, 1, self, system_model))  # Skalar
        self.model.add_equation(Equation(full_name_of_share, self, system_model))
        self.model.eqs[full_name_of_share].add_summand(self.model.variables[full_name_of_share], -1)

        return self.model.eqs[full_name_of_share]


class FeatureInvest(Feature):
    # -> var_name            : z.B. "size", "capacity_inFlowHours"
    # -> investment_size : size, capacity_inFlowHours, ...
    # -> defining_variable         : z.B. flow.model.var_val
    # -> min_rel,max_rel     : ist relatives Min,Max der defining_variable bzgl. investment_size

    @property
    def on_variable_is_used(self):  # existiert On-variable
        return True if self.featureOn is not None and self.featureOn.use_on else False

    def __init__(self,
                 name_of_investment_size: str,
                 owner: Element,
                 invest_parameters: InvestParameters,
                 min_rel: TimeSeries,
                 max_rel: TimeSeries,
                 val_rel: Optional[TimeSeries],
                 investment_size: Optional[Skalar],
                 featureOn: Optional[FeatureOn] = None):
        """
        Parameters
        ----------
        name_of_investment_size : TYPE
            DESCRIPTION.
        owner : TYPE
            owner of this Element
        invest_parameters : InvestParameters
            arguments for modeling
        min_rel : scalar or TS
            given min_rel of defining_variable
            (min = min_rel * investmentSize)
        max_rel : scalar or TS        
            given max_rel of defining_variable
            (max = max_rel * investmentSize)
        val_rel : scalar or TS
            given val_rel of defining_variable
            (val = val_rel * investmentSize)
        investment_size : scalar or None
            value of fixed investmentSize (None if no fixed investmentSize)
            Flow: investmentSize=size
            Storage: investmentSize =
        featureOn : FeatureOn
            FeatureOn of the defining_variable (if it has a cFeatureOn)

        Returns
        -------
        None.
        """

        super().__init__('invest', owner)
        self.name_of_investment_size = name_of_investment_size
        self.owner = owner
        self.invest_parameters = invest_parameters
        self.max_rel = max_rel
        self.min_rel = min_rel
        self.val_rel = val_rel
        self.investment_size = investment_size  # nominalValue
        self.featureOn = featureOn
        self.check_plausibility()

        self.defining_variable = None
        self.defining_on_variable = None

        self.featureLinearSegments = None   # segmented investcosts:
        if self.invest_parameters.effects_in_segments is not None:
            self.featureLinearSegments = FeatureLinearSegmentVars('segmentedInvestcosts', self)
            self.sub_elements.append(self.featureLinearSegments)

    def check_plausibility(self):
        # Check investment_size:
        # todo: vielleicht ist es aber auch ok, wenn der size belegt ist und einfach nicht genutzt wird....
        if self.invest_parameters.fixed_size:
            if self.investment_size in (None, 0):
                raise ValueError(f'In {self.name_of_investment_size=} in {self.owner.label=}: '
                                 f'investment_size must be > 0 if an Investment with a fixed size is made')
        elif self.investment_size is not None:
            raise Exception(f'!{self.name_of_investment_size=} of {self.owner.label_full=} '
                            f'must be None if investment_size is variable')

    def bounds_of_defining_variable(self) -> Tuple[Optional[Numeric], Optional[Numeric], Optional[Numeric]]:

        if self.val_rel is not None:   # Wenn fixer relativer Lastgang:
            # max_rel = min_rel = val_rel !
            min_rel = self.val_rel.active_data
            max_rel = self.val_rel.active_data
        else:
            min_rel = self.min_rel.active_data
            max_rel = self.max_rel.active_data

        on_is_used = self.featureOn is not None and self.featureOn.use_on
        on_is_used_and_val_is_not_fix = (self.val_rel is None) and on_is_used

        # min-Wert:
        if self.invest_parameters.optional or on_is_used_and_val_is_not_fix:
            lower_bound = 0  # can be zero (if no invest) or can switch off
        else:
            if self.invest_parameters.fixed_size:
                lower_bound = min_rel * self.investment_size  # immer an
            else:
                lower_bound = min_rel * self.invest_parameters.minimum_size  # investSize is variabel

        #  max-Wert:
        if self.invest_parameters.fixed_size:
            upper_bound = max_rel * self.investment_size
        else:
            upper_bound = max_rel * self.invest_parameters.maximum_size  # investSize is variabel

        # upper_bound und lower_bound gleich, dann fix:
        if np.all(upper_bound == lower_bound):  # np.all -> kann listen oder werte vergleichen
            fix_value = upper_bound
            upper_bound = None
            lower_bound = None
        else:
            fix_value = None

        return lower_bound, upper_bound, fix_value

    # Variablenreferenz kann erst später hinzugefügt werden, da erst später erstellt:
    # todo-> abändern durch Variable-Dummies
    def set_defining_variables(self, defining_variable: VariableTS, defining_on_variable: Optional[VariableTS]):
        self.defining_variable = defining_variable
        self.defining_on_variable = defining_on_variable

    def declare_vars_and_eqs(self, system_model: SystemModel):

        # Determine lower and upper bounds for investment size
        lower_bound = 0 if self.invest_parameters.optional else (
            self.investment_size if self.invest_parameters.fixed_size else self.invest_parameters.minimum_size
        )
        upper_bound = self.investment_size if self.invest_parameters.fixed_size else self.invest_parameters.maximum_size

        # Define var_investmentSize
        if lower_bound == upper_bound:
            self.model.add_variable(Variable(self.name_of_investment_size, 1, self, system_model,
                                                     value=lower_bound))
        else:
            self.model.add_variable(Variable(self.name_of_investment_size, 1, self, system_model,
                                                     lower_bound=lower_bound, upper_bound=upper_bound))

        # Define var_isInvested if investment is optional
        if self.invest_parameters.optional:
            self.model.add_variable(Variable('isInvested', 1, self, system_model, is_binary=True))

        # Define cost segments if featureLinearSegments is present
        if self.featureLinearSegments is not None:
            self._define_cost_segments(system_model)
            self.featureLinearSegments.declare_vars_and_eqs(system_model)

    # definingInvestcosts in Segments:
    def _define_cost_segments(self, system_model: SystemModel):
        invest_size_segments, effect_value_segments = self.invest_parameters.effects_in_segments
        effect_value_segments = as_effect_dict(effect_value_segments)

        ## 1. create segments for investSize and every effect##
        ## 1.a) add investSize-Variablen-Segmente: ##
        segments_of_variables = {self.model.variables[self.name_of_investment_size]: invest_size_segments}  # i.e. {var_investSize: [0,5, 5,20]}

        ## 1.b) je Effekt -> new Variable und zugehörige Segmente ##
        self.model.var_list_investCoqsts_segmented = []
        self.investVar_effect_dict = {}  # benötigt
        for aEffect, aSegmentCosts in effect_value_segments.items():
            variable_for_segmented_invest_effect = self._create_variable_for_segmented_invest_effect(aEffect, system_model)
            segment = {variable_for_segmented_invest_effect: aSegmentCosts}  # i.e. {var_investCosts_segmented_costs : [0,10, 10,30]}
            segments_of_variables.update(segment)  #
            self.investVar_effect_dict.update({aEffect: variable_for_segmented_invest_effect})

        ## 2. on_var: ##
        if self.invest_parameters.optional:
            var_isInvested = self.model.variables['isInvested']
        else:
            var_isInvested = None

        ## 3. transfer segments_of_variables to FeatureLinearSegmentVars: ##
        self.featureLinearSegments.define_segments(segments_of_variables, binary_variable=var_isInvested,
                                                   vars_for_check=list(segments_of_variables.keys()))

    def _create_variable_for_segmented_invest_effect(self, aEffect, system_model: SystemModel):
        # define cost-Variable (=costs through segmented Investsize-costs):
        from flixOpt.elements import Effect
        if isinstance(aEffect, Effect):
            aStr = aEffect.label
        elif aEffect is None:
            aStr = system_model.system.effects.standard_effect.label  # Standard-Effekt
        else:
            raise Exception('Given effect (' + str(aEffect) + ') is not an effect!')
        # new variable, i.e for costs, CO2,... :
        var_investForEffect = Variable('investCosts_segmented_' + aStr, 1, self, system_model, lower_bound=0)
        self.model.add_variable(var_investForEffect)
        return var_investForEffect

    def do_modeling(self, system_model: SystemModel, time_indices: Union[list[int], range]):
        assert self.defining_variable is not None, 'set_defining_variables() still not executed!'
        # wenn var_isInvested existiert:    
        if self.invest_parameters.optional:
            self._add_defining_var_isInvested(system_model)

        # Bereich von defining_variable in Abh. von var_investmentSize:

        # Wenn fixer relativer Lastgang:
        if self.val_rel is not None:
            self._add_fixEq_of_definingVar_with_var_investmentSize(system_model)
        # Wenn nicht fix:
        else:
            self._add_max_min_of_definingVar_with_var_investmentSize(system_model)

            # if linear Segments defined:
        if self.featureLinearSegments is not None:
            self.featureLinearSegments.do_modeling(system_model, time_indices)

    def _add_fixEq_of_definingVar_with_var_investmentSize(self, system_model: SystemModel):

        ## Gleichung zw. DefiningVar und Investgröße:    
        # eq: defining_variable(t) = var_investmentSize * val_rel
        self.model.add_equation(Equation('fix_via_InvestmentSize', self, system_model, 'eq'))
        self.model.eqs['fix_via_InvestmentSize'].add_summand(self.defining_variable, 1)
        self.model.eqs['fix_via_InvestmentSize'].add_summand(self.model.variables[self.name_of_investment_size], np.multiply(-1, self.val_rel.active_data))

    def _add_max_min_of_definingVar_with_var_investmentSize(self, system_model: SystemModel):

        ## 1. Gleichung: Maximum durch Investmentgröße ##     
        # eq: defining_variable(t) <=                var_investmentSize * max_rel(t)
        # eq: P(t) <= max_rel(t) * P_inv    
        self.model.add_equation(Equation('max_via_InvestmentSize', self, system_model, 'ineq'))
        self.model.eqs['max_via_InvestmentSize'].add_summand(self.defining_variable, 1)
        # TODO: Changed by FB
        # self.eq_max_via_investmentSize.add_summand(self.model.var_investmentSize, np.multiply(-1, self.max_rel.active_data))
        self.model.eqs['max_via_InvestmentSize'].add_summand(self.model.variables[self.name_of_investment_size], np.multiply(-1, self.max_rel.data))
        # TODO: BUGFIX: Here has to be active_data, but it throws an error for storages (length)
        # TODO: Changed by FB

        ## 2. Gleichung: Minimum durch Investmentgröße ##        

        # Glg nur, wenn nicht Kombination On und fixed:
        if not self.on_variable_is_used or not self.invest_parameters.fixed_size:
            self.model.add_equation(Equation('min_via_investmentSize', self, system_model, 'ineq'))

        if self.on_variable_is_used:
            # Wenn InvestSize nicht fix, dann weitere Glg notwendig für Minimum (abhängig von var_investSize)
            if not self.invest_parameters.fixed_size:
                # eq: defining_variable(t) >= Big * (On(t)-1) + investment_size * min_rel(t)
                #     ... mit Big = max(min_rel*P_inv_max, epsilon)
                # (P < min_rel*P_inv -> On=0 | On=1 -> P >= min_rel*P_inv)

                # äquivalent zu:.
                # eq: - defining_variable(t) + Big * On(t) + min_rel(t) * investment_size <= Big

                Big = helpers.max_args(self.min_rel.active_data * self.invest_parameters.maximum_size, system_model.epsilon)

                self.model.eqs['min_via_investmentSize'].add_summand(self.defining_variable, -1)
                self.model.eqs['min_via_investmentSize'].add_summand(self.defining_on_variable, Big)  # übergebene On-Variable
                self.model.eqs['min_via_investmentSize'].add_summand(self.model.variables[self.name_of_investment_size], self.min_rel.active_data)
                self.model.eqs['min_via_investmentSize'].add_constant(Big)
                # Anmerkung: Glg bei Spezialfall min_rel = 0 redundant zu FeatureOn-Glg.
            else:
                pass  # Bereits in FeatureOn mit P>= On(t)*Min ausreichend definiert
        else:
            # eq: defining_variable(t) >= investment_size * min_rel(t)

            self.model.eqs['min_via_investmentSize'].add_summand(self.defining_variable, -1)
            self.model.eqs['min_via_investmentSize'].add_summand(self.model.variables[self.name_of_investment_size], self.min_rel.active_data)

    def _add_defining_var_isInvested(self, system_model: SystemModel):
        if self.invest_parameters.fixed_size:
            # eq: investment_size = isInvested * size
            self.model.add_equation(Equation('isInvested_constraint_1', self, system_model, 'eq'))
            self.model.eqs['isInvested_constraint_1'].add_summand(self.model.variables[self.name_of_investment_size], -1)
            self.model.eqs['isInvested_constraint_1'].add_summand(self.model.variables['isInvested'], self.investment_size)
        else:
            ## 1. Gleichung (skalar):            
            # eq1: P_invest <= isInvested * investSize_max
            # (isInvested = 0 -> P_invest=0  |  P_invest>0 -> isInvested = 1 ->  P_invest < investSize_max )   

            self.model.add_equation(Equation('isInvested_constraint_1', self, system_model, 'ineq'))
            self.model.eqs['isInvested_constraint_1'].add_summand(self.model.variables[self.name_of_investment_size], 1)
            self.model.eqs['isInvested_constraint_1'].add_summand(self.model.variables['isInvested'],
                                             np.multiply(-1, self.invest_parameters.maximum_size))  # Variable ist Skalar!

            ## 2. Gleichung (skalar):                  
            # eq2: P_invest  >= isInvested * max(epsilon, investSize_min)
            # (isInvested = 1 -> P_invest>0  |  P_invest=0 -> isInvested = 0)
            self.model.add_equation(Equation('isInvested_constraint_2', self, system_model, 'ineq'))
            self.model.eqs['isInvested_constraint_2'].add_summand(self.model.variables[self.name_of_investment_size], -1)
            self.model.eqs['isInvested_constraint_2'].add_summand(self.model.variables['isInvested'], max(system_model.epsilon, self.invest_parameters.minimum_size))

    def add_share_to_globals(self, global_comp: Global, system_model: SystemModel):

        # # fix_effects:
        if self.invest_parameters.fix_effects is not None and self.invest_parameters.fix_effects != 0:
            if self.invest_parameters.optional:
                # fix Share to InvestCosts: 
                # share: + isInvested * fix_effects
                global_comp.add_share_to_invest('fix_effects', self.owner, self.model.var_isInvested, self.invest_parameters.fix_effects, 1)
            else:
                # share: + fix_effects
                global_comp.add_constant_share_to_invest('fix_effects', self.owner, self.invest_parameters.fix_effects,
                                                         1)  # fester Wert hinufügen

        # # divest_effects:
        if self.invest_parameters.divest_effects is not None and self.invest_parameters.divestCost != 0:
            if self.invest_parameters.optional:
                # fix Share to InvestCosts: 
                # share: [(1- isInvested) * divest_effects]
                # share: [divest_effects - isInvested * divest_effects]
                # 1. part of share [+ divest_effects]:
                global_comp.add_constant_share_to_invest('divest_effects', self.owner, self.invest_parameters.divest_effects, 1)
                # 2. part of share [- isInvested * divest_effects]:
                global_comp.add_share_to_invest('divestCosts_cancellation', self.owner, self.model.variables['isInvested],'],
                                                self.invest_parameters.divest_effects, -1)
                # TODO : these 2 parts should be one share!

        # # specific_effects:
        if self.invest_parameters.specific_effects is not None:
            # share: + investment_size (=var)   * specific_effects
            global_comp.add_share_to_invest('specific_effects', self.owner, self.model.variables[self.name_of_investment_size],
                                            self.invest_parameters.specific_effects, 1)

        # # segmentedCosts:                                        
        if self.featureLinearSegments is not None:
            for effect, var_investSegs in self.investVar_effect_dict.items():
                global_comp.add_share_to_invest('linearSegments', self.owner, var_investSegs, {effect: 1}, 1)
