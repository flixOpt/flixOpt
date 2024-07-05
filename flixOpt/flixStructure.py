# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:40:23 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import math
import time
import textwrap
import pprint
from typing import List, Set, Tuple, Dict, Union, Optional, Literal, TYPE_CHECKING
import logging

import numpy as np
import yaml  # (für json-Schnipsel-print)

from flixOpt import flixOptHelperFcts as helpers
from flixOpt.basicModeling import LinearModel, Variable, VariableTS, Equation  # Modelliersprache
from flixOpt.flixBasics import TimeSeries, Numeric, Numeric_TS, Skalar, as_effect_dict, as_effect_dict_with_ts
from flixOpt.flixBasicsPublic import InvestParameters, TimeSeriesRaw
if TYPE_CHECKING:  # for type checking and preventing circular imports
    from flixFeatures import FeatureInvest

log = logging.getLogger(__name__)


class SystemModel(LinearModel):
    '''
    Hier kommen die ModellingLanguage-spezifischen Sachen rein
    '''

    @property
    def infos(self):
        infos = super().infos
        # Hauptergebnisse:
        infos['main_results'] = self.main_results_str
        # unten dran den vorhanden rest:
        infos.update(self._infos)  # da steht schon zeug drin
        return infos

    def __init__(self,
                 label: str,
                 modeling_language: Literal['pyomo', 'cvxpy'],
                 system,
                 time_indices: Union[List[int], range],
                 TS_explicit=None):
        super().__init__(label, modeling_language)
        self.system: System = system  # energysystem (wäre Attribut von cTimePeriodModel)
        self.time_indices = time_indices
        self.nrOfTimeSteps = len(time_indices)
        self.TS_explicit = TS_explicit  # für explizite Vorgabe von Daten für TS {TS1: data, TS2:data,...}
        self.models_of_elements: Dict = {}  # dict with all ElementModel's od Elements in System

        self.before_values = None  # hier kommen, wenn vorhanden gegebene Before-Values rein (dominant ggü. before-Werte des energysystems)
        # Zeitdaten generieren:
        (self.time_series, self.time_series_with_end, self.dt_in_hours, self.dt_in_hours_total) = (
            system.get_time_data_from_indices(time_indices))

    # register ModelingElements and belonging Mod:
    def register_element_with_model(self, aModelingElement, aMod):
        # allocation Element -> model
        self.models_of_elements[aModelingElement] = aMod  # aktuelles model hier speichern

    # override:
    def characterize_math_problem(self):  # overriding same method in motherclass!
        # Systembeschreibung abspeichern: (Beachte: system_model muss aktiviert sein)
        # self.system.activate_model()
        self._infos['str_Eqs'] = self.system.description_of_equations()
        self._infos['str_Vars'] = self.system.description_of_variables()

    # 'gurobi'
    def solve(self,
              mip_gap: float = 0.02,
              time_limit_seconds: int = 3600,
              solver_name: str = 'highs',
              solver_output_to_console: bool = True,
              excess_threshold: Union[int, float] = 0.1,
              logfile_name: str = 'solver_log.log',
              **kwargs):
        '''
        Parameters
        ----------
        mip_gap : TYPE, optional
            DESCRIPTION. The default is 0.02.
        time_limit_seconds : TYPE, optional
            DESCRIPTION. The default is 3600.
        solver : TYPE, optional
            DESCRIPTION. The default is 'cbc'.
        solver_output_to_console : TYPE, optional
            DESCRIPTION. The default is True.
        excess_threshold : float, positive!
            threshold for excess: If sum(Excess)>excess_threshold a warning is raised, that an excess occurs
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        main_results_str : TYPE
            DESCRIPTION.

        '''

        # check valid solver options:
        if len(kwargs) > 0:
            for key in kwargs.keys():
                if key not in ['threads']:
                    raise Exception(
                        'no allowed arguments for kwargs: ' + str(key) + '(all arguments:' + str(kwargs) + ')')

        print('')
        print('##############################################################')
        print('##################### solving ################################')
        print('')

        self.printNoEqsAndVars()

        super().solve(mip_gap, time_limit_seconds, solver_name, solver_output_to_console, logfile_name, **kwargs)

        if solver_name == 'gurobi':
            termination_message = self.solver_results['Solver'][0]['Termination message']
        elif solver_name == 'glpk':
            termination_message = self.solver_results['Solver'][0]['Status']
        else:
            termination_message = f'not implemented for solver "{solver_name}" yet'
        print(f'Termination message: "{termination_message}"')

        print('')
        # Variablen-Ergebnisse abspeichern:
        # 1. dict:
        (self.results, self.results_var) = self.system.get_results_after_solve()
        # 2. struct:
        self.results_struct = helpers.createStructFromDictInDict(self.results)

        print('##############################################################')
        print('################### finished #################################')
        print('')
        for aEffect in self.system.global_comp.listOfEffectTypes:
            print(aEffect.label + ' in ' + aEffect.unit + ':')
            print('  operation: ' + str(aEffect.operation.model.var_sum.result))
            print('  invest   : ' + str(aEffect.invest.model.var_sum.result))
            print('  sum      : ' + str(aEffect.all.model.var_sum.result))

        print('SUM              : ' + '...todo...')
        print('penaltyCosts     : ' + str(self.system.global_comp.penalty.model.var_sum.result))
        print('––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
        print('Result of Obj : ' + str(self.objective_result))
        try:
            print('lower bound   : ' + str(self.solver_results['Problem'][0]['Lower bound']))
        except:
            print
        print('')
        for aBus in self.system.buses:
            if aBus.with_excess:
                if any(self.results[aBus.label]['excess_input'] > 1e-6) or any(
                        self.results[aBus.label]['excess_output'] > 1e-6):
                    # if any(aBus.excess_input.result > 0) or any(aBus.excess_output.result > 0):
                    print('!!!!! Attention !!!!!')
                    print('!!!!! Exzess.Value in Bus ' + aBus.label + '!!!!!')

                    # if penalties exist
        if self.system.global_comp.penalty.model.var_sum.result > 10:
            print('Take care: -> high penalty makes the used mip_gap quite high')
            print('           -> real costs are not optimized to mip_gap')

        print('')
        print('##############################################################')

        # str description of results:
        # nested fct:
        def _getMainResultsAsStr():
            main_results_str = {}

            aEffectDict = {}
            main_results_str['Effects'] = aEffectDict
            for aEffect in self.system.global_comp.listOfEffectTypes:
                aDict = {}
                aEffectDict[aEffect.label + ' [' + aEffect.unit + ']'] = aDict
                aDict['operation'] = str(aEffect.operation.model.var_sum.result)
                aDict['invest'] = str(aEffect.invest.model.var_sum.result)
                aDict['sum'] = str(aEffect.all.model.var_sum.result)
            main_results_str['penaltyCosts'] = str(self.system.global_comp.penalty.model.var_sum.result)
            main_results_str['Result of Obj'] = self.objective_result
            if self.solver_name =='highs':
                main_results_str['lower bound'] = self.solver_results.best_objective_bound
            else:
                main_results_str['lower bound'] = self.solver_results['Problem'][0]['Lower bound']
            busesWithExcess = []
            main_results_str['busesWithExcess'] = busesWithExcess
            for aBus in self.system.buses:
                if aBus.with_excess:
                    if sum(self.results[aBus.label]['excess_input']) > excess_threshold or sum(
                            self.results[aBus.label]['excess_output']) > excess_threshold:
                        busesWithExcess.append(aBus.label)

            aDict = {'invested': {},
                     'not invested': {}
                     }
            main_results_str['Invest-Decisions'] = aDict
            for aInvestFeature in self.system.invest_features:
                investValue = aInvestFeature.model.var_investmentSize.result
                investValue = float(investValue)  # bei np.floats Probleme bei Speichern
                # umwandeln von numpy:
                if isinstance(investValue, np.ndarray):
                    investValue = investValue.tolist()
                label = aInvestFeature.owner.label_full
                if investValue > 1e-3:
                    aDict['invested'][label] = investValue
                else:
                    aDict['not invested'][label] = investValue

            return main_results_str

        self.main_results_str = _getMainResultsAsStr()
        helpers.printDictAndList(self.main_results_str)


class Element:
    """
    Element mit Variablen und Gleichungen
    -> besitzt Methoden, die jede Kindklasse ergänzend füllt:
    1. Element.finalize()          --> Finalisieren der Modell-Beschreibung (z.B. notwendig, wenn Bezug zu Elementen, die bei __init__ noch gar nicht bekannt sind)
    2. Element.declare_vars_and_eqs() --> Variablen und Eqs definieren.
    3. Element.do_modeling()        --> Modellierung
    4. Element.add_share_to_globals() --> Beitrag zu Gesamt-Kosten
    """
    system_model: SystemModel

    new_init_args = ['label']
    not_used_args = []

    @classmethod
    def get_init_args(cls):
        '''
        diese (Klassen-)Methode holt aus dieser und den Kindklassen
        alle zulässigen Argumente der Kindklasse!
        '''

        ### 1. Argumente der Mutterklasse (rekursiv) ###
        # wird rekursiv aufgerufen bis man bei Mutter-Klasse cModelingElement ankommt.
        # nur bis zu cArgsClass zurück gehen:
        if hasattr(cls.__base__, 'get_init_args'):  # man könnte auch schreiben: if cls.__name__ == cArgsClass
            allArgsFromMotherClass = cls.__base__.get_init_args()  # rekursiv in Mutterklasse aufrufen

        # wenn cls.__base__ also bereits eine Ebene UNTER cArgsClass:
        else:
            allArgsFromMotherClass = []

            # checken, dass die zwei class-Atributes auch wirklich für jede Klasse (und nicht nur für Mutterklasse) existieren (-> die nimmt er sonst einfach automatisch)
        if (not ('not_used_args' in cls.__dict__)) | (not ('new_init_args' in cls.__dict__)):
            raise Exception(
                'class ' + cls.__name__ + ': you forgot to implement class attribute <not_used_args> or/and <new_int_args>')
        notTransferedMotherArgs = cls.not_used_args

        ### 2. Abziehen der nicht durchgereichten Argumente ###
        # delete not Transfered Args:
        allArgsFromMotherClass = [prop for prop in allArgsFromMotherClass if prop not in notTransferedMotherArgs]

        ### 3. Ergänzen der neuen Argumente ###
        myArgs = cls.new_init_args.copy()  # get all new arguments of __init__() (as a copy)
        # melt lists:
        myArgs.extend(allArgsFromMotherClass)
        return myArgs

    @property
    def label_full(self) -> str:  # standard-Funktion, wird von Kindern teilweise überschrieben
        return self.label  # eigtl später mal rekursiv: return self.owner.label_full + self.label

    @property  # sub_elements of all layers
    def all_sub_elements(self) -> list:  #TODO: List[Element] doesnt work...
        all_sub_elements = []  # wichtig, dass neues Listenobjekt!
        all_sub_elements += self.sub_elements
        for subElem in self.sub_elements:
            # all sub_elements of subElement hinzufügen:
            all_sub_elements += subElem.all_sub_elements
        return all_sub_elements

    # TODO: besser occupied_args
    def __init__(self, label: str, **kwargs):
        self.label = label
        self.TS_list: List[TimeSeries] = []  # = list with ALL timeseries-Values (--> need all classes with .trimTimeSeries()-Method, e.g. TimeSeries)

        self.sub_elements: List[Element] = []  # zugehörige Sub-ModelingElements
        self.system_model: Optional[SystemModel] = None  # hier kommt die aktive system_model rein
        self.model: Optional[ElementModel] = None  # hier kommen alle Glg und Vars rein

        # wenn hier kwargs auftauchen, dann wurde zuviel übergeben:
        if len(kwargs) > 0:
            raise Exception('class and its motherclasses have no allowed arguments for:' + str(kwargs)[:200])

    def __repr__(self):
        return f"<{self.__class__.__name__}> {self.label}"

    def __str__(self):
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
                    f"{textwrap.indent(remaining_data_str, ' ' * 3)}"
                    )

        return str_desc

    # activate inkl. sub_elements:
    def activate_system_model(self, system_model) -> None:
        for element in self.sub_elements:
            element.activate_system_model(system_model)  # inkl. sub_elements
        self.activate_system_model_for_me(system_model)

    # activate ohne SubElements!
    def activate_system_model_for_me(self, system_model) -> None:
        self.system_model = system_model
        self.model = system_model.models_of_elements[self]

    # 1.
    def finalize(self) -> None:
        for element in self.sub_elements:
            element.finalize()

    # 2.
    def create_new_model_and_activate_system_model(self, system_model: SystemModel) -> None:
        # print('new model for ' + self.label)
        # subElemente ebenso:
        element: Element
        for element in self.sub_elements:
            element.create_new_model_and_activate_system_model(system_model)  # rekursiv!

        # create model:
        model = ElementModel(self)
        # register model:
        system_model.register_element_with_model(self, model)

        self.activate_system_model_for_me(system_model)  # sub_elements werden bereits aktiviert über aElement.createNewMod...()

    # 3.
    def declare_vars_and_eqs(self, system_model) -> None:
        #   #   # Features preparing:
        #   # for aFeature in self.features:
        #   #   aFeature.declare_vars_and_eqs(model)
        pass

    # def do_modeling(self,model,time_indices):
    #   # for aFeature in self.features:
    #   aFeature.do_modeling(model, time_indices)

    # Ergebnisse als dict ausgeben:
    def get_results(self) -> Tuple[Dict, Dict]:
        aData = {}
        aVars = {}
        # 1. Unterelemente füllen (rekursiv!):
        for element in self.sub_elements:
            (aData[element.label], aVars[element.label]) = element.get_results()  # rekursiv

        # 2. Variablenwerte ablegen:
        aVar: Variable
        for aVar in self.model.variables:
            # print(aVar.label)
            aData[aVar.label] = aVar.result
            aVars[aVar.label] = aVar  # link zur Variable
            if aVar.is_binary and aVar.length > 1:
                # Bei binären Variablen zusätzlichen Vektor erstellen,z.B. a  = [0, 1, 0, 0, 1]
                #                                                       -> a_ = [nan, 1, nan, nan, 1]
                aData[aVar.label + '_'] = helpers.zerosToNans(aVar.result)
                aVars[aVar.label + '_'] = aVar  # link zur Variable

        # 3. Alle TS übergeben
        aTS: TimeSeries
        for aTS in self.TS_list:
            # print(aVar.label)
            aData[aTS.label] = aTS.data
            aVars[aTS.label] = aTS  # link zur Variable

            # 4. Attribut Group übergeben, wenn vorhanden
            aGroup: str
            if hasattr(self, 'group'):
                if self.group is not None:
                    aData["group"] = self.group
                    aVars["group"] = self.group

        return aData, aVars

    def description_of_equations(self):

        ## subelemente durchsuchen:
        subs = {}
        for aSubElement in self.sub_elements:
            subs[aSubElement.label] = aSubElement.description_of_equations()  # rekursiv
        ## Element:

        # wenn sub-eqs, dann dict:
        if not (subs == {}):
            eqsAsStr = {}
            eqsAsStr['_self'] = self.model.description_of_equations()  # zuerst eigene ...
            eqsAsStr.update(subs)  # ... dann sub-Eqs
        # sonst liste:
        else:
            eqsAsStr = self.model.description_of_equations()

        return eqsAsStr

    def description_of_variables(self) -> List:
        aList = []
        aList += self.model.description_of_variables()
        for aSubElement in self.sub_elements:
            aList += aSubElement.description_of_variables()  # rekursiv

        return aList

    def overview_of_eqs_and_vars(self) -> Dict[str, int]:
        aDict = {}
        aDict['no eqs'] = len(self.model.eqs)
        aDict['no eqs single'] = sum([eq.nr_of_single_equations for eq in self.model.eqs])
        aDict['no inEqs'] = len(self.model.ineqs)
        aDict['no inEqs single'] = sum([ineq.nr_of_single_equations for ineq in self.model.ineqs])
        aDict['no vars'] = len(self.model.variables)
        aDict['no vars single'] = sum([var.length for var in self.model.variables])
        return aDict


class ElementModel:
    '''
    is existing in every Element and owns eqs and vars of the activated calculation
    '''

    def __init__(self, element: Element):
        self.element = element
        # TODO: Dicts instead of Lists for referencing?
        self.variables = []
        self.eqs = []
        self.ineqs = []
        self.objective = None

    # Eqs, Ineqs und Objective als Str-Description:
    def description_of_equations(self) -> List:
        # Wenn Glg vorhanden:
        eq: Equation
        aList = []
        if (len(self.eqs) + len(self.ineqs)) > 0:
            for eq in (self.eqs + self.ineqs):
                aList.append(eq.description())
        if not (self.objective is None):
            aList.append(self.objective.description())
        return aList

    def description_of_variables(self) -> List:
        aList = []
        for aVar in self.variables:
            aList.append(aVar.get_str_description())
        return aList


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
        #TODO: Why as attributes, and not only in sub_elements?
        from flixOpt.flixFeatures import Feature_ShareSum
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
        self.all.add_variable_share('operation', self, self.operation.model.var_sum, 1, 1)
        self.all.add_variable_share('invest', self, self.invest.model.var_sum, 1, 1)
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
        shares_inv = f"Invest Shares={self.specific_share_to_other_effects_invest}"\
            if self.specific_share_to_other_effects_invest != {} else ""

        all_relevant_parts = [info for info in [objective, tot_sum, inv_sum, op_sum, shares_inv, shares_op, standart, desc ] if info != ""]

        full_str =f"{label_unit} {', '.join(all_relevant_parts)}"

        return f"<{self.__class__.__name__}> {full_str}"


# ModelingElement (Element) Klasse zum Summieren einzelner Shares
# geht für skalar und TS
# z.B. invest.costs


# Liste mit zusätzlicher Methode für Rückgabe Standard-Element:
class EffectCollection(List[Effect]):
    '''
    internal effect list for simple handling of effects
    '''

    @property
    def standard_effect(self) -> Optional[Effect]:
        aEffect: Effect
        for aEffectType in self:
            if aEffectType.is_standard:
                return aEffectType

    @property
    def objective_effect(self) -> Effect:
        aEffect: Effect
        aObjectiveEffect = None
        # TODO: eleganter nach attribut suchen:
        for aEffectType in self:
            if aEffectType.is_objective: aObjectiveEffect = aEffectType
        return aObjectiveEffect


EffectTypeDict = Dict[Effect, Numeric_TS]  #Datatype

# Beliebige Komponente (:= Element mit Ein- und Ausgängen)
class Component(Element):
    '''
    basic component class for all components
    '''
    system_model: SystemModel
    new_init_args = ['label', 'on_values_before_begin', 'switch_on_effects', 'switch_on_total_max', 'on_hours_total_min',
                     'on_hours_total_max', 'running_hour_effects', 'exists']
    not_used_args = ['label']

    def __init__(self,
                 label: str,
                 on_values_before_begin:Optional[List[Skalar]] = None,
                 switch_on_effects: Optional[Union[EffectTypeDict, Numeric_TS]] = None,
                 switch_on_total_max: Optional[Skalar] = None,
                 on_hours_total_min: Optional[Skalar] = None,
                 on_hours_total_max: Optional[Skalar] = None,
                 running_hour_effects: Optional[Union[EffectTypeDict, Numeric_TS]] = None,
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
        switch_on_effects : look in Flow for description
        switch_on_total_max : look in Flow for description
        on_hours_total_min : look in Flow for description
        on_hours_total_max : look in Flow for description
        running_hour_effects : look in Flow for description
        exists : array, int, None
            indicates when a component is present. Used for timing of Investments. Only contains blocks of 0 and 1.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if on_hours_total_min is not None:
            raise NotImplementedError("'on_hours_total_min' is not implemented yet for Components. Use Flow directly instead")
        if on_hours_total_max is not None:
            raise NotImplementedError("'on_hours_total_max' is not implemented yet for Components. Use Flow directly instead")
        label = helpers.checkForAttributeNameConformity(label)  # todo: indexierbar / eindeutig machen!
        super().__init__(label, **kwargs)
        self.on_values_before_begin = on_values_before_begin if on_values_before_begin else [0, 0]
        self.switch_on_effects = as_effect_dict_with_ts('switch_on_effects', switch_on_effects, self)
        self.switch_on_max = switch_on_total_max
        self.on_hours_total_min = on_hours_total_min
        self.on_hours_total_max = on_hours_total_max
        self.running_hour_effects = as_effect_dict_with_ts('running_hour_effects', running_hour_effects, self)
        self.exists = TimeSeries('exists', helpers.checkExists(exists), self)

        ## TODO: theoretisch müsste man auch zusätzlich checken, ob ein flow Werte beforeBegin hat!
        # % On Werte vorher durch Flow-values bestimmen:
        # self.on_valuesBefore = 1 * (self.featureOwner.values_before_begin >= np.maximum(model.epsilon,self.flowMin)) für alle Flows!

        #TODO: Dict instead of list?
        self.inputs = []  # list of flows
        self.outputs = []  # list of flows
        self.is_storage = False  #TODO: Use isinstance instead?

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

    def declare_vars_and_eqs_of_flows(self, system_model: SystemModel) -> None:  # todo: macht aber bei Kindklasse Bus keinen Sinn!
        # Flows modellieren:
        for aFlow in self.inputs + self.outputs:
            aFlow.declare_vars_and_eqs(system_model)

    def do_modeling_of_flows(self, system_model: SystemModel, time_indices: Union[list[int], range]) -> None:  # todo: macht aber bei Kindklasse Bus keinen Sinn!
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
        from flixOpt.flixFeatures import FeatureOn

        # feature for: On and SwitchOn Vars
        # (kann erst hier gebaut werden wg. weil input/output Flows erst hier vorhanden)
        flows_defining_on = self.inputs + self.outputs  # Sobald ein input oder  output > 0 ist, dann soll On =1 sein!
        self.featureOn = FeatureOn(self, flows_defining_on, self.on_values_before_begin, self.switch_on_effects,
                                   self.running_hour_effects, on_hours_total_min=self.on_hours_total_min,
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

    def add_share_to_globals_of_flows(self, globalComp, system_model) -> None:
        for aFlow in self.inputs + self.outputs:
            aFlow.add_share_to_globals(globalComp, system_model)

    # wird von Kindklassen überschrieben:
    def add_share_to_globals(self, globalComp, system_model) -> None:
        # Anfahrkosten, Betriebskosten, ... etc ergänzen:
        self.featureOn.add_share_to_globals(globalComp, system_model)

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


# komponenten übergreifende Gleichungen/Variablen/Zielfunktion!
class Global(Element):
    '''
    storing global modeling stuff like effect equations and optimization target
    '''

    def __init__(self, label: str, **kwargs):
        super().__init__(label, **kwargs)

        self.listOfEffectTypes = []  # wird überschrieben mit spezieller Liste

        self.objective = None

    def finalize(self) -> None:
        super().finalize()  # TODO: super-Finalize eher danach?
        from flixOpt.flixFeatures import Feature_ShareSum
        self.penalty = Feature_ShareSum('penalty', self, shares_are_time_series=True)

        # Effekte als Subelemente hinzufügen ( erst hier ist effectTypeList vollständig)
        self.sub_elements.extend(self.listOfEffectTypes)

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
        if variable is None: raise Exception('add_share_to_operation() needs variable or use add_constant_share instead')
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
            effect: Effect
            if effect is None:
                effect = self.listOfEffectTypes.standard_effect
            elif effect not in self.listOfEffectTypes:
                raise Exception('Effect \'' + effect.label + '\' was used but not added to model!')

            if operation_or_invest == 'operation':
                effect.operation.add_share(name_of_share, owner, variable, value,
                                           factor)  # hier darf aVariable auch None sein!
            elif operation_or_invest == 'invest':
                effect.invest.add_share(name_of_share, owner, variable, value,
                                        factor)  # hier darf aVariable auch None sein!
            else:
                raise Exception('operationOrInvest=' + str(operation_or_invest) + ' ist kein zulässiger Wert')

    def declare_vars_and_eqs(self, system_model) -> None:

        # TODO: ggf. Unterscheidung, ob Summen überhaupt als Zeitreihen-Variablen abgebildet werden sollen, oder nicht, wg. Performance.

        super().declare_vars_and_eqs(system_model)

        for effect in self.listOfEffectTypes:
            effect.declare_vars_and_eqs(system_model)
        self.penalty.declare_vars_and_eqs(system_model)

        self.objective = Equation('obj', self, system_model, 'objective')

        # todo : besser wäre objective separat:

    #  eq_objective = Equation('objective',self,model,'objective')
    # todo: hier vielleicht gleich noch eine Kostenvariable ergänzen. Wäre cool!
    def do_modeling(self, system_model, time_indices: Union[list[int], range]) -> None:
        # super().do_modeling(model,time_indices)

        self.penalty.do_modeling(system_model, time_indices)
        ## Gleichungen bauen für Effekte: ##
        effect: Effect
        for effect in self.listOfEffectTypes:
            effect.do_modeling(system_model, time_indices)

        ## Beiträge von Effekt zu anderen Effekten, Beispiel 180 €/t_CO2: ##
        for effectType in self.listOfEffectTypes:
            # Beitrag/Share ergänzen:
            # 1. operation: -> hier sind es Zeitreihen (share_TS)
            # alle specificSharesToOtherEffects durchgehen:
            nameOfShare = 'specific_share_to_other_effects_operation'  # + effectType.label
            for effectTypeOfShare, specShare_TS in effectType.specific_share_to_other_effects_operation.items():
                # Share anhängen (an jeweiligen Effekt):
                shareSum_op = effectTypeOfShare.operation
                shareSum_op: flixOpt.flixFeatures.Feature_ShareSum
                shareHolder = effectType
                shareSum_op.add_variable_share(nameOfShare, shareHolder, effectType.operation.model.var_sum_TS,
                                               specShare_TS, 1)
            # 2. invest:    -> hier ist es Skalar (share)
            # alle specificSharesToOtherEffects durchgehen:
            nameOfShare = 'specificShareToOtherEffects_invest_'  # + effectType.label
            for effectTypeOfShare, specShare in effectType.specific_share_to_other_effects_invest.items():
                # Share anhängen (an jeweiligen Effekt):
                shareSum_inv = effectTypeOfShare.invest
                from flixOpt.flixFeatures import Feature_ShareSum
                shareSum_inv: Feature_ShareSum
                shareHolder = effectType
                shareSum_inv.add_variable_share(nameOfShare, shareHolder, effectType.invest.model.var_sum, specShare, 1)

        # ####### target function  ###########
        # Strafkosten immer:
        self.objective.add_summand(self.penalty.model.var_sum, 1)

        # Definierter Effekt als Zielfunktion:
        objectiveEffect = self.listOfEffectTypes.objective_effect
        if objectiveEffect is None: raise Exception('Kein Effekt als Zielfunktion gewählt!')
        self.objective.add_summand(objectiveEffect.operation.model.var_sum, 1)
        self.objective.add_summand(objectiveEffect.invest.model.var_sum, 1)


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
            self.excess_effects_per_flow_hour = TimeSeries('excess_effects_per_flow_hour', excess_effects_per_flow_hour, self)

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
            self.excess_input = VariableTS('excess_input', len(system_model.time_series), self, system_model, lower_bound=0)
            self.excess_output = VariableTS('excess_output', len(system_model.time_series), self, system_model, lower_bound=0)

    def do_modeling(self, system_model: SystemModel, time_indices: Union[list[int], range]) -> None:
        super().do_modeling(system_model, time_indices)

        # inputs = outputs
        bus_balance = Equation('busBalance', self, system_model)
        for aFlow in self.inputs:
            bus_balance.add_summand(aFlow.model.var_val, 1)
        for aFlow in self.outputs:
            bus_balance.add_summand(aFlow.model.var_val, -1)

        if self.with_excess:   # Fehlerplus/-minus hinzufügen zur Bilanz:
            bus_balance.add_summand(self.excess_output, -1)
            bus_balance.add_summand(self.excess_input, 1)

    def add_share_to_globals(self, global_comp: Global, system_model: SystemModel) -> None:
        super().add_share_to_globals(global_comp, system_model)
        if self.with_excess:   # Strafkosten hinzufügen:
            global_comp.penalty.add_variable_share('excess_effects_per_flow_hour', self, self.excess_input,
                                                   self.excess_effects_per_flow_hour, system_model.dt_in_hours)
            global_comp.penalty.add_variable_share('excess_effects_per_flow_hour', self, self.excess_output,
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
        return self.can_switch_off & np.any(self.min_rel.data > 0)

    def __init__(self, label,
                 bus: Bus = None,  # TODO: Is this for sure Optional?
                 min_rel: Numeric_TS = 0,   # TODO: Rename?
                 max_rel: Numeric_TS = 1,   # TODO: Rename?
                 size: Optional[Skalar] = _default_size,
                 load_factor_min: Optional[Skalar] = None, load_factor_max: Optional[Skalar] = None,
                 #positive_gradient=None,
                 effects_per_flow_hour: Optional[Union[Numeric_TS, EffectTypeDict]] =None,
                 can_switch_off: bool = True,
                 on_hours_total_min: Optional[Skalar] = None,
                 on_hours_total_max: Optional[Skalar] = None,
                 on_hours_min: Optional[Skalar] = None,
                 on_hours_max: Optional[Skalar] = None,
                 off_hours_min: Optional[Skalar] = None,
                 off_hours_max: Optional[Skalar] = None,
                 switch_on_effects: Optional[Union[Numeric_TS, EffectTypeDict]] = None,
                 switch_on_total_max: Optional[Skalar] = None,
                 running_hour_effects: Optional[Union[Numeric_TS, EffectTypeDict]] = None,
                 flow_hours_total_max: Optional[Skalar] = None, flow_hours_total_min: Optional[Skalar] = None,
                 values_before_begin: Optional[List[Skalar]] = None,
                 val_rel: Optional[Numeric_TS] = None,   # TODO: Rename?
                 medium: Optional[str] = None,
                 invest_parameters: Optional[InvestParameters] = None,
                 exists: Numeric_TS = 1,
                 group: Optional[str] = None,
                 **kwargs):
        '''
        Parameters
        ----------
        label : str
            name of flow
        bus : Bus, optional
            bus to which flow is linked
        min_rel : scalar, array, TimeSeriesRaw, optional
            min value is min_rel multiplied by size
        max_rel : scalar, array, TimeSeriesRaw, optional
            max value is max_rel multiplied by size. If size = max then max_rel=1
        size : scalar. None if is a nominal value is a opt-variable, optional
            nominal value/ invest size (linked to min_rel, max_rel and others).
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
            flow can be "off", i.e. be zero (only relevant if min_rel > 0)
            Then a binary var "on" is used.
            If any on/off-forcing parameters like "switch_on_effects", "on_hours_min" etc. are used, then
            this is automatically forced.
        on_hours_total_min : scalar, optional
            min. overall sum of operating hours.
        on_hours_total_max : scalar, optional
            max. overall sum of operating hours.
        on_hours_min : scalar, optional
            min sum of operating hours in one piece
            (last on-time period of timeseries is not checked and can be shorter)
        on_hours_max : scalar, optional
            max sum of operating hours in one piece
        off_hours_min : scalar, optional
            min sum of non-operating hours in one piece
            (last off-time period of timeseries is not checked and can be shorter)
        off_hours_max : scalar, optional
            max sum of non-operating hours in one piece
        switch_on_effects : scalar, array, TimeSeriesRaw, optional
            cost of one switch from off (var_on=0) to on (var_on=1),
            unit i.g. in Euro
        switch_on_total_max : integer, optional
            max nr of switchOn operations
        running_hour_effects : scalar or TS, optional
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
        val_rel : scalar, array, TimeSeriesRaw, optional
            fixed relative values for flow (if given).
            val(t) := val_rel(t) * size(t)
            With this value, the flow-value is no opt-variable anymore;
            (min_rel u. max_rel are making sense anymore)
            used for fixed load profiles, i.g. heat demand, wind-power, solarthermal
            If the load-profile is just an upper limit, use max_rel instead.
        medium: string, None
            medium is relevant, if the linked bus only allows a special defined set of media.
            If None, any bus can be used.
        invest_parameters : None or InvestParameters, optional
            used for investment costs or/and investment-optimization!
        exists : int, array, None
            indicates when a flow is present. Used for timing of Investments. Only contains blocks of 0 and 1.
            max_rel is multiplied with this value before the solve
        group: str, None
            group name to assign flows to groups. Used for later analysis of the results
        '''

        super().__init__(label, **kwargs)
        # args to attributes:
        self.bus = bus
        self.size = size  # skalar!
        self.min_rel = TimeSeries('min_rel', min_rel, self)
        self.max_rel = TimeSeries('max_rel', max_rel, self)

        self.load_factor_min = load_factor_min
        self.load_factor_max = load_factor_max
        #self.positive_gradient = TimeSeries('positive_gradient', positive_gradient, self)
        self.effects_per_flow_hour = as_effect_dict_with_ts('effects_per_flow_hour', effects_per_flow_hour, self)
        self.can_switch_off = can_switch_off
        self.on_hours_total_min = on_hours_total_min
        self.on_hours_total_max = on_hours_total_max
        self.on_hours_min = None if (on_hours_min is None) else TimeSeries('on_hours_min', on_hours_min, self)
        self.on_hours_max = None if (on_hours_max is None) else TimeSeries('on_hours_max', on_hours_max, self)
        self.off_hours_min = None if (off_hours_min is None) else TimeSeries('off_hours_min', off_hours_min, self)
        self.off_hours_max = None if (off_hours_max is None) else TimeSeries('off_hours_max', off_hours_max, self)
        self.switch_on_effects = as_effect_dict_with_ts('switch_on_effects', switch_on_effects, self)
        self.switch_on_total_max = switch_on_total_max
        self.running_hour_effects = as_effect_dict_with_ts('running_hour_effects', running_hour_effects, self)
        self.flow_hours_total_max = flow_hours_total_max
        self.flow_hours_total_min = flow_hours_total_min

        self.exists = TimeSeries('exists', helpers.checkExists(exists), self)
        self.group = group   # TODO: wird überschrieben von Component!
        self.values_before_begin = np.array(values_before_begin) if values_before_begin else np.array([0, 0])  # list -> np-array

        self.invest_parameters = invest_parameters   # Info: Plausi-Checks erst, wenn Flow self.comp kennt.
        self.comp = None   # zugehörige Komponente (wird später von Komponente gefüllt)

        self.val_rel = None
        if val_rel is not None:
            # Wenn noch size noch Default, aber investment_size nicht optimiert werden soll:
            size_is_default = self.size == Flow._default_size
            if size_is_default and self.size_is_fixed:
                raise Exception(
                    'Achtung: Wenn val_rel genutzt wird, muss zugehöriges size definiert werden, da: value = val_rel * size!')
            self.val_rel = TimeSeries('val_rel', val_rel, self)

        self.medium = medium
        if (self.medium is not None) and (not isinstance(self.medium, str)):
            raise Exception('medium must be a string or None')

        # Liste. Ich selbst bin der definierende Flow! (Bei Komponente sind es hingegen alle in/out-flows)
        flows_defining_on = [self]
        # TODO: besser wäre model.epsilon, aber hier noch nicht bekannt!)
        on_values_before_begin = 1 * (self.values_before_begin >= 0.0001)
        # TODO: Wenn can_switch_off = False und min > 0, dann könnte man var_on fest auf 1 setzen um Rechenzeit zu sparen

        #TODO: Why not in sub_elements?
        from flixOpt.flixFeatures import FeatureOn, FeatureInvest
        self.featureOn = FeatureOn(self, flows_defining_on,
                                   on_values_before_begin,
                                   self.switch_on_effects,
                                   self.running_hour_effects,
                                   on_hours_total_min=self.on_hours_total_min,
                                   on_hours_total_max=self.on_hours_total_max,
                                   on_hours_min=self.on_hours_min,
                                   on_hours_max=self.on_hours_max,
                                   off_hours_min=self.off_hours_min,
                                   off_hours_max=self.off_hours_max,
                                   switch_on_total_max=self.switch_on_total_max,
                                   force_on=self.on_variable_is_forced)

        self.featureInvest: Optional[FeatureInvest] = None   # Is defined in finalize()


    def __str__(self):
        details = [
            f"bus={self.bus.label if self.bus else 'None'}",
            f"size={self.size}",
            f"min/max_rel={self.min_rel}-{self.max_rel}",
            f"medium={self.medium}",
            f"invest_parameters={self.invest_parameters.__str__()}" if self.invest_parameters else "",
            f"val_rel={self.val_rel}" if self.val_rel else "",
            f"effects_per_flow_hour={self.effects_per_flow_hour}" if self.effects_per_flow_hour else "",
            f"running_hour_effects={self.running_hour_effects}" if self.running_hour_effects else "",
        ]

        all_relevant_parts = [part for part in details if part != ""]

        full_str =f"{', '.join(all_relevant_parts)}"

        return f"<{self.__class__.__name__}> {self.label}: {full_str}"



    # Plausitest der Eingangsparameter (sollte erst aufgerufen werden, wenn self.comp bekannt ist)
    def plausibility_test(self) -> None:
        # TODO: Incorporate into Variable? (Lower_bound can not be greater than upper bound
        if np.any(self.min_rel.data > self.max_rel.data):
            raise Exception(self.label_full + ': Take care, that min_rel <= max_rel!')

    # bei Bedarf kann von außen Existenz von Binärvariable erzwungen werden:
    def force_on_variable(self) -> None:
        self.featureOn.force_on = True

    def finalize(self) -> None:
        self.plausibility_test()  # hier Input-Daten auf Plausibilität testen (erst hier, weil bei __init__ self.comp noch nicht bekannt)


        # exist-merge aus Flow.exist und Comp.exist
        exists_global = np.multiply(self.exists.data, self.comp.exists.data) # array of 0 and 1
        self.exists_with_comp = TimeSeries('exists_with_comp', helpers.checkExists(exists_global), self)
        # combine max_rel with and exist from the flow and the comp it belongs to
        self.max_rel_with_exists = TimeSeries('max_rel_with_exists', np.multiply(self.max_rel.data, self.exists_with_comp.data), self)
        self.min_rel_with_exists = TimeSeries('min_rel_with_exists', np.multiply(self.min_rel.data, self.exists_with_comp.data), self)

        # prepare invest Feature:
        if self.invest_parameters is not None:
            from flixOpt.flixFeatures import FeatureInvest
            self.featureInvest = FeatureInvest('size', self, self.invest_parameters,
                                               min_rel=self.min_rel_with_exists,
                                               max_rel=self.max_rel_with_exists,
                                               val_rel=self.val_rel,
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
            if self.val_rel is not None:
                # min = max = val !
                lower_bound = None
                upper_bound = None
                fix_value = self.val_rel.active_data * self.size
            else:
                lower_bound = 0 if self.featureOn.use_on else self.min_rel_with_exists.active_data * self.size
                upper_bound = self.max_rel_with_exists.active_data * self.size
                fix_value = None
            return lower_bound, upper_bound, fix_value

        # wenn keine Investrechnung:
        if self.featureInvest is None:
            (lower_bound, upper_bound, fix_value) = bounds_of_defining_variable()
        else:
            (lower_bound, upper_bound, fix_value) = self.featureInvest.bounds_of_defining_variable()

        # TODO --> wird trotzdem modelliert auch wenn value = konst -> Sinnvoll?
        self.model.var_val = VariableTS('val', system_model.nrOfTimeSteps, self, system_model,
                                        lower_bound=lower_bound, upper_bound=upper_bound, value=fix_value)
        self.model.var_sumFlowHours = Variable('sumFlowHours', 1, self, system_model, lower_bound=self.flow_hours_total_min,
                                               upper_bound=self.flow_hours_total_max)
        # ! Die folgenden Variablen müssen erst von featureOn erstellt worden sein:
        self.model.var_on = self.featureOn.getVar_on()  # mit None belegt, falls nicht notwendig
        self.model.var_switchOn, self.model.var_switchOff = self.featureOn.getVars_switchOnOff()  # mit None belegt, falls nicht notwendig

        # erst hier, da defining_variable vorher nicht belegt!
        if self.featureInvest is not None:
            self.featureInvest.set_defining_variables(self.model.var_val, self.model.var_on)
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
            eq_on_hours_total_max.add_summand(self.model.var_on, 1, as_sum=True)
            eq_on_hours_total_max.add_constant(self.on_hours_total_max / system_model.dt_in_hours)

        #
        # ############## on_hours_total_max: ##############
        #

        # ineq: sum(var_on(t)) >= on_hours_total_min

        if self.on_hours_total_min is not None:
            eq_on_hours_total_min = Equation('on_hours_total_min', self, system_model, 'ineq')
            eq_on_hours_total_min.add_summand(self.model.var_on, -1, as_sum=True)
            eq_on_hours_total_min.add_constant(-1 * self.on_hours_total_min / system_model.dt_in_hours)


        #
        # ############## sumFlowHours: ##############
        #

        # eq: var_sumFlowHours - sum(var_val(t)* dt(t) = 0

        eq_sumFlowHours = Equation('sumFlowHours', self, system_model, 'eq')  # general mean
        eq_sumFlowHours.add_summand(self.model.var_val, system_model.dt_in_hours, as_sum=True)
        eq_sumFlowHours.add_summand(self.model.var_sumFlowHours, -1)

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
            eq_flowHoursPerInvestsize_Max.add_summand(self.model.var_sumFlowHours, 1)
            if self.featureInvest is not None:
                eq_flowHoursPerInvestsize_Max.add_summand(self.featureInvest.model.var_investmentSize,
                                                          -1 * flowHoursPerInvestsize_max)
            else:
                eq_flowHoursPerInvestsize_Max.add_constant(self.size * flowHoursPerInvestsize_max)

                ## min load factor:
        #  eq: size * sum(dt)* load_factor_min <= var_sumFlowHours

        if self.load_factor_min is not None:
            flowHoursPerInvestsize_min = system_model.dt_in_hours_total * self.load_factor_min  # = fullLoadHours if investsize in [kW]
            eq_flowHoursPerInvestsize_Min = Equation('load_factor_min', self, system_model, 'ineq')
            eq_flowHoursPerInvestsize_Min.add_summand(self.model.var_sumFlowHours, -1)
            if self.featureInvest is not None:
                eq_flowHoursPerInvestsize_Min.add_summand(self.featureInvest.model.var_investmentSize,
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

    def add_share_to_globals(self, global_comp: Global, system_model) -> None:

        # Arbeitskosten:
        if self.effects_per_flow_hour is not None:
            owner = self
            global_comp.add_share_to_operation(
                'effects_per_flow_hour', owner, self.model.var_val,
                self.effects_per_flow_hour, system_model.dt_in_hours)

        # Anfahrkosten, Betriebskosten, ... etc ergänzen:
        self.featureOn.add_share_to_globals(global_comp, system_model)

        if self.featureInvest is not None:
            self.featureInvest.add_share_to_globals(global_comp, system_model)

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
        if self.medium is None:    # nicht überschreiben, nur wenn leer:
            self.medium = medium


class System:
    '''
    A System holds Elements (Components, Buses, Flows, Effects,...).
    '''

    ## Properties:

    @property
    def elements_of_first_layer_wo_flows(self) -> List[Element]:
        return (self.components + list(self.buses) + [self.global_comp] + self.effects +
                list(self.other_elements))

    @property
    def elements_of_fists_layer(self) -> List[Element]:
        return self.elements_of_first_layer_wo_flows + list(self.flows)

    @property
    def invest_features(self) -> List['FeatureInvest']:
        all_invest_features = []

        def get_invest_features_of_element(element: Element) -> List['FeatureInvest']:
            invest_features = []
            from flixOpt.flixFeatures import FeatureInvest
            for aSubComp in element.all_sub_elements:
                if isinstance(aSubComp, FeatureInvest):
                    invest_features.append(aSubComp)
                invest_features += get_invest_features_of_element(aSubComp)  # recursive!
            return invest_features

        for element in self.elements_of_fists_layer:  # kann in Komponente (z.B. Speicher) oder Flow stecken
            all_invest_features += get_invest_features_of_element(element)

        return all_invest_features

    # Achtung: Funktion wird nicht nur für Getter genutzt.
    @property
    def flows(self) -> Set[Flow]:
        return {flow for comp in self.components for flow in comp.inputs + comp.outputs}

    # get all TS in one list:
    @property
    def all_time_series_in_elements(self) -> List[TimeSeries]:
        element: Element
        all_TS = []
        for element in self.elements_of_fists_layer:
            all_TS += element.TS_list
        return all_TS

    @property
    def buses(self) -> Set[Bus]:
        return {flow.bus for flow in self.flows}

        # time_series: möglichst format ohne pandas-Nutzung bzw.: Ist DatetimeIndex hier das passende Format?

    def __init__(self,
                 time_series: np.ndarray[np.datetime64],
                 last_time_step_hours: Optional[Union[int, float]] = None):
        """
          Parameters
          ----------
          time_series : np.ndarray of datetime64
              timeseries of the data
          last_time_step_hours : for calc
              The duration of last time step.
              Storages needs this time-duration for calculation of charge state
              after last time step.
              If None, then last time increment of time_series is used.
        """
        self.time_series = time_series
        self.last_time_step_hours = last_time_step_hours

        self.time_series_with_end = helpers.getTimeSeriesWithEnd(time_series, last_time_step_hours)
        helpers.checkTimeSeries('global esTimeSeries', self.time_series_with_end)

        # defaults:
        self.components: List[Component] = []
        self.other_elements: Set[Element] = set()  ## hier kommen zusätzliche Elements rein, z.B. aggregation
        self.effects: EffectCollection = EffectCollection()  # Kosten, CO2, Primärenergie, ...
        self.temporary_elements = []  # temporary elements, only valid for one calculation (i.g. aggregation modeling)
        # instanzieren einer globalen Komponente (diese hat globale Gleichungen!!!)
        self.global_comp = Global('global_comp')
        self._finalized = False  # wenn die Elements alle finalisiert sind, dann True
        self.model: Optional[SystemModel] = None  # later activated

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self.components)} components and {len(self.effects)} effects>"

    def __str__(self):
        components = '\n'.join(component.__str__() for component in
                               sorted(self.components, key=lambda component: component.label.upper()))
        effects = '\n'.join(effect.__str__() for effect in
                               sorted(self.effects, key=lambda effect: effect.label.upper()))
        return f"Energy System with components:\n{components}\nand effects:\n{effects}"

    # Effekte registrieren:
    def add_effects(self, *args: Effect) -> None:
        new_effects = list(args)
        for new_effect in new_effects:
            print('Register new effect ' + new_effect.label)
            self._check_if_element_is_unique(new_effect, self.effects)   # check if already exists
            # Wenn Standard-Effekt, und schon einer vorhanden:
            if new_effect.is_standard and self.effects.standard_effect is not None:
                raise Exception(f'standardEffekt ist bereits belegt mit {self.effects.standard_effect.label}')
            # Wenn Objective-Effekt, und schon einer vorhanden:
            if new_effect.is_objective and self.effects.objective_effect is not None:
                raise Exception(f'objectiveEffekt ist bereits belegt mit {self.effects.standard_effect.label}')

            self.effects.append(new_effect)   # in liste ergänzen:

        # TODO: doppelte Haltung in system und global_comp ist so nicht schick.
        self.global_comp.listOfEffectTypes = self.effects   # an global_comp durchreichen

    def add_components(self, *args: Component) -> None:
        # Komponenten registrieren:
        new_components = list(args)
        for new_component in new_components:
            print('Register new Component ' + new_component.label)
            self._check_if_element_is_unique(new_component, self.components)   # check if already exists:
            new_component.register_component_in_flows()   # Komponente in Flow registrieren
            new_component.register_flows_in_bus()   # Flows in Bus registrieren:
        self.components.extend(new_components)   # Add to existing list of components

        # Element registrieren ganz allgemein:

    def add_elements(self, *args: Element) -> None:
        '''
        add all modeling elements, like storages, boilers, heatpumps, buses, ...

        Parameters
        ----------
        *args : childs of   Element like cBoiler, HeatPump, Bus,...
            modeling Elements

        '''

        for new_element in list(args):
            if isinstance(new_element, Component):
                self.add_components(new_element)
            elif isinstance(new_element, Effect):
                self.add_effects(new_element)
            elif isinstance(new_element, Element):
                # check if already exists:
                self._check_if_element_is_unique(new_element, self.other_elements)
                # register Element:
                self.other_elements.add(new_element)
            else:
                raise Exception('argument is not instance of a modeling Element (Element)')

    def add_temporary_elements(self, *args: Element) -> None:
        '''
        add temporary modeling elements, only valid for one calculation,
        i.g. cAggregationModeling-Element

        Parameters
        ----------
        *args : Element
            temporary modeling Elements.

        '''

        self.add_elements(*args)
        self.temporary_elements += args  # Register temporary Elements

    def delete_temporary_elements(self):  # function just implemented, still not used
        '''
        deletes all registered temporary Elements
        '''
        for temporary_element in self.temporary_elements:
            # delete them again in the lists:
            self.components.remove(temporary_element)
            self.buses.remove(temporary_element)
            self.other_elements.remove(temporary_element)
            self.effects.remove(temporary_element)
            self.flows(temporary_element)

    def _check_if_element_is_unique(self, element: Element, existing_elements: List[Element]) -> None:
        '''
        checks if element or label of element already exists in list

        Parameters
        ----------
        aElement : Element
            new element to check
        existing_elements : list
            list of already registered elements
        '''

        # check if element is already registered:
        if element in existing_elements:
            raise Exception('Element \'' + element.label + '\' already added to cEnergysystem!')

        # check if name is already used:
        # TODO: Check all elements instead of only a list that is passed?
        # TODO: An Effect with the same label as another element is not allowed, or is it?
        if element.label in [elem.label for elem in existing_elements]:
            raise Exception('Elementname \'' + element.label + '\' already used in another element!')

    def _plausibility_checks(self) -> None:
        # Check circular loops in effects: (Effekte fügen sich gegenseitig Shares hinzu):

        def error_str(effect_label: str, shareEffect_label: str):
            return (
                f'  {effect_label} -> has share in: {shareEffect_label}\n'
                f'  {shareEffect_label} -> has share in: {effect_label}'
            )

        for effect in self.effects:
            # operation:
            for shareEffect in effect.specific_share_to_other_effects_operation.keys():
                # Effekt darf nicht selber als Share in seinen ShareEffekten auftauchen:
                assert (effect not in shareEffect.specific_share_to_other_effects_operation.keys(),
                        f'Error: circular operation-shares \n{error_str(effect.label, shareEffect.label)}')
            # invest:
            for shareEffect in effect.specific_share_to_other_effects_invest.keys():
                assert (effect not in shareEffect.specific_share_to_other_effects_invest.keys(),
                        f'Error: circular invest-shares \n{error_str(effect.label, shareEffect.label)}')

    # Finalisieren aller ModelingElemente (dabei werden teilweise auch noch sub_elements erzeugt!)
    def finalize(self) -> None:
        print('finalize all Elements...')
        self._plausibility_checks()
        # nur EINMAL ausführen: Finalisieren der Elements:
        if not self._finalized:
            # finalize Elements for modeling:
            for element in self.elements_of_fists_layer:
                print(element.label)   #TODO: Remove this print??
                element.finalize()  # inklusive sub_elements!
            self._finalized = True

    def do_modeling_of_elements(self) -> SystemModel:

        if not self._finalized:
            raise Exception('modeling not possible, because Energysystem is not finalized')

        # Bus-Liste erstellen: -> Wird die denn überhaupt benötigt?

        # TODO: Achtung time_indices kann auch nur ein Teilbereich von time_indices abdecken, z.B. wenn man für die anderen Zeiten anderweitig modellieren will
        # --> ist aber nicht sauber durchimplementiert in den ganzehn add_summand()-Befehlen!!
        time_indices = range(len(self.model.time_indices))

        # globale Modellierung zuerst, damit andere darauf zugreifen können:
        self.global_comp.declare_vars_and_eqs(self.model)  # globale Funktionen erstellen!
        self.global_comp.do_modeling(self.model, time_indices)  # globale Funktionen erstellen!

        # Komponenten-Modellierung (# inklusive sub_elements!)
        for aComp in self.components:
            aComp: Component
            log.debug('model ' + aComp.label + '...')
            # todo: ...OfFlows() ist nicht schön --> besser als rekursive Geschichte aller subModelingElements der Komponente umsetzen z.b.
            aComp.declare_vars_and_eqs_of_flows(self.model)
            aComp.declare_vars_and_eqs(self.model)

            aComp.do_modeling_of_flows(self.model, time_indices)
            aComp.do_modeling(self.model, time_indices)

            aComp.add_share_to_globals_of_flows(self.global_comp, self.model)
            aComp.add_share_to_globals(self.global_comp, self.model)

        # Bus-Modellierung (# inklusive sub_elements!)
        aBus: Bus
        for aBus in self.buses:
            log.debug('model ' + aBus.label + '...')
            aBus.declare_vars_and_eqs(self.model)
            aBus.do_modeling(self.model, time_indices)
            aBus.add_share_to_globals(self.global_comp, self.model)

        # TODO: Currently there are no "other elements"
        # weitere übergeordnete Modellierungen:
        for element in self.other_elements:
            element.declare_vars_and_eqs(self.model)
            element.do_modeling(self.model, time_indices)
            element.add_share_to_globals(self.global_comp, self.model)

            # transform to Math:
        self.model.to_math_model()

        return self.model

    # aktiviere in TS die gewählten Indexe: (wird auch direkt genutzt, nicht nur in activate_system_model)
    def activate_indices_in_time_series(
            self, indices: Union[List[int], range],
            alternative_data_for_time_series: Optional[Dict[TimeSeries, np.ndarray]] = None) -> None:
        # TODO: Aggreagation functionality to other part of framework?
        aTS: TimeSeries
        if alternative_data_for_time_series is None:
            alternative_data_for_time_series = {}

        for aTS in self.all_time_series_in_elements:
            # Wenn explicitData vorhanden:
            if aTS in alternative_data_for_time_series.keys():
                explicitData = alternative_data_for_time_series[aTS]
            else:
                explicitData = None
                # Aktivieren:
            aTS.activate(indices, explicitData)

    def activate_model(self, system_model: SystemModel) -> None:
        self.model = system_model
        system_model: SystemModel
        element: Element

        # hier nochmal TS updaten (teilweise schon für Preprozesse gemacht):
        self.activate_indices_in_time_series(system_model.time_indices, system_model.TS_explicit)

        # Wenn noch nicht gebaut, dann einmalig Element.model bauen:
        if system_model.models_of_elements == {}:
            log.debug('create model-Vars for Elements of EnergySystem')
            for element in self.elements_of_fists_layer:
                # BEACHTE: erst nach finalize(), denn da werden noch sub_elements erst erzeugt!
                if not self._finalized:
                    raise Exception('activate_model(): --> Geht nicht, da System noch nicht finalized!')
                # model bauen und in model registrieren.
                element.create_new_model_and_activate_system_model(self.model)  # inkl. sub_elements
        else:
            # nur Aktivieren:
            for element in self.elements_of_fists_layer:  # TODO: Is This a BUG?
                element.activate_system_model(system_model)  # inkl. sub_elements

    # ! nur nach Solve aufrufen, nicht später nochmal nach activating model (da evtl stimmen Referenzen nicht mehr unbedingt!)
    def get_results_after_solve(self) -> Tuple[Dict, Dict]:
        results = {}  # Daten
        results_var = {}  # zugehörige Variable
        # für alle Komponenten:
        for element in self.elements_of_first_layer_wo_flows:
            # results        füllen:
            (results[element.label], results_var[element.label]) = element.get_results()  # inklusive sub_elements!

        # Zeitdaten ergänzen
        aTime = {}
        results['time'] = aTime
        aTime['time_series_with_end'] = self.model.time_series_with_end
        aTime['time_series'] = self.model.time_series
        aTime['dt_in_hours'] = self.model.dt_in_hours
        aTime['dt_in_hours_total'] = self.model.dt_in_hours_total

        return results, results_var

    def printModel(self) -> None:
        aBus: Bus
        aComp: Component
        print('')
        print('##############################################################')
        print('########## Short String Description of Energysystem ##########')
        print('')

        print(yaml.dump(self.description_of_system()))

    def description_of_system(self, flowsWithBusInfo=False) -> Dict:
        modelDescription = {}

        # Anmerkung buses und comps als dict, weil Namen eindeutig!
        # Buses:
        modelDescription['buses'] = {}
        for aBus in self.buses:
            aBus: Bus
            modelDescription['buses'].update(aBus.description())
        # Comps:
        modelDescription['components'] = {}
        aComp: Component
        for aComp in self.components:
            modelDescription['components'].update(aComp.description())

        # Flows:
        flowList = []
        modelDescription['flows'] = flowList
        aFlow: Flow
        for aFlow in self.flows:
            flowList.append(aFlow.description())

        return modelDescription

    def description_of_equations(self) -> Dict:
        aDict = {}

        # comps:
        aSubDict = {}
        aDict['Components'] = aSubDict
        aComp: Element
        for aComp in self.components:
            aSubDict[aComp.label] = aComp.description_of_equations()

        # buses:
        aSubDict = {}
        aDict['buses'] = aSubDict
        for aBus in self.buses:
            aSubDict[aBus.label] = aBus.description_of_equations()

        # globals:
        aDict['globals'] = self.global_comp.description_of_equations()

        # flows:
        aSubDict = {}
        aDict['flows'] = aSubDict
        for aComp in self.components:
            for aFlow in (aComp.inputs + aComp.outputs):
                aSubDict[aFlow.label_full] = aFlow.description_of_equations()

        # others
        aSubDict = {}
        aDict['others'] = aSubDict
        for element in self.other_elements:
            aSubDict[element.label] = element.description_of_equations()

        return aDict

    def print_equations(self) -> None:

        print('')
        print('##############################################################')
        print('################# Equations of Energysystem ##################')
        print('')

        print(yaml.dump(self.description_of_equations(),
                        default_flow_style=False,
                        allow_unicode=True))

    def description_of_variables(self, structured=True) -> Union[List, Dict]:
        aVar: Variable

        # liste:
        if not structured:
            aList = []
            for aVar in self.model.variables:
                aList.append(aVar.get_str_description())
            return aList

        # struktur:
        else:
            aDict = {}

            # comps (and belonging flows):
            subDict = {}
            aDict['Comps'] = subDict
            # comps:
            for aComp in self.components:
                subDict[aComp.label] = aComp.description_of_variables()
                for aFlow in aComp.inputs + aComp.outputs:
                    subDict[aComp.label] += aFlow.description_of_variables()

            # buses:
            subDict = {}
            aDict['buses'] = subDict
            for bus in self.buses:
                subDict[bus.label] = bus.description_of_variables()

            # globals:
            aDict['globals'] = self.global_comp.description_of_variables()

            # others
            aSubDict = {}
            aDict['others'] = aSubDict
            for element in self.other_elements:
                aSubDict[element.label] = element.description_of_variables()

            return aDict

    def print_variables(self) -> None:
        print('')
        print('##############################################################')
        print('################# Variables of Energysystem ##################')
        print('')
        print('############# a) as list : ################')
        print('')

        yaml.dump(self.description_of_variables(structured=False))

        print('')
        print('############# b) structured : ################')
        print('')

        yaml.dump(self.description_of_variables(structured=True))

    # Datenzeitreihe auf Basis gegebener time_indices aus globaler extrahieren:
    def get_time_data_from_indices(self, time_indices: Union[List[int], range]) -> Tuple[
                                                                                        np.ndarray[np.datetime64],
                                                                                        np.ndarray[np.datetime64],
                                                                                        np.ndarray[np.float64],
                                                                                        np.float64]:
        # if time_indices is None, dann alle : time_indices = range(length(self.time_series))
        # Zeitreihen:
        time_series = self.time_series[time_indices]
        # next timestamp as endtime:
        endTime = self.time_series_with_end[time_indices[-1] + 1]
        time_series_with_end = np.append(time_series, endTime)

        # Zeitdifferenz:
        #              zweites bis Letztes            - erstes bis Vorletztes
        dt = time_series_with_end[1:] - time_series_with_end[0:-1]
        dt_in_hours = dt / np.timedelta64(1, 'h')
        # dt_in_hours    = dt.total_seconds() / 3600
        dt_in_hours_total = sum(dt_in_hours)  # Gesamtzeit
        return (time_series, time_series_with_end, dt_in_hours, dt_in_hours_total)


# Standardoptimierung segmentiert/nicht segmentiert
class Calculation:
    '''
    class for defined way of solving a energy system optimizatino
    '''

    @property
    def infos(self):
        infos = {}

        calcInfos = self._infos
        infos['calculation'] = calcInfos
        calcInfos['name'] = self.label
        calcInfos['no ChosenIndexe'] = len(self.time_indices)
        calcInfos['calculation_type'] = self.calculation_type
        calcInfos['duration'] = self.durations
        infos['system_description'] = self.system.description_of_system()
        infos['system_models'] = {}
        infos['system_models']['duration'] = [system_model.duration for system_model in self.system_models]
        infos['system_models']['info'] = [system_model.infos for system_model in self.system_models]

        return infos

    @property
    def results(self):
        # wenn noch nicht belegt, dann aus system_model holen
        if self._results is None:
            self._results = self.system_models[0].results

            # (bei segmented Calc ist das schon explizit belegt.)
        return self._results

    @property
    def results_struct(self):
        # Wenn noch nicht ermittelt:
        if (self._results_struct is None):
            # Neurechnen (nur bei Segments)
            if (self.calculation_type == 'segmented'):
                self._results_struct = helpers.createStructFromDictInDict(self.results)
            # nur eine system_model vorhanden ('full','aggregated')
            elif len(self.system_models) == 1:
                self._results_struct = self.system_models[0].results_struct
            else:
                raise Exception('calculation_type ' + str(self.calculation_type) + ' not defined')
        return self._results_struct

    # time_indices: die Indexe des Energiesystems, die genutzt werden sollen. z.B. [0,1,4,6,8]
    def __init__(self, label, system: System, modeling_language, time_indices: Optional[list[int]] = None, pathForSaving='results'):
        '''
        Parameters
        ----------
        label : str
            name of calculation
        system : System
            system which should be calculated
        modType : 'pyomo','cvxpy' (not implemeted yet)
            choose optimization modeling language
        time_indices : None, list
            list with indexe, which should be used for calculation. If None, then all timesteps are used.
        pathForSaving : str
            Path for result files. The default is 'results'.

        '''
        self.label = label
        self.nameOfCalc = None  # name for storing results   # TODO: Why nameOfClac and label???
        self.system = system
        self.modeling_language = modeling_language
        self.time_indices = time_indices
        self.path_for_saving = pathForSaving
        self.calculation_type = None  # 'full', 'segmented', 'aggregated'
        self._infos = {}

        self.system_models: List[SystemModel] = []  # liste der ModelBoxes (nur bei Segmentweise mehrere!)
        self.durations = {'modeling': 0, 'solving': 0}   # Dauer der einzelnen Dinge
        self.time_series_for_aggregation = None  # list of timeseries for aggregation

        self.time_indices = time_indices or range(len(system.time_series))  # Wenn time_indices = None, dann alle nehmen
        (self.time_series, self.time_series_with_end, self.dt_in_hours, self.dt_in_hours_total) = (
            system.get_time_data_from_indices(self.time_indices))
        helpers.checkTimeSeries('time_indices', self.time_series)

        self._results = None
        self._results_struct = None  # hier kommen die verschmolzenen Ergebnisse der Segmente rein!
        self.segmented_system_models = []  # model list
        self.aggregation_data = None  # aggregationStuff (if calculation_type = 'aggregated')

    # Variante1:
    def do_modeling_as_one_segment(self):
        '''
          modeling full problem

        '''
        self.check_if_already_modeled()
        self.calculation_type = 'full'
        # System finalisieren:
        self.system.finalize()

        t_start = time.time()
        # Modellierungsbox / TimePeriod-Box bauen:
        system_model = SystemModel(self.label, self.modeling_language, self.system,
                                   self.time_indices)  # alle Indexe nehmen!
        # model aktivieren:
        self.system.activate_model(system_model)
        # modellieren:
        self.system.do_modeling_of_elements()

        self.durations['modeling'] = round(time.time() - t_start, 2)
        self.system_models.append(system_model)
        return system_model

    # Variante2:
    def do_segmented_modeling_and_solving(self, solverProps, segmentLen, nrOfUsedSteps, namePrefix='', nameSuffix='',
                                          aPath='results/'):
        '''
          Dividing and Modeling the problem in (overlapped) time-segments.
          Storage values as result of segment n are overtaken
          to the next segment n+1 for timestep, which is first in segment n+1

          Afterwards timesteps of segments (without overlap)
          are put together to the full timeseries

          Because the result of segment n is used in segment n+1, modeling and
          solving is both done in this method

          Take care:
          Parameters like invest_parameters, loadfactor etc. does not make sense in
          segmented modeling, cause they are newly defined in each segment

          Parameters
          ----------
          solverProps : TYPE
              DESCRIPTION.
          segmentLen : int
              nr Of Timesteps of Segment.
          nrOfUsedSteps : int
              nr of timesteps used/overtaken in resulting complete timeseries
              (the timesteps after these are "overlap" and used for better
              results of chargestate of storages)
          namePrefix : str
              prefix-String for name of calculation. The default is ''.
          nameSuffix : str
              suffix-String for name of calculation. The default is ''.
          aPath : str
              path for output. The default is 'results/'.

          '''
        self.check_if_already_modeled()
        self._infos['segmentedProps'] = {'segmentLen': segmentLen, 'nrUsedSteps': nrOfUsedSteps}
        self.calculation_type = 'segmented'
        print('##############################################################')
        print('#################### segmented Solving #######################')

        t_start = time.time()

        # system finalisieren:
        self.system.finalize()

        if len(self.system.invest_features) > 0:
            raise Exception('segmented calculation with Invest-Parameters does not make sense!')

        assert nrOfUsedSteps <= segmentLen
        assert segmentLen <= len(self.time_series), 'segmentLen must be smaller than (or equal to) the whole nr of timesteps'

        # time_seriesOfSim = self.system.time_series[from_index:to_index+1]

        # Anzahl = Letzte Simulation bis zum Ende plus die davor mit Überlappung:
        nrOfSimSegments = math.ceil((len(self.time_series)) / nrOfUsedSteps)
        self._infos['segmentedProps']['nrOfSegments'] = nrOfSimSegments
        print('indexe        : ' + str(self.time_indices[0]) + '...' + str(self.time_indices[-1]))
        print('segmentLen    : ' + str(segmentLen))
        print('usedSteps     : ' + str(nrOfUsedSteps))
        print('-> nr of Sims : ' + str(nrOfSimSegments))
        print('')

        self._define_path_names(namePrefix, nameSuffix, aPath, saveResults=True, nr_of_system_models=nrOfSimSegments)

        for i in range(nrOfSimSegments):
            startIndex_calc = i * nrOfUsedSteps
            endIndex_calc = min(startIndex_calc + segmentLen - 1, len(self.time_indices) - 1)

            startIndex_global = self.time_indices[startIndex_calc]
            endIndex_global = self.time_indices[endIndex_calc]  # inklusiv
            indexe_global = self.time_indices[startIndex_calc:endIndex_calc + 1]  # inklusive endIndex

            # new realNrOfUsedSteps:
            # if last Segment:
            if i == nrOfSimSegments - 1:
                realNrOfUsedSteps = endIndex_calc - startIndex_calc + 1
            else:
                realNrOfUsedSteps = nrOfUsedSteps

            print(
                str(i) + '. Segment ' + ' (system-indexe ' + str(startIndex_global) + '...' + str(endIndex_global) + ') :')

            # Modellierungsbox / TimePeriod-Box bauen:
            label = self.label + '_seg' + str(i)
            segmentModBox = SystemModel(label, self.modeling_language, self.system, indexe_global)  # alle Indexe nehmen!
            segmentModBox.realNrOfUsedSteps = realNrOfUsedSteps

            # Startwerte übergeben von Vorgänger-system_model:
            if i > 0:
                from flixOpt.basicModeling import BeforeValues
                segmentModBoxBefore = self.segmented_system_models[i - 1]
                segmentModBox.before_values = BeforeValues(segmentModBoxBefore.all_ts_variables,
                                                           segmentModBoxBefore.realNrOfUsedSteps - 1)
                print('### before_values: ###')
                segmentModBox.before_values.print()
                print('#######################')
                # transferStartValues(segment, segmentBefore)

            # model in Energiesystem aktivieren:
            self.system.activate_model(segmentModBox)

            # modellieren:
            t_start_modeling = time.time()
            self.system.do_modeling_of_elements()
            self.durations['modeling'] += round(time.time() - t_start_modeling, 2)
            # system_model in Liste hinzufügen:
            self.segmented_system_models.append(segmentModBox)
            # übergeordnete system_model-Liste:
            self.system_models.append(segmentModBox)

            # Lösen:
            t_start_solving = time.time()

            segmentModBox.solve(**solverProps,
                                logfile_name=self.paths_Log[i])  # keine SolverOutput-Anzeige, da sonst zu viel
            self.durations['solving'] += round(time.time() - t_start_solving, 2)
            ## results adding:
            self._add_segment_results(segmentModBox, startIndex_calc, realNrOfUsedSteps)

        self.durations['model, solve and segmentStuff'] = round(time.time() - t_start, 2)

        self._save_solve_infos()

    def do_aggregated_modeling(self, periodLengthInHours, nr_of_typical_periods,
                               use_extreme_periods, fixStorageFlows,
                               fixBinaryVarsOnly, percentageOfPeriodFreedom=0,
                               costsOfPeriodFreedom=0,
                               addPeakMax=[],
                               addPeakMin=[]):
        '''
        method of aggregated modeling.
        1. Finds typical periods.
        2. Equalizes variables of typical periods.

        Parameters
        ----------
        periodLengthInHours : float
            length of one period.
        nr_of_typical_periods : int
            no of typical periods
        use_extreme_periods : boolean
            True, if periods of extreme values should be explicitly chosen
            Define recognised timeseries in args addPeakMax, addPeakMin!
        fixStorageFlows : boolean
            Defines, wether load- and unload-Flow should be also aggregated or not.
            If all other flows are fixed, it is mathematically not necessary
            to fix them.
        fixBinaryVarsOnly : boolean
            True, if only binary var should be aggregated.
            Additionally choose, wether orginal or aggregated timeseries should
            be chosen for the calculation.
        percentageOfPeriodFreedom : 0...100
            Normally timesteps of all periods in one period-collection
            are all equalized. Here you can choose, which percentage of values
            can maximally deviate from this and be "free variables". The solver
            chooses the "free variables".
        costsOfPeriodFreedom : float
            costs per "free variable". The default is 0.
            !! Warning: At the moment these costs are allocated to
            operation costs, not to penalty!!
        useOriginalTimeSeries : boolean.
            orginal or aggregated timeseries should
            be chosen for the calculation. default is False.
        addPeakMax : list of TimeSeriesRaw
            list of data-timeseries. The period with the max-value are
            chosen as a explicitly period.
        addPeakMin : list of TimeSeriesRaw
            list of data-timeseries. The period with the min-value are
            chosen as a explicitly period.


        Returns
        -------
        system_model : TYPE
            DESCRIPTION.

        '''
        self.check_if_already_modeled()

        self._infos['aggregatedProps'] = {'periodLengthInHours': periodLengthInHours,
                                          'nr_of_typical_periods': nr_of_typical_periods,
                                          'use_extreme_periods': use_extreme_periods,
                                          'fixStorageFlows': fixStorageFlows,
                                          'fixBinaryVarsOnly': fixBinaryVarsOnly,
                                          'percentageOfPeriodFreedom': percentageOfPeriodFreedom,
                                          'costsOfPeriodFreedom': costsOfPeriodFreedom}

        self.calculation_type = 'aggregated'
        t_start_agg = time.time()
        # chosen Indexe aktivieren in TS: (sonst geht Aggregation nicht richtig)
        self.system.activate_indices_in_time_series(self.time_indices)

        # Zeitdaten generieren:
        (chosenTimeSeries, chosenTimeSeriesWithEnd, dt_in_hours, dt_in_hours_total) = (
            self.system.get_time_data_from_indices(self.time_indices))

        # check equidistant timesteps:
        if max(dt_in_hours) - min(dt_in_hours) != 0:
            raise Exception('!!! Achtung Aggregation geht nicht, da unterschiedliche delta_t von ' + str(
                min(dt_in_hours)) + ' bis ' + str(max(dt_in_hours)) + ' h')

        print('#########################')
        print('## TS for aggregation ###')

        ## Daten für Aggregation vorbereiten:
        # TSlist and TScollection ohne Skalare:
        self.time_series_for_aggregation = [item for item in self.system.all_time_series_in_elements if item.is_array]
        from flixOpt.flixBasics import TimeSeriesCollection
        self.TScollectionForAgg = TimeSeriesCollection(self.time_series_for_aggregation,
                                                       addPeakMax_TSraw=addPeakMax,
                                                       addPeakMin_TSraw=addPeakMin,
                                                       )

        self.TScollectionForAgg.print()

        import pandas as pd
        # seriesDict = {i : self.time_series_for_aggregation[i].active_data_vector for i in range(length(self.time_series_for_aggregation))}
        df_OriginalData = pd.DataFrame(self.TScollectionForAgg.seriesDict,
                                       index=chosenTimeSeries)  # eigentlich wäre TS als column schön, aber TSAM will die ordnen können.

        # Check, if timesteps fit in Period:
        stepsPerPeriod = periodLengthInHours / self.dt_in_hours[0]
        if not stepsPerPeriod.is_integer():
            raise Exception('Fehler! Gewählte Periodenlänge passt nicht zur Zeitschrittweite')

        ##########################################################
        # ### Aggregation - creation of aggregated timeseries: ###
        from flixOpt import flixAggregation as flixAgg
        dataAgg = flixAgg.Aggregation('aggregation',
                                      timeseries=df_OriginalData,
                                      hours_per_time_step=self.dt_in_hours[0],
                                      hours_per_period=periodLengthInHours,
                                      hasTSA=False,
                                      nr_of_typical_periods=nr_of_typical_periods,
                                      use_extreme_periods=use_extreme_periods,
                                      weights=self.TScollectionForAgg.weightDict,
                                      addPeakMax=self.TScollectionForAgg.addPeak_Max_labels,
                                      addPeakMin=self.TScollectionForAgg.addPeak_Min_labels)

        dataAgg.cluster()
        self.aggregation_data = dataAgg

        self._infos['aggregatedProps']['periods_order'] = str(list(dataAgg.aggregation.clusterOrder))

        # aggregation_data.aggregation.clusterPeriodIdx
        # aggregation_data.aggregation.clusterOrder
        # aggregation_data.aggregation.clusterPeriodNoOccur
        # aggregation_data.aggregation.predictOriginalData()
        # self.periods_order = aggregation.clusterOrder
        # self.period_occurrences = aggregation.clusterPeriodNoOccur

        # ### Some plot for plausibility check ###

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.title('aggregated series (dashed = aggregated)')
        plt.plot(df_OriginalData.values)
        for label_TS, agg_values in dataAgg.aggregated_timeseries.items():
            # aLabel = str(i)
            # aLabel = self.time_series_for_aggregation[i].label_full
            plt.plot(agg_values.values, '--', label=label_TS)
        if len(self.time_series_for_aggregation) < 10:  # wenn nicht zu viele
            plt.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center')
        plt.show()

        # ### Some infos as print ###

        print('TS Aggregation:')
        for i in range(len(self.time_series_for_aggregation)):
            aLabel = self.time_series_for_aggregation[i].label_full
            print('TS ' + str(aLabel))
            print('  max_agg:' + str(max(dataAgg.aggregated_timeseries[aLabel])))
            print('  max_orig:' + str(max(df_OriginalData[aLabel])))
            print('  min_agg:' + str(min(dataAgg.aggregated_timeseries[aLabel])))
            print('  min_orig:' + str(min(df_OriginalData[aLabel])))
            print('  sum_agg:' + str(sum(dataAgg.aggregated_timeseries[aLabel])))
            print('  sum_orig:' + str(sum(df_OriginalData[aLabel])))

        print('addpeakmax:')
        print(self.TScollectionForAgg.addPeak_Max_labels)
        print('addpeakmin:')
        print(self.TScollectionForAgg.addPeak_Min_labels)

        # ################
        # ### Modeling ###

        aggregationModel = flixAgg.cAggregationModeling('aggregation', self.system,
                                                        indexVectorsOfClusters=dataAgg.indexVectorsOfClusters,
                                                        fixBinaryVarsOnly=fixBinaryVarsOnly,
                                                        fixStorageFlows=fixStorageFlows,
                                                        listOfElementsToClusterize=None,
                                                        percentageOfPeriodFreedom=percentageOfPeriodFreedom,
                                                        costsOfPeriodFreedom=costsOfPeriodFreedom)

        # temporary Modeling-Element for equalizing indices of aggregation:
        self.system.add_temporary_elements(aggregationModel)

        if fixBinaryVarsOnly:
            TS_explicit = None
        else:
            # neue (Explizit)-Werte für TS sammeln::
            TS_explicit = {}
            for i in range(len(self.time_series_for_aggregation)):
                TS = self.time_series_for_aggregation[i]
                # todo: agg-Wert für TS:
                TS_explicit[TS] = dataAgg.aggregated_timeseries[TS.label_full].values  # nur data-array ohne Zeit

        # ##########################
        # ## System finalizing: ##
        self.system.finalize()

        self.durations['aggregation'] = round(time.time() - t_start_agg, 2)

        t_m_start = time.time()
        # Modellierungsbox / TimePeriod-Box bauen: ! inklusive TS_explicit!!!
        system_model = SystemModel(self.label, self.modeling_language, self.system, self.time_indices,
                                   TS_explicit)  # alle Indexe nehmen!
        self.system_models.append(system_model)
        # model aktivieren:
        self.system.activate_model(system_model)
        # modellieren:
        self.system.do_modeling_of_elements()

        self.durations['modeling'] = round(time.time() - t_m_start, 2)
        return system_model

    def solve(self, solverProps, namePrefix='', nameSuffix='', aPath='results/', saveResults=True):

        self._define_path_names(namePrefix, nameSuffix, aPath, saveResults, nr_of_system_models=1)

        if self.calculation_type not in ['full', 'aggregated']:
            raise Exception('calculation_type ' + self.calculation_type + ' needs no solve()-Command (only for ' + str())
        system_model = self.system_models[0]
        system_model.solve(**solverProps, logfile_name=self.paths_Log[0])

        if saveResults:
            self._save_solve_infos()

    def _define_path_names(self, namePrefix, nameSuffix, aPath, saveResults, nr_of_system_models=1):
        import datetime
        import pathlib

        # absoluter Pfad:
        aPath = pathlib.Path.cwd() / aPath
        # Pfad anlegen, fall noch nicht vorhanden:
        aPath.mkdir(parents=True, exist_ok=True)
        self.pathForResults = aPath

        timestamp = datetime.datetime.now()
        timestring = timestamp.strftime('%Y-%m-%data')
        self.nameOfCalc = namePrefix.replace(" ", "") + timestring + '_' + self.label.replace(" ",
                                                                                              "") + nameSuffix.replace(
            " ", "")

        if saveResults:
            filename_Data = self.nameOfCalc + '_data.pickle'
            filename_Info = self.nameOfCalc + '_solvingInfos.yaml'
            if nr_of_system_models == 1:
                filenames_Log = [self.nameOfCalc + '_solver.log']
            else:
                filenames_Log = [(self.nameOfCalc + '_solver_' + str(i) + '.log') for i in range(nr_of_system_models)]

            self.paths_Log = [self.pathForResults / filenames_Log[i] for i in range(nr_of_system_models)]
            self.path_Data = self.pathForResults / filename_Data
            self.path_Info = self.pathForResults / filename_Info
        else:
            self.paths_Log = None
            self.path_Data = None
            self.path_Info = None

    def check_if_already_modeled(self):
        if self.calculation_type is not None:
            raise Exception(
                f'Another modeling-Method ({self.calculation_type=}) was already executed with this Calculation-Object.\n'
                f'Always create a new instance of Calculation for new modeling/solving-command!')

        if self.system.temporary_elements:  # if some element in this list
            raise Exception(
                f'The Energysystem has some temporary modelingElements from previous calculation '
                f'(i.g. aggregation-Modeling-Elements. These must be deleted before new calculation.')

    def _save_solve_infos(self):
        import yaml
        # Daten:
        # with open(yamlPath_Data, 'w') as f:
        #   yaml.dump(self.results, f, sort_keys = False)
        import pickle
        with open(self.path_Data, 'wb') as f:
            pickle.dump(self.results, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Infos:'
        with open(self.path_Info, 'w', encoding='utf-8') as f:
            yaml.dump(self.infos, f,
                      width=1000,  # Verhinderung Zeilenumbruch für lange equations
                      allow_unicode=True,
                      sort_keys=False)

        aStr = f'# saved calculation {self.nameOfCalc} #'
        print('#' * len(aStr))
        print(aStr)
        print('#' * len(aStr))

    def _add_segment_results(self, segment, startIndex_calc, realNrOfUsedSteps):
        # rekursiv aufzurufendes Ergänzen der Dict-Einträge um segment-Werte:

        if (self._results is None):
            self._results = {}  # leeres Dict als Ausgangszustand

        def append_new_results_to_dict_values(result: Dict, result_to_append: Dict, result_to_append_var: Dict):
            if result == {}:
                firstFill = True  # jeweils neuer Dict muss erzeugt werden für globales Dict
            else:
                firstFill = False

            for key, val in result_to_append.items():
                # print(key)

                # Wenn val ein Wert ist:
                if isinstance(val, np.ndarray) or isinstance(val, np.float64) or np.isscalar(val):

                    # Beachte Länge (withEnd z.B. bei Speicherfüllstand)
                    if key in ['time_series', 'dt_in_hours', 'dt_in_hours_total']:
                        withEnd = False
                    elif key in ['time_series_with_end']:
                        withEnd = True
                    else:
                        # Beachte Speicherladezustand und ähnliche Variablen:
                        aReferedVariable = result_to_append_var[key]
                        aReferedVariable: VariableTS
                        withEnd = isinstance(aReferedVariable, VariableTS) \
                                  and aReferedVariable.activated_beforeValues \
                                  and aReferedVariable.before_value_is_start_value

                        # nested:

                    def getValueToAppend(val, withEnd):
                        # wenn skalar, dann Vektor draus machen:
                        # todo: --> nicht so schön!
                        if np.isscalar(val):
                            val = np.array([val])

                        if withEnd:
                            if firstFill:
                                aValue = val[0:realNrOfUsedSteps + 1]  # (inklusive WithEnd!)
                            else:
                                # erstes Element weglassen, weil das schon vom Vorgängersegment da ist:
                                aValue = val[1:realNrOfUsedSteps + 1]  # (inklusive WithEnd!)
                        else:
                            aValue = val[0:realNrOfUsedSteps]  # (nur die genutzten Steps!)
                        return aValue

                    aValue = getValueToAppend(val, withEnd)

                    if firstFill:
                        result[key] = aValue
                    else:  # erstmaliges Füllen. Array anlegen.
                        result[key] = np.append(result[key], aValue)  # Anhängen (nur die genutzten Steps!)

                else:
                    if firstFill: result[key] = {}

                    if (result_to_append_var is not None) and key in result_to_append_var.keys():
                        resultToAppend_sub = result_to_append_var[key]
                    else:  # z.B. bei time (da keine Variablen)
                        resultToAppend_sub = None
                    append_new_results_to_dict_values(result[key], result_to_append[key], resultToAppend_sub)  # hier rekursiv!

        # rekursiv:
        append_new_results_to_dict_values(self._results, segment.results, segment.results_var)

        # results füllen:
        # ....