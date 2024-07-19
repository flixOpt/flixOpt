# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:40:23 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import textwrap
from typing import List, Set, Tuple, Dict, Union, Optional, Literal, TYPE_CHECKING
import logging
import timeit

import numpy as np

from flixOpt import flixOptHelperFcts as helpers
from flixOpt.modeling import LinearModel, Variable, VariableTS, Equation  # Modelliersprache
from flixOpt.core import TimeSeries
if TYPE_CHECKING:  # for type checking and preventing circular imports
    from flixOpt.elements import Flow
    from flixOpt.system import System

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
            print('  operation: ' + str(aEffect.operation.model.variables['sum'].result))
            print('  invest   : ' + str(aEffect.invest.model.variables['sum'].result))
            print('  sum      : ' + str(aEffect.all.model.variables['sum'].result))

        print('SUM              : ' + '...todo...')
        print('penaltyCosts     : ' + str(self.system.global_comp.penalty.model.variables['sum'].result))
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
        if self.system.global_comp.penalty.model.variables['sum'].result > 10:
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
                aDict['operation'] = str(aEffect.operation.model.variables['sum'].result)
                aDict['invest'] = str(aEffect.invest.model.variables['sum'].result)
                aDict['sum'] = str(aEffect.all.model.variables['sum'].result)
            main_results_str['penaltyCosts'] = str(self.system.global_comp.penalty.model.variables['sum'].result)
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
                investValue = aInvestFeature.model.variables[aInvestFeature.name_of_investment_size].result
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

    @property
    def all_variables(self) -> List[Variable]:
        all_vars = []
        for model in self.models_of_elements.values():
            all_vars += [var for var in model.variables.values()]
        return all_vars

    @property
    def all_ts_variables(self) -> List[VariableTS]:
        return [var for var in self.all_variables if isinstance(var, VariableTS)]

    @property
    def all_equations(self) -> List[Equation]:
        all_eqs = []
        for model in self.models_of_elements.values():
            all_eqs += [eq for eq in model.eqs.values()]
        return all_eqs

    @property
    def all_inequations(self) -> List[Equation]:
        all_eqs = []
        for model in self.models_of_elements.values():
            all_eqs += [ineq for ineq in model.ineqs.values()]
        return all_eqs

    def to_math_model(self) -> None:
        t_start = timeit.default_timer()
        eq: Equation
        # Variablen erstellen
        for variable in self.all_variables:
            variable.to_math_model(self)
        # Gleichungen erstellen
        for eq in self.all_equations:
            eq.to_math_model(self)
        # Ungleichungen erstellen:
        for ineq in self.all_inequations:
            ineq.to_math_model(self)
        # Zielfunktion erstellen
        self.objective.to_math_model(self)

        self.duration['to_math_model'] = round(timeit.default_timer() - t_start, 2)


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
               #TODO: Bugfix, as Flow is not imported in this module...
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
    def activate_system_model(self, system_model: SystemModel) -> None:
        for element in self.sub_elements:
            element.activate_system_model(system_model)  # inkl. sub_elements
        self.activate_system_model_for_me(system_model)

    # activate ohne SubElements!
    def activate_system_model_for_me(self, system_model: SystemModel) -> None:
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
    def declare_vars_and_eqs(self, system_model: SystemModel) -> None:
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
        for aVar in self.model.variables.values():
            # print(aVar.label)
            aData[aVar.label] = aVar.result
            aVars[aVar.label] = aVar  # link zur Variable
            if aVar.is_binary and aVar.length > 1:
                # Bei binären Variablen zusätzlichen Vektor erstellen,z.B. a  = [0, 1, 0, 0, 1]
                #                                                       -> a_ = [nan, 1, nan, nan, 1]
                aData[aVar.label + '_'] = helpers.zero_to_nan(aVar.result)
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
        self.variables = {}
        self.eqs = {}
        self.ineqs = {}
        self.objective = None

    def get_var(self, label: str) -> Variable:
        if label in self.variables.keys():
            return self.variables[label]
        raise Exception(f'Variable "{label}" does not exist')

    def get_eq(self, label: str) -> Equation:
        if label in self.eqs.keys():
            return self.eqs[label]
        if label in self.ineqs.keys():
            return self.ineqs[label]
        raise Exception(f'Equation "{label}" does not exist')

    def add_variable(self, variable: Variable) -> None:
        if variable.label not in self.variables.keys():
            self.variables[variable.label] = variable
        else:
            if variable in self.variables.values():
                raise Exception(f'Variable "{variable.label}" already exists')
            else:
                raise Exception(f'A Variable with the label "{variable.label}" already exists')

    def add_equation(self, equation: Equation) -> None:
        if equation.label not in self.eqs.keys():
            self.eqs[equation.label] = equation
        else:
            raise Exception(f'Equation "{equation.label}" already exists')

    def description_of_equations(self) -> List:
        # Wenn Glg vorhanden:
        eq: Equation
        aList = []
        if (len(self.eqs) + len(self.ineqs)) > 0:
            for eq in (list(self.eqs.values()) + list(self.ineqs.values())):
                aList.append(eq.description())
        if not (self.objective is None):
            aList.append(self.objective.description())
        return aList

    def description_of_variables(self) -> List:
        aList = []
        for aVar in self.variables.values():
            aList.append(aVar.get_str_description())
        return aList
