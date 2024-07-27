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

from flixOpt import utils
from flixOpt.math_modeling import MathModel, Variable, VariableTS, Equation  # Modelliersprache
from flixOpt.core import TimeSeries, Skalar

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from flixOpt.elements import Flow
    from flixOpt.flow_system import FlowSystem

logger = logging.getLogger('flixOpt')


class SystemModel(MathModel):
    '''
    Hier kommen die ModellingLanguage-spezifischen Sachen rein
    '''

    def __init__(self,
                 label: str,
                 modeling_language: Literal['pyomo', 'cvxpy'],
                 flow_system,
                 time_indices: Union[List[int], range],
                 TS_explicit=None):
        super().__init__(label, modeling_language)
        self.flow_system: FlowSystem = flow_system  # energysystem (wäre Attribut von cTimePeriodModel)
        self.time_indices = time_indices
        self.nrOfTimeSteps = len(time_indices)
        self.TS_explicit = TS_explicit  # für explizite Vorgabe von Daten für TS {TS1: data, TS2:data,...}
        self.models_of_elements: Dict[Element, ElementModel] = {}  # dict with all ElementModel's of Elements in FlowSystem

        # Zeitdaten generieren:
        (self.time_series, self.time_series_with_end, self.dt_in_hours, self.dt_in_hours_total) = (
            flow_system.get_time_data_from_indices(time_indices))

    @property
    def infos(self):
        infos = super().infos
        infos['str_Eqs'] = self.flow_system.description_of_equations()
        infos['str_Vars'] = self.flow_system.description_of_variables()
        infos['main_results'] = self.main_results_str   # Hauptergebnisse:
        infos.update(self._infos)
        return infos

    # register ModelingElements and belonging Mod:
    def register_element_with_model(self, aModelingElement, aMod):
        # allocation Element -> model
        self.models_of_elements[aModelingElement] = aMod  # aktuelles model hier speichern

    # 'gurobi'
    def solve(self,
              mip_gap: float = 0.02,
              time_limit_seconds: int = 3600,
              solver_name: str = 'highs',
              solver_output_to_console: bool = True,
              excess_threshold: Union[int, float] = 0.1,
              logfile_name: str = 'solver_log.log',
              **kwargs):
        """
        Parameters
        ----------
        mip_gap : TYPE, optional
            DESCRIPTION. The default is 0.02.
        time_limit_seconds : TYPE, optional
            DESCRIPTION. The default is 3600.
        solver_name : TYPE, optional
            DESCRIPTION. The default is 'highs'.
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
        """

        # check valid solver options:
        if len(kwargs) > 0:
            for key in kwargs.keys():
                if key not in ['threads']:
                    raise Exception(
                        'no allowed arguments for kwargs: ' + str(key) + '(all arguments:' + str(kwargs) + ')')

        logger.info(f'{" starting solving ":#^80}')
        logger.info(f'{self.describe()}')

        super().solve(mip_gap, time_limit_seconds, solver_name, solver_output_to_console, logfile_name, **kwargs)

        if solver_name == 'gurobi':
            termination_message = self.solver_results['Solver'][0]['Termination message']
        elif solver_name == 'glpk':
            termination_message = self.solver_results['Solver'][0]['Status']
        else:
            termination_message = f'not implemented for solver "{solver_name}" yet'
        logger.info(f'Termination message: "{termination_message}"')

        # Variablen-Ergebnisse abspeichern:
        # 1. dict:
        (self.results, self.results_var) = self.flow_system.get_results_after_solve()
        # 2. struct:
        self.results_struct = utils.createStructFromDictInDict(self.results)

        def extract_main_results() -> Dict[str, Union[Skalar, Dict]]:
            main_results = {}
            effect_results = {}
            main_results['Effects'] = effect_results
            for effect in self.flow_system.effect_collection.effects:
                effect_results[f'{effect.label} [{effect.unit}]'] = {
                    'operation': float(effect.operation.model.variables['sum'].result),
                    'invest': float(effect.invest.model.variables['sum'].result),
                    'sum': float(effect.all.model.variables['sum'].result)}
            main_results['penalty'] = float(self.flow_system.effect_collection.penalty.model.variables['sum'].result)
            main_results['Result of objective'] = self.objective_result
            if self.solver_name == 'highs':
                main_results['lower bound'] = self.solver_results.best_objective_bound
            else:
                main_results['lower bound'] = self.solver_results['Problem'][0]['Lower bound']
            busesWithExcess = []
            main_results['buses with excess'] = busesWithExcess
            for aBus in self.flow_system.all_buses:
                if aBus.with_excess:
                    if (
                            np.sum(self.results[aBus.label]['excess_input']) > excess_threshold or
                            np.sum(self.results[aBus.label]['excess_output']) > excess_threshold
                    ):
                        busesWithExcess.append(aBus.label)

            invest_decisions = {'invested': {}, 'not invested': {}}
            main_results['Invest-Decisions'] = invest_decisions
            for invest_feature in self.flow_system.all_investments:
                invested_size = invest_feature.model.variables[invest_feature.name_of_investment_size].result
                invested_size = float(invested_size)  # bei np.floats Probleme bei Speichern
                label = invest_feature.owner.label_full
                if invested_size > 1e-3:
                    invest_decisions['invested'][label] = invested_size
                else:
                    invest_decisions['not invested'][label] = invested_size

            return main_results

        self.main_results_str = extract_main_results()

        logger.info(f'{" finished solving ":#^80}')
        logger.info(f'{" Main Results ":#^80}')
        for effect_name, effect_results in self.main_results_str['Effects'].items():
            logger.info(f'{effect_name}:\n'
                        f'  {"operation":<15}: {effect_results["operation"]:>10.2f}\n'
                        f'  {"invest":<15}: {effect_results["invest"]:>10.2f}\n'
                        f'  {"sum":<15}: {effect_results["sum"]:>10.2f}')

        logger.info(
            # f'{"SUM":<15}: ...todo...\n'
            f'{"penalty":<17}: {self.main_results_str["penalty"]:>10.2f}\n'
            f'{"":-^80}\n'
            f'{"Objective":<17}: {self.main_results_str["Result of objective"]:>10.2f}\n'
            f'{"":-^80}')

        logger.info(f'Investment Decisions:')
        logger.info(utils.apply_formating(data_dict={**self.main_results_str["Invest-Decisions"]["invested"],
                                                     **self.main_results_str["Invest-Decisions"]["not invested"]},
                                          key_format="<30", indent=2, sort_by='value'))

        for bus in self.main_results_str['buses with excess']:
            logger.warning(f'Excess Value in Bus {bus.label}!')

        if self.main_results_str["penalty"] > 10:
            logger.warning(f'A total penalty of {self.main_results_str["penalty"]} occurred.'
                           f'This might distort the results')
        logger.info(f'{" End of Main Results ":#^80}')

    @property
    def variables(self) -> List[Variable]:
        all_vars = list(self._variables)
        for model in self.models_of_elements.values():
            all_vars += [var for var in model.variables.values()]
        return all_vars

    @property
    def eqs(self) -> List[Equation]:
        all_eqs = list(self._eqs)
        for model in self.models_of_elements.values():
            all_eqs += [var for var in model.eqs.values()]
        return all_eqs

    @property
    def ineqs(self) -> List[Equation]:
        return [eq for eq in self.eqs if eq.eqType == 'ineq']


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

    @property
    def all_variables_with_sub_elements(self) -> Dict[str, Variable]:
        all_vars = self.model.variables
        for sub_element in self.all_sub_elements:
            all_vars_of_sub_element = sub_element.model.variables
            duplicate_var_names = set(all_vars.keys()) & set(all_vars_of_sub_element.keys())
            if duplicate_var_names:
                raise Exception(f'Variables {duplicate_var_names} of Subelement "{sub_element.label_full}" '
                                f'already exists in Element "{self.label_full}". labels must be unique.')
            all_vars.update(all_vars_of_sub_element)

        return all_vars

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
        logger.debug('new model for ' + self.label)
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
            aData[aVar.label] = aVar.result
            aVars[aVar.label] = aVar  # link zur Variable
            if aVar.is_binary and aVar.length > 1:
                # Bei binären Variablen zusätzlichen Vektor erstellen,z.B. a  = [0, 1, 0, 0, 1]
                #                                                       -> a_ = [nan, 1, nan, nan, 1]
                aData[aVar.label + '_'] = utils.zero_to_nan(aVar.result)
                aVars[aVar.label + '_'] = aVar  # link zur Variable

        # 3. Alle TS übergeben
        aTS: TimeSeries
        for aTS in self.TS_list:
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
        return self.model.description_of_variables() + [
            description for sub_element in self.sub_elements for description in sub_element.description_of_variables()
        ]

    def overview_of_eqs_and_vars(self) -> Dict[str, int]:
        return {'no eqs': len(self.model.eqs),
                'no eqs single': sum(eq.nr_of_single_equations for eq in self.model.eqs.values()),
                'no inEqs': len(self.model.ineqs),
                'no inEqs single': sum(ineq.nr_of_single_equations for ineq in self.model.ineqs.values()),
                'no vars': len(self.model.variables),
                'no vars single': sum(var.length for var in self.model.variables.values())}


class ElementModel:
    '''
    is existing in every Element and owns eqs and vars of the activated calculation
    '''

    def __init__(self, element: Element):
        self.element = element
        self.variables = {}
        self.eqs = {}
        self.objective = None

    def get_var(self, label: str) -> Variable:
        if label in self.variables.keys():
            return self.variables[label]
        raise Exception(f'Variable "{label}" does not exist')

    def get_eq(self, label: str) -> Equation:
        if label in self.eqs.keys():
            return self.eqs[label]
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
        if len(self.eqs) > 0:
            for eq in list(self.eqs.values()):
                aList.append(eq.description())
        if not (self.objective is None):
            aList.append(self.objective.description())
        return aList

    def description_of_variables(self) -> List:
        aList = []
        for aVar in self.variables.values():
            aList.append(aVar.get_str_description())
        return aList

    @property
    def ineqs(self) -> Dict[str, Equation]:
        return {name: eq for name, eq in self.eqs.items() if eq.eqType == 'ineq'}
