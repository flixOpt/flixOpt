# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:40:23 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import textwrap
from typing import List, Set, Tuple, Dict, Union, Optional, Literal, TYPE_CHECKING
import logging

import numpy as np

from flixOpt import utils
from flixOpt.math_modeling import MathModel, Variable, VariableTS, Equation
from flixOpt.core import TimeSeries, Skalar, Numeric, Numeric_TS, as_effect_dict
from flixOpt.interface import InvestParameters, OnOffParameters
from flixOpt.components import Storage

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from flixOpt.elements import Flow, Effect, EffectCollection, Objective, Bus
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
        self.flow_system: FlowSystem = flow_system
        self.time_indices = range(len(time_indices))
        self.nrOfTimeSteps = len(time_indices)
        self.TS_explicit = TS_explicit  # für explizite Vorgabe von Daten für TS {TS1: data, TS2:data,...}

        # Zeitdaten generieren:
        self.time_series, self.time_series_with_end, self.dt_in_hours, self.dt_in_hours_total = (
            flow_system.get_time_data_from_indices(time_indices))

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
            logger.warning(f'Excess Value in Bus {bus}!')

        if self.main_results_str["penalty"] > 10:
            logger.warning(f'A total penalty of {self.main_results_str["penalty"]} occurred.'
                           f'This might distort the results')
        logger.info(f'{" End of Main Results ":#^80}')

    @property
    def infos(self):
        infos = super().infos
        infos['str_Eqs'] = self.flow_system.description_of_equations()
        infos['str_Vars'] = self.flow_system.description_of_variables()
        infos['main_results'] = self.main_results_str   # Hauptergebnisse:
        infos.update(self._infos)
        return infos

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

    @property
    def models_of_elements(self) -> Dict['Element', 'ElementModel']:
        return {element: element.model for element in self.flow_system.all_elements}


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

    def description_of_equations(self) -> Union[List, Dict]:
        sub_element_desc = {sub_elem.label: sub_elem.description_of_equations() for sub_elem in self.sub_elements}

        if sub_element_desc:
            return {'_self': self.model.description_of_equations(), **sub_element_desc}
        else:
            return self.model.description_of_equations()

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

    # 1.
    def finalize(self) -> None:
        """
        Finalizing the creation of all sub_elements in the Elements
        """
        for element in self.sub_elements:
            element.finalize()

    # 2.
    def create_model(self) -> None:
        """
        Create the empty model for each Element and its sub_elements
        """
        for element in self.sub_elements:
            element.create_model()  # rekursiv!
        logger.debug(f'New Model for {self.label_full}')
        self.model = ElementModel(self)

    # 3.
    def declare_vars_and_eqs(self, system_model: SystemModel) -> None:
        """
        Declare variables and equations for all sub elements.
        """
        pass

    def get_results(self) -> Tuple[Dict, Dict]:
        """
        Get results after the solve
        """
        # Ergebnisse als dict ausgeben:
        data, variables = {}, {}

        # 1. Fill sub-elements recursively:
        for element in self.sub_elements:
            data[element.label], variables[element.label] = element.get_results()

        # 2. Store variable values:
        for var in self.model.variables.values():
            data[var.label] = var.result
            variables[var.label] = var  # link to the variable
            if var.is_binary and var.length > 1:
                data[f"{var.label}_"] = utils.zero_to_nan(var.result)   # Additional vector when binary with nan
                variables[f"{var.label}_"] = var  # link to the variable

        # 3. Pass all time series:
        for ts in self.TS_list:
            data[ts.label] = ts.data
            variables[ts.label] = ts  # link to the time series

        # 4. Pass the group attribute, if it exists:
        if hasattr(self, 'group') and self.group is not None:
            data["group"] = self.group
            variables["group"] = self.group

        return data, variables


class ElementModel:
    """
    is existing in every Element and owns eqs and vars of the activated calculation
    """

    def __init__(self, element: Element):
        self.element = element
        self.variables = {}
        self.eqs = {}
        self.objective = None
        self.sub_models = []

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
            aList.append(aVar.description())
        return aList

    def get_var(self, label: str) -> Variable:
        if label in self.variables.keys():
            return self.variables[label]
        raise Exception(f'Variable "{label}" does not exist')

    def get_eq(self, label: str) -> Equation:
        if label in self.eqs.keys():
            return self.eqs[label]
        raise Exception(f'Equation "{label}" does not exist')

    def add_variables(self, *variables: Variable) -> None:
        for variable in variables:
            if variable.label not in self.variables.keys():
                self.variables[variable.label] = variable
            elif variable in self.variables.values():
                raise Exception(f'Variable "{variable.label}" already exists')
            else:
                raise Exception(f'A Variable with the label "{variable.label}" already exists')

    def add_equations(self, *equations: Equation) -> None:
        for equation in equations:
            if equation.label not in self.eqs.keys():
                self.eqs[equation.label] = equation
            else:
                raise Exception(f'Equation "{equation.label}" already exists')

    @property
    def ineqs(self) -> Dict[str, Equation]:
        return {name: eq for name, eq in self.eqs.items() if eq.eqType == 'ineq'}


class FlowModel(ElementModel):
    def __init__(self, element: Flow):
        super().__init__(element)
        self.element: Flow = element
        self.flow_rate: Optional[VariableTS] = None
        self.sum_flow_hours: Optional[Variable] = None

        self._on: Optional[OnOffModel] = None if element.featureOn else OnOffModel(element.featureOn)
        self._investment: Optional[InvestmentModel] = None if isinstance(element.size, Skalar) else InvestmentModel(element.featureInvest, element.featureOn.use_on)
        self.sub_models.extend([model for model in (self._on, self._investment) if model is not None])

    def do_modeling(self, system_model: SystemModel):
        # flow_rate
        if isinstance(self.element.size, Skalar):
            fixed_value = None if self.element.fixed_relative_value is None else self.element.fixed_relative_value * self.element.size
            self.flow_rate = VariableTS('flow_rate', system_model.nrOfTimeSteps, self.element.label_full, system_model,
                                        lower_bound=self.element.relative_minimum_with_exists.active_data * self.element.size,
                                        upper_bound=self.element.relative_maxiumum_with_exists.active_data * self.element.size,
                                        value=fixed_value)
        else:
            self.flow_rate = VariableTS('flow_rate', system_model.nrOfTimeSteps, self.element.label_full, system_model)
        self.add_variables(self.flow_rate)

        # sumFLowHours
        self.sum_flow_hours = Variable('sumFlowHours', 1, self.element.label_full, system_model,
                                       lower_bound=self.element.flow_hours_total_min,
                                       upper_bound=self.element.flow_hours_total_max)
        eq_sum_flow_hours = Equation('sumFlowHours', self, system_model, 'eq')
        eq_sum_flow_hours.add_summand(self.flow_rate, system_model.dt_in_hours, as_sum=True)
        eq_sum_flow_hours.add_summand(self.sum_flow_hours, -1)
        self.add_variables(self.sum_flow_hours)
        self.add_equations(eq_sum_flow_hours)

        self._create_bounds_for_load_factor(system_model)

        self._on.do_modeling(system_model) if self._on is not None else None
        self._investment.do_modeling(system_model, ) if self._investment is not None else None
        self._create_shares(system_model)

    def _create_shares(self, system_model: SystemModel):
        # Arbeitskosten:
        effect_collection = system_model.flow_system.effect_collection
        if self.element.effects_per_flow_hour is not None:
            effect_collection.add_share_to_operation(
                name_of_share='effects_per_flow_hour',
                owner=self.element, variable=self.flow_rate,
                effect_values=self.element.effects_per_flow_hour,
                factor=system_model.dt_in_hours
            )

    def _create_bounds_for_load_factor(self, system_model: SystemModel):
        ## ############## full load fraction bzw. load factor ##############
        # eq: var_sumFlowHours <= size * dt_tot * load_factor_max
        if self.element.load_factor_max is not None:
            flow_hours_per_size_max = system_model.dt_in_hours_total * self.element.load_factor_max
            eq_load_factor_max = Equation('load_factor_max', self, system_model, 'ineq')
            self.add_equations(eq_load_factor_max)
            eq_load_factor_max.add_summand(self.sum_flow_hours, 1)
            # if investment:
            if self._investment is not None:
                eq_load_factor_max.add_summand(self._investment.size, -1 * flow_hours_per_size_max)
            else:
                eq_load_factor_max.add_constant(self.element.size * flow_hours_per_size_max)

        #  eq: size * sum(dt)* load_factor_min <= var_sumFlowHours
        if self.element.load_factor_min is not None:
            flow_hours_per_size_min = system_model.dt_in_hours_total * self.element.load_factor_min
            eq_load_factor_min = Equation('load_factor_min', self, system_model, 'ineq')
            self.add_equations(eq_load_factor_min)
            eq_load_factor_min.add_summand(self.sum_flow_hours, 1)
            if self._investment is not None:
                eq_load_factor_min.add_summand(self._investment.size, flow_hours_per_size_min)
            else:
                eq_load_factor_min.add_constant(-1 * self.element.size * flow_hours_per_size_min)


class EffectModel(ElementModel):
    def __init__(self, element: Effect):
        super().__init__(element)
        self.element: Effect
        self.invest = ShareAllocationModel(self.element, False,
                                           total_max=self.element.maximum_invest,
                                           total_min=self.element.minimum_invest)
        self.operation = ShareAllocationModel(self.element, True,
                                              total_max=self.element.maximum_operation,
                                              total_min=self.element.minimum_operation,
                                              min_per_hour=self.element.minimum_operation_per_hour,
                                              max_per_hour=self.element.maximum_operation_per_hour)
        self.all = ShareAllocationModel(self.element, False,
                                        total_max=self.element.maximum_total,
                                        total_min=self.element.minimum_total)
        self.sub_models.extend([self.invest, self.operation, self.all])

    def do_modeling(self, system_model: SystemModel):
        for model in self.sub_models:
            model.do_modeling(system_model)

        self.all.add_variable_share('operation', self.element, self.operation.sum, 1, 1)
        self.all.add_variable_share('invest', self.element, self.invest.sum, 1, 1)


class EffectCollectionModel(ElementModel):
    #TODO: Maybe all EffectModels should be sub_models of this Model? Including Objective and Penalty?
    def __init__(self, element: EffectCollection, system_model: SystemModel):
        super().__init__(element)
        self.element = element
        self._system_model = system_model
        self._effect_models: Dict[Effect, EffectModel] = {}
        self._penalty: Optional[ShareAllocationModel] = None

    def do_modeling(self, system_model: SystemModel):
        self._effect_models = {effect: EffectModel(effect) for effect in self.element.effects}
        self._penalty = ShareAllocationModel(self.element.penalty, True)
        self.sub_models.extend(list(self._effect_models.values()) + [self._penalty])
        for model in self.sub_models:
            model.do_modeling(system_model)

    def add_share(self,
                  target: Literal['operation', 'invest', 'penalty'],
                  name_of_share: str,
                  owner: Element,
                  effect_values: Union[Numeric, Dict[Optional[Effect], TimeSeries]],
                  factor: Numeric,
                  variable: Optional[Variable] = None) -> None:
        assert target in ('operation', 'invest', 'penalty'), f'{target} not supported'
        effect_values_dict = as_effect_dict(effect_values)

        # an alle Effects, die einen Wert haben, anhängen:
        for effect, value in effect_values_dict.items():
            if effect is None:  # Falls None, dann Standard-effekt nutzen:
                effect = self.element.standard_effect
            assert effect in self.element.effects, f'Effect {effect.label} was used but not added to model!'

            if target == 'operation':
                model = self._effect_models[effect].operation
            elif target == 'invest':
                model = self._effect_models[effect].invest
            elif target == 'penalty':
                model = self._penalty
            else:
                raise ValueError(f'Target {target} not supported!')
            
            if variable is not None:
                model.add_constant_share(name_of_share, owner, value, factor)
            elif isinstance(variable, Variable):
                model.add_variable_share(name_of_share, owner, variable, value, factor)
            else:
                raise TypeError

    def add_share_between_effects(self):
        for origin_effect in self.element.effects:
            # 1. operation: -> hier sind es Zeitreihen (share_TS)
            name_of_share = 'specific_share_to_other_effects_operation'  # + effectType.label
            for target_effect, factor in origin_effect.specific_share_to_other_effects_operation.items():
                target_model = self._effect_models[target_effect].operation
                origin_model = self._effect_models[origin_effect].operation
                target_model.add_variable_share(name_of_share, origin_effect, origin_model.sum_TS, factor, 1)
            # 2. invest:    -> hier ist es Skalar (share)
            name_of_share = 'specificShareToOtherEffects_invest_'  # + effectType.label
            for target_effect, factor in origin_effect.specific_share_to_other_effects_invest.items():
                target_model = self._effect_models[target_effect].invest
                origin_model = self._effect_models[origin_effect].invest
                target_model.add_variable_share(name_of_share, origin_effect, origin_model.sum, factor, 1)


def _extract_sample_points(data: List[Skalar]) -> List[Tuple[Skalar, Skalar]]:
    assert len(data) % 2 == 0, f'Segments must have an even number of start-/endpoints'
    return [(data[i], data[i+1]) for i in range(0, len(data), 2)]


class InvestmentModel(ElementModel):
    """Class for modeling an investment"""
    def __init__(self, element: Union[Flow, Storage],
                 invest_parameters: InvestParameters,
                 defining_variable: Optional[VariableTS],
                 fixed_relative_value: Optional[Numeric],
                 relative_minimum: Numeric,
                 relative_maximum: Numeric,
                 on_variable: Optional[VariableTS]):
        super().__init__(element)
        self.element: Union[Flow, Storage] = element
        self.size: Optional[Union[Skalar, Variable]] = None
        self.is_invested: Optional[Variable] = None

        self._defining_variable = defining_variable
        self._on_variable = on_variable
        self._fixed_relative_value = fixed_relative_value
        self._relative_minimum = relative_minimum
        self._relative_maximum = relative_maximum
        
        self._segments: Optional[SegmentedSharesModel] = None
        
        self._invest_parameters = invest_parameters

    def do_modeling(self, system_model: SystemModel):
        effect_collection = system_model.flow_system.effect_collection
        invest_parameters = self._invest_parameters
        if invest_parameters.fixed_size:
            self.size = Variable('size', 1, self.element.label_full, system_model,
                                 value=invest_parameters.fixed_size)
        else:
            lower_bound = 0 if invest_parameters.optional else invest_parameters.minimum_size
            self.size = Variable('size', 1, self.element.label_full, system_model,
                                 lower_bound=lower_bound, upper_bound=invest_parameters.maximum_size)
        self.add_variables(self.size)
        # Optional
        if invest_parameters.optional:
            self.is_invested = Variable('isInvested', 1, self.element.label_full, system_model,
                                        is_binary=True)
            self.add_variables(self.is_invested)
            self._create_bounds_for_optional_investment(system_model)

        self._create_bounds_for_defining_var(system_model)

        #############################################################
        # fix_effects:
        fix_effects = invest_parameters.fix_effects
        if fix_effects is not None and fix_effects != 0:
            if invest_parameters.optional:  # share: + isInvested * fix_effects
                effect_collection.add_share_to_invest('fix_effects', self.element,
                                                      self.is_invested, fix_effects, 1)
            else:  # share: + fix_effects
                effect_collection.add_constant_share_to_invest('fix_effects', self.element,
                                                               fix_effects,1)
        # divest_effects:
        divest_effects = invest_parameters.divest_effects
        if divest_effects is not None and divest_effects != 0:
            if invest_parameters.optional:  # share: [divest_effects - isInvested * divest_effects]
                # 1. part of share [+ divest_effects]:
                effect_collection.add_constant_share_to_invest('divest_effects', self.element,
                                                               divest_effects, 1)
                # 2. part of share [- isInvested * divest_effects]:
                effect_collection.add_share_to_invest('divest_cancellation_effects', self.element,
                                                      self.is_invested, divest_effects, -1)
                # TODO : these 2 parts should be one share! -> SingleShareModel...?

        # # specific_effects:
        specific_effects = invest_parameters.specific_effects
        if specific_effects is not None:
            # share: + investment_size (=var)   * specific_effects
            effect_collection.add_share_to_invest('specific_effects', self.element,
                                                  self.size, specific_effects, 1)
        # segmented Effects
        invest_segments = invest_parameters.effects_in_segments
        if invest_segments:
            self._segments = SegmentedSharesModel(self.element,
                                                  (self.size, invest_segments[0]),
                                                  invest_segments[1], self.is_invested)
            self.sub_models.append(self._segments)
            self._segments.do_modeling(system_model)

    def _create_bounds_for_optional_investment(self, system_model: SystemModel):
        if self._invest_parameters.fixed_size:
            # eq: investment_size = isInvested * fixed_size
            eq_is_invested = Equation('is_invested', self.element, system_model, 'eq')
            eq_is_invested.add_summand(self.size, -1)
            eq_is_invested.add_summand(self.is_invested, self._invest_parameters.fixed_size)
            self.add_equations(eq_is_invested)
        else:
            # eq1: P_invest <= isInvested * investSize_max
            eq_is_invested_ub = Equation('is_invested_ub', self.element, system_model, 'ineq')
            eq_is_invested_ub.add_summand(self.size, 1)
            eq_is_invested_ub.add_summand(self.is_invested, np.multiply(-1, self._invest_parameters.maximum_size))

            # eq2: P_invest >= isInvested * max(epsilon, investSize_min)
            eq_is_invested_lb = Equation('is_invested_lb', self.element, system_model, 'ineq')
            eq_is_invested_lb.add_summand(self.size, -1)
            eq_is_invested_lb.add_summand(self.is_invested, np.max(system_model.epsilon, self._invest_parameters.minimum_size))
            self.add_equations(eq_is_invested_ub, eq_is_invested_lb)

    def _create_bounds_for_defining_var(self, system_model: SystemModel):
        if self._fixed_relative_value is not None:  # Wenn fixer relativer Lastgang:
            # eq: defining_variable(t) = var_investmentSize * fixed_relative_value
            eq = Equation('fixed_relative_value', self.element, system_model, 'eq')
            eq.add_summand(self._defining_variable, 1)
            eq.add_summand(self.size, np.multiply(-1, self._fixed_relative_value))
            self.add_equations(eq)
        else:
            ## 1. Gleichung: Maximum durch Investmentgröße ##
            eq_upper = Equation('ub_defining_var', self.element, system_model, 'ineq')
            # eq: defining_variable(t)  <= size * relative_maximum(t)
            eq_upper.add_summand(self._defining_variable, 1)
            eq_upper.add_summand(self.size, np.multiply(-1, self._relative_maximum))

            ## 2. Gleichung: Minimum durch Investmentgröße ##
            eq_lower = Equation('lb_defining_var', self, system_model, 'ineq')
            if self._on_variable is None:
                # eq: defining_variable(t) >= investment_size * relative_minimum(t)
                eq_lower.add_summand(self._defining_variable, -1)
                eq_lower.add_summand(self.size, self._relative_minimum)

            # Wenn InvestSize nicht fix, dann weitere Glg notwendig für Minimum (abhängig von var_investSize)
            elif not self._invest_parameters.fixed_size:
                # eq: defining_variable(t) >= mega * (On(t)-1) + size * relative_minimum(t)
                #     ... mit mega = relative_maximum * maximum_size

                # äquivalent zu:.
                # eq: - defining_variable(t) + mega * On(t) + size * relative_minimum(t) <= + mega

                mega = self._relative_maximum * self._invest_parameters.maximum_size
                eq_lower.add_summand(self._defining_variable, -1)
                eq_lower.add_summand(self._on_variable, mega)  # übergebene On-Variable
                eq_lower.add_summand(self.size, self._relative_minimum)
                eq_lower.add_constant(mega)
                # Anmerkung: Glg bei Spezialfall relative_minimum = 0 redundant zu FeatureOn-Glg.
            else:
                pass  # Bereits in FeatureOn mit P>= On(t)*Min ausreichend definiert
                #TODO: CHECK THIS!!!

            self.add_equations(eq_lower, eq_upper)


class OnOffModel(ElementModel):
    """Class for modeling the on and off state of a variable"""
    def __init__(self, element: Element, on_off_parameters: OnOffParameters):
        super().__init__(element)
        self.element = element
        self.on: Optional[VariableTS] = None
        self.total_on_hours: Optional[Variable] = None

        self.consecutive_on_hours: Optional[VariableTS] = None
        self.consecutive_off_hours: Optional[VariableTS] = None

        self.off: Optional[VariableTS] = None

        self.switch_on: Optional[VariableTS] = None
        self.switch_off: Optional[VariableTS] = None
        self.nr_switch_on: Optional[VariableTS] = None

        self._on_off_parameters = on_off_parameters

    def do_modeling(self, system_model: SystemModel):
        if self._on_off_parameters.use_on:
            # Before-Variable:
            self.on = VariableTS('on', system_model.nrOfTimeSteps, self.element.label_full, system_model,
                                 is_binary=True, before_value=self._on_off_parameters.on_values_before_begin[0])
            self.total_on_hours = Variable('onHoursSum', 1, self.element.label_full, system_model,
                                           lower_bound=self._on_off_parameters.on_hours_total_min,
                                           upper_bound=self._on_off_parameters.on_hours_total_max)
            self.add_variables(self.on)
            self.add_variables(self.total_on_hours)

        if self._on_off_parameters.use_off:
            # off-Var is needed:
            self.off = VariableTS('off', system_model.nrOfTimeSteps, self.element.label_full, system_model,
                                  is_binary=True)
            self.add_variables(self.off)

        # onHours:
        #   var_on      = [0 0 1 1 1 1 0 0 0 1 1 1 0 ...]
        #   var_onHours = [0 0 1 2 3 4 0 0 0 1 2 3 0 ...] (bei dt=1)
        if self._on_off_parameters.use_on_hours:
            maximum_consecutive_on_hours = None if self._on_off_parameters.consecutive_on_hours_max is None \
                else self._on_off_parameters.consecutive_on_hours_max.active_data
            self.consecutive_on_hours = VariableTS('onHours', system_model.nrOfTimeSteps,
                                                   self.element.label_full, system_model,
                                                   lower_bound=0,
                                                   upper_bound=maximum_consecutive_on_hours)  # min separat
            self.add_variables(self.consecutive_on_hours)
        # offHours:
        if self._on_off_parameters.use_off_hours:
            maximum_consecutive_off_hours = None if self._on_off_parameters.consecutive_off_hours_max is None \
                else self._on_off_parameters.consecutive_off_hours_max.active_data
            self.consecutive_off_hours = VariableTS('offHours', system_model.nrOfTimeSteps,
                                                   self.element.label_full, system_model,
                                                   lower_bound=0,
                                                   upper_bound=maximum_consecutive_off_hours)  # min separat
            self.add_variables(self.consecutive_off_hours)
        # Var SwitchOn
        if self._on_off_parameters.use_switch_on:
            self.switch_on = VariableTS('switchOn', system_model.nrOfTimeSteps, self.element.label_full, system_model, is_binary=True)
            self.switch_off = VariableTS('switchOff', system_model.nrOfTimeSteps, self.element.label_full, system_model, is_binary=True)
            self.nr_switch_on = Variable('nrSwitchOn', 1, self.element.label_full, system_model,
                                         upper_bound=self._on_off_parameters.switch_on_total_max)
            self.add_variables(self.switch_on)
            self.add_variables(self.switch_off)
            self.add_variables(self.nr_switch_on)

        #TODO: Equations!!


class SegmentModel(ElementModel):
    """Class for modeling a linear segment of one or more variables in parallel"""
    def __init__(self, element: Element, segment_index: Union[int, str],
                 sample_points: Dict[Variable, Tuple[Union[Numeric, TimeSeries], Union[Numeric, TimeSeries]]]):
        super().__init__(element)
        self.element = element
        self.in_segment: Optional[VariableTS] = None
        self.lambda0: Optional[VariableTS] = None
        self.lambda1: Optional[VariableTS] = None

        self._segment_index = segment_index
        self._sample_points = sample_points

    def do_modeling(self, system_model: SystemModel):
        length = system_model.nrOfTimeSteps
        self.in_segment = VariableTS(f'onSeg_{self._segment_index}', length, self.element.label_full, system_model,
                                     is_binary=True)  # Binär-Variable
        self.lambda0 = VariableTS(f'lambda0_{self._segment_index}', length, self.element.label_full, system_model,
                                  lower_bound=0, upper_bound=1)  # Wertebereich 0..1
        self.lambda1 = VariableTS(f'lambda1_{self._segment_index}', length, self.element.label_full, system_model,
                                  lower_bound=0, upper_bound=1)  # Wertebereich 0..1
        self.add_variables(self.in_segment)
        self.add_variables(self.lambda0)
        self.add_variables(self.lambda1)

        # eq: -aSegment.onSeg(t) + aSegment.lambda1(t) + aSegment.lambda2(t)  = 0
        equation = Equation(f'Lambda_onSeg_{self._segment_index}', self, system_model)
        self.add_equations(equation)

        equation.add_summand(self.in_segment, -1)
        equation.add_summand(self.lambda0, 1)
        equation.add_summand(self.lambda1, 1)

        #  eq: - v(t) + (v_0 * lambda_0 + v_1 * lambda_1) = 0       -> v_0, v_1 = Stützstellen des Segments
        for variable, sample_points in self._sample_points.items():
            sample_0, sample_1 = sample_points
            if isinstance(sample_0, TimeSeries):
                sample_0 = sample_0.active_data
                sample_1 = sample_1.active_data
            else:
                sample_0 = sample_0
                sample_1 = sample_1

            lambda_eq = Equation(f'{variable.label_full}_lambda', self, system_model)
            lambda_eq.add_summand(variable, -1)
            lambda_eq.add_summand(self.lambda0, sample_0)
            lambda_eq.add_summand(self.lambda1, sample_1)
            self.add_equations(lambda_eq)


class MultipleSegmentsModel(ElementModel):
    def __init__(self, element: Element,
                 sample_points: Dict[Variable, List[Tuple[Union[Numeric, TimeSeries], Union[Numeric, TimeSeries]]]],
                 outside_segments: Optional[Variable] = None):
        super().__init__(element)
        self.element = element

        self.outside_segments: Optional[VariableTS] = outside_segments  # Variable to allow being outside segments = 0

        self._sample_points = sample_points
        self._segment_models: List[SegmentModel] = []

    def do_modeling(self, system_model: SystemModel):
        restructured_variables_with_segments: List[Dict[Variable, Tuple[Numeric, Numeric]]] = [
            {key: values[i] for key, values in self._sample_points.items()}
            for i in range(self._nr_of_segments)
        ]

        for i, sample_points in enumerate(restructured_variables_with_segments):
            self._segment_models.append(SegmentModel(self.element, i, sample_points))

        for segment_model in self._segment_models:
            segment_model.do_modeling(system_model)

        # Outside of Segments
        if self.outside_segments is None:  # TODO: Make optional
            self.outside_segments = VariableTS(f'outside_segments', system_model.nrOfTimeSteps, self.element.label_full,
                                          system_model, is_binary=True)
            self.add_variables(self.outside_segments)

        # a) eq: Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 1                Aufenthalt nur in Segmenten erlaubt
        # b) eq: -On(t) + Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 0       zusätzlich kann alles auch Null sein
        in_single_segment = Equation('in_single_Segment', self, system_model)
        for segment_model in self._segment_models:
            in_single_segment.add_summand(segment_model.in_segment, 1)
        if self.outside_segments is None:
            in_single_segment.add_constant(1)
        else:
            in_single_segment.add_summand(self.outside_segments, -1)

    @property
    def _nr_of_segments(self):
        return len(next(iter(self._sample_points.values())))


class ShareAllocationModel(ElementModel):
    def __init__(self,
                 element: Element,
                 shares_are_time_series: bool,
                 total_max: Optional[Skalar] = None,
                 total_min: Optional[Skalar] = None,
                 max_per_hour: Optional[TimeSeries] = None,
                 min_per_hour: Optional[TimeSeries] = None):
        super().__init__(element)
        if not shares_are_time_series:  # If the condition is True
            assert max_per_hour is None and min_per_hour is None, \
                "Both max_per_hour and min_per_hour cannot be used when shares_are_time_series is False"
        self.element = element
        self.sum_TS: Optional[VariableTS] = None
        self.sum: Optional[Variable] = None

        self._eq_time_series: Optional[Equation] = None
        self._eq_sum: Optional[Equation] = None

        # Parameters
        self._shares_are_time_series = shares_are_time_series
        self._total_max = total_max
        self._total_min = total_min
        self._max_per_hour = max_per_hour
        self._min_per_hour = min_per_hour

    def do_modeling(self, system_model: SystemModel):
        self.sum = Variable('sum', 1, self.element.label_full, system_model,
                            lower_bound=self._total_min, upper_bound=self._total_max)
        self.add_variables(self.sum)
        # eq: sum = sum(share_i) # skalar
        self._eq_sum = Equation('sum', self, system_model)
        self._eq_sum.add_summand(self.sum, -1)
        self.add_equations(self._eq_sum)

        if self._shares_are_time_series:
            lb_TS = None if (self._min_per_hour is None) else np.multiply(self._min_per_hour.active_data, system_model.dt_in_hours)
            ub_TS = None if (self._max_per_hour is None) else np.multiply(self._max_per_hour.active_data, system_model.dt_in_hours)
            self.sum_TS = VariableTS('sum_TS', system_model.nrOfTimeSteps, self.element.label_full, system_model,
                                     lower_bound=lb_TS, upper_bound=ub_TS)
            self.add_variables(self.sum_TS)

            # eq: sum_TS = sum(share_TS_i) # TS
            self._eq_time_series = Equation('time_series', self, system_model)
            self._eq_time_series.add_summand(self.sum_TS, -1)
            self.add_equations(self._eq_time_series)

            # eq: sum = sum(sum_TS(t)) # additionaly to self.sum
            self._eq_sum.add_summand(self.sum_TS, 1, as_sum=True)

    def add_variable_share(self,
                           system_model: SystemModel,
                           name_of_share: Optional[str],
                           share_holder: Element,
                           variable: Variable,
                           factor1: Numeric_TS,
                           factor2: Numeric_TS):  # if variable = None, then fix Share
        if variable is None:
            raise Exception('add_variable_share() needs variable as input. Use add_constant_share() instead')
        self._add_share(system_model, name_of_share, share_holder, variable, factor1, factor2)

    def add_constant_share(self,
                           system_model: SystemModel,
                           name_of_share: Optional[str],
                           share_holder: Element,
                           factor1: Numeric_TS,
                           factor2: Numeric_TS):
        variable = None
        self._add_share(system_model, name_of_share, share_holder, variable, factor1, factor2)

    def _add_share(self,
                   system_model: SystemModel,
                   name_of_share: Optional[str],
                   share_holder: Element,
                   variable: Optional[Variable],
                   factor1: Numeric_TS,
                   factor2: Numeric_TS):
        #TODO: accept only one factor or accept unlimited factors -> *factors

        # Falls TimeSeries, Daten auslesen:
        if isinstance(factor1, TimeSeries):
            factor1 = factor1.active_data
        if isinstance(factor2, TimeSeries):
            factor2 = factor2.active_data
        total_factor = np.multiply(factor1, factor2)

        # var and eq for publishing share-values in results:
        if name_of_share is not None:  #TODO: is this check necessary?
            new_share = SingleShareModel(share_holder, self._shares_are_time_series, name_of_share)
            new_share.do_modeling(system_model)
            new_share.add_summand_to_share(variable, total_factor)

        # Check to which equation the share should be added
        if self._shares_are_time_series:
            target_eq = self._eq_time_series
        else:
            assert total_factor.shape[0] == 1, (f'factor1 und factor2 müssen Skalare sein, '
                                                f'da shareSum {self.element.label} skalar ist')
            target_eq = self._eq_sum

        if variable is None:  # constant share
            target_eq.add_constant(-1 * total_factor)
        else:  # variable share
            target_eq.add_summand(variable, total_factor)
        #TODO: Instead use new_share.single_share: Variable ?


class SingleShareModel(ElementModel):
    def __init__(self, element: Element, shares_are_time_series: bool, name_of_share: str):
        super().__init__(element)
        self.single_share: Optional[Variable] = None
        self._equation: Optional[Equation] = None
        self._full_name_of_share = f'{element.label_full}_{name_of_share}'
        self._shares_are_time_series = shares_are_time_series
        self._name_of_share = name_of_share

    def do_modeling(self, system_model: SystemModel):
        self.single_share = Variable(self._full_name_of_share, 1, self.element.label_full, system_model)
        self.add_variables(self.single_share)

        self._equation = Equation(self._full_name_of_share, self, system_model)
        self._equation.add_summand(self.single_share, -1)
        self.add_equations(self._equation)

    def add_summand_to_share(self, variable: Optional[Variable], factor: Numeric):
        """share to a sum"""
        if variable is None:  # if constant share:
            constant_value = np.sum(factor) if self._shares_are_time_series else factor
            self._equation.add_constant(-1 * constant_value)
        else:  # if variable share - always as a skalar -> as_sum=True if shares are timeseries
            self._equation.add_summand(variable, factor, as_sum=self._shares_are_time_series)


class SegmentedSharesModel(ElementModel):
    def __init__(self,
                 element: Element,
                 variable_segments: Tuple[Variable, List[Tuple[Skalar, Skalar]]],
                 share_segments: Dict[Effect, List[Tuple[Skalar, Skalar]]],
                 outside_segments: Optional[Variable]):
        super().__init__(element)
        assert len(variable_segments[1]) == len(list(share_segments.values())[0]), \
            f'Segment length of variable_segments and share_segments must be equal'
        self.element: Element
        self._outside_segments = outside_segments
        self._variable_segments = variable_segments
        self._share_segments = share_segments
        self._shares: Optional[Dict[Effect, SingleShareModel]] = None
        self._segments_model: Optional[MultipleSegmentsModel] = None

    def do_modeling(self, system_model: SystemModel):
        self._shares = {effect: SingleShareModel(self.element, False, f'segmented')
                        for effect in self._share_segments}
        for single_share in self._shares.values():
            single_share.do_modeling(system_model)

        segments: Dict[Variable, List[Tuple[Skalar, Skalar]]] = {
            self._shares[effect].single_share: segment for effect, segment in self._share_segments.values()}
        self._segments_model = MultipleSegmentsModel(self.element, self._outside_segments, segments)
        self._segments_model.do_modeling(system_model)

        # Shares
        effect_collection = system_model.flow_system.effect_collection
        for effect, single_share_model in self._shares.items():
            effect_collection.add_share_to_invest(
                name_of_share='segmented_effects',
                owner=self.element, variable=single_share_model.single_share,
                effect_values={effect: 1},
                factor=1
            )


def bounds_of_defining_variable(minimum_size: Skalar,
                                maximum_size: Skalar,
                                size_is_optional: bool,
                                fixed_relative_value: Optional[Numeric],
                                relative_minimum: Numeric,
                                relative_maximum: Numeric,
                                can_be_off: bool,
                                ) -> Tuple[Optional[Numeric], Optional[Numeric], Optional[Numeric]]:

    if fixed_relative_value is not None:   # Wenn fixer relativer Lastgang:
        # relative_maximum = relative_minimum = fixed_relative_value !
        relative_minimum = fixed_relative_value
        relative_maximum = fixed_relative_value

    upper_bound = relative_maximum * maximum_size  # Use maximum of Investment

    # min-Wert:
    if size_is_optional or (can_be_off and fixed_relative_value is None):
        lower_bound = 0  # can be zero (if no invest) or can switch off
    else:
        lower_bound = relative_minimum * minimum_size   # Use minimum of Investment

    # upper_bound und lower_bound gleich, dann fix:
    if np.all(upper_bound == lower_bound):  # np.all -> kann listen oder werte vergleichen
        return None, None, upper_bound
    else:
        return lower_bound, upper_bound, None
