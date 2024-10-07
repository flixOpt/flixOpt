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
from flixOpt.math_modeling import MathModel, Variable, VariableTS, Equation, Summand  # Modelliersprache
from flixOpt.core import TimeSeries, Skalar, Numeric, Numeric_TS, as_effect_dict

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from flixOpt.elements import Flow, Effect
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

    @property
    def ineqs(self) -> Dict[str, Equation]:
        return {name: eq for name, eq in self.eqs.items() if eq.eqType == 'ineq'}


from flixOpt.elements import Effect, EffectCollection, Objective, Flow


class FlowModel(ElementModel):
    def __init__(self, element: Flow):
        super().__init__(element)
        self.element = element
        self.flow_rate: Optional[VariableTS] = None
        self.sum_flow_hours: Optional[Variable] = None

        self._on: Optional[OnModel] = None if element.featureOn else OnModel(element.featureOn)
        self._investment: Optional[InvestmentModel] = None if element.featureInvest is None else InvestmentModel(element.featureInvest, element.featureOn.use_on)

    def create_variables(self, system_model: SystemModel):
        # flow_rate
        if isinstance(self.element.size, Skalar):
            fixed_value = None if self.element.fixed_relative_value is None else self.element.fixed_relative_value * self.element.size
            self.flow_rate = VariableTS('flow_rate', system_model.nrOfTimeSteps, self.element.label_full, system_model,
                                        lower_bound=self.element.relative_minimum_with_exists.active_data * self.element.size,
                                        upper_bound=self.element.relative_maxiumum_with_exists.active_data * self.element.size,
                                        value=fixed_value)
        else:
            self.flow_rate = VariableTS('flow_rate', system_model.nrOfTimeSteps, self.element.label_full, system_model)
        self.add_variable(self.flow_rate)

        # sumFLowHours
        self.sum_flow_hours = Variable('sumFlowHours', 1, self.element.label_full, system_model,
                                       lower_bound=self.element.flow_hours_total_min,
                                       upper_bound=self.element.flow_hours_total_max)
        self.add_variable(self.sum_flow_hours)

        self._on.create_variables(system_model) if self._on is not None else None
        self._investment.create_variables(system_model, ) if self._investment is not None else None

    def create_equations(self, system_model: SystemModel):
        # sumFLowHours
        eq_sumFlowHours = Equation('sumFlowHours', self, system_model, 'eq')
        eq_sumFlowHours.add_summand(self.flow_rate, system_model.dt_in_hours, as_sum=True)
        eq_sumFlowHours.add_summand(self.sum_flow_hours, -1)
        self.add_equation(eq_sumFlowHours)

        # load factor #  eq: var_sumFlowHours <= size * dt_tot * load_factor_max
        if self.element.load_factor_max is not None:
            flow_hours_per_size_max = system_model.dt_in_hours_total * self.element.load_factor_max
            eq_load_factor_max = Equation('load_factor_max', self, system_model, 'ineq')
            self.add_equation(eq_load_factor_max)
            eq_load_factor_max.add_summand(self.sum_flow_hours, 1)
            eq_load_factor_max.add_constant(self.element.size * flow_hours_per_size_max)
            # if investment:
            if self.featureInvest is not None:
                eq_load_factor_max.add_summand(self.featureInvest.model.variables[self.featureInvest.name_of_investment_size],
                                                          -1 * flow_hours_per_size_max)

    def create_shares(self, system_model: SystemModel):
        # Arbeitskosten:
        effect_collection = system_model.flow_system.effect_collection
        if self.element.effects_per_flow_hour is not None:
            effect_collection.add_share_to_operation(
                name_of_share='effects_per_flow_hour',
                owner=self.element, variable=self.flow_rate,
                effect_values=self.element.effects_per_flow_hour,
                factor=system_model.dt_in_hours
            )


class EffectModel(ElementModel):
    def __init__(self, element: Effect):
        super().__init__(element)
        self.element = element
        self._invest = ShareAllocationModel(element.invest)
        self._operation = ShareAllocationModel(element.operation)
        self._all = ShareAllocationModel(element.all)
        self.sub_models.extend([self._invest, self._operation, self._all])

    def create_variables(self, system_model: SystemModel):
        for model in (self._invest, self._operation, self._all):
            model.create_variables(system_model)

    def create_equations(self, system_model: SystemModel):
        for model in (self._invest, self._operation, self._all):
            model.create_equations(system_model)
        self._all.add_variable_share('operation', self.element, self._operation.sum, 1, 1)
        self._all.add_variable_share('invest', self.element, self._invest.sum, 1, 1)


class EffectCollectionModel(ElementModel):
    #TODO: Maybe all EffectModels should be sub_models of this Model? Including Objective and Penalty?
    def __init__(self, element: EffectCollection, system_model: SystemModel):
        super().__init__(element)
        self.element = element
        self._system_model = system_model
        self._effect_models: Dict[Effect, EffectModel] = {}
        self._penalty: Optional[ShareAllocationModel] = None

    def create_variables(self, system_model: SystemModel):
        self._effect_models = {effect: effect.model for effect in self.element.effects}
        for model in self._effect_models.values():
            model.create_variables(system_model)
        self._penalty = ShareAllocationModel(self.element.penalty)
        self._penalty.create_variables(system_model)

    def create_equations(self, system_model: SystemModel):
        for model in self._effect_models.values():
            model.create_equations(system_model)
        self._penalty.create_equations(system_model)

    def add_share(self,
                  operation_or_invest: Literal['operation', 'invest'],
                  name_of_share: str,
                  owner: Element,
                  effect_values: Union[Numeric, Dict[Optional[Effect], TimeSeries]],
                  factor: Numeric,
                  variable: Optional[Variable] = None) -> None:
        assert operation_or_invest in ('operation', 'invest'), f'{operation_or_invest} not supported'
        effect_values_dict = as_effect_dict(effect_values)

        # an alle Effects, die einen Wert haben, anhängen:
        for effect, value in effect_values_dict.items():
            if effect is None:  # Falls None, dann Standard-effekt nutzen:
                effect = self.element.standard_effect
            assert effect in self.element.effects, f'Effect {effect.label} was used but not added to model!'

            if operation_or_invest == 'operation':
                model = self._effect_models[effect]._operation
            else:
                model = self._effect_models[effect]._invest

            model._add_share(self._system_model, name_of_share, owner, variable, value, factor)  # hier darf aVariable auch None sein!

    def add_share_between_effects(self):
        for origin_effect in self.element.effects:
            # 1. operation: -> hier sind es Zeitreihen (share_TS)
            name_of_share = 'specific_share_to_other_effects_operation'  # + effectType.label
            for target_effect, factor in origin_effect.specific_share_to_other_effects_operation.items():
                target_model = self._effect_models[target_effect]._operation
                origin_model = self._effect_models[origin_effect]._operation
                target_model.add_variable_share(name_of_share, origin_effect, origin_model.sum_TS, factor, 1)
            # 2. invest:    -> hier ist es Skalar (share)
            name_of_share = 'specificShareToOtherEffects_invest_'  # + effectType.label
            for target_effect, factor in origin_effect.specific_share_to_other_effects_invest.items():
                target_model = self._effect_models[target_effect]._invest
                origin_model = self._effect_models[origin_effect]._invest
                target_model.add_variable_share(name_of_share, origin_effect, origin_model.sum, factor, 1)


from flixOpt.features import FeatureInvest, FeatureOn, Segment, FeatureLinearSegmentVars, Feature, FeatureShares, Feature_ShareSum


def _extract_sample_points(data: List[Skalar]) -> List[Tuple[Skalar, Skalar]]:
    assert len(data) % 2 == 0, f'Segments must have an even number of start-/endpoints'
    return [(data[i], data[i+1]) for i in range(0, len(data), 2)]


class InvestmentModel(ElementModel):
    """Class for modeling an investment"""
    def __init__(self, element: FeatureInvest, with_on_variable: bool):
        super().__init__(element)
        self.element = element
        self.size: Optional[Union[Skalar, Variable]] = None
        self.is_invested: Optional[Variable] = None

        self._with_on_variable = with_on_variable
        #TODO: Add Submodel for Segmented Costs

    def create_variables(self, system_model: SystemModel):
        if self.element.invest_parameters.fixed_size:
            self.size = Variable('size', 1, self.element.label_full, system_model,
                                 value=self.element.invest_parameters.fixed_size)
        else:
            lower_bound = 0 if self.element.invest_parameters.optional else self.element.invest_parameters.minimum_size
            self.size = Variable('size', 1, self.element.label_full, system_model,
                                 lower_bound=lower_bound, upper_bound=self.element.invest_parameters.maximum_size)
        self.add_variable(self.size)
        # Optional
        if self.element.invest_parameters.optional:
            self.is_invested = Variable('isInvested', 1, self.element.label_full, system_model,
                                        is_binary=True)
        self.add_variable(self.is_invested)

    def create_equations(self, system_model: SystemModel):
        if self.element.invest_parameters.optional:
            self._create_bounds_for_variable_size(system_model)

        if self.element.fixed_relative_value is not None:  # Wenn fixer relativer Lastgang:
            self._create_bounds_for_fixed_defining_var(system_model)
        else:   # Wenn nicht fix:
            self._create_bounds_for_defining_var(system_model, on_is_used=self._with_on_variable)

        if self.featureLinearSegments is not None:  # if linear Segments defined:
            self.featureLinearSegments.do_modeling(system_model)

    def create_shares(self, system_model: SystemModel):
        effect_collection = system_model.flow_system.effect_collection
        invest_parameters = self.element.invest_parameters

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
                effect_collection.add_share_to_invest('divestCosts_cancellation', self.element,
                                                      self.is_invested, divest_effects, -1)
                # TODO : these 2 parts should be one share!

        # # specific_effects:
        specific_effects = invest_parameters.specific_effects
        if specific_effects is not None:
            # share: + investment_size (=var)   * specific_effects
            effect_collection.add_share_to_invest('specific_effects', self.element,
                                                  self.size, specific_effects, 1)

        # # segmentedCosts:
        if self.element.featureLinearSegments is not None:
            for effect, var_investSegs in self.element.investVar_effect_dict.items():
                effect_collection.add_share_to_invest('linearSegments', self.element, var_investSegs, {effect: 1}, 1)


    def _create_bounds_for_variable_size(self, system_model: SystemModel):
        if self.element.invest_parameters.fixed_size:
            # eq: investment_size = isInvested * fixed_size
            isInvested_constraint_1 = Equation('isInvested_constraint_1', self.element, system_model, 'eq')
            self.add_equation(isInvested_constraint_1)
            isInvested_constraint_1.add_summand(self.size, -1)
            isInvested_constraint_1.add_summand(self.is_invested, self.element.invest_parameters.fixed_size)
        else:
            # eq1: P_invest <= isInvested * investSize_max
            self.add_equation(Equation('isInvested_constraint_1', self.element, system_model, 'ineq'))
            self.eqs['isInvested_constraint_1'].add_summand(self.size, 1)
            self.eqs['isInvested_constraint_1'].add_summand(
                self.variables['isInvested'], np.multiply(-1, self.element.invest_parameters.maximum_size)
            )

            # eq2: P_invest >= isInvested * max(epsilon, investSize_min)
            self.add_equation(Equation('isInvested_constraint_2', self.element, system_model, 'ineq'))
            self.eqs['isInvested_constraint_2'].add_summand(self.size, -1)
            self.eqs['isInvested_constraint_2'].add_summand(
                self.variables['isInvested'], max(system_model.epsilon, self.element.invest_parameters.minimum_size))

    def _create_bounds_for_fixed_defining_var(self, system_model: SystemModel):
        # eq: defining_variable(t) = var_investmentSize * fixed_relative_value
        self.add_equation(Equation('fix_via_InvestmentSize', self, system_model, 'eq'))
        self.eqs['fix_via_InvestmentSize'].add_summand(self.element.defining_variable, 1)
        self.eqs['fix_via_InvestmentSize'].add_summand(
            self.size, np.multiply(-1, self.element.fixed_relative_value.active_data)
        )

    def _create_bounds_for_defining_var(self, system_model: SystemModel, on_is_used: bool):

        ## 1. Gleichung: Maximum durch Investmentgröße ##
        # eq: defining_variable(t) <=                var_investmentSize * relative_maximum(t)
        # eq: P(t) <= relative_maximum(t) * P_inv
        self.add_equation(Equation('max_via_InvestmentSize', self, system_model, 'ineq'))
        self.eqs['max_via_InvestmentSize'].add_summand(self.element.defining_variable, 1)
        # self.eqs['max_via_InvestmentSize'].add_summand(self.size, np.multiply(-1, self.element.relative_maximum.active_data))
        self.eqs['max_via_InvestmentSize'].add_summand(self.size, np.multiply(-1, self.element.relative_maximum.data))
        # TODO: BUGFIX: Here has to be active_data, but it throws an error for storages (length)
        # TODO: Changed by FB

        ## 2. Gleichung: Minimum durch Investmentgröße ##

        # Glg nur, wenn nicht Kombination On und fixed:
        if not (on_is_used and self.element.invest_parameters.fixed_size):
            self.add_equation(Equation('min_via_investmentSize', self, system_model, 'ineq'))

        if on_is_used:
            # Wenn InvestSize nicht fix, dann weitere Glg notwendig für Minimum (abhängig von var_investSize)
            if not self.element.invest_parameters.fixed_size:
                # eq: defining_variable(t) >= Big * (On(t)-1) + investment_size * relative_minimum(t)
                #     ... mit Big = max(relative_minimum*P_inv_max, epsilon)
                # (P < relative_minimum*P_inv -> On=0 | On=1 -> P >= relative_minimum*P_inv)

                # äquivalent zu:.
                # eq: - defining_variable(t) + Big * On(t) + relative_minimum(t) * investment_size <= Big

                Big = utils.get_max_value(
                    self.element.relative_minimum.active_data * self.element.invest_parameters.maximum_size,
                    system_model.epsilon)
                self.eqs['min_via_investmentSize'].add_summand(self.element.defining_variable, -1)
                self.eqs['min_via_investmentSize'].add_summand(self.element.defining_on_variable, Big)  # übergebene On-Variable
                self.eqs['min_via_investmentSize'].add_summand(self.size, self.element.relative_minimum.active_data)
                self.eqs['min_via_investmentSize'].add_constant(Big)
                # Anmerkung: Glg bei Spezialfall relative_minimum = 0 redundant zu FeatureOn-Glg.
            else:
                pass  # Bereits in FeatureOn mit P>= On(t)*Min ausreichend definiert
        else:
            # eq: defining_variable(t) >= investment_size * relative_minimum(t)
            self.eqs['min_via_investmentSize'].add_summand(self.element.defining_variable, -1)
            self.eqs['min_via_investmentSize'].add_summand(self.size, self.element.relative_minimum.active_data)



class OnModel(ElementModel):
    """Class for modeling the on and off state of a variable"""
    def __init__(self, element: Element):
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

    def create_variables(self, system_model: SystemModel):
        if self.element.use_on:
            # Before-Variable:
            self.on = VariableTS('on', system_model.nrOfTimeSteps, self.element.label_full, system_model,
                                 is_binary=True, before_value=self.element.on_values_before_begin[0])
            self.total_on_hours = Variable('onHoursSum', 1, self.element.label_full, system_model,
                                           lower_bound=self.element.on_hours_total_min,
                                           upper_bound=self.element.on_hours_total_max)
            self.add_variable(self.on)
            self.add_variable(self.total_on_hours)

        if self.element.use_off:
            # off-Var is needed:
            self.off = VariableTS('off', system_model.nrOfTimeSteps, self.element.label_full, system_model,
                                  is_binary=True)
            self.add_variable(self.off)

        # onHours:
        #   var_on      = [0 0 1 1 1 1 0 0 0 1 1 1 0 ...]
        #   var_onHours = [0 0 1 2 3 4 0 0 0 1 2 3 0 ...] (bei dt=1)
        if self.element.use_on_hours:
            maximum_consecutive_on_hours = None if self.element.consecutive_on_hours_max is None \
                else self.element.consecutive_on_hours_max.active_data
            self.consecutive_on_hours = VariableTS('onHours', system_model.nrOfTimeSteps,
                                                   self.element.label_full, system_model,
                                                   lower_bound=0,
                                                   upper_bound=maximum_consecutive_on_hours)  # min separat
            self.add_variable(self.consecutive_on_hours)
        # offHours:
        if self.element.use_off_hours:
            maximum_consecutive_off_hours = None if self.element.consecutive_off_hours_max is None \
                else self.element.consecutive_off_hours_max.active_data
            self.consecutive_off_hours = VariableTS('offHours', system_model.nrOfTimeSteps,
                                                   self.element.label_full, system_model,
                                                   lower_bound=0,
                                                   upper_bound=maximum_consecutive_off_hours)  # min separat
            self.add_variable(self.consecutive_off_hours)
        # Var SwitchOn
        if self.element.use_switch_on:
            self.switch_on = VariableTS('switchOn', system_model.nrOfTimeSteps, self.element.label_full, system_model, is_binary=True)
            self.switch_off = VariableTS('switchOff', system_model.nrOfTimeSteps, self.element.label_full, system_model, is_binary=True)
            self.nr_switch_on = Variable('nrSwitchOn', 1, self.element.label_full, system_model,
                                         upper_bound=self.element.switch_on_total_max)
            self.add_variable(self.switch_on)
            self.add_variable(self.switch_off)
            self.add_variable(self.nr_switch_on)

    def create_equations(self, system_model: SystemModel):
        raise NotImplementedError


class SegmentModel(ElementModel):
    """Class for modeling a linear segment of one or more variables in parallel"""
    def __init__(self, element: Feature, segment_index: Union[int, str],
                 sample_points: Dict[Variable, Tuple[Union[Numeric, TimeSeries], Union[Numeric, TimeSeries]]]):
        super().__init__(element)
        self.element = element
        self.in_segment: Optional[VariableTS] = None
        self.lambda0: Optional[VariableTS] = None
        self.lambda1: Optional[VariableTS] = None

        self._segment_index = segment_index
        self._sample_points = sample_points

    def create_variables(self, system_model: SystemModel):
        length = system_model.nrOfTimeSteps
        self.in_segment = VariableTS(f'onSeg_{self._segment_index}', length, self.element.label_full, system_model,
                                     is_binary=True)  # Binär-Variable
        self.lambda0 = VariableTS(f'lambda0_{self._segment_index}', length, self.element.label_full, system_model,
                                  lower_bound=0, upper_bound=1)  # Wertebereich 0..1
        self.lambda1 = VariableTS(f'lambda1_{self._segment_index}', length, self.element.label_full, system_model,
                                  lower_bound=0, upper_bound=1)  # Wertebereich 0..1
        self.add_variable(self.in_segment)
        self.add_variable(self.lambda0)
        self.add_variable(self.lambda1)

    def create_equations(self, system_model: SystemModel):
        # eq: -aSegment.onSeg(t) + aSegment.lambda1(t) + aSegment.lambda2(t)  = 0
        name_of_equation = f'Lambda_onSeg_{self._segment_index}'
        equation = Equation(name_of_equation, self, system_model)
        self.add_equation(equation)

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

            lambda_eq = Equation(variable.label_full + '_lambda', self, system_model)
            lambda_eq.add_summand(variable, -1)
            lambda_eq.add_summand(self.lambda0, sample_0)
            lambda_eq.add_summand(self.lambda1, sample_1)
            self.add_equation(lambda_eq)


class MultipleSegmentsModel(ElementModel):
    def __init__(self, element: Element,
                 outside_segments: Optional[Variable],
                 sample_points: Dict[Variable, List[Tuple[Union[Numeric, TimeSeries], Union[Numeric, TimeSeries]]]]):
        super().__init__(element)
        self.element = element

        self.outside_segments: Optional[VariableTS] = outside_segments  # Variable to allow being outside segments = 0

        self._sample_points = sample_points
        self._segment_models: List[SegmentModel] = []

    def create_variables(self, system_model: SystemModel):
        restructured_variables_with_segments: List[Dict[Variable, Tuple[Numeric, Numeric]]] = [
            {key: values[i] for key, values in self._sample_points.items()}
            for i in range(self._nr_of_segments)
        ]

        for i, sample_points in enumerate(restructured_variables_with_segments):
            self._segment_models.append(SegmentModel(self.element, i, sample_points))

        for segment_model in self._segment_models:
            segment_model.create_variables(system_model)

        # Outside of Segments
        if self.outside_segments is None:  # TODO: Make optional
            self.outside_segments = VariableTS(f'outside_segments', system_model.nrOfTimeSteps, self.element.label_full,
                                          system_model, is_binary=True)
            self.add_variable(self.outside_segments)

    def create_equations(self, system_model: SystemModel):
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
    def __init__(self, element: Feature_ShareSum):
        super().__init__(element)
        self.element = element
        self.sum_TS: Optional[VariableTS] = None
        self.sum: Optional[Variable] = None

        self._shares: List[ShareAllocationModel] = []

        self._eq_bilanz: Optional[Equation] = None
        self._eq_sum: Optional[Equation] = None

    def create_variables(self, system_model: SystemModel):
        self.sum = Variable('sum', 1, self.element.label_full, system_model,
                            lower_bound=self.element.total_min, upper_bound=self.element.total_max)
        self.add_variable(self.sum)

        if self.element.shares_are_time_series:
            lb_TS = None if (self.element.min_per_hour is None) else np.multiply(self.element.min_per_hour.active_data, system_model.dt_in_hours)
            ub_TS = None if (self.element.max_per_hour is None) else np.multiply(self.element.max_per_hour.active_data, system_model.dt_in_hours)
            self.sum_TS = VariableTS('sum_TS', system_model.nrOfTimeSteps, self.element.label_full, system_model,
                                     lower_bound=lb_TS, upper_bound=ub_TS)
            self.add_variable(self.sum_TS)

    def create_equations(self, system_model: SystemModel):
        self._eq_sum = Equation('sum', self, system_model)
        self.add_equation(self._eq_sum)
        # eq: sum = sum(share_i) # skalar
        self._eq_sum.add_summand(self.sum, -1)
        if self.element.shares_are_time_series:
            # eq: sum = sum(sum_TS(t)) # additionaly to self.sum
            self._eq_sum.add_summand(self.sum_TS, 1, as_sum=True)

            # eq: sum_TS = sum(share_TS_i) # TS
            self._eq_bilanz = Equation('bilanz', self, system_model)
            self._eq_bilanz.add_summand(self.sum_TS, -1)
            self.add_equation(self._eq_bilanz)

    def add_variable_share(self,
                           name_of_share: Optional[str],
                           share_holder: Element,
                           variable: Variable,
                           factor1: Numeric_TS,
                           factor2: Numeric_TS):  # if variable = None, then fix Share
        if variable is None:
            raise Exception('add_variable_share() needs variable as input. Use add_constant_share() instead')
        self._add_share(name_of_share, share_holder, variable, factor1, factor2)

    def add_constant_share(self,
                           name_of_share: Optional[str],
                           share_holder: Element,
                           factor1: Numeric_TS,
                           factor2: Numeric_TS):
        variable = None
        self._add_share(name_of_share, share_holder, variable, factor1, factor2)

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
            new_share = SingleShareModel(self, self.element.shares_are_time_series)
            new_share.create_variables(system_model, share_holder, name_of_share)
            new_share.create_equations(system_model)
            new_share.add_summand_to_share(name_of_share, variable, total_factor)

        # Check to which equation the share should be added
        if self.element.shares_are_time_series:
            target_eq = self._eq_bilanz
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
        self.element = element
        self.single_share: Optional[Variable] = None
        self._equation: Optional[Equation] = None
        self._full_name_of_share: Optional[str] = None
        self._shares_are_time_series = shares_are_time_series
        self._name_of_share = name_of_share

    def create_variables(self, system_model: SystemModel):
        self._full_name_of_share = f'{self.element.label_full}_{self._name_of_share}'
        self.single_share = Variable(self._full_name_of_share, 1, self.element.label_full, system_model)
        self.add_variable(self.single_share)

    def create_equations(self, system_model: SystemModel):
        self._equation = Equation(self._full_name_of_share, self, system_model)
        self._equation.add_summand(self.single_share, -1)
        self.add_equation(self._equation)

    def add_summand_to_share(self,
                  name_of_share: Optional[str],
                  variable: Optional[Variable],
                  total_factor: Numeric):
        """share to a sum"""
        if name_of_share is not None:
            return

        if variable is None:  # if constant share:
            constant_value = sum(total_factor) if self._shares_are_time_series else total_factor
            self._equation.add_constant(-1 * constant_value)
        else:  # if variable share - always as a skalar -> as_sum=True if shares are timeseries
            self._equation.add_summand(variable, total_factor, as_sum=self._shares_are_time_series)
