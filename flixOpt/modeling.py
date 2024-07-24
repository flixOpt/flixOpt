# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:09:02 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import logging
import re
import timeit
from typing import List, Dict, Optional, Union, Literal

import numpy as np
from pyomo.contrib import appsi

from flixOpt import flixOptHelperFcts as helpers
from flixOpt.core import Skalar, Numeric

pyomoEnv = None  # das ist module, das nur bei Bedarf belegt wird

log = logging.getLogger(__name__)


class LinearModel:
    '''
    Class for equations of the form a_1*x_1 + a_2*x_2 = y
    x_1 and a_1 can be vectors or scalars.

    Model for adding vector variables and scalars:
    Allowed summands:
    - var_vec * factor_vec
    - var_vec * factor
    - factor
    - var * factor
    - var * factor_vec  # Does this make sense? Is this even implemented?
    '''

    def __init__(self,
                 label: str,
                 modeling_language: Literal['pyomo', 'cvxpy'] = 'pyomo'):
        self._infos = {}
        self.label = label
        self.modeling_language = modeling_language

        self.countComp = 0  # ElementeZähler für Pyomo
        self.epsilon = 1e-5  #

        self.solver_name: Optional[str] = None
        self.model = None  # Übergabe später, zumindest für Pyomo notwendig
        self.variables: List[Variable] = []  # Liste aller Variablen
        self.eqs: List[Equation] = []  # Liste aller Gleichungen
        self.ineqs: List[Equation] = []  # Liste aller Ungleichungen
        self.objective = None  # objective-Function
        self.objective_result = None  # Ergebnis
        self.duration = {}  # Laufzeiten
        self.solver_log = None  # logging und parsen des solver-outputs
        self.before_values: Dict[str, Numeric] = {}  # before_values, which overwrite inital before values defined in the Elements.

        if self.modeling_language == 'pyomo':
            global pyomoEnv  # als globale Variable
            import pyomo.environ as pyomoEnv
            log.info('Loaded pyomo modules')
            # für den Fall pyomo wird EIN Modell erzeugt, das auch für rollierende Durchlaufe immer wieder genutzt wird.
            self.model = pyomoEnv.ConcreteModel(name="(Minimalbeispiel)")
        elif self.modeling_language == 'cvxpy':
            raise NotImplementedError('Modeling Language cvxpy is not yet implemented')
        else:
            raise Exception('not defined for modeling_language' + str(self.modeling_language))

    def to_math_model(self) -> None:
        t_start = timeit.default_timer()
        for variable in self.variables:   # Variablen erstellen
            variable.to_math_model(self)
        for eq in self.eqs:   # Gleichungen erstellen
            eq.to_math_model(self)
        for ineq in self.ineqs:   # Ungleichungen erstellen:
            ineq.to_math_model(self)

        self.duration['to_math_model'] = round(timeit.default_timer() - t_start, 2)

    @property
    def nr_of_equations(self) -> int:
        return len(self.eqs)

    @property
    def nr_of_single_equations(self) -> int:
        return sum([eq.nr_of_single_equations for eq in self.eqs])

    @property
    def nr_of_inequations(self) -> int:
        return len(self.ineqs)

    @property
    def nr_of_single_inequations(self) -> int:
        return sum([eq.nr_of_single_equations for eq in self.ineqs])

    @property
    def nr_of_variables(self) -> int:
        return len(self.variables)

    @property
    def nr_of_single_variables(self) -> int:
        return sum([var.length for var in self.variables])


    def solve(self,
              mip_gap: float,
              time_limit_seconds: int,
              solver_name: Literal['highs', 'gurobi', 'cplex', 'glpk', 'cbc'],
              solver_output_to_console: bool,
              logfile_name: str,
              **solver_opt) -> None:
        self.solver_name = solver_name
        t_start = timeit.default_timer()
        for variable in self.variables:
            variable.reset_result()  # altes Ergebnis löschen (falls vorhanden)
        if self.modeling_language == 'pyomo':
            if solver_name == 'highs':
              solver = appsi.solvers.Highs()
            else:
              solver = pyomoEnv.SolverFactory(solver_name)
            if solver_name == 'cbc':
                solver_opt["ratio"] = mip_gap
                solver_opt["sec"] = time_limit_seconds
            elif solver_name == 'gurobi':
                solver_opt["mipgap"] = mip_gap
                solver_opt["TimeLimit"] = time_limit_seconds
            elif solver_name == 'cplex':
                solver_opt["mipgap"] = mip_gap
                solver_opt["timelimit"] = time_limit_seconds
                # todo: threads = ? funktioniert das für cplex?
            elif solver_name == 'glpk':
                # solver_opt = {} # überschreiben, keine kwargs zulässig
                # solver_opt["mipgap"] = mip_gap
                solver_opt['mipgap'] = mip_gap
            elif solver_name == 'highs':
                  solver_opt["mip_rel_gap"] = mip_gap
                  solver_opt["time_limit"] = time_limit_seconds
                  solver_opt["log_file"]= "results/highs.log"
                  solver_opt["parallel"] = "on"
                  solver_opt["presolve"] = "on"
                  solver_opt["threads"] = 4
                  solver_opt["output_flag"] = True
                  solver_opt["log_to_console"] = True
            # logfile_name = "flixSolverLog.log"
            if solver_name == 'highs':
                solver.highs_options=solver_opt
                self.solver_results = solver.solve(self.model)
            else:
                self.solver_results = solver.solve(self.model, options = solver_opt, tee = solver_output_to_console, keepfiles=True, logfile=logfile_name)

            # Log wieder laden:
            if solver_name == 'highs':
                pass
            else:
                self.solver_log = SolverLog(solver_name, logfile_name)
                self.solver_log.parse_infos()
            # Ergebnis Zielfunktion ablegen
            self.objective_result = self.model.objective.expr()

        else:
            raise Exception('not defined for modtype ' + self.modeling_language)

        self.duration['solve'] = round(timeit.default_timer() - t_start, 2)

    @property
    def infos(self) -> Dict:
        infos = {}
        infos['Solver'] = self.solver_name

        info_flixModel = {}
        infos['flixModel'] = info_flixModel

        info_flixModel['no eqs'] = self.nr_of_equations
        info_flixModel['no eqs single'] = self.nr_of_single_equations
        info_flixModel['no inEqs'] = self.nr_of_inequations
        info_flixModel['no inEqs single'] = self.nr_of_single_inequations
        info_flixModel['no vars'] = self.nr_of_variables
        info_flixModel['no vars single'] = self.nr_of_single_variables
        info_flixModel['no vars TS'] = len(self.all_ts_variables)

        if self.solver_log is not None:
            infos['solver_log'] = self.solver_log.infos
        return infos

    @property
    def all_ts_variables(self) -> List:
        return [variable for variable in self.variables if isinstance(variable, VariableTS)]

    def printNoEqsAndVars(self) -> None:
        print('no of Eqs   (single):' + str(self.nr_of_equations) + ' (' + str(self.nr_of_single_equations) + ')')
        print('no of InEqs (single):' + str(self.nr_of_inequations) + ' (' + str(self.nr_of_single_inequations) + ')')
        print('no of Vars  (single):' + str(self.nr_of_variables) + ' (' + str(self.nr_of_single_variables) + ')')

    ##############################################################################################
    ################ pyomo-Spezifisch
    # alle Pyomo Elemente müssen im model registriert sein, sonst werden sie nicht berücksichtigt
    def registerPyComp(self, py_comp, aStr='', oldPyCompToOverwrite=None) -> None:
        # neu erstellen
        if oldPyCompToOverwrite == None:
            self.countComp += 1
            # Komponenten einfach hochzählen, damit eindeutige Namen, d.h. a1_timesteps, a2, a3 ,...
            # Beispiel:
            # model.add_component('a1',py_comp) äquivalent zu model.a1 = py_comp
            self.model.add_component('a' + str(self.countComp) + '_' + aStr, py_comp)  # a1,a2,a3, ...
        # altes überschreiben:
        else:
            self.__overwritePyComp(py_comp, oldPyCompToOverwrite)

            # Komponente löschen:

    def deletePyComp(self, old_py_comp) -> None:
        aName = self.getPyCompStr(old_py_comp)
        aNameOfAdditionalComp = aName + '_index'  # sowas wird bei manchen Komponenten als Komponente automatisch mit erzeugt.
        # sonstige zugehörige Variablen löschen:
        if aNameOfAdditionalComp in self.model.component_map().keys():
            self.model.del_component(aNameOfAdditionalComp)
        self.model.del_component(aName)

        # name of component

    def getPyCompStr(self, aComp) -> str:
        for key, val in self.model.component_map().iteritems():
            if aComp == val:
                return key

    def getPyComp(self, aStr):
        return self.model.component_map()[aStr]

    # gleichnamige Pyomo-Komponente überschreiben (wenn schon vorhanden, sonst neu)
    def __overwritePyComp(self, py_comp, old_py_comp) -> None:
        aName = self.getPyCompStr(old_py_comp)
        # alles alte löschen:
        self.deletePyComp(old_py_comp)
        # überschreiben:
        self.model.add_component(aName, py_comp)

    ######## Other Modeling Languages


class Variable:
    def __init__(self,
                 label: str,
                 length: int,
                 label_of_owner: str,
                 linear_model: LinearModel,
                 is_binary: bool = False,
                 value: Optional[Union[int, float]] = None,
                 lower_bound: Optional[Union[int, float]] = None,
                 upper_bound: Optional[Union[int, float]] = None):
        self.label = label
        self.length = length
        self.linear_model = linear_model
        self.is_binary = is_binary
        self.value = value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.indices = range(self.length)
        self.label_full = label_of_owner + '__' + label
        self.fixed = False
        self._result = None  # Ergebnis

        self._result = None  # Ergebnis-Speicher
        log.debug('Variable created: ' + self.label)

        # Check conformity:
        self.label = helpers.check_name_for_conformity(label)

        # Wenn Vorgabewert vorhanden:
        if value is not None:
            # check if conflict with min/max values
            # Wenn ein Element nicht in min/max-Grenzen

            minOk = (self.lower_bound is None) or (np.all(self.value >= self.lower_bound))  # prüft elementweise
            maxOk = (self.upper_bound is None) or (np.all(self.value <= self.upper_bound))  # prüft elementweise
            if (not (minOk)) or (not (maxOk)):
                raise Exception('Variable.value' + self.label_full + ' not inside set bounds')

            # Werte in Variable festsetzen:
            self.fixed = True
            self.value = helpers.as_vector(value, length)

        # Register Element:
        # owner .variables.append(self) # Komponentenliste
        #linear_model.variables.append(self)  # linear_model-Liste mit allen vars
        #owner.model.variables.append(self)  # TODO: not nice, that this specific thing for energysystems is done here

    def to_math_model(self, baseModel: LinearModel):
        self.linear_model = baseModel

        # TODO: self.var ist hier einziges Attribut, das linear_model-spezifisch ist: --> umbetten in linear_model!
        if baseModel.modeling_language == 'pyomo':
            if self.is_binary:
                self.var = pyomoEnv.Var(self.indices, domain=pyomoEnv.Binary)
                # self.var = Var(linear_model.timesteps,domain=Binary)
            else:
                self.var = pyomoEnv.Var(self.indices, within=pyomoEnv.Reals)

            # Register in pyomo-model:
            aNameSuffixInPyomo = 'var__' + self.label_full
            baseModel.registerPyComp(self.var, aNameSuffixInPyomo)

            lower_bound_vector = helpers.as_vector(self.lower_bound, self.length)
            upper_bound_vector = helpers.as_vector(self.upper_bound, self.length)
            value_vector = helpers.as_vector(self.value, self.length)
            for i in self.indices:
                # Wenn Vorgabe-Wert vorhanden:
                if self.fixed and (value_vector[i] != None):
                    # Fixieren:
                    self.var[i].value = value_vector[i]
                    self.var[i].fix()
                else:
                    # Boundaries:
                    self.var[i].setlb(lower_bound_vector[i])  # min
                    self.var[i].setub(upper_bound_vector[i])  # max


        elif baseModel.modeling_language == 'vcxpy':
            raise Exception('not defined for modtype ' + baseModel.modeling_language)
        else:
            raise Exception('not defined for modtype ' + baseModel.modeling_language)

    def reset_result(self):
        self._result = None

    @property
    def result(self) -> Union[int, float, np.ndarray]:
        # wenn noch nicht abgefragt: (so wird verhindert, dass für jede Abfrage jedesMal neuer Speicher bereitgestellt wird.)
        if self._result is None:
            if self.linear_model.modeling_language == 'pyomo':
                # get Data:
                values = self.var.get_values().values()  # .values() of dict, because {0:0.1, 1:0.3,...}
                # choose dataType:
                if self.is_binary:
                    dType = np.int8  # geht das vielleicht noch kleiner ???
                else:
                    dType = float
                # transform to np-array (fromiter() is 5-7x faster than np.array(list(...)) )
                self._result = np.fromiter(values, dtype=dType)
                # Falls skalar:
                if len(self._result) == 1:
                    self._result = self._result[0]

            elif self.linear_model.modeling_language == 'vcxpy':
                raise Exception('not defined for modtype ' + self.linear_model.modeling_language)
            else:
                raise Exception('not defined for modtype ' + self.linear_model.modeling_language)

        return self._result

    def get_str_description(self) -> str:
        maxChars = 50  # länge begrenzen falls vector-Darstellung
        aStr = 'var'

        if isinstance(self, VariableTS):
            aStr += ' TS'
        else:
            aStr += '   '

        if self.is_binary:
            aStr += ' bin '
        else:
            aStr += '     '

        aStr += self.label_full + ': ' + 'length=' + str(self.length)
        if self.fixed:
            aStr += ', fixed =' + str(self.value)[:maxChars]

        aStr += ' min = ' + str(self.lower_bound)[:maxChars] + ', max = ' + str(self.upper_bound)[:maxChars]

        return aStr


# Timeseries-Variable, optional mit Before-Werten:
class VariableTS(Variable):
    def __init__(self,
                 label: str,
                 length: int,
                 label_of_owner: str,
                 linear_model: LinearModel,
                 is_binary: bool = False,
                 value: Optional[Union[int, float, np.ndarray]] = None,
                 lower_bound: Optional[Union[int, float, np.ndarray]] = None,
                 upper_bound: Optional[Union[int, float, np.ndarray]] = None,
                 before_value: Optional[Union[int, float, np.ndarray, List[Union[int, float]]]] = None,
                 before_value_is_start_value: bool = False):
        assert length > 1, 'length is one, that seems not right for VariableTS'
        super().__init__(label, length, label_of_owner, linear_model, is_binary=is_binary, value=value, lower_bound=lower_bound, upper_bound=upper_bound)
        self._before_value = before_value
        self.before_value_is_start_value = before_value_is_start_value

    @property
    def before_value(self) -> Optional[Union[int, float, np.ndarray, List[Union[int, float]]]]:
        # Return value if found in before_values, else return stored value
        return self.linear_model.before_values.get(self.label_full) or self._before_value

    @before_value.setter
    def before_value(self, value: Union[int, float, np.ndarray, List[Union[int, float]]]):
        self._before_value = value

    def get_before_value_for_next_segment(self, last_index_of_segment: int) -> Skalar:
        # hole Startwert/letzten Wert für nächstes Segment:
        if self.before_value_is_start_value:   # Wenn Speicherladezustand o.ä.
            index = last_index_of_segment + 1  # = Ladezustand zum Startzeitpunkt des nächsten Segments
        else:
            index = last_index_of_segment  # Leistungswert beim Zeitpunkt VOR Startzeitpunkt vom nächsten Segment
        return self.result[index]


# class cInequation(Equation):
#   def __init__(self, label, owner, linear_model):
#     super().__init__(label, owner, linear_model, eqType='ineq')

class Equation:
    def __init__(self,
                 label: str,
                 owner,
                 baseModel: LinearModel,
                 eqType: Literal['eq', 'ineq', 'objective'] = 'eq'):
        self.label = label
        self.listOfSummands = []
        self.constant = 0  # rechte Seite

        self.nr_of_single_equations = 1  # Anzahl der Gleichungen
        self.constant_vector = np.array([0])
        self.parts_of_constant = []  # liste mit shares von constant
        self.eqType = eqType
        self.myMom = owner
        self.eq = None  # z.B. für pyomo : pyomoComponente

        log.debug('equation created: ' + str(label))


    def add_summand(self,
                    variable: Variable,
                    factor: Union[int, float, np.ndarray],
                    indices_of_variable: Optional[Union[int, np.ndarray, range, List[int]]] = None,
                    as_sum: bool = False) -> None:
        """
        Adds a summand to the equation.

        This method creates a summand from the given variable and factor, optionally summing over all indices of the variable.
        The summand is then added to the list of summands for the equation.

        Parameters:
        -----------
        variable : Variable
            The variable to be used in the summand.
        factor : Union[int, float, np.ndarray]
            The factor by which the variable is multiplied.
        indices_of_variable : Optional[Union[int, float, np.ndarray]], optional
            Specific indices of the variable to be used. If not provided, all indices are used.
        as_sum : bool, optional
            If True, the summand is treated as a sum over all indices of the variable.

        Raises:
        -------
        TypeError
            If the provided variable is not an instance of the Variable class.
        Exception
            If the variable is None and as_sum is True.
        """
        # TODO: Functionality to create A Sum of Summand over a specified range of indices? For Limiting stuff per one year...?
        if not isinstance(variable, Variable):
            raise TypeError(f'Error in Equation "{self.label}": no variable given (variable = "{variable}")')
        # Wenn nur ein Wert, dann Liste mit einem Eintrag drausmachen:
        if np.isscalar(indices_of_variable):
            indices_of_variable = [indices_of_variable]

        if not as_sum:
            summand = Summand(variable, factor, indices=indices_of_variable)
        else:
            if variable is None:
                raise Exception(f'Error in Equation "{self.label}": variable = None! is not allowed if the variable is summed up!')
            summand = SumOfSummand(variable, factor, indices=indices_of_variable)

        # Check Variablen-Länge:
        self._update_nr_of_single_equations(summand.length, summand.variable.label)
        # zu Liste hinzufügen:
        self.listOfSummands.append(summand)

    def add_constant(self, value: Union[int, float, np.ndarray]) -> None:
        """
          constant value of the right side,
          if method is executed several times, than values are summed up.

          Parameters
          ----------
          value : float or array
              constant-value of equation [A*x = constant] or [A*x <= constant]

          Returns
          -------
          None.

          """
        self.constant = np.add(self.constant, value)  # Adding to current constant
        self.parts_of_constant.append(value)   # Adding to parts of constants

        length = 1 if np.isscalar(self.constant) else len(self.constant)
        self._update_nr_of_single_equations(length, 'constant')   # Update
        self.constant_vector = helpers.as_vector(self.constant, self.nr_of_single_equations)  # Update

    def to_math_model(self, baseModel: LinearModel) -> None:
        log.debug('eq ' + self.label + '.to_math_model()')

        # constant_vector hier erneut erstellen, da Anz. Glg. vorher noch nicht bekannt:
        self.constant_vector = helpers.as_vector(self.constant, self.nr_of_single_equations)

        if baseModel.modeling_language == 'pyomo':
            # 1. Constraints:
            if self.eqType in ['eq', 'ineq']:

                # lineare Summierung für i-te Gleichung:
                def linear_sum_pyomo_rule(model, i):
                    lhs = 0
                    aSummand: Summand
                    for aSummand in self.listOfSummands:
                        lhs += aSummand.math_expression(i)  # i-te Gleichung (wenn Skalar, dann wird i ignoriert)
                    rhs = self.constant_vector[i]
                    # Unterscheidung return-value je nach typ:
                    if self.eqType == 'eq':
                        return lhs == rhs
                    elif self.eqType == 'ineq':
                        return lhs <= rhs

                # TODO: self.eq ist hier einziges Attribut, das linear_model-spezifisch ist: --> umbetten in linear_model!
                self.eq = pyomoEnv.Constraint(range(self.nr_of_single_equations),
                                              rule=linear_sum_pyomo_rule)  # Nebenbedingung erstellen
                # Register im Pyomo:
                baseModel.registerPyComp(self.eq,
                                         'eq_' + self.myMom.label + '_' + self.label)  # in pyomo-Modell mit eindeutigem Namen registrieren

            # 2. Zielfunktion:
            elif self.eqType == 'objective':
                # Anmerkung: nrOfEquation - Check könnte auch weiter vorne schon passieren!
                if self.nr_of_single_equations > 1:
                    raise Exception('Equation muss für objective ein Skalar ergeben!!!')

                # Summierung der Skalare:
                def linearSumRule_Skalar(model):
                    skalar = 0
                    for aSummand in self.listOfSummands:
                        skalar += aSummand.math_expression(baseModel.modeling_language)  # kein i übergeben, da skalar
                    return skalar

                self.eq = pyomoEnv.Objective(rule=linearSumRule_Skalar, sense=pyomoEnv.minimize)
                # Register im Pyomo:
                baseModel.model.objective = self.eq

                # 3. Undefined:
            else:
                raise Exception('equation.eqType= ' + str(self.eqType) + ' nicht definiert')
        elif baseModel.modeling_language == 'vcxpy':
            raise Exception('not defined for modtype ' + baseModel.modeling_language)
        else:
            raise Exception('not defined for modtype ' + baseModel.modeling_language)

            # print i-th equation:

    def description(self, equation_nr: int = 0) -> str:
        equation_nr = min(equation_nr, self.nr_of_single_equations - 1)

        aStr = ''
        # header:
        if self.eqType == 'objective':
            aStr += 'obj' + ': '  # leerzeichen wichtig, sonst im yaml interpretation als dict
        else:
            aStr += 'eq ' + self.label + '[' + str(equation_nr) + ' of ' + str(self.nr_of_single_equations) + ']: '

        # Summanden:
        first = True
        for aSummand in self.listOfSummands:
            if not first: aStr += ' + '
            first = False
            if aSummand.length == 1:
                i = 0
            else:
                i = equation_nr
            #      i     = min(equation_nr, aSummand.length-1) # wenn zu groß, dann letzter Eintrag ???
            index = aSummand.indices[i]
            # factor formatieren:
            factor = aSummand.factor_vec[i]
            factor_str = str(factor) if isinstance(factor, int) else "{:.6}".format(float(factor))
            # Gesamtstring:
            aElementOfSummandStr = factor_str + '* ' + aSummand.variable.label_full + '[' + str(index) + ']'
            if isinstance(aSummand, SumOfSummand):
                aStr += '∑('
                if i > 0:
                    aStr += '..+'
                aStr += aElementOfSummandStr
                if i < aSummand.length:
                    aStr += '+..'
                aStr += ')'
            else:
                aStr += aElementOfSummandStr

                # = oder <= :
        if self.eqType in ['eq', 'objective']:
            aStr += ' = '
        elif self.eqType == 'ineq':
            aStr += ' <= '
        else:
            aStr += ' ? '

            # right side:
        aStr += str(self.constant_vector[equation_nr])  # todo: hier könnte man noch aufsplitten nach parts_of_constant

        return aStr

    def _update_nr_of_single_equations(self, length_of_summand: int, label_of_summand: str) -> None:
        """Checks if the new Summand is compatible with the existing Summands"""
        if self.nr_of_single_equations == 1:
            self.nr_of_single_equations = length_of_summand  # first Summand defines length of equation
            self.constant_vector = helpers.as_vector(self.constant, self.nr_of_single_equations)  # Update
        elif (length_of_summand != 1) & (length_of_summand != self.nr_of_single_equations):
            raise Exception(f'Variable {label_of_summand} hat eine nicht passende Länge für Gleichung {self.label}')


# Beachte: Muss auch funktionieren für den Fall, dass variable.var fixe Werte sind.
class Summand:
    def __init__(self,
                 variable: Variable,
                 factor: Union[int, float, np.ndarray],
                 indices: Optional[Union[int, np.ndarray, range, List[int]]] = None):  # indices_of_variable default : alle
        self.variable = variable
        self.factor = factor
        self.indices = indices

        # wenn nicht definiert, dann alle Indexe
        if self.indices is None:
            self.indices = variable.indices  # alle indices

        # Länge ermitteln:
        self.length = self._check_length()

        # Faktor als Vektor:
        self.factor_vec = helpers.as_vector(factor, self.length)

    def _check_length(self):
        """
        Determines and returns the length of the summand by comparing the lengths of the factor and the variable indices.
        Sets the attribute .length to this value.

        Returns:
        --------
        int
            The length of the summand, which is the length of the indices if they match the length of the factor,
            or the length of the longer one if one of them is a scalar.

        Raises:
        -------
        Exception
            If the lengths of the factor and the variable indices do not match and neither is a scalar.
        """
        length_of_factor = 1 if np.isscalar(self.factor) else len(self.factor)
        length_of_indices = len(self.indices)
        if length_of_indices == length_of_factor:
            return length_of_indices
        elif length_of_factor == 1:
            return length_of_indices
        elif length_of_indices == 1:
            return length_of_factor
        else:
            raise Exception(f'Variable {self.variable.label_full} (length={length_of_indices}) und '
                            f'Faktor (length={length_of_factor}) müssen gleiche Länge haben oder Skalar sein')

    # Ausdruck für i-te Gleichung (falls Skalar, dann immer gleicher Ausdruck ausgegeben)
    def math_expression(self, at_index: int = 0):
        if self.length == 1:
            return self.variable.var[self.indices[0]] * self.factor_vec[0]  # ignore argument at_index, because Skalar is used for every single equation
        if len(self.indices) == 1:
            return self.variable.var[self.indices[0]] * self.factor_vec[at_index]
        return self.variable.var[self.indices[at_index]] * self.factor_vec[at_index]

class SumOfSummand(Summand):
    def __init__(self,
                 variable: Variable,
                 factor: Union[int, float, np.ndarray],
                 indices: Optional[Union[int, np.ndarray, range, List[int]]] = None):  # indices_of_variable default : alle
        super().__init__(variable, factor, indices)

        self._math_expression = None
        self.length = 1

    def math_expression(self, at_index=0):
        # at index doesn't do anything. Can be removed, but induces changes elsewhere (Inherritance)
        if self._math_expression is not None:
            return self._math_expression
        else:
            self._math_expression = sum(self.variable.var[self.indices[j]] * self.factor_vec[j] for j in self.indices)
            return self._math_expression


class SolverLog:
    def __init__(self, solver_name: str, filename: str):
        with open(filename, 'r') as file:
            self.log = file.read()

        self.solver_name = solver_name

        self.presolved_rows = None
        self.presolved_cols = None
        self.presolved_nonzeros = None

        self.presolved_continuous = None
        self.presolved_integer = None
        self.presolved_binary = None

    @property
    def infos(self):
        infos = {}
        aPreInfo = {}
        infos['presolved'] = aPreInfo
        aPreInfo['cols'] = self.presolved_cols
        aPreInfo['continuous'] = self.presolved_continuous
        aPreInfo['integer'] = self.presolved_integer
        aPreInfo['binary'] = self.presolved_binary
        aPreInfo['rows'] = self.presolved_rows
        aPreInfo['nonzeros'] = self.presolved_nonzeros

        return infos

    # Suche infos aus log:
    def parse_infos(self):
        if self.solver_name == 'gurobi':

            # string-Schnipsel 1:
            '''
            Optimize a model with 285 rows, 292 columns and 878 nonzeros
            Model fingerprint: 0x1756ffd1
            Variable types: 202 continuous, 90 integer (90 binary)        
            '''
            # string-Schnipsel 2:
            '''
            Presolve removed 154 rows and 172 columns
            Presolve time: 0.00s
            Presolved: 131 rows, 120 columns, 339 nonzeros
            Variable types: 53 continuous, 67 integer (67 binary)
            '''
            # string: Presolved: 131 rows, 120 columns, 339 nonzeros\n
            match = re.search('Presolved: (\d+) rows, (\d+) columns, (\d+) nonzeros' +
                              '\\n\\n' +
                              'Variable types: (\d+) continuous, (\d+) integer \((\d+) binary\)', self.log)
            if not match is None:
                # string: Presolved: 131 rows, 120 columns, 339 nonzeros\n
                self.presolved_rows = int(match.group(1))
                self.presolved_cols = int(match.group(2))
                self.presolved_nonzeros = int(match.group(3))
                # string: Variable types: 53 continuous, 67 integer (67 binary)
                self.presolved_continuous = int(match.group(4))
                self.presolved_integer = int(match.group(5))
                self.presolved_binary = int(match.group(6))

        elif self.solver_name == 'cbc':

            # string: Presolve 1623 (-1079) rows, 1430 (-1078) columns and 4296 (-3306) elements
            match = re.search('Presolve (\d+) \((-?\d+)\) rows, (\d+) \((-?\d+)\) columns and (\d+)', self.log)
            if not match is None:
                self.presolved_rows = int(match.group(1))
                self.presolved_cols = int(match.group(3))
                self.presolved_nonzeros = int(match.group(5))

            # string: Presolved problem has 862 integers (862 of which binary)
            match = re.search('Presolved problem has (\d+) integers \((\d+) of which binary\)', self.log)
            if not match is None:
                self.presolved_integer = int(match.group(1))
                self.presolved_binary = int(match.group(2))
                self.presolved_continuous = self.presolved_cols - self.presolved_integer

        elif self.solver_name == 'glpk':
            print('######################################################')
            print('### No solver-log parsing implemented for glpk yet! ###')
        else:
            raise Exception('SolverLog.parse_infos() is not defined for solver ' + self.solver_name)
