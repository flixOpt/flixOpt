# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:09:02 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import logging
import re
import timeit
from typing import List, Dict, Optional, Union, Literal, Any
from abc import ABC, abstractmethod

import numpy as np
from pyomo.contrib import appsi

from flixOpt import utils
from flixOpt.core import Skalar, Numeric

pyomoEnv = None  # das ist module, das nur bei Bedarf belegt wird

logger = logging.getLogger('flixOpt')


class Variable:
    """
    Regular single Variable
    """
    def __init__(self,
                 label: str,
                 length: int,
                 math_model: 'MathModel',
                 label_short: Optional[str] = None,
                 is_binary: bool = False,
                 fixed_value: Optional[Numeric] = None,
                 lower_bound: Optional[Numeric] = None,
                 upper_bound: Optional[Numeric] = None):
        """
        label: full label of the variable
        label_short: short label of the variable

        # TODO: Allow for None values in fixed_value. If None, the index gets not fixed!
        """
        self.label = label
        self.label_short = label_short or label
        self.length = length
        self.math_model = math_model
        self.is_binary = is_binary
        self.fixed_value = fixed_value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.indices = range(self.length)
        self.fixed = False

        self.result = None  # Ergebnis-Speicher

        if self.fixed_value is not None:   # Check if value is within bounds, element-wise
            above = self.lower_bound is None or np.all(np.asarray(self.fixed_value) >= np.asarray(self.lower_bound))
            below = self.upper_bound is None or np.all(np.asarray(self.fixed_value) <= np.asarray(self.upper_bound))
            if not (above and below):
                raise Exception(f'Fixed value of Variable {self.label} not inside set bounds:'
                                f'\n{self.fixed_value=};\n{self.lower_bound=};\n{self.upper_bound=}')

            # Mark as fixed
            self.fixed = True

        logger.debug('Variable created: ' + self.label)

    def description(self, max_length_ts=60) -> str:
        bin_type = 'bin' if self.is_binary else '   '

        header = f'Var {bin_type} x {self.length:<6} "{self.label}"'
        if self.fixed:
            description = f'{header:<40}: fixed={str(self.fixed_value)[:max_length_ts]:<10}'
        else:
            description = (f'{header:<40}: min={str(self.lower_bound)[:max_length_ts]:<10}, '
                           f'max={str(self.upper_bound)[:max_length_ts]:<10}')
        return description

    def reset_result(self):
        self._result = None


class VariableTS(Variable):
    """
    # Timeseries-Variable, optionally with previous_values
    """
    def __init__(self,
                 label: str,
                 length: int,
                 math_model: 'MathModel',
                 label_short: Optional[str] = None,
                 is_binary: bool = False,
                 fixed_value: Optional[Numeric] = None,
                 lower_bound: Optional[Numeric] = None,
                 upper_bound: Optional[Numeric] = None,
                 previous_values: Optional[Numeric] = None):
        assert length > 1, 'length is one, that seems not right for VariableTS'
        super().__init__(label, length, math_model, label_short, is_binary=is_binary, fixed_value=fixed_value,
                         lower_bound=lower_bound, upper_bound=upper_bound)
        self.previous_values = previous_values


class Equation:
    """
    Representing a single equation or, with the Variable being a VariableTS, a set of equations

    """
    def __init__(self,
                 label: str,
                 label_short: Optional[str] = None,
                 eqType: Literal['eq', 'ineq', 'objective'] = 'eq'):
        """
        Equation of the form: ∑(<summands>) = <constant>        type: 'eq'
        Equation of the form: ∑(<summands>) <= <constant>       type: 'ineq'
        Equation of the form: ∑(<summands>) = <constant>        type: 'objective'

        Parameters
        ----------
            label: full label of the variable
            label_short: short label of the variable. If None, the the full label is used
            eqType: Literal['eq', 'ineq', 'objective']
        """
        self.label = label
        self.label_short = label_short or label
        self.summands: List[SumOfSummand] = []
        self.parts_of_constant: List[Numeric] = []
        self.constant: Numeric = 0  # Total of right side

        self.length = 1  # Anzahl der Gleichungen
        self.eqType = eqType

        self._pyomo_comp = None  # z.B. für pyomo : pyomoComponente

        logger.debug(f'Equation created: {self.label}')

    def add_summand(self,
                    variable: Variable,
                    factor: Numeric,
                    indices_of_variable: Optional[Union[int, np.ndarray, range, List[int]]] = None,
                    as_sum: bool = False) -> None:
        """
        Adds a summand to the left side of the equation.

        This method creates a summand from the given variable and factor, optionally summing over all given indices.
        The summand is then added to the summands of the equation, which represent the left side.

        Parameters:
        -----------
        variable : Variable
            The variable to be used in the summand.
        factor : Numeric
            The factor by which the variable is multiplied.
        indices_of_variable : Optional[Numeric], optional
            Specific indices of the variable to be used. If not provided, all indices are used.
        as_sum : bool, optional
            If True, the summand is treated as a sum over all indices of the variable.

        Raises:
        -------
        TypeError
            If the provided variable is not an instance of the Variable class.
        ValueError
            If the variable is None and as_sum is True.
        ValueError
            If the length doesnt match the Equation's length.
        """
        # TODO: Functionality to create A Sum of Summand over a specified range of indices? For Limiting stuff per one year...?
        if not isinstance(variable, Variable):
            raise TypeError(f'Error in Equation "{self.label}": no variable given (variable = "{variable}")')
        if variable is None and as_sum:
            raise ValueError(f'Error in Equation "{self.label}": Variable can not be None and be summed up!')

        if np.isscalar(indices_of_variable):   # Wenn nur ein Wert, dann Liste mit einem Eintrag drausmachen:
            indices_of_variable = [indices_of_variable]

        if as_sum:
            summand = SumOfSummand(variable, factor, indices=indices_of_variable)
        else:
            summand = Summand(variable, factor, indices=indices_of_variable)

        try:
            self._update_length(summand.length)   # Check Variablen-Länge:
        except ValueError as e:
            raise ValueError(f'Length of Summand with variable "{variable.label}" '
                             f'does not fit equation "{self.label}": {e}')
        self.summands.append(summand)

    def add_constant(self, value: Numeric) -> None:
        """
        Adds a constant value to the rigth side of the equation

        Parameters
        ----------
        value : float or array
          constant-value of equation [A*x = constant] or [A*x <= constant]

        Returns
        -------
        None.

        Raises:
        -------
        ValueError
            If the length doesnt match the Equation's length.

        """
        self.constant = np.add(self.constant, value)  # Adding to current constant
        self.parts_of_constant.append(value)   # Adding to parts of constants

        length = 1 if np.isscalar(self.constant) else len(self.constant)
        try:
            self._update_length(length)
        except ValueError as e:
            raise ValueError(f'Length of Constant {value=} does not fit: {e}')

    def description(self, at_index: int = 0) -> str:
        equation_nr = min(at_index, self.length - 1)

        # Name and index
        if self.eqType == 'objective':
            name = 'OBJ'
            index_str = ''
        else:
            name = f'EQ {self.label}'
            index_str = f'[{equation_nr+1}/{self.length}]'

        # Summands:
        summand_strings = []
        for idx, summand in enumerate(self.summands):
            i = 0 if summand.length == 1 else equation_nr
            index = summand.indices[i]
            factor = summand.factor_vec[i]
            factor_str = str(factor) if isinstance(factor, int) else f"{factor:.6}"
            single_summand_str = f"{factor_str} * {summand.variable.label}[{index}]"

            if isinstance(summand, SumOfSummand):
                summand_strings.append(
                    f"∑({('..+' if i > 0 else '')}{single_summand_str}{('+..' if i < summand.length else '')})")
            else:
                summand_strings.append(single_summand_str)

        all_summands_string = ' + '.join(summand_strings)

        # Equation type:
        signs = {'eq': '= ', 'ineq': '=>', 'objective': '= '}
        sign = signs.get(self.eqType, '? ')

        constant = self.constant_vector[equation_nr]

        header_width = 30
        header = f"{name:<{header_width-len(index_str)-1}} {index_str}"
        return f'{header:<{header_width}}: {constant:>8} {sign} {all_summands_string}'

    def _update_length(self, new_length: int) -> None:
        """
        Passes if the new_length is 1, the current length is 1 or new_length matches the existing length of the Equation
        """
        if self.length == 1:  # First Summand sets length
            self.length = new_length
        elif new_length == 1 or new_length == self.length:  # Length 1 is always possible
            pass
        else:
            raise ValueError(f'The length of the new element {new_length=} doesnt match the existing '
                             f'length of the Equation {self.length=}!')

    @property
    def constant_vector(self) -> Numeric:
        return utils.as_vector(self.constant, self.length)


class Summand:
    """
    Part of an equation. Either with a single Variable or a VariableTS
    """
    def __init__(self,
                 variable: Variable,
                 factor: Numeric,
                 indices: Optional[Union[int, np.ndarray, range, List[int]]] = None):  # indices_of_variable default : alle
        self.variable = variable
        self.factor = factor
        self.indices = indices if indices is not None else variable.indices    # wenn nicht definiert, dann alle Indexe

        self.length = self._check_length()   # Länge ermitteln:

        self.factor_vec = utils.as_vector(factor, self.length)   # Faktor als Vektor:

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
            raise Exception(f'Variable {self.variable.label} (length={length_of_indices}) und '
                            f'Faktor (length={length_of_factor}) müssen gleiche Länge haben oder Skalar sein')


class SumOfSummand(Summand):
    """
    Part of an Equation. Summing up all parts of a regular Summand of a regular Summand
    'sum(factor[i]*variable[i] for i in all_indexes)'
    """
    def __init__(self,
                 variable: Variable,
                 factor: Numeric,
                 indices: Optional[Union[int, np.ndarray, range, List[int]]] = None):  # indices_of_variable default : alle
        super().__init__(variable, factor, indices)

        self.length = 1


class MathModel:
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
        self.modeling_language: str = modeling_language

        self.epsilon = 1e-5

        self.solver: Optional[Solver] = None
        self.model: Optional[ModelingLanguage] = None

        self._variables: List[Variable] = []
        self._eqs: List[Equation] = []
        self._ineqs: List[Equation] = []
        self.result_of_objective = None  # Ergebnis
        self.duration = {}  # Laufzeiten

    def add(self, *args: Union[Variable, Equation]) -> None:
        if not isinstance(args, list):
            args = list(args)
        for arg in args:
            if isinstance(arg, Variable):
                self._variables.append(arg)
            elif isinstance(arg, Equation):
                if arg.eqType == 'eq':
                    self._eqs.append(arg)
                elif arg.eqType == 'ineq':
                    self._ineqs.append(arg)
                else:
                    raise Exception(f'{arg} cant be added this way!')
            else:
                raise Exception(f'{arg} cant be added this way!')

    def describe(self) -> str:
        return (f'no of Eqs   (single): {self.nr_of_equations} ({self.nr_of_single_equations})\n'
                f'no of InEqs (single): {self.nr_of_inequations} ({self.nr_of_single_inequations})\n'
                f'no of Vars  (single): {self.nr_of_variables} ({self.nr_of_single_variables})')

    def to_math_model(self) -> None:
        t_start = timeit.default_timer()
        if self.modeling_language == 'pyomo':
            self.model = PyomoModel()
            self.model.translate_model(self)
        else:
            raise NotImplementedError('Modeling Language cvxpy is not yet implemented')
        self.duration['Translation'] = round(timeit.default_timer() - t_start, 2)

    def solve(self, solver: 'Solver') -> None:
        self.solver = solver
        t_start = timeit.default_timer()
        for variable in self.variables:
            variable.reset_result()  # altes Ergebnis löschen (falls vorhanden)
        self.model.solve(self, solver)
        self.duration['solve'] = round(timeit.default_timer() - t_start, 2)

    @property
    def infos(self) -> Dict:
        infos = {}
        infos['Solver'] = self.solver.__repr__()

        info_flixModel = {}
        infos['flixModel'] = info_flixModel

        info_flixModel['no eqs'] = self.nr_of_equations
        info_flixModel['no eqs single'] = self.nr_of_single_equations
        info_flixModel['no inEqs'] = self.nr_of_inequations
        info_flixModel['no inEqs single'] = self.nr_of_single_inequations
        info_flixModel['no vars'] = self.nr_of_variables
        info_flixModel['no vars single'] = self.nr_of_single_variables
        info_flixModel['no vars TS'] = len(self.ts_variables)

        if self.solver.log is not None:
            infos['solver_log'] = self.solver.log
        return infos

    @property
    def variables(self) -> List[Variable]:
        return self._variables

    @property
    def eqs(self) -> List[Equation]:
        return self._eqs

    @property
    def ineqs(self) -> List[Equation]:
        return self._ineqs

    @property
    def ts_variables(self) -> List[VariableTS]:
        return [variable for variable in self.variables if isinstance(variable, VariableTS)]

    @property
    def nr_of_variables(self) -> int:
        return len(self.variables)

    @property
    def nr_of_equations(self) -> int:
        return len(self.eqs)

    @property
    def nr_of_inequations(self) -> int:
        return len(self.ineqs)

    @property
    def nr_of_single_variables(self) -> int:
        return sum([var.length for var in self.variables])

    @property
    def nr_of_single_equations(self) -> int:
        return sum([eq.length for eq in self.eqs])

    @property
    def nr_of_single_inequations(self) -> int:
        return sum([eq.length for eq in self.ineqs])

    def results(self):
        return {variable.label: variable.result for variable in self.variables}


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
            logger.warning(f'{"":#^80}\n')
            logger.warning(f'{" No solver-log parsing implemented for glpk yet! ":#^80}\n')
        else:
            raise Exception('SolverLog.parse_infos() is not defined for solver ' + self.solver_name)


class Solver(ABC):
    """ Abstract class for Solvers """
    def __init__(self,
                 mip_gap: float,
                 solver_output_to_console: bool,
                 logfile_name: str,
                 ):
        self.mip_gap = mip_gap
        self.solver_output_to_console = solver_output_to_console
        self.logfile_name = logfile_name

        self.result: Optional[float, str] = None
        self.best_bound: Optional[float, str] = None
        self.termination_message: Optional[str] = None
        self.log = None

        self._solver = None

    def solve(self, model: 'pyomoEnv.ConcreteModel'):
        raise NotImplementedError(f' Solving is not possible with this Abstract class')

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"mip_gap={self.mip_gap}, "
                f"solver_output_to_console={self.solver_output_to_console}, "
                f"logfile_name='{self.logfile_name}', "
                f"result={self.result!r}, "
                f"best_bound={self.best_bound!r}, "
                f"termination_message={self.termination_message!r})")


class GurobiSolver(Solver):
    def __init__(self,
                 mip_gap: float,
                 time_limit_seconds: int,
                 solver_output_to_console: bool,
                 logfile_name: str,
                 ):
        super().__init__(mip_gap, solver_output_to_console, logfile_name)
        self.time_limit_seconds = time_limit_seconds

    def solve(self, modeling_language: 'ModelingLanguage'):
        if isinstance(modeling_language, PyomoModel):
            self._solver = pyomoEnv.SolverFactory('gurobi')
            self.result = self._solver.solve(
                modeling_language.model, tee=self.solver_output_to_console, keepfiles=True, logfile=self.logfile_name,
                options={"mipgap": self.mip_gap, "TimeLimit": self.time_limit_seconds}
            )
        else:
            raise NotImplementedError(f'Only Pyomo is implemented for GUROBI solver.')


class CplexSolver(Solver):
    def __init__(self,
                 mip_gap: float,
                 time_limit_seconds: int,
                 solver_output_to_console: bool,
                 logfile_name: str,
                 ):
        super().__init__(mip_gap, solver_output_to_console, logfile_name)
        self.time_limit_seconds = time_limit_seconds

    def solve(self, modeling_language: 'ModelingLanguage'):
        if isinstance(modeling_language, PyomoModel):
            self._solver = pyomoEnv.SolverFactory('cplex')
            self.result = self._solver.solve(
                modeling_language.model, tee=self.solver_output_to_console, keepfiles=True, logfile=self.logfile_name,
                options={"mipgap": self.mip_gap, "timelimit": self.time_limit_seconds}
            )
        else:
            raise NotImplementedError(f'Only Pyomo is implemented for GUROBI solver.')


class HighsSolver(Solver):
    def __init__(self,
                 mip_gap: float,
                 time_limit_seconds: int,
                 solver_output_to_console: bool,
                 threads: int,
                 logfile_name: str,
                 ):
        super().__init__(mip_gap, solver_output_to_console, logfile_name)
        self.time_limit_seconds = time_limit_seconds
        self.threads = threads

    def solve(self, modeling_language: 'ModelingLanguage'):
        if isinstance(modeling_language, PyomoModel):
            self._solver = appsi.solvers.Highs()
            self._solver.highs_options = {"mip_rel_gap": self.mip_gap,
                                          "time_limit": self.time_limit_seconds,
                                          "log_file": "results/highs.log",
                                          "log_to_console": self.solver_output_to_console,
                                          "threads": self.threads,
                                          "parallel": "on",
                                          "presolve": "on",
                                          "output_flag": True}
            self.result = self._solver.solve(modeling_language.model)
        else:
            raise NotImplementedError(f'Only Pyomo is implemented for HIGHS solver.')


class CbcSolver(Solver):
    def __init__(self,
                 mip_gap: float,
                 time_limit_seconds: int,
                 solver_output_to_console: bool,
                 logfile_name: str,
                 ):
        super().__init__(mip_gap, solver_output_to_console, logfile_name)
        self.time_limit_seconds = time_limit_seconds

    def solve(self, modeling_language: 'ModelingLanguage'):
        if isinstance(modeling_language, PyomoModel):
            self._solver = pyomoEnv.SolverFactory('cbc')
            self.result = self._solver.solve(
                modeling_language.model, tee=self.solver_output_to_console, keepfiles=True, logfile=self.logfile_name,
                options={"ratio": self.mip_gap, "sec": self.time_limit_seconds}
            )
        else:
            raise NotImplementedError(f'Only Pyomo is implemented for Cbc solver.')


class GlpkSolver(Solver):
    def solve(self, modeling_language: 'ModelingLanguage'):
        if isinstance(modeling_language, PyomoModel):
            self._solver = pyomoEnv.SolverFactory('glpk')
            self.result = self._solver.solve(
                modeling_language.model, tee=self.solver_output_to_console, keepfiles=True, logfile=self.logfile_name,
                options={"mipgap": self.mip_gap}
            )
        else:
            raise NotImplementedError(f'Only Pyomo is implemented for Cbc solver.')


class ModelingLanguage(ABC):
    @abstractmethod
    def translate_model(self, model: MathModel):
        raise NotImplementedError

    def solve(self, math_model: MathModel, solver_opt: Dict):
        raise NotImplementedError


class PyomoModel(ModelingLanguage):

    def __init__(self):
        global pyomoEnv  # als globale Variable
        import pyomo.environ as pyomoEnv
        logger.debug('Loaded pyomo modules')
        # für den Fall pyomo wird EIN Modell erzeugt, das auch für rollierende Durchläufe immer wieder genutzt wird.
        self.model = pyomoEnv.ConcreteModel(name="(Minimalbeispiel)")

        self.mapping: Dict[Union[Variable, Equation], Any] = {}  # Mapping to Pyomo Units
        self._counter = 0

    def solve(self, math_model: MathModel, solver: Solver):
        self.translate_model(math_model)
        solver.solve(self)

        # write results
        math_model.result_of_objective = self.model.objective.expr()
        for variable in math_model.variables:
            raw_results = self.mapping[variable].get_values().values()  # .values() of dict, because {0:0.1, 1:0.3,...}
            if variable.is_binary:
                dtype = np.int8  # geht das vielleicht noch kleiner ???
            else:
                dtype = float
            # transform to np-array (fromiter() is 5-7x faster than np.array(list(...)) )
            result = np.fromiter(raw_results, dtype=dtype)
            # Falls skalar:
            if len(result) == 1:
                variable.result = result[0]
            else:
                variable.result = result

    def translate_model(self, math_model: MathModel):
        for variable in math_model.variables:   # Variablen erstellen
            logger.debug(f'VAR {variable.label} gets translated to Pyomo')
            self.translate_variable(variable)
        for eq in math_model.eqs:   # Gleichungen erstellen
            logger.debug(f'EQ {eq.label} gets translated to Pyomo')
            self.translate_equation(eq)
        for ineq in math_model.ineqs:   # Ungleichungen erstellen:
            logger.debug(f'INEQ {ineq.label} gets translated to Pyomo')
            self.translate_equation(ineq)

    def translate_variable(self, variable: Variable):
        assert isinstance(variable, Variable), 'Wrong type of variable'

        if variable.is_binary:
            pyomo_comp = pyomoEnv.Var(variable.indices, domain=pyomoEnv.Binary)
        else:
            pyomo_comp = pyomoEnv.Var(variable.indices, within=pyomoEnv.Reals)
        self.mapping[variable] = pyomo_comp

        # Register in pyomo-model:
        self._register_pyomo_comp(pyomo_comp, variable)

        lower_bound_vector = utils.as_vector(variable.lower_bound, variable.length)
        upper_bound_vector = utils.as_vector(variable.upper_bound, variable.length)
        fixed_value_vector = utils.as_vector(variable.fixed_value, variable.length)
        for i in variable.indices:
            # Wenn Vorgabe-Wert vorhanden:
            if variable.fixed and (fixed_value_vector[i] != None):
                # Fixieren:
                pyomo_comp[i].value = fixed_value_vector[i]
                pyomo_comp[i].fix()
            else:
                # Boundaries:
                pyomo_comp[i].setlb(lower_bound_vector[i])  # min
                pyomo_comp[i].setub(upper_bound_vector[i])  # max

    def translate_equation(self, equation: Equation):

        # constant_vector hier erneut erstellen, da Anz. Glg. vorher noch nicht bekannt:
        constant_vector = equation.constant_vector
        # 1. Constraints:
        if equation.eqType in ['eq', 'ineq']:
            model = self.model

            # lineare Summierung für i-te Gleichung:
            def linear_sum_pyomo_rule(model, i):
                lhs = 0
                aSummand: Summand
                for aSummand in equation.summands:
                    lhs += self._summand_math_expression(aSummand, i)  # i-te Gleichung (wenn Skalar, dann wird i ignoriert)
                rhs = constant_vector[i]
                # Unterscheidung return-value je nach typ:
                if equation.eqType == 'eq':
                    return lhs == rhs
                elif equation.eqType == 'ineq':
                    return lhs <= rhs

            pyomo_comp = pyomoEnv.Constraint(range(equation.length),
                                             rule=linear_sum_pyomo_rule)  # Nebenbedingung erstellen

            self._register_pyomo_comp(pyomo_comp, equation)

        # 2. Zielfunktion:
        elif equation.eqType == 'objective':
            # Anmerkung: nrOfEquation - Check könnte auch weiter vorne schon passieren!
            if equation.length > 1:
                raise Exception('Equation muss für objective ein Skalar ergeben!!!')

            # Summierung der Skalare:
            def linearSumRule_Skalar(model):
                skalar = 0
                for aSummand in equation.summands:
                    skalar += self._summand_math_expression(aSummand)
                return skalar

            self.model.objective = pyomoEnv.Objective(rule=linearSumRule_Skalar, sense=pyomoEnv.minimize)
            self.mapping[equation] = self.model.objective

        # 3. Undefined:
        else:
            raise Exception('equation.eqType= ' + str(equation.eqType) + ' nicht definiert')

    def _summand_math_expression(self, summand: Summand, at_index: int = 0) -> 'pyomoEnv.Expression':
        pyomo_variable = self.mapping[summand.variable]
        if isinstance(summand, SumOfSummand):
            return sum(pyomo_variable[summand.indices[j]] * summand.factor_vec[j] for j in summand.indices)

        # Ausdruck für i-te Gleichung (falls Skalar, dann immer gleicher Ausdruck ausgegeben)
        if summand.length == 1:
            # ignore argument at_index, because Skalar is used for every single equation
            return pyomo_variable[summand.indices[0]] * summand.factor_vec[0]
        if len(summand.indices) == 1:
            return pyomo_variable[summand.indices[0]] * summand.factor_vec[at_index]
        return pyomo_variable[summand.indices[at_index]] * summand.factor_vec[at_index]

    def _register_pyomo_comp(self, pyomo_comp, part: Union[Variable, Equation]) -> None:
        self._counter += 1  # Komponenten einfach hochzählen, damit eindeutige Namen, d.h. a1_timesteps, a2, a3 ,...
        self.model.add_component(f'a{self._counter}__{part.label}', pyomo_comp)  # a1,a2,a3, ...
        self.mapping[part] = pyomo_comp
