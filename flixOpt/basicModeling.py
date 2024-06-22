# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:09:02 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import logging
import time
import re
from typing import List, Dict, Optional, Union, Tuple, Literal

import numpy as np
from pyomo.contrib import appsi

from flixOpt import flixOptHelperFcts as helpers

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

        self.model = None  # Übergabe später, zumindest für Pyomo notwendig
        self.variables: List[Variable] = []  # Liste aller Variablen
        self.eqs: List[Equation] = []  # Liste aller Gleichungen
        self.ineqs: List[Equation] = []  # Liste aller Ungleichungen
        self.objective = None  # objective-Function
        self.objective_result = None  # Ergebnis
        self.duration = {}  # Laufzeiten
        self.solver_log = None  # logging und parsen des solver-outputs

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

    def transform2MathModel(self) -> None:

        self.characterize_math_problem()

        t_start = time.time()
        eq: Equation
        # Variablen erstellen
        for variable in self.variables:
            variable.transform2MathModel(self)
        # Gleichungen erstellen
        for eq in self.eqs:
            eq.transform2MathModel(self)
        # Ungleichungen erstellen:
        for ineq in self.ineqs:
            ineq.transform2MathModel(self)
        # Zielfunktion erstellen
        self.objective.transform2MathModel(self)

        self.duration['transform2MathModel'] = round(time.time() - t_start, 2)

    # Attention: is overrided by childclass:
    def characterize_math_problem(self) -> None:
        eq: Equation
        var: Variable

        self.noOfEqs = len(self.eqs)
        self.noOfSingleEqs = sum([eq.nrOfSingleEquations for eq in self.eqs])

        self.noOfIneqs = len(self.ineqs)
        self.noOfSingleIneqs = sum([eq.nrOfSingleEquations for eq in self.ineqs])

        self.noOfVars = len(self.variables)
        self.noOfSingleVars = sum([var.len for var in self.variables])

    def solve(self, gapfrac, timelimit, solver_name, displaySolverOutput, logfileName, **solver_opt) -> None:
        self.solver_name = solver_name
        t_start = time.time()
        for variable in self.variables:
            variable.resetResult()  # altes Ergebnis löschen (falls vorhanden)
        if self.modeling_language == 'pyomo':
            if solver_name == 'highs':
              solver = appsi.solvers.Highs()
            else:
              solver = pyomoEnv.SolverFactory(solver_name)
            if solver_name == 'cbc':
                solver_opt["ratio"] = gapfrac
                solver_opt["sec"] = timelimit
            elif solver_name == 'gurobi':
                solver_opt["mipgap"] = gapfrac
                solver_opt["TimeLimit"] = timelimit
            elif solver_name == 'cplex':
                solver_opt["mipgap"] = gapfrac
                solver_opt["timelimit"] = timelimit
                # todo: threads = ? funktioniert das für cplex?
            elif solver_name == 'glpk':
                # solver_opt = {} # überschreiben, keine kwargs zulässig
                # solver_opt["mipgap"] = gapfrac
                solver_opt['mipgap'] = gapfrac
            elif solver_name == 'highs':
                  solver_opt["mip_rel_gap"] = gapfrac
                  solver_opt["time_limit"] = timelimit
                  solver_opt["log_file"]= "results/highs.log"
                  solver_opt["parallel"] = "on"
                  solver_opt["presolve"] = "on"
                  solver_opt["threads"] = 4
                  solver_opt["output_flag"] = True
                  solver_opt["log_to_console"] = True
            # logfileName = "flixSolverLog.log"
            if solver_name == 'highs':
                solver.highs_options=solver_opt
                self.solver_results = solver.solve(self.model)
            else:
                self.solver_results = solver.solve(self.model, options = solver_opt, tee = displaySolverOutput, keepfiles=True, logfile=logfileName)

            # Log wieder laden:
            if solver_name == 'highs':
                pass
            else:
                self.solver_log = SolverLog(solver_name, logfileName)
                self.solver_log.parseInfos()
            # Ergebnis Zielfunktion ablegen
            self.objective_result = self.model.objective.expr()

        else:
            raise Exception('not defined for modtype ' + self.modeling_language)

        self.duration['solve'] = round(time.time() - t_start, 2)


    @property
    def infos(self) -> Dict:
        infos = {}
        infos['Solver'] = self.solver_name

        info_flixModel = {}
        infos['flixModel'] = info_flixModel

        info_flixModel['no eqs'] = self.noOfEqs
        info_flixModel['no eqs single'] = self.noOfSingleEqs
        info_flixModel['no inEqs'] = self.noOfIneqs
        info_flixModel['no inEqs single'] = self.noOfSingleIneqs
        info_flixModel['no vars'] = self.noOfVars
        info_flixModel['no vars single'] = self.noOfSingleVars
        info_flixModel['no vars TS'] = len(self.all_ts_variables)

        if self.solver_log is not None:
            infos['solver_log'] = self.solver_log.infos
        return infos

    @property
    def all_ts_variables(self) -> List:
        return [variable for variable in self.variables if isinstance(variable, VariableTS)]

    def printNoEqsAndVars(self) -> None:
        print('no of Eqs   (single):' + str(self.noOfEqs) + ' (' + str(self.noOfSingleEqs) + ')')
        print('no of InEqs (single):' + str(self.noOfIneqs) + ' (' + str(self.noOfSingleIneqs) + ')')
        print('no of Vars  (single):' + str(self.noOfVars) + ' (' + str(self.noOfSingleVars) + ')')

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


class Variable:
    def __init__(self, label: str, len: int, myMom, baseModel: LinearModel, isBinary: bool = False,
                 value: Optional[Union[int, float]] = None,
                 min: Optional[Union[int, float]] = None, max: Optional[Union[int, float]] = None):  #TODO: Rename max and min!!
        self.label = label
        self.len = len
        self.myMom = myMom
        self.baseModel = baseModel
        self.isBinary = isBinary
        self.value = value
        self.min = min
        self.max = max

        self.indexe = range(self.len)
        self.label_full = myMom.label + '.' + label
        self.fixed = False
        self.result = None  # Ergebnis

        self.value_vec = helpers.getVector(value, self.len)
        # boundaries:
        self.min_vec = helpers.getVector(min, self.len)  # note: auch aus None wird None-Vektor
        self.max_vec = helpers.getVector(max, self.len)
        self.__result = None  # Ergebnis-Speicher
        log.debug('Variable created: ' + self.label)

        # Check conformity:
        self.label = helpers.checkForAttributeNameConformity(label)

        # Wenn Vorgabewert vorhanden:
        if not (value is None):
            # check if conflict with min/max values
            # Wenn ein Element nicht in min/max-Grenzen

            minOk = (self.min is None) or (np.all(self.value >= self.min_vec))  # prüft elementweise
            maxOk = (self.max is None) or (np.all(self.value <= self.max_vec))  # prüft elementweise
            if (not (minOk)) or (not (maxOk)):
                raise Exception('Variable.value' + self.label_full + ' nicht in min/max Grenzen')

            # Werte in Variable festsetzen:
            self.fixed = True
            self.value = helpers.getVector(value, len)

        # Register Element:
        # myMom .variables.append(self) # Komponentenliste
        baseModel.variables.append(self)  # baseModel-Liste mit allen vars
        myMom.model.variables.append(self)  # TODO: not nice, that this specific thing for energysystems is done here

    def transform2MathModel(self, baseModel: LinearModel):
        self.baseModel = baseModel

        # TODO: self.var ist hier einziges Attribut, das baseModel-spezifisch ist: --> umbetten in baseModel!
        if baseModel.modeling_language == 'pyomo':
            if self.isBinary:
                self.var = pyomoEnv.Var(self.indexe, domain=pyomoEnv.Binary)
                # self.var = Var(baseModel.timesteps,domain=Binary)
            else:
                self.var = pyomoEnv.Var(self.indexe, within=pyomoEnv.Reals)

            # Register in pyomo-model:
            aNameSuffixInPyomo = 'var_' + self.myMom.label + '_' + self.label  # z.B. KWK1_On
            baseModel.registerPyComp(self.var, aNameSuffixInPyomo)

            for i in self.indexe:
                # Wenn Vorgabe-Wert vorhanden:
                if self.fixed and (self.value[i] != None):
                    # Fixieren:
                    self.var[i].value = self.value[i]
                    self.var[i].fix()
                else:
                    # Boundaries:
                    self.var[i].setlb(self.min_vec[i])  # min
                    self.var[i].setub(self.max_vec[i])  # max


        elif baseModel.modeling_language == 'vcxpy':
            raise Exception('not defined for modtype ' + baseModel.modeling_language)
        else:
            raise Exception('not defined for modtype ' + baseModel.modeling_language)

    def resetResult(self):
        self.__result = None

    def getResult(self):
        # wenn noch nicht abgefragt: (so wird verhindert, dass für jede Abfrage jedesMal neuer Speicher bereitgestellt wird.)
        if self.__result is None:
            if self.baseModel.modeling_language == 'pyomo':
                # get Data:
                values = self.var.get_values().values()  # .values() of dict, because {0:0.1, 1:0.3,...}
                # choose dataType:
                if self.isBinary:
                    dType = np.int8  # geht das vielleicht noch kleiner ???
                else:
                    dType = float
                # transform to np-array (fromiter() is 5-7x faster than np.array(list(...)) )
                self.__result = np.fromiter(values, dtype=dType)
                # Falls skalar:
                if len(self.__result) == 1:
                    self.__result = self.__result[0]

            elif self.baseModel.modeling_language == 'vcxpy':
                raise Exception('not defined for modtype ' + self.baseModel.modeling_language)
            else:
                raise Exception('not defined for modtype ' + self.baseModel.modeling_language)

        return self.__result

    def print(self, shiftChars):
        # aStr = ' ' * len(shiftChars)
        aStr = shiftChars + self.getStrDescription()
        print(aStr)

    def getStrDescription(self):
        maxChars = 50  # länge begrenzen falls vector-Darstellung
        aStr = 'var'

        if isinstance(self, VariableTS):
            aStr += ' TS'
        else:
            aStr += '   '

        if self.isBinary:
            aStr += ' bin '
        else:
            aStr += '     '

        aStr += self.label_full + ': ' + 'len=' + str(self.len)
        if self.fixed:
            aStr += ', fixed =' + str(self.value)[:maxChars]

        aStr += ' min = ' + str(self.min)[:maxChars] + ', max = ' + str(self.max)[:maxChars]

        return aStr


# TODO:
# class cTS_Variable (Variable):
#   valuesIsPostTimeStep = False # für Speicherladezustände true!!!
#   oneValDependsOnPrevious : Bool, optional
#             if every value depends on previous -> not fixed in aggregation mode. The default is True.
#   # beforeValues 


# Timeseries-Variable, optional mit Before-Werten:
class VariableTS(Variable):
    def __init__(self, label: str, len: int, myMom, baseModel: LinearModel, isBinary: bool = False,
                 value: Optional[Union[int, float, np.ndarray]] = None,
                 min: Optional[Union[int, float, np.ndarray]] = None,
                 max: Optional[Union[int, float, np.ndarray]] = None):
        assert len > 1, 'len is one, that seems not right for CVariable_TS'
        self.activated_beforeValues = False
        super().__init__(label, len, myMom, baseModel, isBinary=isBinary, value=value, min=min, max=max)

    # aktiviere Before-Werte. ZWINGENDER BEFEHL bei before-Werten
    def activateBeforeValues(self, esBeforeValue,
                             beforeValueIsStartValue) -> None:  # beforeValueIsStartValue heißt ob es Speicherladezustand ist oder Nicht
        # TODO: Achtung: private Variablen wären besser, aber irgendwie nimmt er die nicht. Ich vermute, das liegt am fehlenden init
        self.beforeValueIsStartValue = beforeValueIsStartValue
        self.esBeforeValue = esBeforeValue  # Standardwerte für Simulationsstart im Energiesystem
        self.activated_beforeValues = True

    def transform2MathModel(self, baseModel:LinearModel) -> None:
        super().transform2MathModel(baseModel)

    # hole Startwert/letzten Wert vor diesem Segment:
    def beforeVal(self):
        assert self.activated_beforeValues, 'activateBeforeValues() not executed'
        # wenn beforeValue-Datensatz für baseModel gegeben:
        if self.baseModel.beforeValueSet is not None:
            # für Variable rausziehen:
            (value, time) = self.baseModel.beforeValueSet.getBeforeValues(self)
            return value
        # sonst Standard-BeforeValues von Energiesystem verwenden:
        else:
            return self.esBeforeValue

    # hole Startwert/letzten Wert für nächstes Segment:
    def getBeforeValueForNEXTSegment(self, lastUsedIndex) -> Tuple:
        assert self.activated_beforeValues, 'activateBeforeValues() not executed'
        # Wenn Speicherladezustand o.ä.
        if self.beforeValueIsStartValue:
            index = lastUsedIndex + 1  # = Ladezustand zum Startzeitpunkt des nächsten Segments
        # sonst:
        else:
            index = lastUsedIndex  # Leistungswert beim Zeitpunkt VOR Startzeitpunkt vom nächsten Segment
        aTime = self.baseModel.timeSeriesWithEnd[index]
        aValue = self.getResult()[index]
        return (aValue, aTime)


# managed die Before-Werte des segments:
class StartValue:
    def __init__(self, fromBaseModel, lastUsedIndex):
        self.fromBaseModel = fromBaseModel
        self.beforeValues = {}
        # Sieht dann so aus = {(Element1, aVar1.name): (value, time),
        #                      (Element2, aVar2.name): (value, time),
        #                       ...                       }
        for aVar in self.fromBaseModel.all_ts_variables:
            aVar: VariableTS
            if aVar.activated_beforeValues:
                # Before-Value holen:
                (aValue, aTime) = aVar.getBeforeValueForNEXTSegment(lastUsedIndex)
                self.addBeforeValues(aVar, aValue, aTime)

    def addBeforeValues(self, aVar, aValue, aTime):
        element = aVar.myMom
        aKey = (element, aVar.label)  # hier muss label genommen werden, da aVar sich ja ändert je baseModel!
        # beforeValues = aVar.getResult(aValue) # letzten zwei Werte

        if aKey in self.beforeValues.keys():
            raise Exception('setBeforeValues(): Achtung Wert würde überschrieben, Wert ist schon belegt!')
        else:
            self.beforeValues.update({aKey: (aValue, aTime)})

    # return (value, time)
    def getBeforeValues(self, aVar):
        element = aVar.myMom
        aKey = (element, aVar.label)  # hier muss label genommen werden, da aVar sich ja ändert je baseModel!
        if aKey in self.beforeValues.keys():
            return self.beforeValues[aKey]  # returns (value, time)
        else:
            return None

    def print(self):
        for (element, varName) in self.beforeValues.keys():
            print(element.label + '.' + varName + ' = ' + str(self.beforeValues[(element, varName)]))


# class cInequation(Equation):
#   def __init__(self, label, myMom, baseModel):
#     super().__init__(label, myMom, baseModel, eqType='ineq')    

class Equation:
    def __init__(self, label: str, myMom, baseModel: LinearModel, eqType: Literal['eq', 'ineq', 'objective'] = 'eq'):
        self.label = label
        self.listOfSummands = []
        self.nrOfSingleEquations = 1  # Anzahl der Gleichungen
        self.y = 0  # rechte Seite
        self.y_shares = []  # liste mit shares von y
        self.y_vec = helpers.getVector(self.y, self.nrOfSingleEquations)
        self.eqType = eqType
        self.myMom = myMom
        self.eq = None  # z.B. für pyomo : pyomoComponente

        log.debug('equation created: ' + str(label))

        ## Register Element:
        # Equation:
        if eqType == 'ineq':  # lhs <= rhs
            # myMom .ineqs.append(self) # Komponentenliste
            baseModel.ineqs.append(self)  # baseModel-Liste mit allen ineqs
            myMom.model.ineqs.append(self)
        # Inequation:
        elif eqType == 'eq':
            # myMom .eqs.append(self) # Komponentenliste
            baseModel.eqs.append(self)  # baseModel-Liste mit allen eqs
            myMom.model.eqs.append(self)
        # Objective:
        elif eqType == 'objective':
            if baseModel.objective == None:
                baseModel.objective = self
                myMom.model.objective = self
            else:
                raise Exception('baseModel.objective ist bereits belegt!')
        # Undefined:
        else:
            raise Exception('Equation.eqType ' + str(self.eqType) + ' nicht definiert!')

        # in Matlab noch:
        # B; % B of this object (related to x)!
        # B_visual; % mit Spaltenüberschriften!
        # maxElementsOfVisualCell = 1e4; % über 10000 Spalten wählt Matlab komische Darstellung

    def addSummand(self, variable, factor, indexeOfVariable=None):
        # input: 3 Varianten entscheiden über Elemente-Anzahl des Summanden:
        #   len(variable) = 1             -> len = len(factor)
        #   len(factor)   = 1             -> len = len(variable)
        #   len(factor)   = len(variable) -> len = len(factor)

        isSumOf_Type = False  # kein SumOf_Type!
        self.addUniversalSummand(variable, factor, isSumOf_Type, indexeOfVariable)

    def addSummandSumOf(self, variable, factor, indexeOfVariable=None):
        isSumOf_Type = True  # SumOf_Type!

        if variable is None:
            raise Exception('Fehler in eq ' + str(self.label) + ': variable = None!')
        self.addUniversalSummand(variable, factor, isSumOf_Type, indexeOfVariable)

    def addUniversalSummand(self, variable, factor, isSumOf_Type, indexeOfVar):
        if not isinstance(variable, Variable):
            raise Exception('error in eq ' + self.label + ' : no variable given (variable = ' + str(variable) + ')')
        # Wenn nur ein Wert, dann Liste mit einem Eintrag drausmachen:
        if np.isscalar(indexeOfVar):
            indexeOfVar = [indexeOfVar]
        # Vektor/Summand erstellen:
        aVector = Summand(variable, factor, indexeOfVariable=indexeOfVar)

        if isSumOf_Type:
            aVector.sumOf()  # Umwandlung zu Sum-Of-Skalar
        # Check Variablen-Länge:
        self.__UpdateLengthOfEquations(aVector.len, aVector.variable.label)
        # zu Liste hinzufügen:
        self.listOfSummands.append(aVector)

    def addRightSide(self, aValue):
        '''
          value of the right side,
          if method is executed several times, than values are summed up.

          Parameters
          ----------
          aValue : float or array
              y-value of equation [A*x = y] or [A*x <= y]

          Returns
          -------
          None.

          '''
        # Wert ablegen
        self.y_shares.append(aValue)

        # Wert hinzufügen:
        self.y = np.add(self.y, aValue)  # Addieren
        # Check Variablen-Länge:
        if np.isscalar(self.y):
            y_len = 1
        else:
            y_len = len(self.y)
        self.__UpdateLengthOfEquations(y_len, 'y')

        # hier erstellen (z.B. für StrDescription notwendig)
        self.y_vec = helpers.getVector(self.y, self.nrOfSingleEquations)

        # Umsetzung in der gewählten Modellierungssprache:

    def transform2MathModel(self, baseModel: LinearModel):
        log.debug('eq ' + self.label + '.transform2MathModel()')

        # y_vec hier erneut erstellen, da Anz. Glg. vorher noch nicht bekannt:
        self.y_vec = helpers.getVector(self.y, self.nrOfSingleEquations)

        if baseModel.modeling_language == 'pyomo':
            # 1. Constraints:
            if self.eqType in ['eq', 'ineq']:

                # lineare Summierung für i-te Gleichung:
                def linearSumRule(model, i):
                    lhs = 0
                    aSummand: Summand
                    for aSummand in self.listOfSummands:
                        lhs += aSummand.getMathExpression_Pyomo(baseModel.modeling_language,
                                                                i)  # i-te Gleichung (wenn Skalar, dann wird i ignoriert)
                    rhs = self.y_vec[i]
                    # Unterscheidung return-value je nach typ:
                    if self.eqType == 'eq':
                        return lhs == rhs
                    elif self.eqType == 'ineq':
                        return lhs <= rhs

                # TODO: self.eq ist hier einziges Attribut, das baseModel-spezifisch ist: --> umbetten in baseModel!
                self.eq = pyomoEnv.Constraint(range(self.nrOfSingleEquations),
                                              rule=linearSumRule)  # Nebenbedingung erstellen
                # Register im Pyomo:
                baseModel.registerPyComp(self.eq,
                                         'eq_' + self.myMom.label + '_' + self.label)  # in pyomo-Modell mit eindeutigem Namen registrieren

            # 2. Zielfunktion:
            elif self.eqType == 'objective':
                # Anmerkung: nrOfEquation - Check könnte auch weiter vorne schon passieren!
                if self.nrOfSingleEquations > 1:
                    raise Exception('Equation muss für objective ein Skalar ergeben!!!')

                # Summierung der Skalare:
                def linearSumRule_Skalar(model):
                    skalar = 0
                    for aSummand in self.listOfSummands:
                        skalar += aSummand.getMathExpression_Pyomo(baseModel.modeling_language)  # kein i übergeben, da skalar
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

    def print(self, shiftChars, eqNr=0):
        aStr = ' ' * len(shiftChars) + self.getStrDescription(eqNr=eqNr)
        print(aStr)

    def getStrDescription(self, eqNr=0):
        eqNr = min(eqNr, self.nrOfSingleEquations - 1)

        aStr = ''
        # header:
        if self.eqType == 'objective':
            aStr += 'obj' + ': '  # leerzeichen wichtig, sonst im yaml interpretation als dict
        else:
            aStr += 'eq ' + self.label + '[' + str(eqNr) + ' of ' + str(self.nrOfSingleEquations) + ']: '

        # Summanden:
        first = True
        for aSummand in self.listOfSummands:
            if not first: aStr += ' + '
            first = False
            if aSummand.len == 1:
                i = 0
            else:
                i = eqNr
            #      i     = min(eqNr, aSummand.len-1) # wenn zu groß, dann letzter Eintrag ???
            index = aSummand.indexeOfVariable[i]
            # factor formatieren:
            factor = aSummand.factor_vec[i]
            factor_str = str(factor) if isinstance(factor, int) else "{:.6}".format(float(factor))
            # Gesamtstring:
            aElementOfSummandStr = factor_str + '* ' + aSummand.variable.label_full + '[' + str(index) + ']'
            if aSummand.isSumOf_Type:
                aStr += '∑('
                if i > 0:
                    aStr += '..+'
                aStr += aElementOfSummandStr
                if i < aSummand.len:
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
        aStr += str(self.y_vec[eqNr])  # todo: hier könnte man noch aufsplitten nach y_shares

        return aStr

    ##############################################
    # private Methods:

    # Anzahl Gleichungen anpassen und check, ob Länge des neuen Vektors ok ist:
    def __UpdateLengthOfEquations(self, lenOfSummand, SummandLabel):
        if self.nrOfSingleEquations == 1:
            # Wenn noch nicht Länge der Vektoren abgelegt, dann bestimmt der erste Vektor-Summand:
            self.nrOfSingleEquations = lenOfSummand
            # Update der rechten Seite:
            self.y_vec = helpers.getVector(self.y, self.nrOfSingleEquations)
        else:
            # Wenn Variable länger als bisherige Vektoren in der Gleichung:
            if (lenOfSummand != 1) & (
                    lenOfSummand != self.nrOfSingleEquations):  # Wenn kein Skalar & nicht zu Länge der anderen Vektoren passend:
                raise Exception(
                    'Variable ' + SummandLabel + ' hat eine nicht passende Länge für Gleichung ' + self.label + '!')

            # Vektor aus Vektor-Variable und Faktor!


# Beachte: Muss auch funktionieren für den Fall, dass variable.var fixe Werte sind.
class Summand:
    def __init__(self, variable, factor, indexeOfVariable=None):  # indexeOfVariable default : alle
        self.__dict__.update(**locals())
        self.isSumOf_Type = False  # Falls Skalar durch Summation über alle Indexe
        self.sumOfExpr = None  # Zwischenspeicher für Ergebniswert (damit das nicht immer wieder berechnet wird)

        # wenn nicht definiert, dann alle Indexe
        if self.indexeOfVariable is None:
            self.indexeOfVariable = variable.indexe  # alle indexe

        self.len_var_indexe = len(self.indexeOfVariable)

        # Länge ermitteln:
        self.len = self.getAndCheckLength(self.len_var_indexe, factor)

        # Faktor als Vektor:
        self.factor_vec = helpers.getVector(factor, self.len)

    # @staticmethod
    def getAndCheckLength(self, len_var_indexe, factor):
        len_factor = 1 if np.isscalar(factor) else len(factor)
        if len_var_indexe == len_factor:
            aLen = len_var_indexe
        elif len_factor == 1:
            aLen = len_var_indexe
        elif len_var_indexe == 1:
            aLen = len_factor
        else:
            raise Exception(
                'Variable ' + self.variable.label_full + '(len=' + str(len_var_indexe) + ') und Faktor (len=' + str(
                    len_factor) + ') müssen gleiche Länge haben oder Skalar sein')
        return aLen

    # Umwandeln zu Summe aller Elemente:
    def sumOf(self):
        if len == 1:
            print(
                'warning: Summand.sumOf() senceless für Variable ' + self.variable.label + ', because only one vector-element already')
        self.isSumOf_Type = True
        self.len = 1  # jetzt nur noch Skalar!
        return self

    # Ausdruck für i-te Gleichung (falls Skalar, dann immer gleicher Ausdruck ausgegeben)
    def getMathExpression_Pyomo(self, modType, nrOfEq=0):
        # TODO: alles noch nicht sonderlich schön, weil viele Schleifen --> Performance!
        # Wenn SumOfType:
        if self.isSumOf_Type:
            # Wenn Zwischenspeicher leer, dann füllen:
            if self.sumOfExpr is None:
                self.sumOfExpr = sum(
                    self.variable.var[self.indexeOfVariable[j]] * self.factor_vec[j] for j in self.indexeOfVariable)
            expr = self.sumOfExpr
        # Wenn Skalar oder Vektor:
        else:
            # Wenn Skalar:
            if self.len == 1:
                # ignore argument nrOfEq, because Skalar is used for every single equation
                nrOfEq = 0

                # Wenn nur Skalare-Variable bzw. ein Index:
            if self.len_var_indexe == 1:
                indexeOfVar = 0
            else:
                indexeOfVar = nrOfEq

            ## expression:
            expr = self.variable.var[self.indexeOfVariable[indexeOfVar]] * self.factor_vec[nrOfEq]
        return expr


class SolverLog():
    def __init__(self, solver_name, filename, string=None):

        if filename is None:

            self.log = string
        else:
            file = open(filename, 'r')
            self.log = file.read()
            file.close()

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
    def parseInfos(self):
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
            raise Exception('SolverLog.parseInfos() is not defined for solver ' + self.solver_name)
