# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:09:02 2020

@author: Panitz
"""

# todo:

# timesteps als pyomo-RangeSet für alle festlegen, damit keine redundanten Sets für jede Variable
# -> overloaded Operators für Summanden/Vektoren

# -> sollte man alles auf np.arrays umstellen? z.B. helpers.getVector()

# -> ist das fixing der Variablen eigentlich performance-Aufwändig --> ggf. Umsetzung ändern? (erweitert das erstmal die Matrix oder wird die vorher verkürzt?)

# -> cVariable nicht beides lagern: min_vec, -> nur zweites nutzen

# für pyomo-Modelliierung: Verhindern, dass time-index für jede Variable separat erzeugt wird!

# binäre Variablen, immer zwei Ergebniss-Varianten angeben: on = 0/1 , on_nan = nan/1 (wie in FAkS)
  
import logging
import numpy as np
import importlib
import gurobipy
import time

pyomoEnv = None # das ist module, das nur bei Bedarf belegt wird

log = logging.getLogger(__name__)

import flixOptHelperFcts as helpers

# Klasse für Gleichungen a_1.*x_1+a_2.*x_2 = y 
# x_1, a_1 können Vektoren oder Skalare sein.

# Modell zum Addieren von Vektor-Variablen und Skalaren : 
# zulässige Summanden:
# + var_vec * factor_vec
# + var_vec * factor 
# +           factor 
# + var     * factor
# + var     * factor_vec  # macht das Sinn? Ist das überhaupt implementiert?

      
class cBaseModel:
  
  @property
  def infos(self):    
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
    
    if self.solverLog is not None:      
      infos['solverLog'] = self.solverLog.infos

    
    return infos  
  def __init__(self, label, aModType):
    self._infos      = {}
    self.label        = label
    self.modType      = aModType

    self.countComp    = 0;    # ElementeZähler für Pyomo
    self.model        = None; # Übergabe später, zumindest für Pyomo notwendig
    
    self.epsilon    = 1e-5 # 

    self.variables  = [] # Liste aller Variablen
    self.eqs        = [] # Liste aller Gleichungen
    self.ineqs      = [] # Liste aller Ungleichungen
   
    self.objective       = None # objective-Function
    self.objective_value = None # Ergebnis
        
    self.duration        = {} # Laufzeiten    
    self.solverLog       = None # logging und parsen des solver-outputs
    
    if self.modType == 'pyomo':
      global pyomoEnv # als globale Variable
      import pyomo.environ as pyomoEnv
      # pyomoEnv = importlib.import_module('pyomo.environ', package = None)
      # global pyomoEnv # mache es global
      # import pyomo.environ as pyomoEnv #Set,Param,Var,AbstractModel,Objective,Constraint,maximize 
      log.info('pyomo Module geladen')
    else:
      raise Exception('not defined for modType' + str(self.modType))
    ########################################
    # globales Zeugs :
    if   self.modType == 'pyomo':
      # für den Fall pyomo wird EIN Modell erzeugt, das auch für rollierende Durchlaufe immer wieder genutzt wird.
      self.model     = pyomoEnv.ConcreteModel(name="(Minimalbeispiel)") 

      # todo, generelles timestep-Set für alle
      # self.timesteps = pyomoEnv.RangeSet(0,len(self.timeSeries)-1) # Start-index = 0, weil np.arrays auch so
      # # initialisieren:
      # self.registerPyComp(self.timesteps)             
    elif self.modType == 'cvxpy':
      pass
    else:
      pass
  
  def printNoEqsAndVars(self):
      print('no of Eqs   (single):' + str(self.noOfEqs)   + ' (' + str(self.noOfSingleEqs)   + ')')
      print('no of InEqs (single):' + str(self.noOfIneqs) + ' (' + str(self.noOfSingleIneqs) + ')')
      print('no of Vars  (single):' + str(self.noOfVars)  + ' (' + str(self.noOfSingleVars)  + ')')
  
 ##############################################################################################
  ################ pyomo-Spezifisch 
  # alle Pyomo Elemente müssen im model registriert sein, sonst werden sie nicht berücksichtigt 
  def registerPyComp(self,py_comp,aStr='',oldPyCompToOverwrite=None):       
    # neu erstellen
    if oldPyCompToOverwrite == None :
      self.countComp += 1
      # Komponenten einfach hochzählen, damit eindeutige Namen, d.h. a1_timesteps, a2, a3 ,...
      # Beispiel: 
      # model.add_component('a1',py_comp) äquivalent zu model.a1 = py_comp
      self.model.add_component('a' + str(self.countComp) + '_' + aStr , py_comp) #a1,a2,a3, ...        
    # altes überschreiben:
    else: 
      self.__overwritePyComp(py_comp,oldPyCompToOverwrite)      
  
  # Komponente löschen:
  def deletePyComp(self,py_comp):
    aName = self.getPyCompStr(old_py_comp)    
    aNameOfAdditionalComp = aName + '_index' # sowas wird bei manchen Komponenten als Komponente automatisch mit erzeugt.       
    # sonstige zugehörige Variablen löschen:
    if aNameOfAdditionalComp in self.model.component_map().keys():
      self.model.del_component(aNameOfAdditionalComp)
    self.model.del_component(aName)  
    
  # name of component
  def getPyCompStr(self,aComp):
    for key,val in self.model.component_map().iteritems():
      if aComp == val:
        return key
      
  def getPyComp(self,aStr):
    return self.model.component_map()[aStr]
  
  # gleichnamige Pyomo-Komponente überschreiben (wenn schon vorhanden, sonst neu)
  def __overwritePyComp(self,py_comp,old_py_comp):            
    aName = self.getPyCompStr(old_py_comp)  
    # alles alte löschen:
    self.deletePyComp(old_py_comp)
    # überschreiben:    
    self.model.add_component(aName, py_comp)
    
  def transform2MathModel(self):
    
    self._charactarizeProblem()
    
    t_start = time.time()    
    eq : cEquation
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
    
    self.duration['transform2MathModel'] = round(time.time()-t_start,2)


  # Attention: is overrided by childclass:
  def _charactarizeProblem(self):
    eq : cEquation
    var : cVariable
  
    
    self.noOfEqs       = len(self.eqs)
    self.noOfSingleEqs = sum([eq.nrOfSingleEquations for eq in self.eqs])

    self.noOfIneqs       = len(self.ineqs)
    self.noOfSingleIneqs = sum([eq.nrOfSingleEquations for eq in self.ineqs])    
    
    self.noOfVars       = len(self.variables)
    self.noOfSingleVars = sum([var.len for var in self.variables])    
    
  def solve(self,gapfrac,timelimit, solver_name, displaySolverOutput, **kwargs):        
    self.solver_name = solver_name
    t_start = time.time()
    for variable in self.variables:
      variable.resetResult() # altes Ergebnis löschen (falls vorhanden)
    if self.modType == 'pyomo' :
      solver = pyomoEnv.SolverFactory(solver_name)
      
      solver_opt = kwargs # kwargs werden schon mal übernommen
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

      logfileName = "flixSolverLog.log"
      self.solver_results = solver.solve(self.model, options = solver_opt, tee = displaySolverOutput, keepfiles=True, logfile=logfileName)     

      # Log laden:
      self.solverLog = cSolverLog(solver_name,logfileName)
      self.solverLog.parseInfos()
      # Ergebnis Zielfunktion ablegen
      self.objective_value = self.model.objective.expr()
    
    else: 
      raise Exception('not defined for modtype ' + self.modType)

    self.duration['solve'] = round(time.time()-t_start,2)


class cVariable : 
  def __init__(self, label, len, myMom, baseModel, isBinary = False, indexe = None, value = None, min = None , max = None): # indexe müssen nicht übergeben werden   
    self.label = label
    self.len   = len
    self.myMom = myMom
    self.baseModel = baseModel
    self.isBinary = isBinary
    self.indexe = indexe
    self.value  = value
    self.min    = min
    self.max    = max
    
    self.label_full = myMom.label + '.' + label
    self.fixed = False    
    self.result = None # Ergebnis
    
    self.value_vec = helpers.getVector(value, self.len)
    # boundaries:
    self.min_vec = helpers.getVector(min, self.len) # note: auch aus None wird None-Vektor
    self.max_vec = helpers.getVector(max, self.len)       
    self.__result = None # Ergebnis-Speicher
    log.debug('Variable created: ' + self.label)

    # Check conformity:
    self.label = helpers.checkForAttributeNameConformity(label)  
    
    
    # Wenn indexe nicht explizit gegeben:     
    if self.indexe == None:
      self.indexe = range(self.len)
    # wenn explizit geben:
    else :
      # check len:
      if len != len(indexe):
        raise Exception('len und len(indexe) passt nicht zusammen!')
    
    # Wenn Vorgabewert vorhanden:
    if not (value is None) :
      # check if conflict with min/max values        
      # Wenn ein Element nicht in min/max-Grenzen
      
      minOk = (self.min is None) or (np.all(self.value >= self.min_vec)) # prüft elementweise
      maxOk = (self.max is None) or (np.all(self.value <= self.max_vec)) # prüft elementweise
      if (not(minOk)) or (not(maxOk)): 
        raise Exception('cVariable.value' + self.label_full + ' nicht in min/max Grenzen')
           
      # Werte in Variable festsetzen:            
      self.fixed = True
      self.value = helpers.getVector(value, len)
        
    # Register me:
    # myMom .variables.append(self) # Komponentenliste
    baseModel   .variables.append(self) # baseModel-Liste mit allen vars
    myMom.mod.variables.append(self)
   
  def transform2MathModel(self,baseModel:cBaseModel):
    self.baseModel = baseModel
    
    # TODO: self.var ist hier einziges Attribut, das baseModel-spezifisch ist: --> umbetten in baseModel!
    if baseModel.modType == 'pyomo':
      
      if self.isBinary:
        self.var = pyomoEnv.Var(self.indexe, domain = pyomoEnv.Binary)
        # self.var = Var(baseModel.timesteps,domain=Binary)
      else:
        self.var = pyomoEnv.Var(self.indexe, within = pyomoEnv.Reals)

      # Register in pyomo-model:
      aNameSuffixInPyomo = 'var_' + self.myMom.label + '_' + self.label # z.B. KWK1_On      
      baseModel.registerPyComp(self.var, aNameSuffixInPyomo) 

      
      for i in self.indexe:          
        # Wenn Vorgabe-Wert vorhanden:          
        if self.fixed and (self.value[i] != None) : 
          # Fixieren:
          self.var[i].value = self.value[i] 
          self.var[i].fix()
        else:
          # Boundaries:
          self.var[i].setlb(self.min_vec[i]) # min
          self.var[i].setub(self.max_vec[i]) # max

      
    elif baseModel.modType == 'vcxpy':
      raise Exception('not defined for modtype ' + baseModel.modType)
    else:
      raise Exception('not defined for modtype ' + baseModel.modType)
      
  def resetResult(self):
    self.__result = None
    
  def getResult(self):
    # wenn noch nicht abgefragt: (so wird verhindert, dass für jede Abfrage jedesMal neuer Speicher bereitgestellt wird.)
    if self.__result is None : 
      if self.baseModel.modType == 'pyomo':
        # get Data:
        values = self.var.get_values().values() # .values() of dict, because {0:0.1, 1:0.3,...}
        # choose dataType:
        if self.isBinary :
          dType = np.int8 # geht das vielleicht noch kleiner ???
        else:
          dType = float
        # transform to np-array (fromiter() is 5-7x faster than np.array(list(...)) )
        self.__result = np.fromiter(values, dtype = dType)  
        # Falls skalar:
        if len(self.__result) == 1:
          self.__result = self.__result[0]
          
      elif self.baseModel.modType == 'vcxpy':
        raise Exception('not defined for modtype ' + baseModel.modType)
      else: 
        raise Exception('not defined for modtype ' + baseModel.modType)

    return self.__result
      
  def print(self,shiftChars):
    # aStr = ' ' * len(shiftChars)
    aStr = shiftChars + self.getStrDescription()
    print(aStr)
    
  def getStrDescription(self):
    maxChars = 50 #länge begrenzen falls vector-Darstellung
    aStr = ''
    if self.isBinary:        
      aStr += 'var bin '
    else:
      aStr += 'var     '  
    aStr += self.label_full + ': ' + 'len=' + str(self.len)
    if self.fixed :
      aStr += ', fixed =' + str(self.value)[:maxChars]
    
    aStr += ' min = ' + str(self.min)[:maxChars] + ', max = ' + str(self.max)[:maxChars]
    
    return aStr
  
# TODO:
# class cTS_Variable (cVariable):  
#   valuesIsPostTimeStep = False # für Speicherladezustände true!!!
#   # beforeValues 
  
# TODO man könnten noch dazwischen allgemeine Klasse cTS_Variable bauen
# variable with Before-Values:
  
# Variable mit Before-Werten:
class cVariableB (cVariable): 

  #######################################
  # gleiches __init__ wie cVariable!
  #######################################
  
  # aktiviere Before-Werte. ZWINGENDER BEFEHL bei cVariableB
  def activateBeforeValues(self, esBeforeValue, beforeValueIsStartValue):  # beforeValueIsStartValue heißt ob es Speicherladezustand ist oder Nicht
    # TODO: Achtung: private Variablen wären besser, aber irgendwie nimmt er die nicht. Ich vermute, das liegt am fehlenden init
    self.beforeValueIsStartValue = beforeValueIsStartValue
    self.esBeforeValue  = esBeforeValue  # Standardwerte für Simulationsstart im Energiesystem
    self.activated_B = True
    
  def transform2MathModel(self,baseModel):
    assert hasattr(self, 'activated_B') and (self.activated_B) , ('var ' + self.label + ':activateBeforeValues() nicht ausgeführt.')
    super().transform2MathModel(baseModel)

  # hole Startwert/letzten Wert vor diesem Segment:
  def beforeVal(self):    
    # wenn beforeValue-Datensatz für baseModel gegeben:
    if self.baseModel.beforeValueSet is not None   : 
      # für Variable rausziehen: 
      (value,time) =  self.baseModel.beforeValueSet.getBeforeValues(self)    
      return value
    # sonst Standard-BeforeValues von Energiesystem verwenden:
    else:
      return self.esBeforeValue
  
  # hole Startwert/letzten Wert für nächstes Segment:
  def getBeforeValueForNEXTSegment(self, lastUsedIndex):     
    # Wenn Speicherladezustand o.ä.
    if self.beforeValueIsStartValue : 
      index = lastUsedIndex + 1 # = Ladezustand zum Startzeitpunkt des nächsten Segments
    # sonst:
    else:
      index = lastUsedIndex     # Leistungswert beim Zeitpunkt VOR Startzeitpunkt vom nächsten Segment    
    aTime  = self.baseModel.timeSeriesWithEnd[index]
    aValue = self.getResult()  [index]    
    return (aValue, aTime)


# managed die Before-Werte des segments:
class cBeforeValueSet : 
  def __init__(self, fromBaseModel, lastUsedIndex):
    self.fromBaseModel = fromBaseModel
    self.beforeValues = {} 
    # Sieht dann so aus = {(aME1, aVar1.name): (value, time),
    #                      (aME2, aVar2.name): (value, time),
    #                       ...                       }
    for aVar in self.fromBaseModel.variables :
      if isinstance(aVar, cVariableB):
         # Before-Value holen:
        (aValue, aTime) = aVar.getBeforeValueForNEXTSegment(lastUsedIndex)   
        self.addBeforeValues(aVar, aValue, aTime)
  
  def addBeforeValues(self, aVar, aValue, aTime):    
    aME     = aVar.myMom
    aKey    = (aME, aVar.label) # hier muss label genommen werden, da aVar sich ja ändert je baseModel!
    # beforeValues = aVar.getResult(aValue) # letzten zwei Werte
    
    if aKey in self.beforeValues.keys():
      raise Exception('setBeforeValues(): Achtung Wert würde überschrieben, Wert ist schon belegt!')    
    else:
      self.beforeValues.update({aKey : (aValue, aTime)})

  # return (value, time)
  def getBeforeValues(self, aVar) :
    aME  = aVar.myMom
    aKey = (aME, aVar.label) # hier muss label genommen werden, da aVar sich ja ändert je baseModel!
    if aKey in self.beforeValues.keys():
      return self.beforeValues[aKey] # returns (value, time)
    else :
      return None    
    
  def print(self):
    for (aME,varName) in self.beforeValues.keys():
      print(aME.label + '.' + varName + ' = ' + str(self.beforeValues[(aME,varName)]))
      
      
      
# class cInequation(cEquation):
#   def __init__(self, label, myMom, baseModel):
#     super().__init__(label, myMom, baseModel, eqType='ineq')    
    
class cEquation :
  def __init__(self, label, myMom, baseModel, eqType='eq'):
    self.label               = label
    self.listOfSummands      = []
    self.nrOfSingleEquations = 1    # Anzahl der Gleichungen     
    self.y                   = 0    # rechte Seite
    self.y_shares            = []   # liste mit shares von y
    self.y_vec               = helpers.getVector(self.y, self.nrOfSingleEquations)
    self.eqType              = eqType
    self.myMom               = myMom
    self.eq                  = None # z.B. für pyomo : pyomoComponente
    
    log.debug('equation created: ' + str(label))
    
    ## Register Me:   
    # Equation:
    if eqType == 'ineq':
      # myMom .ineqs.append(self) # Komponentenliste
      baseModel   .ineqs.append(self) # baseModel-Liste mit allen ineqs
      myMom.mod.ineqs.append(self)
    # Inequation:
    elif eqType == 'eq':
      # myMom .eqs.append(self) # Komponentenliste
      baseModel   .eqs.append(self) # baseModel-Liste mit allen eqs
      myMom.mod.eqs.append(self)
    # Objective:
    elif eqType == 'objective':
      if baseModel.objective == None:
        baseModel.objective = self
        myMom.mod.objective = self
      else :
        raise Exception('baseModel.objective ist bereits belegt!')
    # Undefined:
    else : 
      raise Exception('cEquation.eqType ' + str(self.eqType) + ' nicht definiert!')
    
    # in Matlab noch:
    # B; % B of this object (related to x)!
    # B_visual; % mit Spaltenüberschriften!
    # maxElementsOfVisualCell = 1e4; % über 10000 Spalten wählt Matlab komische Darstellung      
  
  def addSummand         (self, variable, factor, indexeOfVariable=None):           
    # input: 3 Varianten entscheiden über Elemente-Anzahl des Summanden:
    #   len(variable) = 1             -> len = len(factor) 
    #   len(factor)   = 1             -> len = len(variable)
    #   len(factor)   = len(variable) -> len = len(factor)
      
    isSumOf_Type = False # kein SumOf_Type!
    self.addUniversalSummand(variable, factor, isSumOf_Type, indexeOfVariable)

  def addSummandSumOf    (self, variable, factor, indexeOfVariable=None):
    isSumOf_Type = True # SumOf_Type!
    
    if variable is None:
      raise Exception('Fehler in eq ' + str(self.label) + ': variable = None!')
    self.addUniversalSummand(variable, factor, isSumOf_Type, indexeOfVariable)

  def addUniversalSummand(self, variable, factor, isSumOf_Type, indexeOfVar):    
    if not isinstance(variable, cVariable) :
      raise Exception('error in eq ' + self.label + ' : no variable given (variable = '  + str(variable) + ')')
    # Wenn nur ein Wert, dann Liste mit einem Eintrag drausmachen:
    if np.isscalar(indexeOfVar) :
      indexeOfVar = [indexeOfVar]
    # Vektor/Summand erstellen:
    aVector = cVector(variable, factor, indexeOfVariable = indexeOfVar)    

    if isSumOf_Type :
      aVector.sumOf() # Umwandlung zu Sum-Of-Skalar
    # Check Variablen-Länge:    
    self.__UpdateLengthOfEquations(aVector.len, aVector.variable.label)
    # zu Liste hinzufügen:
    self.listOfSummands.append(aVector)  
 
     
  def addRightSide(self,aValue):
    # Wert ablegen    
    self.y_shares.append(aValue)    

    # Wert hinzufügen:
    self.y = np.add(self.y, aValue) # Addieren
    # Check Variablen-Länge:  
    if np.isscalar(self.y) :     
      y_len = 1
    else : 
      y_len = len(self.y)    
    self.__UpdateLengthOfEquations(y_len, 'y')
    
    # hier erstellen (z.B. für StrDescription notwendig)
    self.y_vec = helpers.getVector(self.y, self.nrOfSingleEquations)    
 
       
  # Umsetzung in der gewählten Modellierungssprache:  
  def transform2MathModel(self,baseModel:cBaseModel):
    log.debug('eq ' + self.label + '.transform2MathModel()' )
    
    # y_vec hier erneut erstellen, da Anz. Glg. vorher noch nicht bekannt:
    self.y_vec = helpers.getVector(self.y, self.nrOfSingleEquations)
    
    if baseModel.modType == 'pyomo':
      # 1. Constraints:
      if self.eqType in ['eq','ineq'] :
         
        # lineare Summierung für i-te Gleichung:
        def linearSumRule(model, i):
          lhs = 0
          aSummand : cVector
          for aSummand in self.listOfSummands:
            lhs += aSummand.getMathExpression_Pyomo(baseModel.modType, i) # i-te Gleichung (wenn Skalar, dann wird i ignoriert)
          rhs = self.y_vec[i]
          # Unterscheidung return-value je nach typ:
          if self.eqType == 'eq':            
            return lhs == rhs
          elif self.eqType == 'ineq':
            return lhs <= rhs
        # TODO: self.eq ist hier einziges Attribut, das baseModel-spezifisch ist: --> umbetten in baseModel!
        self.eq = pyomoEnv.Constraint(range(self.nrOfSingleEquations),rule=linearSumRule)   # Nebenbedingung erstellen
        # Register im Pyomo:
        baseModel.registerPyComp(self.eq,'eq_' + self.myMom.label + '_' + self.label) # in pyomo-Modell mit eindeutigem Namen registrieren    
      
      # 2. Zielfunktion:
      elif self.eqType == 'objective':
        # Anmerkung: nrOfEquation - Check könnte auch weiter vorne schon passieren!
        if self.nrOfSingleEquations > 1 : 
          raise Exception('cEquation muss für objective ein Skalar ergeben!!!')
        
        # Summierung der Skalare:
        def linearSumRule_Skalar(model):        
          skalar = 0
          for aSummand in self.listOfSummands:
            skalar += aSummand.getMathExpression_Pyomo(baseModel.modType) # kein i übergeben, da skalar
          return skalar
        self.eq = pyomoEnv.Objective(rule=linearSumRule_Skalar, sense=pyomoEnv.minimize)  
        # Register im Pyomo:
        baseModel.model.objective = self.eq    
        
      # 3. Undefined:
      else :
        raise Exception('equation.eqType= '  + str(self.eqType) + ' nicht definiert')       
    elif baseModel.modType == 'vcxpy':
      raise Exception('not defined for modtype ' + baseModel.modType)    
    else:
      raise Exception('not defined for modtype ' + baseModel.modType)    
      

  # print i-th equation:
  def print(self,shiftChars,eqNr=0):   
    aStr = ' '  * len(shiftChars) + self.getStrDescription(eqNr = eqNr)
    print(aStr)
    
  
  def getStrDescription(self,eqNr=0):
    eqNr = min(eqNr, self.nrOfSingleEquations-1)
    
    aStr = ''
    # header:      
    if self.eqType == 'objective':
      aStr += 'obj' + ': ' # leerzeichen wichtig, sonst im yaml interpretation als dict
    else: 
      aStr += 'eq ' + self.label + '[' + str(eqNr) + ' of '+ str(self.nrOfSingleEquations) +  ']: '
    
    # Summanden:
    first = True
    for aSummand in self.listOfSummands :      
      if not first : aStr += ' + ' 
      first = False
      if aSummand.len == 1:
        i = 0
      else: 
        i = eqNr 
#      i     = min(eqNr, aSummand.len-1) # wenn zu groß, dann letzter Eintrag ???
      index = aSummand.indexeOfVariable[i]
      # factor formatieren:
      factor     = aSummand.factor_vec[i]
      factor_str = str(factor) if isinstance(factor,int) else "{:.6}".format(float(factor))
      # Gesamtstring:
      aElementOfSummandStr = factor_str + '* ' + aSummand.variable.label_full + '[' + str(index) + ']'
      if aSummand.isSumOf_Type :
        aStr += '∑('
        if i > 0 :
          aStr += '..+'
        aStr += aElementOfSummandStr
        if i < aSummand.len :
          aStr += '+..'
        aStr +=')'
      else : 
        aStr +=            aElementOfSummandStr      
    
    # = oder <= :  
    if self.eqType in ['eq', 'objective']: 
      aStr += ' = '
    elif self.eqType == 'ineq':
      aStr += ' <= '
    else:
      aStr += ' ? ' 
    
    # right side:
    aStr += str(self.y_vec[eqNr]) # todo: hier könnte man noch aufsplitten nach y_shares
    
    return aStr
  
  
  ##############################################  
  # private Methods:
  
  # Anzahl Gleichungen anpassen und check, ob Länge des neuen Vektors ok ist:        
  def __UpdateLengthOfEquations(self,lenOfSummand,SummandLabel):
    if self.nrOfSingleEquations == 1 :
      # Wenn noch nicht Länge der Vektoren abgelegt, dann bestimmt der erste Vektor-Summand:
      self.nrOfSingleEquations = lenOfSummand
      # Update der rechten Seite:
      self.y_vec = helpers.getVector(self.y, self.nrOfSingleEquations)
    else : 
      # Wenn Variable länger als bisherige Vektoren in der Gleichung:
      if (lenOfSummand != 1) & (lenOfSummand != self.nrOfSingleEquations): #  Wenn kein Skalar & nicht zu Länge der anderen Vektoren passend:
        raise Exception('Variable ' + SummandLabel + ' hat eine nicht passende Länge für Gleichung ' + self.label + '!')          
        
 
# Vektor aus Vektor-Variable und Faktor!
# Beachte: Muss auch funktionieren für den Fall, dass variable.var fixe Werte sind.
class cVector :
  def __init__(self, variable, factor, indexeOfVariable = None): # indexeOfVariable default : alle
    self.__dict__.update(**locals())        
    self.isSumOf_Type = False # Falls Skalar durch Summation über alle Indexe
    self.sumOfExpr    = None  # Zwischenspeicher für Ergebniswert (damit das nicht immer wieder berechnet wird)


    # wenn nicht definiert, dann alle Indexe
    if self.indexeOfVariable is None:
      self.indexeOfVariable = variable.indexe # alle indexe
    
    self.len_var_indexe = len(self.indexeOfVariable)

    # Länge ermitteln:
    self.len = self.getAndCheckLength(self.len_var_indexe, factor)   
        
    # Faktor als Vektor:
    self.factor_vec = helpers.getVector(factor, self.len)
    

          
  # @staticmethod
  def getAndCheckLength(self,len_var_indexe,factor):
    len_factor = 1 if np.isscalar(factor) else len(factor)    
    if len_var_indexe == len_factor:
      aLen = len_var_indexe
    elif len_factor == 1:
      aLen = len_var_indexe
    elif len_var_indexe == 1:
      aLen = len_factor
    else:
      raise Exception('Variable ' + self.variable.label_full + '(len='+ str(len_var_indexe) + ') und Faktor (len='+str(len_factor)+') müssen gleiche Länge haben oder Skalar sein')
    return aLen
  
  # Umwandeln zu Summe aller Elemente:
  def sumOf(self): 
    if len == 1 : 
      print('warning: cVector.sumOf() senceless für Variable ' + self.variable.label + ', because only one vector-element already')
    self.isSumOf_Type = True
    self.len          = 1 # jetzt nur noch Skalar!
    return self
  
  # Ausdruck für i-te Gleichung (falls Skalar, dann immer gleicher Ausdruck ausgegeben)
  def getMathExpression_Pyomo(self,modType, nrOfEq=0): 
    # TODO: alles noch nicht sonderlich schön, weil viele Schleifen --> Performance!      
    # Wenn SumOfType:
    if self.isSumOf_Type :
      # Wenn Zwischenspeicher leer, dann füllen:
      if self.sumOfExpr is None :
        self.sumOfExpr = sum(self.variable.var[self.indexeOfVariable[j]] * self.factor_vec[j] for j in self.indexeOfVariable)     
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
      else :
        indexeOfVar =nrOfEq

      ## expression:      
      expr   = self.variable.var[self.indexeOfVariable[indexeOfVar]] * self.factor_vec[nrOfEq]     
    return expr 



import re 
class cSolverLog():
    def __init__(self,solver_name, filename, string = None):
        
      if filename is None :
         
         self.log = string
      else:
          file = open(filename,'r')
          self.log = file.read()
          file.close()
          
      self.solver_name = solver_name


      self.presolved_rows = None
      self.presolved_cols     = None
      self.presolved_nonzeros = None
      
      self.presolved_continuous  = None
      self.presolved_integer     = None
      self.presolved_binary      = None
      
    @property
    def infos(self):
      infos = {}
      aPreInfo = {}
      infos['presolved'] = aPreInfo
      aPreInfo['cols']     = self.presolved_cols
      aPreInfo['continuous'] = self.presolved_continuous
      aPreInfo['integer']   = self.presolved_integer
      aPreInfo['binary']    = self.presolved_binary           
      aPreInfo['rows']     = self.presolved_rows
      aPreInfo['nonzeros'] = self.presolved_nonzeros
      
      return infos
    
    # Suche infos aus log:
    def parseInfos(self):
      if self.solver_name == 'gurobi':

        #string-Schnipsel 1:
        '''
        Optimize a model with 285 rows, 292 columns and 878 nonzeros
        Model fingerprint: 0x1756ffd1
        Variable types: 202 continuous, 90 integer (90 binary)        
        '''
        #string-Schnipsel 2:
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
          self.presolved_rows     = int(match.group(1))
          self.presolved_cols     = int(match.group(2))
          self.presolved_nonzeros = int(match.group(3))
          # string: Variable types: 53 continuous, 67 integer (67 binary)
          self.presolved_continuous  = int(match.group(4))
          self.presolved_integer     = int(match.group(5))
          self.presolved_binary      = int(match.group(6))   

      elif self.solver_name == 'cbc':

        # string: Presolve 1623 (-1079) rows, 1430 (-1078) columns and 4296 (-3306) elements          
        match = re.search('Presolve (\d+) \((-?\d+)\) rows, (\d+) \((-?\d+)\) columns and (\d+)',self.log)
        if not match is None:
          self.presolved_rows       = int(match.group(1))
          self.presolved_cols       = int(match.group(3))
          self.presolved_nonzeros   = int(match.group(5))
        
        # string: Presolved problem has 862 integers (862 of which binary)
        match = re.search('Presolved problem has (\d+) integers \((\d+) of which binary\)',self.log)
        if not match is None:
          self.presolved_integer     = int(match.group(1))
          self.presolved_binary      = int(match.group(2))   
          self.presolved_continuous  = self.presolved_cols - self.presolved_integer
      else :
        raise Exception('cSolverLog.parseInfos() is not defined for solver ' + self.solver_name)

