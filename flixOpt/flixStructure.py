# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:40:23 2020

@author: Panitz
"""



#########################
#### todos

#  results: Kessel.Q_th und Waermebus.Kessel_Q_th haben noch unterschiedliche IDs!!!

# -> Bei MultiPeriod-Simulation sollten in results Skalare nicht jeweils angehangen werden, z.B. bei invest.var_sum (-> wenn man Variablen definiert als entweder timeseries oder summe, dann sollte das gehen)

# -> STruktur nochmal überlegen. Vielleicht doch edges, bzw. flows, die sich Variablen teilen, wenn Komponenten direkt gekoppelt werden.

# -> sauberes Erstellen von Listen in cEnergysystem für Busse, Flows und Components in !!!
#    eventuell mit Knode als Unterklasse, mit Methoden: .getBuses(), .getFlows()

# -> make flows unidirectional (min = 0)

# -> flow.doModeling() sollte nur einmal aufgerufen werden können!!!

# -> Trennung von bus und Brennstoff!

# -> maximum -> verpflichtend setzen (oemof macht das m.e. mit nominalValue. Dann hat man immer ein Maximum, was man in den gleichungne benutzen kann)

# -> überall korrigieren: self.__dict__.update(**locals()) erstellt auch self.self --> sinnlows

# -> epsilon variabel gestalten (anpassen an Problem)

# --> addShareToGlobalsOfFlows (-> sollte wirklcih modBox übergeben werden, oder sollte einfach jeder Flow sein timeIndexe und dt haben --> vermutlich generischer!)

# -> wenn Flow fixed, dann lieber als Wert statt Variable an pyomo übergeben (peter: weil sonst variable(fix)*variable(nonfix) nicht geht) + Performance!!

# --> Komponenten können aktuell noch gleiche Labels haben (getestet im Beispiel)

# --> eqs sollten in modbox umziehen. Sonst besteht Gefahr, dass zumindest bei printEqs() alte Glg. von Vorgänger-Modbox geprinted werden.

# bei rollierender Optimierung: 
#  1. Schauen, ob man nicht doch lieber, alle Variablen nur für den Zeitbereich definiert und cVariable.lastValues einrichtet, und dort Vorgängerzeitschritte übergibt
#     --> Vorteil: Gleiche Prozedur für 1. Zeistreifen wie für alle anderen!
#  2. Alle Dinge, die auf historische Werte zugreifen kennzeichnen (Attribut) und warnen/sperrenz.B. 
#    Speicher, On/Off, 
#    maximale FlowArbeit,... -> geht wahrscheinlich gar nicht gut!
#    helpers.checkBoundsOfParameters -> vielleicht besser in cBaseComponent integrieren, weil dort self.label 

##########################
## Features to do:
# -> Standardbuse festlegen für Komponenten (Wenn Eingabe, dann Überschreiben der alten)
# -> statt indexen direkt timeseries(ts) als Index überall verwenden 
#        also: Elemente mit indexe=ts[0,4] statt indexe=range(0,4) aufrufen--> Hätte den Vorteil, dass Fehlermeldungen kommen, falls konkrete Zeit z.B. für eine Variable nicht existiert
# -> kosten als flows mit bus und dann vielleicht auch zielfunktion als Komponent --> damit könnte man Kosten sogar zeitlich auflösen!
# -> Kosten als Variable!
# -> Verbindungen mit VL / RL definieren. RL optional. Dann mit Enthalpien rechnen. mulitlayered storage
# -> Integrate pytest!!!
# --> Komponenten sollten ohne Buse verbunden werden können.
# --> flows mit Temperaturen (VL/RL vorsehen) -> Buses sind dann mixing points!
# --> schedule for flow (oemof --> Caterina Köhl von Consolinno!)

# -> sollte man sich mit dem ganzen an solph.network andocken?  
#    --> dort sind einige coole Features (mapping von nodes und edges), 
#    --> oemof tabular hat das auch gemacht, aber aufwendig (Problem irgendwie das verküpfungen eher erstellt werden als Komponenten)

# SOS - special ordered sets Löser direkt übergeben 

import numpy  as np

import copy
import math
import time
import yaml #(für json-Schnipsel-print)

import gurobipy as gurobi
from gurobipy import GRB


import flixOptHelperFcts as helpers

from basicModeling import * # Modelliersprache
from flixBasics    import *
import logging

log = logging.getLogger(__name__)
# TODO: 
  # -> results sollte mit objekt und! label mappen, also results[aKWK] und! results['KWK1']
   
# TODO: 1. cTimePeriodModel -> hat für Zeitbereich die Modellierung timePeriodModel1, timePeriodModel2,...
class cModelBoxOfES(cBaseModel): # Hier kommen die ModellingLanguage-spezifischen Sachen rein
  
  @property
  def infos(self):    
    infos = super().infos
    # Hauptergebnisse:
    infos['main_results'] = self.main_results_str # 
    # unten dran den vorhanden rest:
    infos.update(self._infos) # da steht schon zeug drin
  
    return infos 
  
  def __init__(self, label, aModType, es, esTimeIndexe, TS_explicit = None):
    super().__init__(label, aModType)
    self.es           : cEnergySystem
    self.es           = es # energysystem (wäre Attribut von cTimePeriodModel)
    self.esTimeIndexe = esTimeIndexe
    self.nrOfTimeSteps = len(esTimeIndexe)
    self.TS_explicit   = TS_explicit # für explizite Vorgabe von Daten für TS {TS1: data, TS2:data,...}
    # self.epsilon    = 1e-5 # 
    # self.variables  = [] # Liste aller Variablen
    # self.eqs        = [] # Liste aller Gleichungen
    # self.ineqs      = [] # Liste aller Ungleichungen
    self.ME_mod     = {} # dict mit allen mods der MEs
    
    # self.objective       = None # objective-Function
    # self.objective_value = None # Ergebnis
    
    self.beforeValueSet  = None # hier kommen, wenn vorhanden gegebene Before-Values rein (dominant ggü. before-Werte des energysystems)
    # Zeitdaten generieren:
    (self.timeSeries, self.timeSeriesWithEnd, self.dtInHours, self.dtInHours_tot) =  es.getTimeDataOfTimeIndexe(esTimeIndexe)


  # mod auslesen:
  def getModOfME(self, aModelingElement):
    return self.ME_mod[aModelingElement]

  # register ModelingElements and belonging Mod:
  def registerMEandMod(self, aModelingElement, aMod):
    # Zuordnung ME -> mod
    self.ME_mod[aModelingElement] = aMod # aktuelles mod hier speichern
 
  # override:
  def _charactarizeProblem(self): # overriding same method in motherclass!
      
    super()._charactarizeProblem()
    
    # Systembeschreibung abspeichern: (Beachte: modbox muss aktiviert sein)
    # self.es.activateModBox()
    self._infos['str_Eqs']   = self.es.getEqsAsStr()
    self._infos['str_Vars']  = self.es.getVarsAsStr()

  
  #'gurobi'
  def solve(self, gapFrac = 0.02,timelimit = 3600, solver ='cbc', displaySolverOutput = True, excessThreshold = 0.1, **kwargs):      
      '''
      

      Parameters
      ----------
      gapFrac : TYPE, optional
          DESCRIPTION. The default is 0.02.
      timelimit : TYPE, optional
          DESCRIPTION. The default is 3600.
      solver : TYPE, optional
          DESCRIPTION. The default is 'cbc'.
      displaySolverOutput : TYPE, optional
          DESCRIPTION. The default is True.
      excessThreshold : float, positive!
          threshold for excess: If sum(Excess)>excessThreshold a warning is raised, that an excess is occurs
      **kwargs : TYPE
          DESCRIPTION.

      Returns
      -------
      main_results_str : TYPE
          DESCRIPTION.

      '''
      
          
      
      # check auf valide Solver-Optionen:
      
      if len(kwargs) > 0 :
        for key in kwargs.keys():
          if key not in ['threads']:
            raise Exception('no allowed arguments for kwargs: ' + str(key) + '(all arguments:' +  str(kwargs) +')')
      
      print('')
      print('##############################################################')
      print('##################### solving ################################')
      print('')
      
      self.printNoEqsAndVars()            
      
      
      super().solve(gapFrac, timelimit, solver, displaySolverOutput, **kwargs)
      
      print('termination message: "' + self.solver_results['Solver'][0]['Termination message'] + '"')
      print('')    
      # Variablen-Ergebnisse abspeichern:      
      # 1. dict:  
      (self.results, self.results_var)  = self.es.getResultsAfterSolve()
      # 2. struct:
      self.results_struct = helpers.createStructFromDictInDict(self.results)
      

      print('##############################################################')
      print('################### finished #################################')
      print('')
      for aEffect in self.es.globalComp.listOfEffectTypes:
        print(aEffect.label +  ' in ' + aEffect.unit + ':')
        print('  operation: ' + str(aEffect.operation.mod.var_sum.getResult())) 
        print('  invest   : ' + str(aEffect.invest   .mod.var_sum.getResult())) 
        print('  sum      : ' + str(aEffect.all      .mod.var_sum.getResult())) 
        
        
      print('SUM              : ' + '...todo...')
      print('penaltyCosts     : ' + str(self.es.globalComp.penalty.mod.var_sum.getResult()  ))
      print('––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
      print('Result of Obj : ' + str(self.objective_value                  ))            
      print('lower bound   : ' + str(self.solver_results['Problem'][0]['Lower bound']))
      print('')
      for aBus in self.es.setOfBuses:
        if aBus.withExcess : 
          if any(self.results[aBus.label]['excessIn'] > 0) or any(self.results[aBus.label]['excessOut'] > 0):
          # if any(aBus.excessIn.getResult() > 0) or any(aBus.excessOut.getResult() > 0):
            print('!!!!! Attention !!!!!')
            print('!!!!! Exzess.Value in Bus ' + aBus.label + '!!!!!')          
      
      # Wenn Strafen vorhanden
      if self.es.globalComp.penalty.mod.var_sum.getResult() > 10 :
        print('Take care: -> high penalty makes the used gapFrac quite high')
        print('           -> real costs are not optimized to gapfrac')
        
      print('')
      print('##############################################################')            

      # str description of results:
      # nested fct:
      def _getMainResultsAsStr():
        main_results_str = {}

  
        aEffectDict = {}      
        main_results_str['Effekte'] = aEffectDict
        for aEffect in self.es.globalComp.listOfEffectTypes:
          aDict = {}
          aEffectDict[aEffect.label +' ['+ aEffect.unit + ']'] = aDict
          aDict['operation']= str(aEffect.operation.mod.var_sum.getResult())
          aDict['invest']   = str(aEffect.invest   .mod.var_sum.getResult())
          aDict['sum']      = str(aEffect.all      .mod.var_sum.getResult())
        main_results_str['Effekte']
        main_results_str['penaltyCosts']  = str(self.es.globalComp.penalty.mod.var_sum.getResult())
        main_results_str['Result of Obj'] = self.objective_value
        main_results_str['lower bound']   = self.solver_results['Problem'][0]['Lower bound']
        
        busesWithExcess = []
        main_results_str['busesWithExcess'] = busesWithExcess
        for aBus in self.es.setOfBuses:
          if aBus.withExcess : 
            if sum(self.results[aBus.label]['excessIn']) > excessThreshold or sum(self.results[aBus.label]['excessOut']) > excessThreshold:
              busesWithExcess.append(aBus.label)      
        
        aDict = {'invested':{},
                 'not invested':{}
                 }
        main_results_str['Invest-Decisions']=aDict
        for aInvestFeature in self.es.allInvestFeatures:
            investValue = aInvestFeature.mod.var_investmentSize.getResult()
            investValue = float(investValue) # bei np.floats Probleme bei Speichern
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
      
      
class cEnergySystem:

    ## Properties:  
  
    @property
    def allMEsOfFirstLayerWithoutFlows(self):
      allMEs = self.listOfComponents + list(self.setOfBuses) + [self.globalComp] + self.listOfEffectTypes +  list(self.setOfOtherMEs)
      return allMEs      

    @property
    def allMEsOfFirstLayer(self):
      allMEs = self.allMEsOfFirstLayerWithoutFlows + list(self.setOfFlows) 
      return allMEs    

    @property
    def allInvestFeatures(self):
      allInvestFeatures = []
       
      for aME in self.allMEsOfFirstLayer: # kann in Komponente (z.B. Speicher) oder Flow stecken
        for aSubComp in aME.subElements_all:
          if isinstance(aSubComp,cFeatureInvest):
            allInvestFeatures.append(aSubComp)
      return allInvestFeatures
    
    # Achtung: Funktion wird nicht nur für Getter genutzt.
    def getFlows(self,listOfComps = None) :      
      setOfFlows = set()
      # standardmäßig Flows aller Komponenten:
      if listOfComps is None:
        listOfComps = self.listOfComponents
      # alle comps durchgehen:
      for comp in listOfComps:
        newFlows = comp.inputs + comp.outputs
        setOfFlows = setOfFlows | set(newFlows)
      return setOfFlows
    
    setOfFlows = property(getFlows)

    
    # get all TS in one lis:
    @property
    def allTSinMEs(self):
      ME : cMEModel
      allTS = []
      for ME in self.allMEsOfFirstLayer:
        allTS += ME.TS_list
      return allTS
      
        
    # aktuelles Bus-Set ausgeben (generiert sich aus dem setOfFlows):  
    @property
    def setOfBuses(self):      
      setOfBuses = set()
      # Flow-Liste durchgehen::
      for aFlow in self.setOfFlows:
        setOfBuses.add(aFlow.bus)
      return setOfBuses     
    

    # timeSeries: möglichst format ohne pandas-Nutzung bzw.: Ist DatetimeIndex hier das passende Format?
    def __init__(self, timeSeries, dt_last=None):
      self.timeSeries = timeSeries
      self.dt_last    = dt_last

      self.timeSeriesWithEnd = helpers.getTimeSeriesWithEnd(timeSeries, dt_last)
      helpers.checkTimeSeries('global esTimeSeries', self.timeSeriesWithEnd)
      
      # defaults: 
      self.listOfComponents = []        
      self.setOfOtherMEs    = set()  ## hier kommen zusätzliche MEs rein, z.B. aggregation
      self.listOfEffectTypes  = cEffectTypeList() # Kosten, CO2, Primärenergie, ...
      self.standardEffect   = None # Standard-Effekt, zumeist Kosten
      self.objectiveEffect  = None # Zielfunktions-Effekt, z.B. Kosten oder CO2
      # instanzieren einer globalen Komponente (diese hat globale Gleichungen!!!)
      self.globalComp       = cGlobal('globalComp')
      self.__finalized      = False # wenn die MEs alle finalisiert sind, dann True
      self.modBox           = None # later activated
      # # global sollte das erste Element sein, damit alle anderen Componenten darauf zugreifen können: 
      # self.addComponents(self.globalComp)

    

      
    # Effekte registrieren:
    def addEffects(self,*args):
      newListOfEffects = list(args)
      aNewEffect : cEffect
      for aNewEffect in newListOfEffects:       
        print('Register new effect ' + aNewEffect.label)
        # Wenn bereits vorhanden:
        if aNewEffect in self.listOfEffectTypes:
          raise Exception('Effekt bereits in cEnergysystem eingefügt')

        # Wenn Standard-Effekt, und schon einer vorhanden:
        if (aNewEffect.isStandard) and (self.listOfEffectTypes.standardType() is not None):
            raise Exception('standardEffekt ist bereits belegt mit ' + self.standardEffect.label)
        # Wenn Objective-Effekt, und schon einer vorhanden:
        if (aNewEffect.isObjective) and (self.listOfEffectTypes.objectiveEffect() is not None):
            raise Exception('objectiveEffekt ist bereits belegt mit ' + self.objectiveEffect.label)
        
        # in liste ergänzen:
        self.listOfEffectTypes.append(aNewEffect)
      
      # an globalComp durchreichen: TODO: doppelte Haltung in es und globalComp ist so nicht schick.
      self.globalComp.listOfEffectTypes = self.listOfEffectTypes
            
    # Komponenten registrieren:
    def addComponents(self,*args):
      
      newListOfComps = list(args)
      aNewComp : cBaseComponent
      # für alle neuen Komponenten:
      for aNewComp in newListOfComps:                      
        # Check ob schon vorhanden:
        print('Register new Component ' + aNewComp.label)
        if aNewComp in self.listOfComponents:
          raise Exception('Komponente bereits in cEnergysystem eingefügt!')                
        
        # Check ob Name schon vergeben:
        if aNewComp.label in [aComp.label for aComp in self.listOfComponents]:
          raise Exception('Komponentenname \'' + aNewComp.label + '\' bereits vergeben!')  

        # # base in Komponente registrieren:
        # aNewComp.addEnergySystemIBelongTo(self)        
        
                
        # Komponente in Flow registrieren
        aNewComp.registerMeInFlows()
        
        # Flows in Bus registrieren:
        aNewComp.registerFlowsInBus()

      # register components:       
      self.listOfComponents.extend(newListOfComps)      
               
    # ME registrieren ganz allgemein:
    def addElements(self,*args):
      newList = list(args)
      for aNewME in newList:
        if    isinstance(aNewME, cBaseComponent):
          self.addComponents(aNewME)
        elif isinstance(aNewME, cEffectType):
          self.addEffects(aNewME)
        else: 
          # register ME:
          self.setOfOtherMEs.add(aNewME)
        
      
    
    def __plausibilityChecks(self):
      # Check circular loops in effects: (Effekte fügen sich gegenseitig Shares hinzu):
      def getErrorStr():
          return \
          '  ' + effect     .label + ' -> has share in: ' + shareEffect.label + '\n' \
          '  ' + shareEffect.label + ' -> has share in: ' + effect.label 
          
      for effect in self.listOfEffectTypes:
        # operation:
        for shareEffect in effect.specificShareToOtherEffects_operation.keys():
          # Effekt darf nicht selber als Share in seinen ShareEffekten auftauchen:
          assert (effect not in shareEffect.specificShareToOtherEffects_operation.keys()), 'Error: circular operation-shares \n' + getErrorStr()
        # invest:
        for shareEffect in effect.specificShareToOtherEffects_invest.keys():
          assert (effect not in shareEffect.specificShareToOtherEffects_invest.keys()), 'Error: circular invest-shares \n' + getErrorStr()

                       
    
    # Finalisieren aller ModelingElemente (dabei werden teilweise auch noch subMEs erzeugt!)
    def finalize(self):
      print('finalize all MEs...')
      self.__plausibilityChecks()
      # nur EINMAL ausführen: Finalisieren der MEs:
      if not self.__finalized:
        # finalize MEs for modeling:
        for aME in self.allMEsOfFirstLayer:
          aME.finalize() # inklusive subMEs!
        self.__finalized = True
      
    def doModelingOfElements(self):                        
      
      if not self.__finalized : 
        raise Exception('modeling not possible, because Energysystem is not finalized')
  
      # Bus-Liste erstellen: -> Wird die denn überhaupt benötigt?
  
      # TODO: Achtung timeIndexe kann auch nur ein Teilbereich von chosenEsTimeIndexe abdecken, z.B. wenn man für die anderen Zeiten anderweitig modellieren will
      # --> ist aber nicht sauber durchimplementiert in den ganzehn addSummand()-Befehlen!!
      timeIndexe = range(len(self.modBox.esTimeIndexe))  
      
      # globale Modellierung zuerst, damit andere darauf zugreifen können:       
      self.globalComp.declareVarsAndEqs(self.modBox) # globale Funktionen erstellen!
      self.globalComp.doModeling(self.modBox,timeIndexe) # globale Funktionen erstellen!
      
      # Komponenten-Modellierung (# inklusive subMEs!)
      for aComp in self.listOfComponents: 
        aComp : cBaseComponent
        log.debug('model ' + aComp.label + '...')
        # todo: ...OfFlows() ist nicht schön --> besser als rekursive Geschichte aller subModelingElements der Komponente umsetzen z.b. 
        aComp.declareVarsAndEqsOfFlows(self.modBox)
        aComp.declareVarsAndEqs       (self.modBox)            
                 
        aComp.doModelingOfFlows     (self.modBox, timeIndexe) 
        aComp.doModeling            (self.modBox, timeIndexe)      

        aComp.addShareToGlobalsOfFlows (self.globalComp, self.modBox)
        aComp.addShareToGlobals        (self.globalComp, self.modBox)
                
      # Bus-Modellierung (# inklusive subMEs!)
      aBus : cBus
      for aBus in self.setOfBuses:
        log.debug('model ' + aBus.label + '...')
        aBus.declareVarsAndEqs(  self.modBox)
        aBus.doModeling(       self.modBox, timeIndexe)      
        aBus.addShareToGlobals(self.globalComp, self.modBox)


      # weitere übergeordnete Modellierungen:
      for aME in self.setOfOtherMEs:
        aME.declareVarsAndEqs(self.modBox)
        aME.doModeling(self.modBox, timeIndexe)      
        aME.addShareToGlobals(self.globalComp, self.modBox)                  
        
      # transform to Math:
      self.modBox.transform2MathModel()
            
      return self.modBox

    
    # aktiviere in TS die gewählten Indexe: (wird auch direkt genutzt, nicht nur in activateModbox)
    def activateInTS(self, chosenTimeIndexe, dictOfTSAndExplicitData = None):
      aTS: cTS_vector      
      if dictOfTSAndExplicitData is None : 
        dictOfTSAndExplicitData = {}
      
      for aTS in self.allTSinMEs:         
        # Wenn explicitData vorhanden:
        if aTS in dictOfTSAndExplicitData.keys():
          explicitData = dictOfTSAndExplicitData[aTS]            
        else :
          explicitData = None          
        # Aktivieren:
        aTS.activate(chosenTimeIndexe, explicitData)        
          
    def activateModBox(self, aModBox) :   
      self.modBox = aModBox
      aModBox : cModelBoxOfES      
      aME     : cME
      
      # hier nochmal TS updaten (teilweise schon für Preprozesse gemacht):
      self.activateInTS(aModBox.esTimeIndexe, aModBox.TS_explicit)
      
      # Wenn noch nicht gebaut, dann einmalig ME.mod bauen:
      if aModBox.ME_mod == {} :
        log.debug('create mod-Vars for MEs of EnergySystem')  
        for aME in self.allMEsOfFirstLayer:
          # BEACHTE: erst nach finalize(), denn da werden noch subMEs erst erzeugt!      
          if not self.__finalized: 
            raise Exception('activateModBox(): --> Geht nicht, da System noch nicht finalized!')
          # mod bauen und in modBox registrieren. 
          aME.createNewModAndActivateModBox(self.modBox) # inkl. subMEs
      else:          
        # nur Aktivieren:
        for aME in allMEsOfFirstLayer:
          aME.activateModbox(aModBox) # inkl. subMEs

  
    # ! nur nach Solve aufrufen, nicht später nochmal nach activating modBox (da evtl stimmen Referenzen nicht mehr unbedingt!)
    def getResultsAfterSolve(self):
      results        = {} # Daten
      results_var    = {} # zugehörige Variable
      # für alle Komponenten:
      for aME in self.allMEsOfFirstLayerWithoutFlows:
        # results        füllen:
        (results[aME.label], results_var[aME.label]) = aME.getResults() # inklusive subMEs!

      #Zeitdaten ergänzen
      aTime = {}
      results['time'] = aTime 
      aTime['timeSeriesWithEnd'] = self.modBox.timeSeriesWithEnd
      aTime['timeSeries'] = self.modBox.timeSeries
      aTime['dtInHours'] = self.modBox.dtInHours
      aTime['dtInHours_tot'] = self.modBox.dtInHours_tot
        
      return results, results_var
           
    def printModel(self):   
      aBus  : cBus
      aComp : cBaseComponent
      print('')               
      print('##############################################################')
      print('########## Short String Description of Energysystem ##########')
      print('')                     
      
      print(yaml.dump(self.getSystemDescr()))
      # print('buses: ')
      # for aBus in self.setOfBuses:     
      #   aBus.print('  ')

      # print('components: ')
      # for aComp in self.listOfComponents:
      #   aComp.print('  ')
    
    def getSystemDescr(self, flowsWithBusInfo = False):
      modelDescription = {}
      
      # Anmerkung buses und comps als dict, weil Namen eindeutig!
      # Buses:
      modelDescription['buses'] = {}
      for aBus in self.setOfBuses:     
        aBus:cBus
        modelDescription['buses'].update(aBus.getDescrAsStr())
      # Comps:
      modelDescription['components'] = {}
      aComp : cBaseComponent
      for aComp in self.listOfComponents:        
        modelDescription['components'].update(aComp.getDescrAsStr())

      
      # Flows: 
      flowList = []
      modelDescription['flows'] = flowList
      aFlow : cFlow
      for aFlow in self.setOfFlows:
        flowList.append(aFlow.getStrDescr())
      
      return modelDescription
      
    def getEqsAsStr(self):


      aDict = {}

      # comps:
      aSubDict = {}      
      aDict['Components'] = aSubDict
      aComp : cME      
      for aComp in self.listOfComponents:
        aSubDict[aComp.label] = aComp.getEqsAsStr()

      # buses:    
      aSubDict = {}      
      aDict['buses'] = aSubDict
      for aBus in self.setOfBuses:
        aSubDict[aBus.label] = aBus.getEqsAsStr()
              
      # globals:
      aDict['globals'] = self.globalComp.getEqsAsStr()

      # flows:
      aSubDict = {}      
      aDict['flows'] = aSubDict      
      for aComp in self.listOfComponents:
        for aFlow in (aComp.inputs + aComp.outputs):        
          aSubDict[aFlow.label_full]=aFlow.getEqsAsStr()
      
      #others
      aSubDict = {}      
      aDict['others'] = aSubDict
      for aME in self.setOfOtherMEs:
        aSubDict[aME.label]=aME.getEqsAsStr()
      
      return aDict
        
              
    
    def printEquations(self):
      
      print('')               
      print('##############################################################')
      print('################# Equations of Energysystem ##################')      
      print('')               
      
      
      print(yaml.dump(self.getEqsAsStr(),
                      default_flow_style = False, 
                      allow_unicode = True))
              
      
    def getVarsAsStr(self, structured = True):
      aVar : cVariable
      
      # liste:
      if not structured:
        aList = []
        for aVar in self.modBox.variables:
          aList.append(aVar.getStrDescription())          
        return aList
      
      #struktur:
      else:        
        aDict = {}
        
        # comps (and belonging flows):
        subDict = {}
        aDict['Comps'] =subDict
        # comps:
        for aComp in self.listOfComponents:
          subDict[aComp.label] = aComp.getVarsAsStr()                              
          for aFlow in aComp.inputs + aComp.outputs:
            subDict[aComp.label] += aFlow.getVarsAsStr()
        
        # buses:
        subDict = {}
        aDict['buses'] =subDict
        for aME in self.setOfBuses:
          subDict[aME.label] = aME.getVarsAsStr()
                  
        # globals:
        aDict['globals'] = self.globalComp.getVarsAsStr()

        #others
        aSubDict = {}      
        aDict['others'] = aSubDict
        for aME in self.setOfOtherMEs:
          aSubDict[aME.label]=aME.getVarsAsStr()
          
        return aDict
      

    def printVariables(self):
      print('')               
      print('##############################################################')
      print('################# Variables of Energysystem ##################')            
      print('')
      print('############# a) as list : ################')
      print('')                     
      
      yaml.dump(self.getVarsAsStr(structured = False))
        
      print('')
      print('############# b) structured : ################')
      print('')

      yaml.dump(self.getVarsAsStr(structured = True))
  
    # Datenzeitreihe auf Basis gegebener esTimeIndexe aus globaler extrahieren:
    def getTimeDataOfTimeIndexe(self, chosenEsTimeIndexe):
      # if chosenEsTimeIndexe is None, dann alle : chosenEsTimeIndexe = range(len(self.timeSeries))
      # Zeitreihen:     
      timeSeries    = self.timeSeries       [chosenEsTimeIndexe]       
      # next timestamp as endtime:
      endTime       = self.timeSeriesWithEnd[chosenEsTimeIndexe[-1] + 1] 
      timeSeriesWithEnd = np.append(timeSeries,endTime)
          
      # Zeitdifferenz:
      #              zweites bis Letztes            - erstes bis Vorletztes
      dt           = timeSeriesWithEnd[1:] - timeSeriesWithEnd[0:-1]
      dtInHours    = dt/np.timedelta64(1, 'h')
      # dtInHours    = dt.total_seconds() / 3600    
      dtInHours_tot = sum(dtInHours) # Gesamtzeit
      return (timeSeries, timeSeriesWithEnd, dtInHours, dtInHours_tot)

# Standardoptimierung segmentiert/nicht segmentiert
class cCalculation :

  @property
  def infos(self):
    infos = {}
    
    calcInfos = self._infos
    infos['calculation'] = calcInfos    
    calcInfos['name'] = self.label
    calcInfos['no ChosenIndexe'] = len(self.chosenEsTimeIndexe)
    calcInfos['calcType'] = self.calcType    
    calcInfos['duration'] = self.durations
    infos['system_description'] = self.es.getSystemDescr()
    infos['modboxes'] = {}        
    infos['modboxes']['duration']= [aModbox.duration for aModbox in self.listOfModbox]               
    infos['modboxes']['info']    = [aModBox.infos for aModBox in self.listOfModbox]    
    
    return infos
  
  
  @property
  def results(self):
    # wenn noch nicht belegt, dann aus modbox holen
    if self.__results is None:
      self.__results = self.listOfModbox[0].results    
    
    # (bei segmented Calc ist das schon explizit belegt.)
    return self.__results

  @property
  def results_struct(self):    
    # Wenn noch nicht ermittelt:
    if (self.__results_struct is None) :      
        #Neurechnen (nur bei Segments)  
        if  (self.calcType == 'segmented'):
            self.__results_struct = helpers.createStructFromDictInDict(self.results)
        # nur eine Modbox vorhanden ('full','aggregated')
        elif len(self.listOfModbox) == 1:
            self.__results_struct = self.listOfModbox[0].results_struct
        else:
            raise Exception ('calcType ' + str(self.calcType) + ' not defined')
    return self.__results_struct
  
  es: cEnergySystem
  # chosenEsTimeIndexe: die Indexe des Energiesystems, die genutzt werden sollen. z.B. [0,1,4,6,8]
  def __init__(self, label, es, modType, pathForSaving = '/results', chosenEsTimeIndexe = None):
    self.label = label
    self.nameOfCalc = None # name for storing results
    self.es = es
    self.modType    = modType
    self.chosenEsTimeIndexe = chosenEsTimeIndexe
    self.pathForSaving = pathForSaving
    self.calcType = None # 'full', 'segmented', 'aggregated'
    self._infos = {}
    
    self.listOfModbox = [] # liste der ModelBoxes (nur bei Segmentweise mehrere!)
    self.durations = {} # Dauer der einzelnen Dinge
    self.durations['modeling'] = 0
    self.durations['solving'] = 0
    self.TSlistForAggregation = None # list of timeseries for aggregation
    # assert from_index < to_index
    # assert from_index >= 0
    # assert to_index <= len(self.es.timeSeries)-1    
    
    # Wenn chosenEsTimeIndexe = None, dann alle nehmen
    if self.chosenEsTimeIndexe is None : self.chosenEsTimeIndexe = range(len(es.timeSeries))
    (self.timeSeries, self.timeSeriesWithEnd, self.dtInHours, self.dtInHours_tot) = es.getTimeDataOfTimeIndexe(self.chosenEsTimeIndexe)        
    helpers.checkTimeSeries('chosenEsTimeIndexe', self.timeSeries)
    
    self.nrOfTimeSteps = len(self.timeSeries)    

    self.__results        = None
    self.__results_struct = None # hier kommen die verschmolzenen Ergebnisse der Segmente rein!
    self.segmentModBoxList = [] # modBox list
  
  # Variante1:
  def doModelingAsOneSegment(self):
    self.checkIfAlreadyModeled()
    self.calcType = 'full'
    # System finalisieren:
    self.es.finalize()
    
    t_start = time.time()
    # Modellierungsbox / TimePeriod-Box bauen:
    aModBox  = cModelBoxOfES(self.label, self.modType, self.es, self.chosenEsTimeIndexe) # alle Indexe nehmen!               
    # modBox aktivieren:
    self.es.activateModBox(aModBox)
    # modellieren:
    self.es.doModelingOfElements()
    
    self.durations['modeling'] = round(time.time()-t_start,2)
    self.listOfModbox.append(aModBox)
    return aModBox
    
  
  # Variante2:
  def doSegmentedModelingAndSolving(self, solverProps, segmentLen, nrOfUsedSteps, namePrefix = '', nameSuffix ='', aPath = 'results/'):
    self.checkIfAlreadyModeled()
    self._infos['segmentedProps'] = {'segmentLen':segmentLen, 'nrUsedSteps':  nrOfUsedSteps}

    
    self.calcType = 'segmented'
    print('##############################################################')
    print('#################### segmented Solving #######################')
    
    
    t_start = time.time()
    # system finalisieren:
    self.es.finalize()
    
    # nrOfTimeSteps = self.to_index - self.from_index +1    
    
    assert nrOfUsedSteps <= segmentLen
    assert segmentLen    <= self.nrOfTimeSteps, 'segmentLen must be smaller than (or equal to) the whole nr of timesteps'

    # timeSeriesOfSim = self.es.timeSeries[from_index:to_index+1]
       
    # Anzahl = Letzte Simulation bis zum Ende plus die davor mit Überlappung:
    nrOfSimSegments = math.ceil((self.nrOfTimeSteps - segmentLen) / nrOfUsedSteps) + 1 
    self._infos['segmentedProps']['nrOfSegments'] = nrOfSimSegments
    print('indexe        : ' + str(self.chosenEsTimeIndexe[0]) + '...' + str(self.chosenEsTimeIndexe[-1]))
    print('segmentLen    : ' + str(segmentLen))
    print('usedSteps     : ' + str(nrOfUsedSteps))
    print('-> nr of Sims : ' + str(nrOfSimSegments))
    print('')
    
    for i in range(nrOfSimSegments):      
      startIndex_calc   = i * nrOfUsedSteps
      endIndex_calc     = min(startIndex_calc + segmentLen - 1, len(self.chosenEsTimeIndexe) - 1)
            
      startIndex_global = self.chosenEsTimeIndexe[startIndex_calc]
      endIndex_global   = self.chosenEsTimeIndexe[endIndex_calc] # inklusiv
      indexe_global     = self.chosenEsTimeIndexe[startIndex_calc:endIndex_calc + 1] # inklusive endIndex



      # new realNrOfUsedSteps:
      # if last Segment:
      if i == nrOfSimSegments-1:
        realNrOfUsedSteps     = endIndex_calc - startIndex_calc + 1
      else:
        realNrOfUsedSteps     = nrOfUsedSteps
      
      print(str(i) +  '. Segment ' + ' (es-indexe ' + str(startIndex_global) + '...' + str(endIndex_global) + ') :')                 
   
      # Modellierungsbox / TimePeriod-Box bauen:
      label = self.label + '_seg' + str(i)
      segmentModBox  = cModelBoxOfES(label, self.modType, self.es, indexe_global) # alle Indexe nehmen!       
      segmentModBox.realNrOfUsedSteps = realNrOfUsedSteps

      # Startwerte übergeben von Vorgänger-Modbox:
      if i > 0 : 
        segmentModBoxBefore = self.segmentModBoxList[i-1]        
        segmentModBox.beforeValueSet = cBeforeValueSet(segmentModBoxBefore, segmentModBoxBefore.realNrOfUsedSteps -1)
        print('### beforeValueSet: ###')        
        segmentModBox.beforeValueSet.print()
        print('#######################')
        # transferStartValues(segment, segmentBefore)
      
      # modBox in Energiesystem aktivieren:
      self.es.activateModBox(segmentModBox)

      # modellieren:
      t_start_modeling = time.time()
      self.es.doModelingOfElements()
      self.durations['modeling'] += round(time.time()-t_start_modeling,2)
      # modbox in Liste hinzufügen:      
      self.segmentModBoxList.append(segmentModBox)
      # übergeordnete Modbox-Liste:
      self.listOfModbox.append(segmentModBox)
      
      # Lösen:
      t_start_solving = time.time()
      segmentModBox.solve(**solverProps)# keine SolverOutput-Anzeige, da sonst zu viel
      self.durations['solving'] += round(time.time()-t_start_solving,2)
      ## results adding:      
      self.__addSegmentResults(segmentModBox, startIndex_calc, realNrOfUsedSteps)
      
    
    self.durations['model, solve and segmentStuff'] = round(time.time()-t_start,2)
    
    self._saveSolveInfos(namePrefix=namePrefix, nameSuffix=nameSuffix, aPath=aPath)
  
  def doAggregatedModeling(self, periodLengthInHours, noTypicalPeriods, 
                           useExtremePeriods, fixStorageFlows,
                           fixBinaryVarsOnly, percentageOfPeriodFreedom = 0,
                           costsOfPeriodFreedom = 0,
                           addPeakMax = [],
                           addPeakMin = []):
    '''
      

      Parameters
      ----------
      periodLengthInHours : TYPE
          DESCRIPTION.
      noTypicalPeriods : TYPE
          DESCRIPTION.
      useExtremePeriods : TYPE
          DESCRIPTION.
      fixStorageFlows : TYPE
          DESCRIPTION.
      fixBinaryVarsOnly : TYPE
          DESCRIPTION.
      percentageOfPeriodFreedom : TYPE, optional
          DESCRIPTION. The default is 0.
      costsOfPeriodFreedom : TYPE, optional
          DESCRIPTION. The default is 0.

      addPeakMax : list of cTSraw ...
          
      addPeakMin : list of cTSraw
          

      Raises
      ------
      Exception
          DESCRIPTION.

      Returns
      -------
      aModBox : TYPE
          DESCRIPTION.

      '''
    self.checkIfAlreadyModeled()
    

    
    self._infos['aggregatedProps']={'periodLengthInHours': periodLengthInHours,
                                     'noTypicalPeriods'   : noTypicalPeriods,
                                     'useExtremePeriods'  : useExtremePeriods,
                                     'fixStorageFlows'    : fixStorageFlows,
                                     'fixBinaryVarsOnly'  : fixBinaryVarsOnly,
                                     'percentageOfPeriodFreedom' : percentageOfPeriodFreedom,
                                     'costsOfPeriodFreedom' : costsOfPeriodFreedom}
    
    self.calcType = 'aggregated'
    t_start_agg = time.time()
    # chosen Indexe aktivieren in TS: (sonst geht Aggregation nicht richtig)
    self.es.activateInTS(self.chosenEsTimeIndexe)
        
    # Zeitdaten generieren:
    (chosenTimeSeries, chosenTimeSeriesWithEnd, dtInHours, dtInHours_tot) = self.es.getTimeDataOfTimeIndexe(self.chosenEsTimeIndexe)    
    
    # check equidistant timesteps:
    if max(dtInHours)-min(dtInHours) != 0:
        raise Exception('!!! Achtung Aggregation geht nicht, da unterschiedliche delta_t von ' + str(min(dtInHours)) + ' bis ' + str(max(dtInHours)) + ' h')
    
    
    print('#########################')
    print('## TS for aggregation ###')

    ## Daten für Aggregation vorbereiten:    
    # TSlist and TScollection ohne Skalare:
    self.TSlistForAggregation = [item for item in self.es.allTSinMEs if item.isArray]
    self.TScollectionForAgg = cTS_collection(self.TSlistForAggregation, 
                                             addPeakMax_TSraw = addPeakMax, 
                                             addPeakMin_TSraw = addPeakMin,
                                             )
    self.TScollectionForAgg.print()   

    import pandas as pd    
    # seriesDict = {i : self.TSlistForAggregation[i].d_i_raw_vec for i in range(len(self.TSlistForAggregation))}    
    dfSeries = pd.DataFrame(self.TScollectionForAgg.seriesDict, index = chosenTimeSeries)# eigentlich wäre TS als column schön, aber TSAM will die ordnen können.       

    # Check, if timesteps fit in Period:
    stepsPerPeriod = periodLengthInHours / self.dtInHours[0] 
    if not stepsPerPeriod.is_integer():
      raise Exception('Fehler! Gewählte Periodenlänge passt nicht zur Zeitschrittweite')
    
    import flixAggregation as flixAgg
    # Aggregation:
    
    dataAgg = flixAgg.flixAggregation('aggregation',
                                      timeseries = dfSeries,
                                      hoursPerTimeStep = self.dtInHours[0],
                                      hoursPerPeriod = periodLengthInHours,
                                      hasTSA = False,
                                      noTypicalPeriods = noTypicalPeriods,
                                      useExtremePeriods = useExtremePeriods,
                                      weightDict = self.TScollectionForAgg.weightDict,
                                      addPeakMax = self.TScollectionForAgg.addPeak_Max_numbers,
                                      addPeakMin = self.TScollectionForAgg.addPeak_Min_numbers)
    
    
    
    dataAgg.cluster()   
    # dataAgg.aggregation.clusterPeriodIdx
    # dataAgg.aggregation.clusterOrder
    # dataAgg.aggregation.clusterPeriodNoOccur
    # dataAgg.aggregation.predictOriginalData()    
    # self.periodsOrder = aggregation.clusterOrder
    # self.periodOccurances = aggregation.clusterPeriodNoOccur
    
    aggregationModel = flixAgg.cAggregationModeling('aggregation',self.es,
                                                indexVectorsOfClusters = dataAgg.indexVectorsOfClusters,
                                                fixBinaryVarsOnly = fixBinaryVarsOnly, 
                                                fixStorageFlows   = fixStorageFlows,
                                                listOfMEsToClusterize = None,
                                                percentageOfPeriodFreedom = percentageOfPeriodFreedom,
                                                costsOfPeriodFreedom = costsOfPeriodFreedom)
    
    # todo: das muss irgendwie nur temporär sein und nicht dauerhaft in es hängen!
    self.es.addElements(aggregationModel)
    
    if fixBinaryVarsOnly:
      TS_explicit = None
    else:
      # neue (Explizit)-Werte für TS sammeln::        
      TS_explicit = {}
      for i in range(len(self.TSlistForAggregation)):
        TS = self.TSlistForAggregation[i]
        # todo: agg-Wert für TS:
        TS_explicit[TS] = dataAgg.totalTimeseries[i].values # nur data-array ohne Zeit   
    
    
    import matplotlib.pyplot as plt        
    plt.title('aggregated series')
    plt.plot(dfSeries.values)
    for i in range(len(self.TSlistForAggregation)):
        plt.plot(dataAgg.totalTimeseries[i].values,'--', label = str(i))
    if len(self.TSlistForAggregation) < 10: # wenn nicht zu viele
        plt.legend()
    plt.show()
    
    ##########################
    # System finalisieren:
    self.es.finalize()
    
    self.durations['aggregation']= round(time.time()- t_start_agg ,2)



    t_m_start = time.time()    
    # Modellierungsbox / TimePeriod-Box bauen: ! inklusive TS_explicit!!!
    aModBox  = cModelBoxOfES(self.label, self.modType, self.es, self.chosenEsTimeIndexe, TS_explicit) # alle Indexe nehmen!               
    self.listOfModbox.append(aModBox)
    # modBox aktivieren:
    self.es.activateModBox(aModBox)
    # modellieren:
    self.es.doModelingOfElements()
    
    
    self.durations['modeling'] = round(time.time()-t_m_start,2)
    return aModBox    
  
  def solve(self, solverProps, namePrefix = '', nameSuffix ='', aPath = 'results/', saveResults = True):
      
      if self.calcType not in ['full','aggregated']:
        raise Exeption('calcType ' + self.calcType + ' needs no solve()-Command (only for ' + str())
      aModbox = self.listOfModbox[0]
      aModbox.solve(**solverProps)
      
      if saveResults:
        self._saveSolveInfos(namePrefix=namePrefix, nameSuffix=nameSuffix, aPath=aPath)
      
  
  def checkIfAlreadyModeled(self):
    
    if self.calcType is not None:
      raise Exception('An other modeling-Method (calctype: ' + self.calcType + ') was already executed with this cCalculation-Object. \n Always create a new instance of cCalculation for new modeling/solving-command!')

  def _saveSolveInfos(self, namePrefix = '', nameSuffix ='', aPath = 'results/'):
      import datetime
      import yaml      
      import pathlib
      
      # wenn "/" am Anfang, dann löschen (sonst kommt pathlib durcheinander):
      aPath = aPath[0].replace("/","") + aPath[1:]
      # absoluter Pfad:
      aPath = pathlib.Path.cwd() / aPath
      # Pfad anlegen, fall noch nicht vorhanden:
      aPath.mkdir(parents=True, exist_ok=True)       
           
      timestamp = datetime.datetime.now()
      timestring = timestamp.strftime('%Y-%m-%d')             
      self.nameOfCalc = namePrefix.replace(" ","") + timestring + '_' + self.label.replace(" ", "") + nameSuffix.replace(" ","")
      
      filename_Data = self.nameOfCalc + '_data.pickle'
      filename_Info = self.nameOfCalc + '_solvingInfos.yaml'
            
      self.path_Data = aPath / filename_Data
      self.path_Info = aPath / filename_Info
      
      #Daten:
      # with open(yamlPath_Data, 'w') as f:
      #   yaml.dump(self.results, f, sort_keys = False)
      import pickle
      with open(self.path_Data,'wb') as f:
        pickle.dump(self.results, f, protocol=pickle.HIGHEST_PROTOCOL)
      
      #Infos:'
      with open(self.path_Info, 'w',encoding='utf-8') as f:
          yaml.dump(self.infos, f, 
                    width=1000, # Verhinderung Zeilenumbruch für lange equations
                    allow_unicode = True,
                    sort_keys = False)
      
      aStr = '# saved calculation ' + self.nameOfCalc + ' #'
      print('#'*len(aStr))      
      print(aStr)
      print('#'*len(aStr))
           
    
  
  def __addSegmentResults(self, segment, startIndex_calc, realNrOfUsedSteps):    
    # rekursiv aufzurufendes Ergänzen der Dict-Einträge um segment-Werte:

    if (self.__results is None) :
      self.__results = {} # leeres Dict als Ausgangszustand      
    
    def appendNewResultsToDictValues(result, resultToAppend, resultToAppendVar):
      if result == {} : 
        firstFill = True # jeweils neuer Dict muss erzeugt werden für globales Dict
      else:
        firstFill = False
            
      for key, val in resultToAppend.items():
        # print(key)
          
        # Wenn val ein Wert ist:
        if isinstance(val, np.ndarray) or isinstance(val, np.float64) or np.isscalar(val): 
          
          
          # Beachte Länge (withEnd z.B. bei Speicherfüllstand)
          if key in ['timeSeries','dtInHours','dtInHours_tot']:
            withEnd = False
          elif key in ['timeSeriesWithEnd']:
            withEnd = True
          else:
            # Beachte Speicherladezustand und ähnliche Variablen:                      
            aReferedVariable = resultToAppendVar[key]
            withEnd = isinstance(aReferedVariable, cVariableB) and aReferedVariable.beforeValueIsStartValue

          # nested:
          def getValueToAppend(val, withEnd):
            # wenn skalar, dann Vektor draus machen: 
            # todo: --> nicht so schön!
            if np.isscalar(val):
              val = np.array([val])
              
            if withEnd:
              if firstFill:
                aValue = val[0:realNrOfUsedSteps +1] # (inklusive WithEnd!)
              else:
                # erstes Element weglassen, weil das schon vom Vorgängersegment da ist:
                aValue = val[1:realNrOfUsedSteps +1] # (inklusive WithEnd!)
            else:
              aValue = val[0:realNrOfUsedSteps] # (nur die genutzten Steps!)            
            return aValue            
                     
          aValue = getValueToAppend(val, withEnd)
          
          if firstFill: 
            result[key] = aValue
          else:# erstmaliges Füllen. Array anlegen.
            result[key] = np.append(result[key], aValue) # Anhängen (nur die genutzten Steps!)
          
        else :
          if firstFill: result[key] = {}
          
          if (resultToAppendVar is not None) and key in resultToAppendVar.keys():
            resultToAppend_sub = resultToAppendVar[key]
          else: # z.B. bei time (da keine Variablen)
            resultToAppend_sub = None
          appendNewResultsToDictValues(result[key], resultToAppend[key], resultToAppend_sub) # hier rekursiv!
    
    # rekursiv:
    appendNewResultsToDictValues(self.__results, segment.results, segment.results_var)
    
    # results füllen:
    # ....
  
# mod: -> wird eingefügt in jedes ModelingElement und besitzt die modBox-spezifischen Dinge
class cMEModel:
  def __init__(self, ME):
    self.ME        = ME
    self.variables = []
    self.eqs       = []
    self.ineqs     = []
    self.objective = None
  def getVar(self, label):
    return next((x for x in  self.variables         if x.label == label), None)
  
  def getEq (self, label):
    return next((x for x in (self.eqs + self.ineqs) if x.label == label), None)
  
  # Eqs, Ineqs und Objective als Str-Description:
  def getEqsAsStr(self):
    # Wenn Glg vorhanden:    
    eq: cEquation
    aList = []
    if (len(self.eqs) + len(self.ineqs)) > 0:
      for eq in (self.eqs + self.ineqs) :
        aList.append(eq.getStrDescription())          
    if not(self.objective is None):
      aList.append(self.objective.getStrDescription())
    return aList
  
  def getVarsAsStr(self):
    aList = []
    for aVar in self.variables:
      aList.append(aVar.getStrDescription())
    return aList
  
  def printEqs(self, shiftChars):    
    yaml.dump(self.getEqsAsStr(),              
              allow_unicode=True)
          
  def printVars(self, shiftChars):
    yaml.dump(self.getVarsAsStr(),              
              allow_unicode=True)
      

    
# Element mit Variablen und Gleichungen (ME = Modeling Element):
#  -> besitzt Methoden, die jede Kindklasse ergänzend füllt:
#    1. cME.finalize()          --> Finalisieren der Modell-Beschreibung (z.B. notwendig, wenn Bezug zu Elementen, die bei __init__ noch gar nicht bekannt sind)
#    2. cME.declareVarsAndEqs() --> Variablen und Eqs definieren.
#    3. cME.doModeling()        --> Modellierung
#    4. cME.addShareToGlobals() --> Beitrag zu Gesamt-Kosten
  
class cME(cArgsClass): 
  modBox : cModelBoxOfES
  
  new_init_args = [cArg('label','param','str', 'Bezeichnung')]
  not_used_args = [] 

  @property
  def label_full(self): #standard-Funktion, wird von Kindern teilweise überschrieben
    return self.label   # eigtl später mal rekursiv: return self.owner.label_full + self.label
  
  @property # subElements of all layers
  def subElements_all(self):    
    allSubElements = [] # wichtig, dass neues Listenobjekt!
    allSubElements += self.subElements 
    for subElem in self.subElements:
      # all subElements of subElement hinzufügen:
      allSubElements += subElem.subElements_all
    return allSubElements
  
  # TODO: besser occupied_args
  def __init__(self,label, **kwargs):
    self.label     = label
    self.TS_list   = []#   = list with ALL timeseries-Values (--> need all classes with .trimTimeSeries()-Method, e.g. cTS_vector)
    
    self.subElements = [] # zugehörige Sub-ModelingElements
    self.modBox      = None # hier kommt die aktive ModBox rein
    self.mod         = None # hier kommen alle Glg und Vars rein
    super().__init__(**kwargs)
        
    
  # activate inkl. subMEs:
  def activateModbox(self, modBox):
    
    for aME in self.subElements:
      aME.activateModbox(modBox) # inkl. subMEs
    self._activateModBox_ForMeOnly(modBox)

  # activate ohne SubMEs!
  def _activateModBox_ForMeOnly(self, modBox):
    self.modBox     = modBox
    self.mod        = modBox.getModOfME(self)

  # 1.
  def finalize(self):
    #print('finalize ' + self.label)
    # gleiches für alle sub MEs:
    for aME in self.subElements:
      aME.finalize()

  # 2.
  def createNewModAndActivateModBox(self, modBox):    
    # print('new mod for ' + self.label)
    # subElemente ebenso:
    aME:cME
    for aME in self.subElements:
      aME.createNewModAndActivateModBox(modBox) # rekursiv!
    
    # create mod:
    aMod = cMEModel(self)
    # register mod:
    modBox.registerMEandMod(self, aMod)         
    
    self._activateModBox_ForMeOnly(modBox)# subMEs werden bereits aktiviert über aME.createNewMod...()
    
  # 3.
  def declareVarsAndEqs(self, modBox):
    #   #   # Features preparing:
    #   # for aFeature in self.features:
    #   #   aFeature.declareVarsAndEqs(modBox)  
    pass


  # def doModeling(self,modBox,timeIndexe):
  #   # for aFeature in self.features:
    #   aFeature.doModeling(modBox, timeIndexe)        
    
    
  # Ergebnisse als dict ausgeben:    
  def getResults(self):
    aData = {}
    aVars = {}
    # 1. Unterelemente füllen (rekursiv!):
    for aME in self.subElements:
      (aData[aME.label], aVars[aME.label]) = aME.getResults() # rekursiv

    # 2. Variablenwerte ablegen:
    aVar : cVariable
    for aVar in self.mod.variables :
      # print(aVar.label)
      aData[aVar.label] = aVar.getResult()
      aVars[aVar.label] = aVar # link zur Variable
      if aVar.isBinary and aVar.len >1:
        # Bei binären Variablen zusätzlichen Vektor erstellen,z.B. a  = [0, 1, 0, 0, 1] 
        #                                                       -> a_ = [nan, 1, nan, nan, 1]        
        aData[aVar.label + '_'] = helpers.zerosToNans(aVar.getResult())   
        aVars[aVar.label + '_'] = aVar # link zur Variable

    return aData, aVars

  # so kurze Schreibweise wie möglich, daher:
  def var(self, label):
    self.mod.getVar(label)
  def eq(self, label):
    self.mod.getEq (label)


  def getEqsAsStr(self):
  
    ## subelemente durchsuchen:
    subs = {}
    for aSubElement in self.subElements:                
      subs[aSubElement.label] = aSubElement.getEqsAsStr() # rekursiv
    ## me:
    
    # wenn sub-eqs, dann dict:
    if not(subs == {}): 
      eqsAsStr = {}
      eqsAsStr['_self'] = self.mod.getEqsAsStr() # zuerst eigene ...
      eqsAsStr.update(subs) # ... dann sub-Eqs
    # sonst liste:
    else:
      eqsAsStr = self.mod.getEqsAsStr()
  
    return eqsAsStr
  

  def getVarsAsStr(self):
    aList = []
    
    aList += self.mod.getVarsAsStr()    
    for aSubElement in self.subElements:      
      aList += aSubElement.getVarsAsStr() # rekursiv   
    
    return aList
  
  def printEqs(self, shiftChars):
    print(shiftChars + '·' + self.label + ':')
    print(yaml.dump(self.getEqsAsStr(),
              allow_unicode = True))
          
  def printVars(self, shiftChars):
    print(shiftChars + self.label + ':')
    print(yaml.dump(self.getVarsAsStr(),
              allow_unicode = True))
  def getEqsVarsOverview(self):
    aDict = {}
    aDict['no eqs']          = len(self.mod.eqs)
    aDict['no eqs single']   = sum([eq.nrOfSingleEquations for eq in self.mod.eqs])
    aDict['no inEqs']        = len(self.mod.ineqs)
    aDict['no inEqs single'] = sum([ineq.nrOfSingleEquations for ineq in self.mod.ineqs])
    aDict['no vars']         = len(self.mod.variables)
    aDict['no vars single']  =  sum([var.len for var in self.mod.variables])  
    return aDict

# Definition Effekt (Kosten, CO2,...)     
class cEffectType(cME) : 
  
   
  new_init_args = [cArg('label'                                ,'param', 'str'     , 'Bezeichnung'),
                   cArg('unit'                                 , '?'   , 'str'     , 'Einheit des Effekts, z.B. €, kWh,...'),
                   cArg('description'                          , '?'   , 'str'     , 'Langbezeichnung'),
                   cArg('isStandard'                           , '?'   , 'boolean' , 'true, wenn Standard-Effekt (wenn direkte Eingabe bei Kostenpositionen ohne dict) , sonst false'),
                   cArg('isObjective'                          , '?'   , 'boolean' , 'true, wenn Zielfunktion'),
                   cArg('specificShareToOtherEffects_operation', '?' , '{effectType: TS, ...}'    , 'Beiträge zu anderen Effekten (nur operation), z.B. 180 €/t_CO2, Angabe als {costs: 180}'),
                   cArg('specificShareToOtherEffects_invest'   , '?' , '{effectType: skalar, ...}', 'Beiträge zu anderen Effekten (nur invest), z.B. 180 €/t_CO2, Angabe als {costs: 180}'),
                   cArg('min_operationSum'                     , '?' , 'skalar' , 'minimale Summe (nur operation) des Effekts'),
                   cArg('max_operationSum'                     , '?' , 'skalar' , 'maximale Summe (nur operation) des Effekts'),
                   cArg('min_investSum'                        , '?' , 'skalar' , 'minimale Summe (nur invest) des Effekts'),
                   cArg('max_investSum'                        , '?' , 'skalar' , 'maximale Summe (nur invest) des Effekts'),
                   cArg('min_Sum'                              , '?' , 'skalar' , 'minimale Summe des Effekts'),                   
                   cArg('max_Sum'                              , '?' , 'skalar' , 'maximale Summe des Effekts'),
                   
                   ] 
                   # todo: effects, die noch nicht instanziert sind, können hier auch keine shares zugeordnet werden!
  not_used_args = ['label']

  # isStandard -> Standard-Effekt (bei Eingabe eines skalars oder TS (statt dict) wird dieser automatisch angewendet)
  def __init__(self, label, unit, description, isStandard = False, isObjective = False, specificShareToOtherEffects_operation = {}, specificShareToOtherEffects_invest = {}, 
               min_operationSum = None, max_operationSum = None, 
               min_investSum = None, max_investSum = None,
               min_Sum = None, max_Sum = None,
               **kwargs) :    
    super().__init__(label, **kwargs)    
    self.label       = label
    self.unit        = unit
    # self.allFlow     = allFlow
    self.description = description
    self.isStandard  = isStandard
    self.isObjective = isObjective
    self.specificShareToOtherEffects_operation = specificShareToOtherEffects_operation
    self.specificShareToOtherEffects_invest    = specificShareToOtherEffects_invest
    
    self.min_operationSum = min_operationSum
    self.max_operationSum = max_operationSum
    self.min_investSum    = min_investSum
    self.max_investSum    = max_investSum
    self.min_Sum          = min_Sum
    self.max_Sum          = max_Sum
    
    
    #  operation-Effect-shares umwandeln in TS (invest bleibt skalar ):
    for effectType, share in self.specificShareToOtherEffects_operation.items() :
      # value überschreiben durch TS:
      TS_name = 'specificShareToOtherEffect' +'_'+ effectType.label
      self.specificShareToOtherEffects_operation[effectType] = cTS_vector(TS_name, specificShareToOtherEffects_operation[effectType], self)

    # ShareSums:
    self.operation = cFeature_ShareSum(label = 'operation', owner = self, sharesAreTS = True,  minOfSum = self.min_operationSum, maxOfSum = self.max_operationSum)
    self.invest    = cFeature_ShareSum(label = 'invest'   , owner = self, sharesAreTS = False, minOfSum = self.min_investSum   , maxOfSum = self.max_investSum)
    self.all       = cFeature_ShareSum(label = 'all'      , owner = self, sharesAreTS = False, minOfSum = self.min_Sum         , maxOfSum = self.max_Sum)
        
  
  def declareVarsAndEqs(self,modBox):
    super().declareVarsAndEqs(modBox)
    self.operation.declareVarsAndEqs(modBox)
    self.invest   .declareVarsAndEqs(modBox)
    self.all      .declareVarsAndEqs(modBox)    
  
  def doModeling(self, modBox, timeIndexe):
    print('modeling ' + self.label)
    super().declareVarsAndEqs(modBox)
    self.operation.doModeling(modBox, timeIndexe)
    self.invest   .doModeling(modBox, timeIndexe)
    self.all      .doModeling(modBox, timeIndexe)
    # Gleichung für Summe Operation und Invest:
    # eq: shareSum = effect.operation_sum + effect.operation_invest
    self.all.addVariableShare(self.operation.mod.var_sum, 1, 1)
    self.all.addVariableShare(self.invest   .mod.var_sum, 1, 1)
    
# ModelingElement (ME) Klasse zum Summieren einzelner Shares
# geht für skalar und TS
# z.B. invest.costs 


# Liste mit zusätzlicher Methode für Rückgabe Standard-Element:
class cEffectTypeList(list):    

  # return standard effectType:
  def standardType(self):
    aEffect : cEffectType
    aStandardEffect = None
    # TODO: eleganter nach attribut suchen:
    for aEffectType in self:
      if aEffectType.isStandard : aStandardEffect = aEffectType
    return aStandardEffect
  
  def objectiveEffect(self):
    aEffect : cEffectType
    aObjectiveEffect = None
    # TODO: eleganter nach attribut suchen:
    for aEffectType in self:
      if aEffectType.isObjective : aObjectiveEffect = aEffectType
    return aObjectiveEffect
    
    
from flixFeatures import * 
# Beliebige Komponente (:= Element mit Ein- und Ausgängen)
class cBaseComponent(cME):
    modBox : cModelBoxOfES
  
    new_init_args = [cArg('on_valuesBeforeBegin', 'initial', 'list', 'Ein(1)/Aus(0)-Wert vor Zeitreihe'),
                     cArg('switchOnCosts'       , 'costs'  , 'TS'  , 'Einschaltkosten z.B. in €'),
                     cArg('switchOn_maxNr'      , 'param', 'skalar', 'max. zulässige Anzahl Starts'),
                     cArg('onHours_min'         , 'param', 'skalar', 'min. Betriebsstunden'),
                     cArg('onHours_max'         , 'param', 'skalar', 'max. Betriebsstunden'),
                     cArg('costsPerRunningHour' , 'costs'  , 'TS'  , 'Betriebskosten z.B. in €/h')]

    not_used_args = []
        
    
    def __init__(self, label, on_valuesBeforeBegin = [0,0], switchOnCosts = None, switchOn_maxNr = None, onHours_min = None, onHours_max = None, costsPerRunningHour = None, **kwargs) :            
      label    =  helpers.checkForAttributeNameConformity(label)          # todo: indexierbar / eindeutig machen!      
      super().__init__(label, **kwargs)
      self.on_valuesBeforeBegin = on_valuesBeforeBegin  
      self.switchOnCosts        = transFormEffectValuesToTSDict('switchOnCosts'      , switchOnCosts       , self)
      self.switchOn_maxNr       = switchOn_maxNr
      self.onHours_min          = onHours_min
      self.onHours_max          = onHours_max
      self.costsPerRunningHour  = transFormEffectValuesToTSDict('costsPerRunningHOur', costsPerRunningHour , self)     
      
      ## TODO: theoretisch müsste man auch zusätzlich checken, ob ein flow Werte beforeBegin hat!
      # % On Werte vorher durch Flow-values bestimmen:    
      # self.on_valuesBefore = 1 * (self.featureOwner.valuesBeforeBegin >= np.maximum(modBox.epsilon,self.flowMin)) für alle Flows!
           
      self.inputs   = [] # list of flows
      self.outputs  = [] # list of flows
      self.isStorage = False

    
      # self.base = None # Energysystem I Belong to     
            
      self.subComps = [] # list of subComponents # für mögliche Baumstruktur!     
        
    # # TODO: ist das noch notwendig?:
    # def addEnergySystemIBelongTo(self,base): 
    #   if self.base is not None :
    #     raise Exception('Komponente ' + self.label + ' wird bereits in anderem Energiesystem verwendet!')
    #   self.base = base
    #   # falls subComps existieren:
    #   for aComp in self.subComps :
    #     aComp.addEnergySystemIBelongTo(base)
    
    def registerMeInFlows(self):
      for aFlow in self.inputs + self.outputs:
        aFlow.comp = self  

    def registerFlowsInBus(self): # todo: macht aber bei Kindklasse cBus keinen Sinn!
      #          
      # ############## register in Bus: ##############
      #          
      # input ist output von Bus:
      for aFlow in self.inputs:
        aFlow.bus.registerOutputFlow(aFlow) # ist das schön programmiert?
      # output ist input von Bus:
      for aFlow in self.outputs:
        aFlow.bus.registerInputFlow(aFlow)


    def declareVarsAndEqsOfFlows(self,modBox): # todo: macht aber bei Kindklasse cBus keinen Sinn!
      # Flows modellieren:        
      for aFlow in self.inputs + self.outputs:
        aFlow.declareVarsAndEqs(modBox)
      
    def doModelingOfFlows(self,modBox,timeIndexe): # todo: macht aber bei Kindklasse cBus keinen Sinn!
      # Flows modellieren:        
      for aFlow in self.inputs + self.outputs:
        aFlow.doModeling(modBox,timeIndexe)
        
      
    def getResults(self):     
      # Variablen der Komponente:
      (results, results_var) = super().getResults()
      
      # Variablen der In-/Out-Puts ergänzen:
      for aFlow in self.inputs + self.outputs :        
        # z.B. results['Q_th'] = {'val':..., 'on': ..., ...}
        if isinstance(self, cBus):
          flowLabel = aFlow.label_full # Kessel_Q_th
        else:
          flowLabel = aFlow.label      # Q_th
        (results[flowLabel], results_var[flowLabel]) = aFlow.getResults()
      return results, results_var

      
    def finalize(self):      
      super().finalize()
      
      # feature for: On and SwitchOn Vars      
      # (kann erst hier gebaut werden wg. weil input/output Flows erst hier vorhanden)
      flowsDefiningOn = self.inputs + self.outputs # Sobald ein input oder  output > 0 ist, dann soll On =1 sein!
      self.featureOn = cFeatureOn(self, flowsDefiningOn, self.on_valuesBeforeBegin, self.switchOnCosts, self.costsPerRunningHour, onHours_min = self.onHours_min, onHours_max = self.onHours_max, switchOn_maxNr = self.switchOn_maxNr)            
    
    def declareVarsAndEqs(self,modBox):
      super().declareVarsAndEqs(modBox)   

      self.featureOn.declareVarsAndEqs(modBox)

      # Binärvariablen holen (wenn vorh., sonst None):
      #   (hier und nicht erst bei doModeling, da linearSegments die Variable zum Modellieren benötigt!)
      self.mod.var_on                               = self.featureOn.getVar_on()           # mit None belegt, falls nicht notwendig           
      self.mod.var_switchOn, self.mod.var_switchOff = self.featureOn.getVars_switchOnOff() # mit None belegt, falls nicht notwendig                  

      # super().declareVarsAndEqs(modBox)
      

    def doModeling(self,modBox,timeIndexe):
      log.debug(str(self.label) + 'doModeling()')
      # super().doModeling(modBox,timeIndexe)
          
      #          
      # ############## Constraints für Binärvariablen : ##############
      #
      self.featureOn.doModeling(modBox,timeIndexe)
   
    def addShareToGlobalsOfFlows(self,globalComp,modBox):
      for aFlow in self.inputs + self.outputs:
        aFlow.addShareToGlobals(globalComp,modBox)

    # wird von Kindklassen überschrieben:
    def addShareToGlobals(self, globalComp, modBox):
      # Anfahrkosten, Betriebskosten, ... etc ergänzen: 
      self.featureOn.addShareToGlobals(globalComp,modBox)
    
    def getDescrAsStr(self):
      
      descr = {}
      inhalt = {'In-Flows':[],'Out-Flows':[]}
      aFlow : cFlow
      
      descr[self.label] = inhalt
      
      if isinstance(self, cBus): 
        descrType = 'for bus-list'
      else :
        descrType = 'for comp-list'
                     
      for aFlow in self.inputs:
        inhalt['In-Flows'].append(aFlow.getStrDescr(type = descrType))#'  -> Flow: '))
      for aFlow in self.outputs:
        inhalt['Out-Flows'].append(aFlow.getStrDescr(type = descrType))#'  <- Flow: '))
                       
      if self.isStorage :
          inhalt['isStorage'] = self.isStorage
      
      return descr
    
    def print(self,shiftChars):
      aFlow : cFlow
      print(yaml.dump(self.getDescrAsStr(), allow_unicode = True))
      


# komponenten übergreifende Gleichungen/Variablen/Zielfunktion!
class cGlobal(cME):
  
  def __init__(self, label, **kwargs):
    super().__init__(label, **kwargs)
    
    self.listOfEffectTypes = [] # wird überschrieben mit spezieller Liste

    self.objective = None

  def finalize(self):
    super().finalize() # TODO: super-Finalize eher danach?
    self.penalty   = cFeature_ShareSum('penalty', self, sharesAreTS = True)
    
    # Effekte als Subelemente hinzufügen ( erst hier ist effectTypeList vollständig)
    self.subElements.extend(self.listOfEffectTypes)
   
  
  # Beiträge registrieren:
  # effectValues kann sein 
  #   1. {effecttype1 : TS, effectType2: : TS} oder 
  #   2. TS oder skalar 
  #     -> Zuweisung zu Standard-EffektType      
    
  def addShareToOperation(self, aVariable, effect_values, factor):
    if aVariable is None: raise Exception('addShareToOperation() needs variable or use addConstantShare instead')
    self.__addShare('operation', effect_values, factor, aVariable)

  def addConstantShareToOperation(self, effect_values, factor):
    self.__addShare('operation', effect_values, factor)
    
  def addShareToInvest   (self, aVariable, effect_values, factor):
    if aVariable is None: raise Exception('addShareToOperation() needs variable or use addConstantShare instead')
    self.__addShare('invest'   , effect_values, factor, aVariable)
    
  def addConstantShareToInvest   (self, effect_values, factor):
    self.__addShare('invest'   , effect_values, factor)    
  
  # wenn aVariable = None, dann constanter Share
  def __addShare(self, operationOrInvest, effect_values, factor, aVariable = None):
    aEffectSum : cFeature_ShareSum
    
    effect_values_dict = getEffectDictOfEffectValues(effect_values)
              
    # an alle Effekttypen, die einen Wert haben, anhängen:
    for effectType, value in effect_values_dict.items():                 
      # Falls None, dann Standard-effekt nutzen:
      effectType : cEffectType
      if effectType is None : effectType = self.listOfEffectTypes.standardType()
      elif effectType not in self.listOfEffectTypes:
        raise Exception('Effect \'' + effectType.label + '\' was not added to model (but used in some costs)!')        
        
      if operationOrInvest   == 'operation':        
        effectType.operation.addShare(aVariable, value, factor) # hier darf aVariable auch None sein!
      elif operationOrInvest == 'invest':
        effectType.invest   .addShare(aVariable, value, factor) # hier darf aVariable auch None sein!
      else:
        raise Exception('operationOrInvest=' + str(operationOrInvest) + ' ist kein zulässiger Wert')

  def declareVarsAndEqs(self,modBox):
    
    # TODO: ggf. Unterscheidung, ob Summen überhaupt als Zeitreihen-Variablen abgebildet werden sollen, oder nicht, wg. Performance.

    super().declareVarsAndEqs(modBox)

    
    for effect in self.listOfEffectTypes:
      effect.declareVarsAndEqs(modBox)
    self.penalty  .declareVarsAndEqs(modBox)

    self.objective  = cEquation('obj',self,modBox,'objective')   
   
    # todo : besser wäre objective separat:
   #  eq_objective = cEquation('objective',self,modBox,'objective')   
    # todo: hier vielleicht gleich noch eine Kostenvariable ergänzen. Wäre cool!
  def doModeling(self,modBox,timeIndexe):
    # super().doModeling(modBox,timeIndexe)
    
    self.penalty  .doModeling(modBox,timeIndexe)
    ## Gleichungen bauen für Effekte: ##       
    for effect in self.listOfEffectTypes:
      effect.doModeling(modBox,timeIndexe)
    
    
    ## Beiträge von Effekt zu anderen Effekten, Beispiel 180 €/t_CO2: ##    
    for effectType in self.listOfEffectTypes :                      
      # Beitrag/Share ergänzen:
      # 1. operation: -> hier sind es Zeitreihen (share_TS)
      # alle specificSharesToOtherEffects durchgehen:
      for effectTypeOfShare, specShare_TS in effectType.specificShareToOtherEffects_operation.items():               
        # Share anhängen (an jeweiligen Effekt):
        effectTypeOfShare.operation.addVariableShare(effectType.operation.mod.var_sum_TS, specShare_TS, 1)
      # 2. invest:    -> hier ist es Skalar (share)
      # alle specificSharesToOtherEffects durchgehen:
      for effectTypeOfShare, specShare in effectType.specificShareToOtherEffects_invest.items():                     
        # Share anhängen (an jeweiligen Effekt):
        effectTypeOfShare.invest.addVariableShare(effectType.invest.mod.var_sum   , specShare   , 1)
                       
      
    
    # ####### Zielfunktion ###########       
    # Strafkosten immer:
    self.objective.addSummand(self.penalty  .mod.var_sum, 1) 
    
    # Definierter Effekt als Zielfunktion:
    objectiveEffect = self.listOfEffectTypes.objectiveEffect()
    if objectiveEffect is None : raise Exception('Kein Effekt als Zielfunktion gewählt!')
    self.objective.addSummand(objectiveEffect.operation.mod.var_sum, 1)
    self.objective.addSummand(objectiveEffect.invest   .mod.var_sum ,1)
          
class cBus(cBaseComponent): # sollte das wirklich geerbt werden oder eher nur cME???

  new_init_args = [cArg('label'                 , 'param', 'str', 'Bezeichnung'),
                   cArg('typ'                   , 'param', 'str', 'Zusatzbeschreibung, sonst kein Einfluss'),
                   cArg('excessCostsPerFlowHour', 'costs', 'TS' , 'Exzesskosten (none/ 0 -> kein Exzess; > 0 -> berücksichtigt')]

  not_used_args = ['label']

  # --> excessCostsPerFlowHour
  #        none/ 0 -> kein Exzess berücksichtigt
  #        > 0 berücksichtigt
    
  def __init__(self, typ, label, excessCostsPerFlowHour = 1e5, **kwargs):   
    super().__init__(label,**kwargs)  
    self.typ = typ
    self.medium = helpers.InfiniteFullSet()
    if  (excessCostsPerFlowHour is not None) and (excessCostsPerFlowHour > 0) :
      self.withExcess = True
      self.excessCostsPerFlowHour = cTS_vector('excessCostsPerFlowHour', excessCostsPerFlowHour, self)      
    else: 
      self.withExcess = False

    
  def registerInputFlow(self, aFlow):
    self.inputs.append(aFlow)
    self.checkMedium(aFlow)
    
  def registerOutputFlow(self,aFlow):
    self.outputs.append(aFlow) 
    self.checkMedium(aFlow)

  def checkMedium(self,aFlow):
    # Wenn noch nicht belegt
    if aFlow.medium is not None : 
      # set gemeinsamer Medien:
      newMedium = self.medium & aFlow.medium
      # wenn mindestens ein Eintrag:    
      if newMedium : 
        self.medium = newMedium
      else : 
        raise Exception('in cBus ' + self.label + ' : registerFlow(): medium ' + str(aFlow.medium) + ' of ' + aFlow.label + ' passt nicht zu bisherigen Flows ' + str(self.medium) ) 

  def declareVarsAndEqs(self, modBox):
    super().declareVarsAndEqs(modBox)
    # Fehlerplus/-minus:
    if self.withExcess:
      # Fehlerplus und -minus definieren
      self.excessIn  =  cVariable('excessIn' , len(modBox.timeSeries), self, modBox, min=0)   
      self.excessOut =  cVariable('excessOut', len(modBox.timeSeries), self, modBox, min=0)         
    
  def doModeling(self, modBox, timeIndexe):        
    super().doModeling(modBox, timeIndexe)
      
    # inputs = outputs 
    eq_busbalance = cEquation('busBalance', self, modBox)
    for aFlow in self.inputs:
      eq_busbalance.addSummand(aFlow.mod.var_val, 1)
    for aFlow in self.outputs:
      eq_busbalance.addSummand(aFlow.mod.var_val,-1)
    
    # Fehlerplus/-minus:
    if self.withExcess:
      # Hinzufügen zur Bilanz:
      eq_busbalance.addSummand(self.excessOut, -1) 
      eq_busbalance.addSummand(self.excessIn ,  1)      

  def addShareToGlobals(self,globalComp,modBox) :
    super().addShareToGlobals(globalComp, modBox)
    # Strafkosten hinzufügen:
    if self.withExcess :      
      globalComp.penalty.addVariableShare(self.excessIn , self.excessCostsPerFlowHour, modBox.dtInHours)
      globalComp.penalty.addVariableShare(self.excessOut, self.excessCostsPerFlowHour, modBox.dtInHours)
      # globalComp.penaltyCosts_eq.addSummand(self.excessIn , np.multiply(self.excessCostsPerFlowHour, modBox.dtInHours))
      # globalComp.penaltyCosts_eq.addSummand(self.excessOut, np.multiply(self.excessCostsPerFlowHour, modBox.dtInHours))
    
  def print(self,shiftChars):
    print(shiftChars + str(self.label) + ' - ' + str(len(self.inputs)) + ' In-Flows / ' +  str(len(self.outputs)) + ' Out-Flows registered' )        
    
    print(shiftChars + '   medium: ' + str(self.medium))
    super().print(shiftChars)

   
    

 # Medien definieren:
class cMediumCollection :
  # single medium:
  heat    = set(['heat'])
  cold    = set(['cold'])
  el      = set(['el'])
  gas     = set(['gas'])
  lignite = set(['lignite'])
  biomass = set(['biomass'])
  ash     = set(['ash'])
  # groups: 
  fu        = gas | lignite | biomass
  fossil_fu = gas | lignite
  
  # neues Medium hinzufügen:
  def addMedium(attrName, aSetOfStrs):
    cMediumCollection.setattr(attrName,aSetOfStrs)
    
  # checkifFits(medium1,medium2,...)
  def checkIfFits(*args):
    aCommonMedium = helpers.InfiniteFullSet()
    for aMedium in args:
      if aMedium is not None : aCommonMedium = aCommonMedium & aMedium 
    if aCommonMedium : return True
    else             : return False
    
# input/output-dock (TODO:
class cIO(): 
  pass
  # -> wäre cool, damit Komponenten auch auch ohne Knoten verbindbar
  # input wären wie cFlow,aber statt bus : connectsTo -> hier andere cIO oder aber Bus (dort keine cIO, weil nicht notwendig)
  
  
# todo: könnte Flow nicht auch von Basecomponent erben. Hat zumindest auch Variablen und Eqs  
# Fluss/Strippe
class cFlow(cME):
    ## Parameter       
    # valuesBeforeBegin -> Liste mit letzten 2 Werten davor!
    new_init_args = [cArg('label'               , 'param', 'str',        'Bezeichnung'),
                     cArg('bus'                 , 'bus'  , 'bus',        'Bus-Komponente, mit der der flow verknüpft wird, angeben'),
                     cArg('min_rel'             , 'param', 'TS',         'minimaler Wert (relativ) := Flow_min/Nominal_val'),
                     cArg('max_rel'             , 'param', 'TS',         'maximaler Wert (relativ) := Flow_max/Nominal_val; (wenn Nennwert = max, dann max_rel = 1'),
                     cArg('nominal_val'         , 'param', 'skalar',     'Investgröße/ Nennwert z.B. kW, Fläche, Volumen, Stck,  möglichst immer so stark wie möglich einschränken (wg. Rechenzeit bzw. Binär-Ungenauigkeits-Problem!)'),
                     cArg('loadFactor_min'      , 'param', 'skalar',     'minimaler nomineller Auslastungsgrad (equivalenter Vollbenutzungsgrad), general: avg Flow per Investsize (e.g. solarthermal: kW/m²; def: load_factor:= sumFlowHours/ (nominal_val*dt_tot)'),
                     cArg('loadFactor_max'      , 'param', 'skalar',     'maximaler nomineller Auslastungsgrad (equivalenter Vollbenutzungsgrad), general: avg Flow per Investsize (e.g. solarthermal: kW/m²; def: load_factor:= sumFlowHours/ (nominal_val*dt_tot)'),                     
                     cArg('positive_gradient'   , 'param', 'TS',         '! noch nicht implementiert !'),
                     cArg('costsPerFlowHour'    , 'costs', 'TS',         'Kosten pro Flow-Arbeit, z.B. €/kWh'),
                     cArg('iCanSwitchOff'       , 'param', 'True/False', 'Kann flow "aus gehen", also auf Null gehen (nur relevant wenn min > 0) -> BinärVariable wird genutzt'),
                     cArg('onHours_min'         , 'param', 'skalar',     'min. Betriebsstunden'),
                     cArg('onHours_max'         , 'param', 'skalar',     'max. Betriebsstunden'),
                     cArg('switchOnCosts'       , 'costs', 'TS'    ,     'Einschaltkosten z.B. in €'),
                     cArg('switchOn_maxNr'      , 'param', 'skalar',     'max. zulässige Anzahl Starts'),
                     cArg('costsPerRunningHour' , 'costs', 'TS'    ,     'Kosten für den reinen Betrieb, z.B. €/h'),
                     cArg('valuesBeforeBegin'   , 'param', 'list'  ,     'Flow-Werte vor Beginn (zur Berechnung verschied. Dinge im ersten Zeitschritt, z.B. switchOn, gradient,...)'),
                     cArg('val_rel'             , 'param', 'TS'    ,     'fixe Werte für Flow (falls gegeben). Damit ist Flow-Wert dann keine freie Optimierungsvariable mehr.; min_rel u. max_rel werden nicht mehr genutzt'),
                     cArg('investArgs'          , 'obj'  , 'cInvestArgs','None or Investitionsparameter'),
                     cArg('sumFlowHours_max'    , 'param', 'skalar',     'maximale FlowHours in Zeitbereich (if nominal_val is not const, better use loadFactor_max!)'),
                     cArg('sumFlowHours_min'    , 'param', 'skalar',     'minimale FlowHours in Zeitbereich (if nominal_val is not const, better use loadFactor_min!)'),                     
                     ]
    
    not_used_args = ['label']   

    @property
    def label_full(self):
      # Wenn im Erstellungsprozess comp noch nicht bekannt:
      if self.comp is None:
        comp_label = 'unknownComp'
      else:
        comp_label = self.comp.label
      
      separator = '_' # wichtig, sonst geht results_struct nicht
      return comp_label + separator + self.label # z.B. für results_struct (deswegen auch _  statt . dazwischen)

    @property #Richtung
    def isInputInComp(self):
      comp : cBaseComponent
      if self in self.comp.inputs:
        return True
      else:
        return False

    @property
    def investmentSize_is_fixed(self):
      #Wenn kein investArg existiert:
      if self.investArgs is None:
        is_fixed = True # keine variable var_InvestSize
      else:
        is_fixed = self.investArgs.investmentSize_is_fixed      
      return is_fixed
        
    @property
    def invest_is_optional(self):
      #Wenn kein investArg existiert:
      if self.investArgs is None:
        is_optional = False # keine variable var_isInvested
      else:
        is_optional = self.investArgs.investment_is_optional      
      return is_optional
    
    #static var:
    __nominal_val_default = 1e9 # Großer Gültigkeitsbereich als Standard
    
    def __init__(self,label, bus:cBus=None , min_rel = 0, max_rel = 1, nominal_val = __nominal_val_default , loadFactor_min = None, loadFactor_max = None, positive_gradient = None, costsPerFlowHour = None ,iCanSwitchOff = True,
                 onHours_min = None, onHours_max= None, switchOnCosts = None, switchOn_maxNr = None, costsPerRunningHour =None, sumFlowHours_max = None, sumFlowHours_min = None, valuesBeforeBegin = [0,0], val_rel = None, investArgs = None, **kwargs):
      
      super().__init__(label, **kwargs)   
      # args to attributes:
      self.bus                 = bus
      self.nominal_val         = nominal_val # skalar!
      self.min_rel             = cTS_vector('min_rel', min_rel              , self)        
      self.max_rel             = cTS_vector('max_rel', max_rel              , self)
      self.loadFactor_min      = loadFactor_min
      self.loadFactor_max      = loadFactor_max
      self.positive_gradient   = cTS_vector('positive_gradient', positive_gradient, self)
      self.costsPerFlowHour    = transFormEffectValuesToTSDict('costsPerFlowHour',costsPerFlowHour , self)
      self.iCanSwitchOff       = iCanSwitchOff
      self.onHours_min         = onHours_min
      self.onHours_max         = onHours_max

      self.switchOnCosts       = transFormEffectValuesToTSDict('switchOnCosts'      , switchOnCosts       , self)
      self.switchOn_maxNr      = switchOn_maxNr
      self.costsPerRunningHour = transFormEffectValuesToTSDict('costsPerRunningHour', costsPerRunningHour , self)
      self.sumFlowHours_max = sumFlowHours_max
      self.sumFlowHours_min = sumFlowHours_min 

      
      self.valuesBeforeBegin   = np.array(valuesBeforeBegin)   # list -> np-array
      
      if val_rel is None:
        self.val_rel = None # damit man noch einfach rauskriegt, ob es belegt wurde
      else:
        # Check:
        # Wenn noch nominal_val noch Default, aber investmentSize nicht optimiert werden soll:
        if (self.nominal_val == cFlow.__nominal_val_default) and \
           ((investArgs is None) or (investArgs.investmentSize_is_fixed == True)): 
          # Fehlermeldung:
          raise Exception('Achtung: Wenn val_ref genutzt wird, muss zugehöriges nominal_val definiert werden, da: value = val_ref * nominal_val!')
        
        self.val_rel = cTS_vector('val_rel',val_rel, self)
      
      self.investArgs          = investArgs
      # Info: Plausi-Checks erst, wenn Flow self.comp kennt.

      # zugehörige Komponente (wird später von Komponente gefüllt)
      self.comp = None
      
      # defaults:
      self.medium    = None
                        
      # Wenn Min-Wert > 0 wird binäre On-Variable benötigt (nur bei flow!):
      self.__useOn_fromProps = iCanSwitchOff & (min_rel > 0)
      
                     
      # self.prepared          = False # ob __declareVarsAndEqs() ausgeführt

      # feature for: On and SwitchOn Vars (builds only if necessary)
      # -> feature bereits hier, da andere Elemente featureOn.activateOnValue() nutzen wollen
      flowsDefiningOn      = [self] # Liste. Ich selbst bin der definierende Flow! (Bei Komponente sind es hingegen alle in/out-flows)      
      on_valuesBeforeBegin = 1 * (self.valuesBeforeBegin >= 0.0001 ) # TODO: besser wäre modBox.epsilon, aber hier noch nicht bekannt!)       
      # TODO: Wenn iCanSwitchOff = False und min > 0, dann könnte man var_on fest auf 1 setzen um Rechenzeit zu sparen
      
      self.featureOn = cFeatureOn(self, flowsDefiningOn, on_valuesBeforeBegin, self.switchOnCosts, self.costsPerRunningHour, onHours_min = self.onHours_min, onHours_max = self.onHours_max, switchOn_maxNr = self.switchOn_maxNr, useOn_explicit = self.__useOn_fromProps)

      if self.investArgs is None:
        self.featureInvest = None # 
      else:
        self.featureInvest = cFeatureInvest('nominal_val', self, self.investArgs, 
                                            min_rel = self.min_rel,
                                            max_rel = self.max_rel,
                                            val_rel = self.val_rel,
                                            investmentSize = self.nominal_val,
                                            featureOn      = self.featureOn)
        
    # Plausitest der Eingangsparameter (sollte erst aufgerufen werden, wenn self.comp bekannt ist)
    def plausiTest(self):
      # Plausi-Check min < max:
      if np.any(np.asarray(self.min_rel.d) > np.asarray(self.max_rel.d)):       
      # if np.any(np.asarray(np.asarray(self.min_rel.d) > np.asarray(self.max_rel.d) )): 
        raise Exception(self.label_full + ': Take care, that min_rel <= max_rel!')

    # bei Bedarf kann von außen Existenz von Binärvariable erzwungen werden:
    def activateOnValue(self):
      self.featureOn.activateOnValueExplicitly()

    def finalize(self):
      self.plausiTest() # hier Input-Daten auf Plausibilität testen (erst hier, weil bei __init__ self.comp noch nicht bekannt)
      super().finalize()

      
    def declareVarsAndEqs(self, modBox:cModelBoxOfES):
      print('declareVarsAndEqs ' + self.label)
      super().declareVarsAndEqs(modBox)
      
      self.featureOn.declareVarsAndEqs(modBox) # TODO: rekursiv aufrufen für subElements
      
      self.modBox = modBox
      # Skalare zu Vektoren # 
      # -> schöner wäre das bei Init, aber da gibt es noch keine Info über Länge)
      # -> überprüfen, ob nur für pyomo notwendig!

      # timesteps = model.timesteps  
      ############################           

      ## min/max Werte:
      #  min-Wert:
      
      def getMinMaxOfDefiningVar():
        # Wenn fixer Lastgang:
        if self.val_rel is not None:
          # min = max = val !
          fix_value = self.val_rel.d_i * self.nominal_val
          lb = None
          ub = None
        else:
          if self.featureOn.useOn:
            lb = 0
          else:
            lb = self.min_rel.d_i * self.nominal_val # immer an       
          ub = self.max_rel.d_i * self.nominal_val     
          fix_value = None
        return (lb, ub, fix_value)
      
      # wenn keine Investrechnung:
      if self.featureInvest is None:  
        (lb, ub, fix_value) = getMinMaxOfDefiningVar()
      else:
        (lb, ub, fix_value) = self.featureInvest.getMinMaxOfDefiningVar()
            
      # TODO --> wird trotzdem modelliert auch wenn value = konst -> Sinnvoll?        
      self.mod.var_val = cVariable('val', modBox.nrOfTimeSteps, self, modBox, min = lb, max = ub, value = fix_value)
      self.mod.var_sumFlowHours = cVariable('sumFlowHours', 1, self, modBox, min = self.sumFlowHours_min, max = self.sumFlowHours_max)
      # ! Die folgenden Variablen müssen erst von featureOn erstellt worden sein:
      self.mod.var_on                               = self.featureOn.getVar_on()           # mit None belegt, falls nicht notwendig           
      self.mod.var_switchOn, self.mod.var_switchOff = self.featureOn.getVars_switchOnOff() # mit None belegt, falls nicht notwendig           
 
      # erst hier, da definingVar vorher nicht belegt!
      if self.featureInvest is not None: 
        self.featureInvest.setDefiningVar(self.mod.var_val, self.mod.var_on)
        self.featureInvest.declareVarsAndEqs(modBox)
        
    def doModeling(self,modBox:cModelBoxOfES,timeIndexe):        
        # super().doModeling(modBox,timeIndexe)
 
        # for aFeature in self.features:
        #   aFeature.doModeling(modBox,timeIndexe)
                  
        #
        # ############## Variablen aktivieren: ##############
        #
        
        # todo -> für pyomo: fix()        

        #
        # ############## sumFlowHours: ##############
        #        

        # eq: var_sumFlowHours - sum(var_val(t)* dt(t) = 0
        
        eq_sumFlowHours = cEquation('sumFlowHours',self,modBox,'eq') # general mean 
        eq_sumFlowHours.addSummandSumOf(self.mod.var_val, modBox.dtInHours)  
        eq_sumFlowHours.addSummand(self.mod.var_sumFlowHours,-1)
        
       
        #          
        # ############## Constraints für Binärvariablen : ##############
        #
        
        self.featureOn.doModeling(modBox, timeIndexe) # TODO: rekursiv aufrufen für subElements
        
        
        #          
        # ############## Glg. für Investition : ##############
        #
        
        if self.featureInvest is not None:
          self.featureInvest.doModeling(modBox, timeIndexe)
        
        ## ############## full load fraction bzw. load factor ##############
                
        ## max load factor:
        #  eq: var_sumFlowHours <= nominal_val * dt_tot * load_factor_max

        if self.loadFactor_max is not None:  
          flowHoursPerInvestsize_max = modBox.dtInHours_tot * self.loadFactor_max # = fullLoadHours if investsize in [kW]
          eq_flowHoursPerInvestsize_Max = cEquation('loadFactor_max',self,modBox,'ineq') # general mean 
          eq_flowHoursPerInvestsize_Max.addSummand(self.mod.var_sumFlowHours, 1)  
          if self.featureInvest is not None:
            eq_flowHoursPerInvestsize_Max.addSummand(self.featureInvest.mod.var_investmentSize, -1 * flowHoursPerInvestsize_max) 
          else:
            eq_flowHoursPerInvestsize_Max.addRightSide(self.nominal_val * flowHoursPerInvestsize_max)          
        
        ## min load factor:
        #  eq: nominal_val * sum(dt)* load_factor_min <= var_sumFlowHours
        
        if self.loadFactor_min is not None:
          flowHoursPerInvestsize_min = modBox.dtInHours_tot * self.loadFactor_min # = fullLoadHours if investsize in [kW]
          eq_flowHoursPerInvestsize_Min = cEquation('loadFactor_min',self,modBox,'ineq')
          eq_flowHoursPerInvestsize_Min.addSummand(self.mod.var_sumFlowHours, -1)          
          if self.featureInvest is not None:
            eq_flowHoursPerInvestsize_Min.addSummand(self.featureInvest.mod.var_investmentSize,  flowHoursPerInvestsize_min) 
          else:
            eq_flowHoursPerInvestsize_Min.addRightSide(-1 * self.nominal_val * flowHoursPerInvestsize_min)
        
        # ############## positiver Gradient ######### 
        
        '''        
        if self.positive_gradient == None :                    
          if modBox.modType == 'pyomo':
            def positive_gradient_rule(t):
              if t == 0:
                return (self.mod.var_val[t] - self.val_initial) / modBox.dtInHours[t] <= self.positive_gradient[t] #             
              else: 
                return (self.mod.var_val[t] - self.mod.var_val[t-1])    / modBox.dtInHours[t] <= self.positive_gradient[t] #
  
            # Erster Zeitschritt beachten:          
            if (self.val_initial == None) & (start == 0):
              self.positive_gradient_constr =  Constraint([start+1:end]        ,rule = positive_gradient_rule)          
            else:
              self.positive_gradient_constr =  Constraint(modBox.timestepsOfRun,rule = positive_gradient_rule)   # timestepsOfRun = [start:end]
              # raise error();
            modbox.registerPyComp(self.positive_gradient_constr, self.label + '_positive_gradient_constr')
          elif modBox.modType == 'vcxpy':
            raise Exception('not defined for modtype ' + modBox.modType)
          else:
            raise Exception('not defined for modtype ' + modBox.modType)'''
        
                
        # ############# Beiträge zu globalen constraints ############
        
        # z.B. max_PEF, max_CO2, ...
              
      
    def addShareToGlobals(self, globalComp:cGlobal, modBox) :
        # Arbeitskosten:
        if self.costsPerFlowHour is not None: 
          # globalComp.addEffectsForVariable(aVariable, aEffect, aFactor)
          # variable_costs          = cVector(self.mod.var_val, np.multiply(self.costsPerFlowHour, modBox.dtInHours))  
          # globalComp.costsOfOperating_eq.addSummand(self.mod.var_val, np.multiply(self.costsPerFlowHour.d_i, modBox.dtInHours)) # np.multiply = elementweise Multiplikation          
  
          globalComp.addShareToOperation(self.mod.var_val, self.costsPerFlowHour, modBox.dtInHours)
            
        # Anfahrkosten, Betriebskosten, ... etc ergänzen: 
        self.featureOn.addShareToGlobals(globalComp,modBox)

        if self.featureInvest is not None:
          self.featureInvest.addShareToGlobals(globalComp, modBox)
          
        ''' in oemof gibt es noch 
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
        '''


    def getStrDescr(self, type = 'full'):            
      aDescr = {}
      if type == 'for bus-list':
        # aDescr = str(self.comp.label) + '.'
        aDescr['comp']=self.comp.label        
        aDescr = {str(self.label):aDescr} # label in front of
      elif type == 'for comp-list':
        # aDescr += ' @Bus ' + str(self.bus.label)      
        aDescr['bus']=self.bus.label            
        aDescr = {str(self.label):aDescr} # label in front of
      elif type == 'full':
        aDescr['label']=self.label
        aDescr['comp']=self.comp.label        
        aDescr['bus']=self.bus.label            
        aDescr['isInputInComp'] = self.isInputInComp
      else: 
        raise Exception('type = \'' + str(type) + '\' is not defined')

      return aDescr
    
    # def printWithBus(self):
    #   return (str(self.label) + ' @Bus ' + str(self.bus.label))
    # def printWithComp(self):
    #   return (str(self.comp.label) + '.' +  str(self.label))
    def setMediumIfNotSet(self,medium):
      # nicht überschreiben, nur wenn leer:
      if self.medium is None: self.medium = medium
        

# class cBeforeValue :
  
#   def __init__(self, modelingElement, var, esBeforeValues, beforeValueIsStartValue):
#     self.esBeforeValues  = esBeforeValues # Standardwerte für Simulationsstart im Energiesystem
#     self.modelingElement = modelingElement 
#     self.var             = var
#     self.beforeValueIsStartValue =beforeValueIsStartValue
  
#   def getBeforeValue(self):
#     if 


  


