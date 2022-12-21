# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:40:23 2020

@author: Panitz
"""

# ########################
# ### todos

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
#  1. Schauen, ob man nicht doch lieber, alle Variablen nur für den Zeitbereich definiert und cVariable_TS.lastValues einrichtet, und dort Vorgängerzeitschritte übergibt
#     --> Vorteil: Gleiche Prozedur für 1. Zeistreifen wie für alle anderen!
#  2. Alle Dinge, die auf historische Werte zugreifen kennzeichnen (Attribut) und warnen/sperrenz.B. 
#    Speicher, On/Off, 
#    maximale FlowArbeit,... -> geht wahrscheinlich gar nicht gut!
#    helpers.checkBoundsOfParameters -> vielleicht besser in cBaseComponent integrieren, weil dort self.label 
# TODO: einen Bus automatisch in cEffect erzeugen.

# mögliche Testszenarien für testing-tool:
   # abschnittsweise linear testen
   # Komponenten mit offenen Flows 
   # Binärvariablen ohne max-Wert-Vorgabe des Flows (Binärungenauigkeitsproblem)
   # Medien-zulässigkeit 
   
##########################
# # Features to do:
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

import numpy as np

import math
import time
import yaml  # (für json-Schnipsel-print)

import flixOptHelperFcts as helpers

from basicModeling import * # Modelliersprache
from flixBasics import *
import logging

log = logging.getLogger(__name__)
# TODO:
# -> results sollte mit objekt und! label mappen, also results[aKWK] und! results['KWK1']

# TODO: 1. cTimePeriodModel -> hat für Zeitbereich die Modellierung timePeriodModel1, timePeriodModel2,...
class cModelBoxOfES(cBaseModel):  
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
    
    def __init__(self, label, aModType, es, esTimeIndexe, TS_explicit = None):
        super().__init__(label, aModType)
        self.es: cEnergySystem
        self.es = es  # energysystem (wäre Attribut von cTimePeriodModel)
        self.esTimeIndexe = esTimeIndexe
        self.nrOfTimeSteps = len(esTimeIndexe)
        self.TS_explicit = TS_explicit  # für explizite Vorgabe von Daten für TS {TS1: data, TS2:data,...}
        # self.epsilon    = 1e-5 # 
        # self.variables  = [] # Liste aller Variablen
        # self.eqs        = [] # Liste aller Gleichungen
        # self.ineqs      = [] # Liste aller Ungleichungen
        self.ME_mod = {}  # dict mit allen mods der MEs
        
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
    def solve(self, gapFrac = 0.02,timelimit = 3600, solver ='cbc', displaySolverOutput = True, excessThreshold = 0.1, logfileName = 'solverLog.log', **kwargs):      
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
        
        
        super().solve(gapFrac, timelimit, solver, displaySolverOutput, logfileName, **kwargs)
        
        if solver == 'gurobi': 
            termination_message = self.solver_results['Solver'][0]['Termination message']
        elif solver == 'glpk':
            termination_message = self.solver_results['Solver'][0]['Status']
        else:
            termination_message = 'not implemented for solver yet'
        print('termination message: "' + termination_message + '"')    
        
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
        try:
            print('lower bound   : ' + str(self.solver_results['Problem'][0]['Lower bound']))
        except:
            print
        print('')
        for aBus in self.es.setOfBuses:
            if aBus.withExcess : 
                if any(self.results[aBus.label]['excessIn'] > 1e-6) or any(self.results[aBus.label]['excessOut'] > 1e-6):
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
            main_results_str['Effects'] = aEffectDict
            for aEffect in self.es.globalComp.listOfEffectTypes:
                aDict = {}
                aEffectDict[aEffect.label +' ['+ aEffect.unit + ']'] = aDict
                aDict['operation']= str(aEffect.operation.mod.var_sum.getResult())
                aDict['invest']   = str(aEffect.invest   .mod.var_sum.getResult())
                aDict['sum']      = str(aEffect.all      .mod.var_sum.getResult())
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
    
    ''' 
    Handles the energy system as a model.
    '''

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
        
        def getInvestFeaturesOfME(aME):
            investFeatures = []
            for aSubComp in aME.subElements_all:
                if isinstance(aSubComp, cFeatureInvest):
                    investFeatures.append(aSubComp)
                investFeatures += getInvestFeaturesOfME(aSubComp)  # recursive!
            return investFeatures
        
        for aME in self.allMEsOfFirstLayer: # kann in Komponente (z.B. Speicher) oder Flow stecken
            allInvestFeatures += getInvestFeaturesOfME(aME)

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
        '''        
          Parameters
          ----------
          timeSeries : np.array of datetime64
              timeseries of the data
          dt_last : for calc
              The duration of last time step. 
              Storages needs this time-duration for calculation of charge state
              after last time step. 
              If None, then last time increment of timeSeries is used.
          
        '''
        self.timeSeries = timeSeries
        self.dt_last    = dt_last
          
        self.timeSeriesWithEnd = helpers.getTimeSeriesWithEnd(timeSeries, dt_last)
        helpers.checkTimeSeries('global esTimeSeries', self.timeSeriesWithEnd)
        
        # defaults: 
        self.listOfComponents = []        
        self.setOfOtherMEs    = set()  ## hier kommen zusätzliche MEs rein, z.B. aggregation
        self.listOfEffectTypes  = cEffectTypeList() # Kosten, CO2, Primärenergie, ...
        self.AllTempMEs       = [] # temporary elements, only valid for one calculation (i.g. aggregation modeling)
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
        # check if already exists:
        self._checkIfUniqueElement(aNewEffect, self.listOfEffectTypes)
        
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
        # check if already exists:
        self._checkIfUniqueElement(aNewComp, self.listOfComponents)

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
        '''
        add all modeling elements, like storages, boilers, heatpumps, buses, ... 

        Parameters
        ----------
        *args : childs of   cME like cBoiler, cHeatPump, cBus,...
            modeling Elements

        '''
        
        newList = list(args)
        for aNewME in newList:
            if    isinstance(aNewME, cBaseComponent):
                self.addComponents(aNewME)
            elif isinstance(aNewME, cEffectType):
                self.addEffects(aNewME)
            elif isinstance(aNewME, cME):
                # check if already exists:
                self._checkIfUniqueElement(aNewME, self.setOfOtherMEs)
                # register ME:
                self.setOfOtherMEs.add(aNewME)

            else: 
                raise Exception('argument is not instance of a modeling Element (cME)')
    
    def addTemporaryElements(self,*args):
        '''
        add temporary modeling elements, only valid for one calculation, 
        i.g. cAggregationModeling-Element

        Parameters
        ----------
        *args : cME
            temporary modeling Elements.

        '''

        self.addElements(*args)
        self.AllTempMEs += args # Register temporary Elements
    
    def deleteTemporaryElements(self): # function just implemented, still not used
        '''        
        deletes all registered temporary Elements
        '''        
        for tempME in self.AllTempMEs:
            # delete them again in the lists:
            self.listOfComponents.remove(tempME)
            self.setOfBuses.remove(tempME)
            self.setOfOtherMEs.remove(tempME)
            self.listOfEffectTypes.remove(tempME)
            self.setOfFlows(tempME)

    def _checkIfUniqueElement(self, aElement, listOfExistingLists):
        '''
        checks if element or label of element already exists in list

        Parameters
        ----------
        aElement : cME
            new element to check
        listOfExistingLists : list
            list of already registered elements
        '''
        
        # check if element is already registered:        
        if aElement in listOfExistingLists:
            raise Exception('Element \'' + aElement.label + '\' already added to cEnergysystem!')                
        
        # check if name is already used:
        if aElement.label in [elem.label for elem in listOfExistingLists]:
            raise Exception('Elementname \'' + aElement.label + '\' already used in another element!')  
            
    
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
            print(aME.label)
            type(aME)
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
    '''
    class for defined way of solving a energy system optimizatino
    '''
    

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
    def __init__(self, label, es : cEnergySystem, modType, chosenEsTimeIndexe = None, pathForSaving = '/results',):
        '''
        Parameters
        ----------
        label : str
            name of calculation
        es : cEnergySystem
            energysystem which should be calculated
        modType : 'pyomo','cvxpy' (not implemeted yet)
            choose optimization modeling language
        chosenEsTimeIndexe : None, list
            list with indexe, which should be used for calculation. If None, then all timesteps are used.
        pathForSaving : str
            Path for result files. The default is '/results'.

        '''
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
        self.dataAgg = None # aggregationStuff (if calcType = 'aggregated')
    
    # Variante1:
    def doModelingAsOneSegment(self):
      '''
        modeling full problem

      '''
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
      '''
        Dividing and Modeling the problem in (overlapped) time-segments. 
        Storage values as result of segment n are overtaken 
        to the next segment n+1 for timestep, which is first in segment n+1
        
        Afterwards timesteps of segments (without overlap) 
        are put together to the full timeseries
        
        Because the result of segment n is used in segment n+1, modeling and 
        solving is both done in this method
        
        Take care: 
        Parameters like investArgs, loadfactor etc. does not make sense in 
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
      self.checkIfAlreadyModeled()
      self._infos['segmentedProps'] = {'segmentLen':segmentLen, 'nrUsedSteps':  nrOfUsedSteps}
      self.calcType = 'segmented'
      print('##############################################################')
      print('#################### segmented Solving #######################')
      
      
      t_start = time.time()
      
      # system finalisieren:
      self.es.finalize()

      if len(self.es.allInvestFeatures) > 0:
          raise Exception('segmented calculation with Invest-Parameters does not make sense!')
      
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

      self._definePathNames(namePrefix, nameSuffix, aPath, saveResults = True, nrOfModBoxes = nrOfSimSegments)
      
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
        
        segmentModBox.solve(**solverProps, logfileName=self.paths_Log[i])# keine SolverOutput-Anzeige, da sonst zu viel
        self.durations['solving'] += round(time.time()-t_start_solving,2)
        ## results adding:      
        self.__addSegmentResults(segmentModBox, startIndex_calc, realNrOfUsedSteps)
        
      
      self.durations['model, solve and segmentStuff'] = round(time.time()-t_start,2)
      
      self._saveSolveInfos()
    
    def doAggregatedModeling(self, periodLengthInHours, noTypicalPeriods, 
                             useExtremePeriods, fixStorageFlows,
                             fixBinaryVarsOnly, percentageOfPeriodFreedom = 0,
                             costsOfPeriodFreedom = 0,
                             addPeakMax = [],
                             addPeakMin = []):
        '''
        method of aggregated modeling. 
        1. Finds typical periods.
        2. Equalizes variables of typical periods. 
  
        Parameters
        ----------
        periodLengthInHours : float
            length of one period.
        noTypicalPeriods : int
            no of typical periods
        useExtremePeriods : boolean
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
        addPeakMax : list of cTSraw 
            list of data-timeseries. The period with the max-value are 
            chosen as a explicitly period.            
        addPeakMin : list of cTSraw
            list of data-timeseries. The period with the min-value are 
            chosen as a explicitly period.            
          
  
        Returns
        -------
        aModBox : TYPE
            DESCRIPTION.
  
        '''
        self.checkIfAlreadyModeled()
        
    
        
        self._infos['aggregatedProps'] ={'periodLengthInHours': periodLengthInHours,
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
        df_OriginalData = pd.DataFrame(self.TScollectionForAgg.seriesDict, index = chosenTimeSeries)# eigentlich wäre TS als column schön, aber TSAM will die ordnen können.       
    
        # Check, if timesteps fit in Period:
        stepsPerPeriod = periodLengthInHours / self.dtInHours[0] 
        if not stepsPerPeriod.is_integer():
          raise Exception('Fehler! Gewählte Periodenlänge passt nicht zur Zeitschrittweite')
        
        
        ##########################################################
        # ### Aggregation - creation of aggregated timeseries: ###
        import flixAggregation as flixAgg        
        dataAgg = flixAgg.flixAggregation('aggregation',
                                          timeseries = df_OriginalData,
                                          hoursPerTimeStep = self.dtInHours[0],
                                          hoursPerPeriod = periodLengthInHours,
                                          hasTSA = False,
                                          noTypicalPeriods = noTypicalPeriods,
                                          useExtremePeriods = useExtremePeriods,
                                          weightDict = self.TScollectionForAgg.weightDict,
                                          addPeakMax = self.TScollectionForAgg.addPeak_Max_labels,
                                          addPeakMin = self.TScollectionForAgg.addPeak_Min_labels)
        
                
        dataAgg.cluster()   
        self.dataAgg = dataAgg
        
        self._infos['aggregatedProps']['periodsOrder'] = str(list(dataAgg.aggregation.clusterOrder))
        
        # dataAgg.aggregation.clusterPeriodIdx
        # dataAgg.aggregation.clusterOrder
        # dataAgg.aggregation.clusterPeriodNoOccur
        # dataAgg.aggregation.predictOriginalData()    
        # self.periodsOrder = aggregation.clusterOrder
        # self.periodOccurances = aggregation.clusterPeriodNoOccur
        

        # ### Some plot for plausibility check ###
        
        import matplotlib.pyplot as plt        
        plt.figure(figsize=(8,6))
        plt.title('aggregated series (dashed = aggregated)')
        plt.plot(df_OriginalData.values)
        for label_TS, agg_values in dataAgg.totalTimeseries.items():
            # aLabel = str(i)
            # aLabel = self.TSlistForAggregation[i].label_full
            plt.plot(agg_values.values,'--', label = label_TS)
        if len(self.TSlistForAggregation) < 10: # wenn nicht zu viele
            plt.legend(bbox_to_anchor =(0.5,-0.05), loc='upper center')
        plt.show()                                            

        # ### Some infos as print ###
        
        print('TS Aggregation:')
        for i in  range(len(self.TSlistForAggregation)):
            aLabel = self.TSlistForAggregation[i].label_full
            print('TS ' + str(aLabel))
            print('  max_agg:' + str(max(dataAgg.totalTimeseries[aLabel])))
            print('  max_orig:' + str(max(df_OriginalData[aLabel])))
            print('  min_agg:' + str(min(dataAgg.totalTimeseries[aLabel])))
            print('  min_orig:' + str(min(df_OriginalData[aLabel])))
            print('  sum_agg:' + str(sum(dataAgg.totalTimeseries[aLabel])))
            print('  sum_orig:' + str(sum(df_OriginalData[aLabel])))
            
        print('addpeakmax:')
        print(self.TScollectionForAgg.addPeak_Max_labels)
        print('addpeakmin:')
        print(self.TScollectionForAgg.addPeak_Min_labels)
        
        # ################
        # ### Modeling ###
        
        aggregationModel = flixAgg.cAggregationModeling('aggregation',self.es,
                                                    indexVectorsOfClusters = dataAgg.indexVectorsOfClusters,
                                                    fixBinaryVarsOnly = fixBinaryVarsOnly, 
                                                    fixStorageFlows   = fixStorageFlows,
                                                    listOfMEsToClusterize = None,
                                                    percentageOfPeriodFreedom = percentageOfPeriodFreedom,
                                                    costsOfPeriodFreedom = costsOfPeriodFreedom)        
        
        # temporary Modeling-Element for equalizing indices of aggregation:
        self.es.addTemporaryElements(aggregationModel)
        
        if fixBinaryVarsOnly:
          TS_explicit = None
        else:
          # neue (Explizit)-Werte für TS sammeln::        
          TS_explicit = {}
          for i in range(len(self.TSlistForAggregation)):
            TS = self.TSlistForAggregation[i]
            # todo: agg-Wert für TS:
            TS_explicit[TS] = dataAgg.totalTimeseries[TS.label_full].values # nur data-array ohne Zeit   
        
              
        # ##########################
        # ## System finalizing: ##
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
    
        self._definePathNames(namePrefix, nameSuffix, aPath, saveResults, nrOfModBoxes=1)

        if self.calcType not in ['full','aggregated']:            
          raise Exception('calcType ' + self.calcType + ' needs no solve()-Command (only for ' + str())
        aModbox = self.listOfModbox[0]
        aModbox.solve(**solverProps, logfileName = self.paths_Log[0])
        
        if saveResults:
          self._saveSolveInfos()

        
    def _definePathNames(self, namePrefix, nameSuffix, aPath, saveResults, nrOfModBoxes=1):
        import datetime
        import pathlib               

        # wenn "/" am Anfang, dann löschen (sonst kommt pathlib durcheinander):
        aPath = aPath[0].replace("/","") + aPath[1:]
        # absoluter Pfad:
        aPath = pathlib.Path.cwd() / aPath
        # Pfad anlegen, fall noch nicht vorhanden:
        aPath.mkdir(parents=True, exist_ok=True)       
        self.pathForResults = aPath             
        
        timestamp = datetime.datetime.now()
        timestring = timestamp.strftime('%Y-%m-%d')                     
        self.nameOfCalc = namePrefix.replace(" ","") + timestring + '_' + self.label.replace(" ", "") + nameSuffix.replace(" ","")        
    
        if saveResults:
            filename_Data = self.nameOfCalc + '_data.pickle'
            filename_Info = self.nameOfCalc + '_solvingInfos.yaml'
            if nrOfModBoxes ==1:
                filenames_Log = [self.nameOfCalc + '_solver.log']
            else:
                filenames_Log = [(self.nameOfCalc + '_solver_' + str(i) + '.log') for i in range(nrOfModBoxes)]
                
            self.paths_Log = [self.pathForResults / filenames_Log[i] for i in range(nrOfModBoxes)]
            self.path_Data = self.pathForResults / filename_Data
            self.path_Info = self.pathForResults / filename_Info
        else:            
            self.paths_Log = None
            self.path_Data = None
            self.path_Info = None
    
    def checkIfAlreadyModeled(self):
      
      if self.calcType is not None:
        raise Exception('An other modeling-Method (calctype: ' + self.calcType + ') was already executed with this cCalculation-Object. \n Always create a new instance of cCalculation for new modeling/solving-command!')
  
      if self.es.AllTempMEs: # if some element in this list
          raise Exception('the Energysystem has some temporary modelingElements from previous calculation (i.g. aggregation-Modeling-Elements. These must be deleted before new calculation.')
          
    def _saveSolveInfos(self):
        import yaml
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
              aReferedVariable : cVariable_TS
              withEnd = isinstance(aReferedVariable, cVariable_TS) \
                  and aReferedVariable.activated_beforeValues \
                  and aReferedVariable.beforeValueIsStartValue 
                  
  
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
  
class cMEModel:
    '''
    is existing in every cME and owns eqs and vars of the activated calculation
    '''
    
    
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


class cME(cArgsClass):
    """
    Element mit Variablen und Gleichungen (ME = Modeling Element)
    -> besitzt Methoden, die jede Kindklasse ergänzend füllt:
    1. cME.finalize()          --> Finalisieren der Modell-Beschreibung (z.B. notwendig, wenn Bezug zu Elementen, die bei __init__ noch gar nicht bekannt sind)
    2. cME.declareVarsAndEqs() --> Variablen und Eqs definieren.
    3. cME.doModeling()        --> Modellierung
    4. cME.addShareToGlobals() --> Beitrag zu Gesamt-Kosten
    """
    modBox : cModelBoxOfES
    
    new_init_args = ['label']
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
                        allow_unicode=True))     

    def printVars(self, shiftChars):
        print(shiftChars + self.label + ':')
        print(yaml.dump(self.getVarsAsStr(),
                        allow_unicode=True))

    def getEqsVarsOverview(self):
        aDict = {}
        aDict['no eqs']          = len(self.mod.eqs)
        aDict['no eqs single']   = sum([eq.nrOfSingleEquations for eq in self.mod.eqs])
        aDict['no inEqs']        = len(self.mod.ineqs)
        aDict['no inEqs single'] = sum([ineq.nrOfSingleEquations for ineq in self.mod.ineqs])
        aDict['no vars']         = len(self.mod.variables)
        aDict['no vars single']  =  sum([var.len for var in self.mod.variables])  
        return aDict

    
class cEffectType(cME):
    '''
    Effect, i.g. costs, CO2 emissions, area, ...
    can be used later afterwards for allocating effects to compontents and flows.
    '''
  
    # isStandard -> Standard-Effekt (bei Eingabe eines skalars oder TS (statt dict) wird dieser automatisch angewendet)
    def __init__(self, label, unit, description, 
                 isStandard = False, 
                 isObjective = False, 
                 specificShareToOtherEffects_operation = {}, 
                 specificShareToOtherEffects_invest = {}, 
                 min_operationSum = None, max_operationSum = None, 
                 min_investSum = None, max_investSum = None,
                 min_Sum = None, max_Sum = None,
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
        isStandard : boolean, optional
            true, if Standard-Effect (for direct input of value without effect (alternatively to dict)) , else false
        isObjective : boolean, optional
            true, if optimization target
        specificShareToOtherEffects_operation : {effectType: TS, ...}, i.g. 180 €/t_CO2, input as {costs: 180}, optional
            share to other effects (only operation)
        specificShareToOtherEffects_invest : {effectType: TS, ...}, i.g. 180 €/t_CO2, input as {costs: 180}, optional
            share to other effects (only invest).
        min_operationSum : scalar, optional
            minimal sum (only operation) of the effect
        max_operationSum : scalar, optional
            maximal sum (nur operation) of the effect.
        min_investSum : scalar, optional
            minimal sum (only invest) of the effect
        max_investSum : scalar, optional
            maximal sum (only invest) of the effect
        min_Sum : sclalar, optional
            min sum of effect (invest+operation).
        max_Sum : scalar, optional
            max sum of effect (invest+operation).
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
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

        # Gleichung für Summe Operation und Invest:
        # eq: shareSum = effect.operation_sum + effect.operation_invest
        self.all.addVariableShare('operation', self, self.operation.mod.var_sum, 1, 1)
        self.all.addVariableShare('invest', self, self.invest   .mod.var_sum, 1, 1)
        self.all.doModeling(modBox, timeIndexe)
      
# ModelingElement (ME) Klasse zum Summieren einzelner Shares
# geht für skalar und TS
# z.B. invest.costs 


# Liste mit zusätzlicher Methode für Rückgabe Standard-Element:
class cEffectTypeList(list):
    '''
    internal effect list for simple handling of effects
    '''
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
    ''' 
    basic component class for all components
    '''    
    modBox : cModelBoxOfES
    new_init_args = ['label', 'on_valuesBeforeBegin', 'switchOnCosts', 'switchOn_maxNr', 'onHoursSum_min','onHoursSum_max', 'costsPerRunningHour']
    not_used_args = ['label']
    def __init__(self, label, on_valuesBeforeBegin = [0,0], switchOnCosts = None, switchOn_maxNr = None, onHoursSum_min = None, onHoursSum_max = None, costsPerRunningHour = None, **kwargs) :            
        '''
        

        Parameters
        ----------
        label : str
            name.
        
        Parameters of on/off-feature 
        ----------------------------
        (component is off, if all flows are zero!)

        on_valuesBeforeBegin :  array (TODO: why not scalar?)
            Ein(1)/Aus(0)-Wert vor Zeitreihe
        switchOnCosts : look in cFlow for description
        switchOn_maxNr : look in cFlow for description
        onHoursSum_min : look in cFlow for description
        onHoursSum_max : look in cFlow for description
        costsPerRunningHour : look in cFlow for description
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        label    =  helpers.checkForAttributeNameConformity(label)          # todo: indexierbar / eindeutig machen!      
        super().__init__(label, **kwargs)
        self.on_valuesBeforeBegin = on_valuesBeforeBegin  
        self.switchOnCosts        = transFormEffectValuesToTSDict('switchOnCosts'      , switchOnCosts       , self)
        self.switchOn_maxNr       = switchOn_maxNr
        self.onHoursSum_min       = onHoursSum_min
        self.onHoursSum_max       = onHoursSum_max
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
      self.featureOn = cFeatureOn(self, flowsDefiningOn, self.on_valuesBeforeBegin, self.switchOnCosts, self.costsPerRunningHour, onHoursSum_min = self.onHoursSum_min, onHoursSum_max = self.onHoursSum_max, switchOn_maxNr = self.switchOn_maxNr)            
    
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
      inhalt['class'] = type(self).__name__
        
      return descr
    
    def print(self,shiftChars):
      aFlow : cFlow
      print(yaml.dump(self.getDescrAsStr(), allow_unicode = True))
      


# komponenten übergreifende Gleichungen/Variablen/Zielfunktion!
class cGlobal(cME):
    ''' 
    storing global modeling stuff like effect equations and optimization target
    '''
  
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
      
    def addShareToOperation(self, nameOfShare, shareHolder, aVariable, effect_values, factor):
      if aVariable is None: raise Exception('addShareToOperation() needs variable or use addConstantShare instead')
      self.__addShare('operation', nameOfShare, shareHolder, effect_values, factor, aVariable)
  
    def addConstantShareToOperation(self, nameOfShare, shareHolder, effect_values, factor):
      self.__addShare('operation', nameOfShare, shareHolder, effect_values, factor)
      
    def addShareToInvest   (self, nameOfShare, shareHolder, aVariable, effect_values, factor):
      if aVariable is None: raise Exception('addShareToOperation() needs variable or use addConstantShare instead')
      self.__addShare('invest'   , nameOfShare, shareHolder, effect_values, factor, aVariable)
      
    def addConstantShareToInvest   (self, nameOfShare, shareHolder, effect_values, factor):
      self.__addShare('invest'   , nameOfShare, shareHolder, effect_values, factor)    
    
    # wenn aVariable = None, dann constanter Share
    def __addShare(self, operationOrInvest, nameOfShare, shareHolder, effect_values, factor, aVariable = None):
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
          effectType.operation.addShare(nameOfShare, shareHolder, aVariable, value, factor) # hier darf aVariable auch None sein!
        elif operationOrInvest == 'invest':
          effectType.invest   .addShare(nameOfShare, shareHolder, aVariable, value, factor) # hier darf aVariable auch None sein!
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
        nameOfShare = 'specificShareToOtherEffects_operation'# + effectType.label
        for effectTypeOfShare, specShare_TS in effectType.specificShareToOtherEffects_operation.items():               
          # Share anhängen (an jeweiligen Effekt):
          shareSum_op = effectTypeOfShare.operation
          shareSum_op : cFeature_ShareSum
          shareHolder = effectType
          shareSum_op.addVariableShare(nameOfShare, shareHolder, effectType.operation.mod.var_sum_TS, specShare_TS, 1)
        # 2. invest:    -> hier ist es Skalar (share)
        # alle specificSharesToOtherEffects durchgehen:
        nameOfShare = 'specificShareToOtherEffects_invest_'# + effectType.label
        for effectTypeOfShare, specShare in effectType.specificShareToOtherEffects_invest.items():                     
          # Share anhängen (an jeweiligen Effekt):
          shareSum_inv = effectTypeOfShare.invest
          shareSum_inv : cFeature_ShareSum
          shareHolder = effectType
          shareSum_inv.addVariableShare(nameOfShare, shareHolder, effectType.invest.mod.var_sum   , specShare   , 1)
                         
        
      
      # ####### target function  ###########       
      # Strafkosten immer:
      self.objective.addSummand(self.penalty  .mod.var_sum, 1) 
      
      # Definierter Effekt als Zielfunktion:
      objectiveEffect = self.listOfEffectTypes.objectiveEffect()
      if objectiveEffect is None : raise Exception('Kein Effekt als Zielfunktion gewählt!')
      self.objective.addSummand(objectiveEffect.operation.mod.var_sum, 1)
      self.objective.addSummand(objectiveEffect.invest   .mod.var_sum ,1)
            
class cBus(cBaseComponent): # sollte das wirklich geerbt werden oder eher nur cME???
    '''
    realizing balance of all linked flows
    (penalty flow is excess can be activated)
    '''
  
    # --> excessCostsPerFlowHour
    #        none/ 0 -> kein Exzess berücksichtigt
    #        > 0 berücksichtigt
      
    new_init_args = ['media', 'label', 'excessCostsPerFlowHour']
    not_used_args = ['label']    
    
    def __init__(self, media, label, excessCostsPerFlowHour = 1e5, **kwargs):   
        '''
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
        excessCostsPerFlowHour : none or scalar, array or cTSraw
            excess costs / penalty costs (bus balance compensation)
            (none/ 0 -> no penalty). The default is 1e5.
        **kwargs : TYPE
            DESCRIPTION.
        '''
        
        super().__init__(label,**kwargs)  
        if media is None: 
            self.media = media # alle erlaubt
        elif isinstance(media,str):
            self.media = {media} # convert to set
        elif isinstance(media,set):
            self.media = media
        else:
            raise Exception('no valid input for argument media!')
            
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
        # commonMedium = self.media & aFlow.medium
        # wenn leer, d.h. kein gemeinsamer Eintrag:    
        if (aFlow.medium is not None) and (self.media is not None) and \
            (not (aFlow.medium in self.media)):
            raise Exception('in cBus ' + self.label + ' : registerFlow(): medium \'' 
                            + str(aFlow.medium) + '\' of ' + aFlow.label_full + 
                            ' and media ' + str(self.media) + ' of bus ' + 
                            self.label_full + '  have no common medium!' + 
                            ' -> Check if the flow is connected correctly OR append flow-medium to the allowed bus-media in bus-definition! OR generally deactivat media-check by setting media in bus-definition to None'
                            ) 
  
    def declareVarsAndEqs(self, modBox):
      super().declareVarsAndEqs(modBox)
      # Fehlerplus/-minus:
      if self.withExcess:
        # Fehlerplus und -minus definieren
        self.excessIn  =  cVariable_TS('excessIn' , len(modBox.timeSeries), self, modBox, min=0)   
        self.excessOut =  cVariable_TS('excessOut', len(modBox.timeSeries), self, modBox, min=0)         
      
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
        globalComp.penalty.addVariableShare('excessCostsPerFlowHour', self, self.excessIn , self.excessCostsPerFlowHour, modBox.dtInHours)
        globalComp.penalty.addVariableShare('excessCostsPerFlowHour', self, self.excessOut, self.excessCostsPerFlowHour, modBox.dtInHours)
        # globalComp.penaltyCosts_eq.addSummand(self.excessIn , np.multiply(self.excessCostsPerFlowHour, modBox.dtInHours))
        # globalComp.penaltyCosts_eq.addSummand(self.excessOut, np.multiply(self.excessCostsPerFlowHour, modBox.dtInHours))
      
    def print(self,shiftChars):
      print(shiftChars + str(self.label) + ' - ' + str(len(self.inputs)) + ' In-Flows / ' +  str(len(self.outputs)) + ' Out-Flows registered' )        
      
      print(shiftChars + '   medium: ' + str(self.medium))
      super().print(shiftChars)


 # Medien definieren:
class cMediumCollection:
    '''
    define possible domains for flow (not tested!) TODO!
    '''
    # single medium:
    heat    = 'heat' # set(['heat'])
    # cold    = set(['cold'])
    el      = 'el' # set(['el'])
    # gas     = set(['gas'])
    # lignite = set(['lignite'])
    # biomass = set(['biomass'])
    # ash     = set(['ash'])
    # groups: 
    fuel      = 'fuel' # gas | lignite | biomass
    # fossil_fu = gas | lignite
    
    # # neues Medium hinzufügen:
    # def addMedium(attrName, aSetOfStrs):
    #     cMediumCollection.setattr(attrName,aSetOfStrs)
      
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
    '''
    flows are inputs and outputs of components
    '''
    

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
    
    def __init__(self,label, 
                 bus:cBus=None , 
                 min_rel = 0, max_rel = 1, 
                 nominal_val = __nominal_val_default , 
                 loadFactor_min = None, loadFactor_max = None, 
                 positive_gradient = None, 
                 costsPerFlowHour = None ,
                 iCanSwitchOff = True,
                 onHoursSum_min = None, onHoursSum_max= None, 
                 onHours_min = None, onHours_max = None,
                 offHours_min = None, offHours_max = None,
                 switchOnCosts = None, 
                 switchOn_maxNr = None, 
                 costsPerRunningHour =None, 
                 sumFlowHours_max = None, sumFlowHours_min = None, 
                 valuesBeforeBegin = [0,0], 
                 val_rel = None, 
                 medium = None,
                 investArgs = None, 
                 **kwargs):
        '''
        Parameters
        ----------
        label : str
            name of flow
        bus : cBus, optional
            bus to which flow is linked
        min_rel : scalar, array, cTSraw, optional
            min value is min_rel multiplied by nominal_val
        max_rel : scalar, array, cTSraw, optional
            max value is max_rel multiplied by nominal_val. If nominal_val = max then max_rel=1
        nominal_val : scalar. None if is a nominal value is a opt-variable, optional
            nominal value/ invest size (linked to min_rel, max_rel and others). 
            i.g. kW, area, volume, pieces, 
            möglichst immer so stark wie möglich einschränken 
            (wg. Rechenzeit bzw. Binär-Ungenauigkeits-Problem!)
        loadFactor_min : scalar, optional
            minimal load factor  general: avg Flow per nominalVal/investSize 
            (e.g. boiler, kW/kWh=h; solarthermal: kW/m²; 
             def: :math:`load\_factor:= sumFlowHours/ (nominal\_val \cdot \Delta t_{tot})`
        loadFactor_max : scalar, optional
            maximal load factor (see minimal load factor)
        positive_gradient : TYPE, optional
           not implemented yet
        costsPerFlowHour : scalar, array, cTSraw, optional
            operational costs, costs per flow-"work"
        iCanSwitchOff : boolean, optional
            can flow be off, i.e. be zero (only relevant if min_rel > 0) 
            Then binary var is used.
        onHoursSum_min : scalar, optional
            min. overall sum of operating hours.
        onHoursSum_max : scalar, optional
            max. overall sum of operating hours.
        onHours_min : scalar, optional
            min sum of operating hours in one piece
        onHours_max : scalar, optional
            max sum of operating hours in one piece
        offHours_min : scalar, optional
            - not implemented yet - 
            min sum of non-operating hours in one piece
        offHours_max : scalar, optional
            - not implemented yet - 
            max sum of non-operating hours in one piece
        switchOnCosts : scalar, array, cTSraw, optional
            cost of one switch from off (var_on=0) to on (var_on=1), 
            unit i.g. in Euro
        switchOn_maxNr : integer, optional
            max nr of switchOn operations
        costsPerRunningHour : costs-types, optional
            costs for operating, i.g. in € per hour
        sumFlowHours_max : TYPE, optional
            maximum flow-hours ("flow-work") 
            (if nominal_val is not const, maybe loadFactor_max fits better for you!)
        sumFlowHours_min : TYPE, optional
            minimum flow-hours ("flow-work") 
            (if nominal_val is not const, maybe loadFactor_min fits better for you!)
        valuesBeforeBegin : list (TODO: why not scalar?), optional
            Flow-value before begin (for calculation of i.g. switchOn for first time step, gradient for first time step ,...)'), 
            # TODO: integration of option for 'first is last'
        val_rel : scalar, array, cTSraw, optional
            fixed relative values for flow (if given). 
            val(t) := val_rel(t) * nominal_val(t)
            With this value, the flow-value is no opt-variable anymore;
            (min_rel u. max_rel are making sense anymore)
            used for fixed load profiles, i.g. heat demand, wind-power, solarthermal
            If the load-profile is just an upper limit, use max_rel instead.
        medium: string, None
            medium is relevant, if the linked bus only allows a special defined set of media.
            If None, any bus can be used.            
        investArgs : None or cInvestargs, optional
            used for investment costs or/and investment-optimization!
        '''
        

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
        self.onHoursSum_min      = onHoursSum_min
        self.onHoursSum_max      = onHoursSum_max
        self.onHours_min         = None if (onHours_min is None) else cTS_vector('onHours_min', onHours_min, self)
        self.onHours_max         = None if (onHours_max is None) else cTS_vector('onHours_max', onHours_max, self)
        self.offHours_min        = None if (offHours_min is None) else cTS_vector('offHours_min', offHours_min, self)
        self.offHours_max        = None if (offHours_max is None) else cTS_vector('offHours_max', offHours_max, self)
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
        if (medium is not None) and (not isinstance(medium, str)):
            raise Exception('medium must be a string or None')
        else:
            self.medium = medium
        # defaults:
                          
        # Wenn Min-Wert > 0 wird binäre On-Variable benötigt (nur bei flow!):
        self.__useOn_fromProps = iCanSwitchOff & (min_rel > 0)
        
                       
        # self.prepared          = False # ob __declareVarsAndEqs() ausgeführt
    
        # feature for: On and SwitchOn Vars (builds only if necessary)
        # -> feature bereits hier, da andere Elemente featureOn.activateOnValue() nutzen wollen
        flowsDefiningOn      = [self] # Liste. Ich selbst bin der definierende Flow! (Bei Komponente sind es hingegen alle in/out-flows)      
        on_valuesBeforeBegin = 1 * (self.valuesBeforeBegin >= 0.0001 ) # TODO: besser wäre modBox.epsilon, aber hier noch nicht bekannt!)       
        # TODO: Wenn iCanSwitchOff = False und min > 0, dann könnte man var_on fest auf 1 setzen um Rechenzeit zu sparen
        
        self.featureOn = cFeatureOn(self, flowsDefiningOn,
                                    on_valuesBeforeBegin, 
                                    self.switchOnCosts, 
                                    self.costsPerRunningHour, 
                                    onHoursSum_min = self.onHoursSum_min, 
                                    onHoursSum_max = self.onHoursSum_max, 
                                    onHours_min    = self.onHours_min,
                                    onHours_max    = self.onHours_max,
                                    offHours_min   = self.offHours_min,
                                    offHours_max   = self.offHours_max,
                                    switchOn_maxNr = self.switchOn_maxNr, 
                                    useOn_explicit = self.__useOn_fromProps)
    
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
        self.mod.var_val = cVariable_TS('val', modBox.nrOfTimeSteps, self, modBox, min = lb, max = ub, value = fix_value)
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
          shareHolder = self
          globalComp.addShareToOperation('costsPerFlowHour', shareHolder, self.mod.var_val, self.costsPerFlowHour, modBox.dtInHours)
            
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
    
    # Preset medium (only if not setted explicitly by user)
    def setMediumIfNotSet(self, medium):
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




