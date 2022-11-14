# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 10:23:49 2021

@author: Panitz
"""
# -*- coding: utf-8 -*-

"""
Modul zur Berechnung eines Energiesystemmodells.
"""

# Paket-Importe
#from component import Storage, Trading, Converter
import pandas as pd
import numpy as np
import tsam.timeseriesaggregation as tsam
import pyomo.environ as pyo
import pyomo.opt as opt
from pyomo.util.infeasible import log_infeasible_constraints
import time
from datetime import datetime
import yaml
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import copy


class flixAggregation:
    """
    Klasse EnergySystemModel
    """

    def __init__(self,
                 name,
                 timeseries,
                 hoursPerTimeStep,
                 hoursPerPeriod,
                 hasTSA=False,
                 noTypicalPeriods=8,
                 useExtremePeriods=True,
                 weightDict = None,
                 addPeakMax = None,
                 addPeakMin = None
                 ):

        """ 
        Konstruktor für die Klasse EnergySystemModel
        """

        self.results = None

        self.name = name
        self.timeseries = copy.deepcopy(timeseries)
        self.weightDict = weightDict
        self.addPeakMax = addPeakMax   
        self.addPeakMin = addPeakMin
        self.hoursPerTimeStep = hoursPerTimeStep
        self.hoursPerPeriod = hoursPerPeriod
        self.hasTSA = hasTSA
        self.noTypicalPeriods = noTypicalPeriods
        self.useExtremePeriods = useExtremePeriods

        # Wenn Extremperioden eingebunden werden sollen, nutze die Methode 'new_cluster_center' aus tsam
        self.extremePeriodMethod = 'None'
        if self.useExtremePeriods:            
            self.extremePeriodMethod = 'new_cluster_center'
            # check:
            if self.addPeakMax == [] and self.addPeakMin == []:
                raise Exception('no addPeakMax or addPeakMin timeseries given for extremePeriods!')

        # Initiales Setzen von Zeitreiheninformationen; werden überschrieben, falls Zeitreihenaggregation
        self.numberOfTimeSteps = len(self.timeseries.index)
        self.numberOfTimeStepsPerPeriod = self.numberOfTimeSteps
        self.totalPeriods = 1

        # Timeseries Index anpassen, damit gesamter Betrachtungszeitraum als eine lange Periode + entsprechender Zeitschrittanzahl interpretiert wird
        self.timeseriesIndex = self.timeseries.index  # ursprünglichen Index in Array speichern für späteres Speichern
        periodIndex, stepIndex = [], []
        for ii in range(0, self.numberOfTimeSteps):
            periodIndex.append(0)
            stepIndex.append(ii)
        self.timeseries.index = pd.MultiIndex.from_arrays([periodIndex, stepIndex],
                                                          names=['Period', 'TimeStep'])

        # Setzen der Zeitreihendaten für Modell
        # werden später überschrieben, falls Zeitreihenaggregation
        self.typicalPeriods = [0]
        self.periods, self.periodsOrder, self.periodOccurances = [0], [0], [1]
        self.totalTimeSteps = list(range(self.numberOfTimeSteps))  # gesamte Anzahl an Zeitschritten
        self.timeStepsPerPeriod = list(range(self.numberOfTimeSteps))  # Zeitschritte pro Periode, ohne ZRA = gesamte Anzahl
        self.interPeriodTimeSteps = list(range(int(len(self.totalTimeSteps) / len(self.timeStepsPerPeriod)) + 1))


    def cluster(self):
       
        """
        Durchführung der Zeitreihenaggregation
        """


        tClusterStart = time.time()

        # Neu berechnen der numberOfTimeStepsPerPeriod
        self.numberOfTimeStepsPerPeriod = int(self.hoursPerPeriod / self.hoursPerTimeStep)

       # Erstellen des aggregation objects
        aggregation = tsam.TimeSeriesAggregation(self.timeseries,
                                                 noTypicalPeriods=self.noTypicalPeriods,
                                                 hoursPerPeriod=self.hoursPerPeriod,
                                                 resolution=self.hoursPerTimeStep,
                                                 clusterMethod='k_means',
                                                 extremePeriodMethod=self.extremePeriodMethod, #flixi: 'None'/'new_cluster_center'
                                                 weightDict=self.weightDict,    
                                                 addPeakMax = self.addPeakMax, #['P_Netz/MW', 'Q_Netz/MW', 'Strompr.€/MWh'],
                                                 addPeakMin= self.addPeakMin
                                                 #addPeakMin=['Strompr.€/MWh']
                                                 )
        self.aggregation = aggregation
        # Ausführen der Aggregation/Clustering
        res_data = aggregation.createTypicalPeriods()
        # res_data.index = pd.MultiIndex.from_arrays([newPeriodIndex, newStepIndex],
        #                                            names=['Period', 'TimeStep'])        

        # Hochrechnen der aggregierten Daten, um sie später zu speichern
        predictedPeriods = aggregation.predictOriginalData()
        
        # ERGEBNISSE:
        self.totalTimeseries   = predictedPeriods
        self.totalTimeseries_t = self.totalTimeseries.set_index(self.timeseriesIndex,inplace=False)# neue DF erstellen

        periodsIndexVectorsOfClusters = {}
               
        ###############
        # Zuordnung der Indexe erstellen: 
        # {cluster 0: [index_vector_3, index_vector_7]
        #  cluster 1: [index_vector_1]
        #  cluster 2: ...}
        
        self.indexVectorsOfClusters = self.getIndexVectorsOfClusters()
        # print(self.indexVectorsOfClusters)        
        self.printDescriptionOfClusters()
        
        ##############
                       


        # Überschreiben der Zeitreiheninformationen
        # self.totalPeriods = int((self.numberOfTimeSteps * self.hoursPerTimeStep) / self.hoursPerPeriod)
        # self.typicalPeriods = aggregation.clusterPeriodIdx
        # self.periodsOrder = aggregation.clusterOrder
        # self.periodOccurances = aggregation.clusterPeriodNoOccur


        # self.timeStepsPerPeriod = list(range(self.numberOfTimeStepsPerPeriod))
        # self.periods = list(range(int(len(self.totalTimeSteps) / len(self.timeStepsPerPeriod))))

        
        # Zeit messen:
        tClusterEnd = time.time()
        self.tCluster = tClusterEnd - tClusterStart


    def getIndexVectorsOfClusters(self):
        ###############
        # Zuordnung der Indexe erstellen: 
        # {cluster 0: [index_vector_3, index_vector_7]
        #  cluster 1: [index_vector_1]
        #  cluster 2: ...} 
        
        # Beachte: letzte Periode muss nicht vollgefüllt sein!
        clusterList = self.aggregation.clusterPeriodNoOccur.keys()
        # Leerer Dict:
        indexVectorsOfClusters = {cluster: [] for cluster in clusterList}
        period_len = len(self.aggregation.stepIdx)
        stepIdx = np.array(self.aggregation.stepIdx) # nur Umwandlen in array
        for period in range(len(self.aggregation.clusterOrder)):
          clusterNr = self.aggregation.clusterOrder[period]
          
          periodStartIndex = period*period_len
          periodEndIndex   = min(periodStartIndex + period_len-1, len(self.timeseries)-1) # Beachtet auch letzte Periode
          indexVector = np.array(range(periodStartIndex,periodEndIndex+1))
          
          indexVectorsOfClusters[clusterNr].append(indexVector)      

        return indexVectorsOfClusters

    def printDescriptionOfClusters(self):      
        print ('#########################')        
        print ('###### Clustering #######')
        print('periodsOrder:')
        print(self.aggregation.clusterOrder)
        print('clusterPeriodNoOccur:')
        print(self.aggregation.clusterPeriodNoOccur)      
        print('indexVectorsOfClusters:')
        aVisual = {}
        for cluster in self.indexVectorsOfClusters.keys():
          aVisual[cluster] = [str(indexVector[0]) + '...' + str(indexVector[-1]) for indexVector in self.indexVectorsOfClusters[cluster]]
        print (aVisual)                        
        print ('########################')
        
        if self.useExtremePeriods:
          print('extremePeriods:')
          # Zeitreihe rauslöschen:                        
          extremePeriods = self.aggregation.extremePeriods.copy() 
          for key,val in extremePeriods.items():
            del(extremePeriods[key]['profile'])
          print(extremePeriods)
          print ('########################')            
        
    def declareTimeSets(self):
        """ 
        Deklarieren der timeSets, die alle Zeitschritte als Periode + Timestep enthalten
        timeSet enthält alle Zeitschritte
        interTimeStepsSet enthält alle Schritte zwischen den Zeitschritten -> für Speicher
        """



    def addTimeseriesData(self):
        """
        Geclusterte Zeitreihen über die timeSets zugänglich machen
        """
        # self.loadTh = dict(zip(model.timeSet, pd.Series(self.timeseries['Q_Netz/MW'])))
        # self.loadEl = dict(zip(model.timeSet, pd.Series(self.timeseries['P_Netz/MW'])))
        # self.priceGas = dict(zip(model.timeSet, pd.Series(self.timeseries['Gaspr.HWA€/MWh'])))
        # self.pricePower = dict(zip(model.timeSet, pd.Series(self.timeseries['Strompr.€/MWh'])))
        # self.pricePowerIn = dict(zip(model.timeSet, pd.Series(self.timeseries['Strompr.€/MWh'] + 0.5)))
        # self.pricePowerOut = dict(zip(model.timeSet, pd.Series(self.timeseries['Strompr.€/MWh'] - 0.5)))
        # self.tradingPrices = dict(zip(model.tradingSet.data(), [self.pricePowerIn, self.pricePowerOut]))
        # self.priceCoal = dict(zip(model.timeSet, pd.Series(4.6 for x in range(len(self.timeseries.index)))))

        # self.fuelCosts['Kohle'] = self.priceCoal
        # self.fuelCosts['Gas'] = self.priceGas


    def saveResults(self,addTimeStamp = True):
        """
        Speichern
        """

        timestamp = datetime.now()
        timestring = timestamp.strftime('%Y-%m-%d')
        
        # Rahmenbedingungen und Ergebnisse speichern
        periodsOrderString = ','.join(map(str, self.periodsOrder))

        # Anzahl Perioden mit ggfs. Extremperioden
        noPer = self.noTypicalPeriods
        if self.useExtremePeriods:
            noPer += 4

        parameterDict = {
            'date': timestring,
            'name': self.name,
            'start': str(self.timeseriesIndex[0]),
            'end': str(self.timeseriesIndex[-1]),
            'hoursPerTimeStep': self.hoursPerTimeStep,
            'useExtremePeriods': self.useExtremePeriods,
            'hasTSA': self.hasTSA,
            'hoursPerPeriod': self.hoursPerPeriod,
            'noTypicalPeriods': noPer,
            'periodsOrder': periodsOrderString,
         #   'periodsOrder_raw': list(self.periodsOrder),
            'cluster time': self.tCluster,


            #'best bound': float(str(self.results['Problem']).split('\n')[2].split(' ')[4]),
            #'best objective': float(str(self.results['Problem']).split('\n')[3].split(' ')[4])
        }

        for key,value in {'a':1,'b':2,'c':3}.items():
            parameterDict[key] = value

        filename = self.name.replace(" ", "") + '_Parameters.yaml'
        if addTimeStamp : filename = timestring + '_' + filename
        yamlPath = './results/Parameters/' + filename
        with open(yamlPath, 'w') as f:
            yaml.dump(parameterDict, f)

    def saveAgg(self,addTimeStamp = True):
        """
        Speichern der aggregierten Zeitreihen.
        """
        timestamp = datetime.now()
        timestring = timestamp.strftime('%Y-%m-%d')
        
        
        
        filename =  self.name.replace(" ", "") + '_Timeseries.csv'        
        if addTimeStamp : filename = timestring + '_' + filename                     
                       
        pathAgg = './results/aggTimeseries/'
        self.totalTimeseries.to_csv(pathAgg + filename)



import flixStructure
import flixComps
from basicModeling import *
# ModelingElement mit Zusatz-Glg. und Variablen für aggregierte Berechnung
class cAggregationModeling(flixStructure.cME):     
  def __init__(self,label, es, indexVectorsOfClusters, fixStorageFlows = True, fixBinaryVarsOnly=True, listOfMEsToClusterize = None, percentageOfPeriodFreedom = 0, costsOfPeriodFreedom = 0, **kwargs):
      '''
      

      Parameters
      ----------
      label : TYPE
          DESCRIPTION.
      es : TYPE
          DESCRIPTION.
      indexVectorsOfClusters : TYPE
          DESCRIPTION.
      fixStorageFlows : TYPE, optional
          DESCRIPTION. The default is True.
      fixBinaryVarsOnly : TYPE, optional
          DESCRIPTION. The default is True.
      listOfMEsToClusterize : TYPE, optional
          DESCRIPTION. The default is None.
      percentageOfPeriodFreedom : TYPE, optional
          DESCRIPTION. The default is 0.
      costsOfPeriodFreedom : TYPE, optional
          DESCRIPTION. The default is 0.
      **kwargs : TYPE
          DESCRIPTION.

      Returns
      -------
      None.

      '''
      es: flixStructure.cEnergySystem           
      self.es = es
      self.indexVectorsOfClusters = indexVectorsOfClusters
      self.fixStorageFlows = fixStorageFlows
      self.fixBinaryVarsOnly = fixBinaryVarsOnly
      self.listOfMEsToClusterize = listOfMEsToClusterize
      
      self.percentageOfPeriodFreedom = percentageOfPeriodFreedom
      self.costsOfPeriodFreedom = costsOfPeriodFreedom
      
      super().__init__(label, **kwargs)   
      # args to attributes:
      
      self.var_K_list = []  

      # self.subElements.append(self.featureOn)  
  
  
  def finalize(self):
    super().finalize()

    
  def declareVarsAndEqs(self, modBox:flixStructure.cModelBoxOfES):
    super().declareVarsAndEqs(modBox)
   
      
  def doModeling(self,modBox:flixStructure.cModelBoxOfES,timeIndexe):        

    if self.listOfMEsToClusterize is None:
      # Alle:
      compSet = set(self.es.listOfComponents)
      flowSet = self.es.setOfFlows
    else:
      # Ausgewählte:
      compSet = set(self.listOfMEsToClusterize)
      flowSet = self.es.getFlows(listOfMEsToClusterize)
    

    
    flow : flixStructure.cFlow
    
    # todo: hier anstelle alle MEs durchgehen, nicht nur flows und comps:
    for aME in flowSet | compSet :
      # Wenn StorageFlows nicht gefixt werden sollen und flow ein storage-Flow ist:      
      if (not self.fixStorageFlows) and hasattr(aME,'comp') and (isinstance(aME.comp, flixComps.cStorage)):
          pass # flow hier nicht fixen!
      else:              
        # On-Variablen:
        if aME.mod.var_on is not None:
          aVar = aME.mod.var_on
          aEq = self.getEqForLinkedIndexe(aVar, modBox,fixFirstIndexOfPeriod = True)       
        # SwitchOn-Variablen:
        if aME.mod.var_switchOn is not None:
          aVar = aME.mod.var_switchOn 
          # --> hier ersten Index weglassen:
          aEq = self.getEqForLinkedIndexe(aVar, modBox,fixFirstIndexOfPeriod = False)       
        if aME.mod.var_switchOff is not None:
          aVar = aME.mod.var_switchOff 
          # --> hier ersten Index weglassen:
          aEq = self.getEqForLinkedIndexe(aVar, modBox,fixFirstIndexOfPeriod = False)       
        
        # todo: nicht schön! Zugriff muss über alle cTSVariablen erfolgen!        
        # Nicht-Binär-Variablen:        
        if not self.fixBinaryVarsOnly:
          # Value-Variablen:
          if hasattr(aME.mod,'var_val'):
            aVar = aME.mod.var_val
            aEq = self.getEqForLinkedIndexe(aVar, modBox,fixFirstIndexOfPeriod = True)       
          
  
  def getEqForLinkedIndexe(self,aVar,modBox,fixFirstIndexOfPeriod):
    aVar : cVariable
    # todo!: idx_var1/2 wird jedes mal gemacht! nicht schick
    
    # Gleichung: 
    # eq1: x(p1,t) - x(p3,t) = 0 # wobei p1 und p3 im gleichen Cluster sind und t = 0..N_p
    idx_var1 = np.array([])
    idx_var2 = np.array([])
    for clusterNr in self.indexVectorsOfClusters.keys():
      listOfIndexVectors = self.indexVectorsOfClusters[clusterNr]
      # alle Indexvektor-Tupel durchgehen:
      for i in range(len(listOfIndexVectors)-1): # ! Nur wenn cluster mehr als eine Periode enthält:        
        # Falls eine Periode nicht ganz voll (eigl. nur bei letzter Periode möglich)  
        v1 = listOfIndexVectors[0]
        v2 = listOfIndexVectors[i+1]
        if not fixFirstIndexOfPeriod : 
          v1 = v1[1:] # erstes Element immer weglassen
          v2 = v2[1:]
        minLen = min(len(v1),len(v2))
        idx_var1 = np.append(idx_var1, v1[:minLen])
        idx_var2 = np.append(idx_var2, v2[:minLen])
  
    eq = flixStructure.cEquation('equalIdx_' + aVar.label_full, self , modBox, eqType = 'eq')
    eq.addSummand(aVar,  1,  indexeOfVariable = idx_var1)
    eq.addSummand(aVar, -1,  indexeOfVariable = idx_var2)    
    
    
    # Korrektur: (bisher nur für Binärvariablen:)
    if aVar.isBinary and self.percentageOfPeriodFreedom > 0:
      # correction-vars (so viele wie Indexe in eq:)
      var_K1 = cVariable('Korr1_' + aVar.label_full.replace('.','_'), eq.nrOfSingleEquations, self , modBox, isBinary = True)
      var_K0 = cVariable('Korr0_' + aVar.label_full.replace('.','_'), eq.nrOfSingleEquations, self , modBox, isBinary = True)
      # equation extends ... 
      # --> On(p3) can be 0/1 independent of On(p1,t)!
      # eq1: On(p1,t) - On(p3,t) + K1(p3,t) - K0(p3,t) = 0 
      # --> correction On(p3) can be:
      #  On(p1,t) = 1 -> On(p3) can be 0 -> K0=1 (,K1=0)
      #  On(p1,t) = 0 -> On(p3) can be 1 -> K1=1 (,K0=1)
      eq.addSummand(var_K1, +1)    
      eq.addSummand(var_K0, -1)    
      self.var_K_list.append(var_K1)
      self.var_K_list.append(var_K0)
    
    
      # interlock var_K1 and var_K2:
      # eq: var_K0(t)+var_K1(t) <= 1.1
      eq_lock = flixStructure.cEquation('lock_K0andK1' + aVar.label_full, self , modBox, eqType = 'ineq')
      eq_lock.addSummand(var_K0,1)
      eq_lock.addSummand(var_K1,1)
      eq_lock.addRightSide(1.1)          
    
      # Begrenzung der Korrektur-Anzahl:
      # eq: sum(K) <= n_Corr_max
      self.noOfCorrections = round(self.percentageOfPeriodFreedom/100 * var_K1.len)              
      eq_max = flixStructure.cEquation('maxNoOfCorrections_' + aVar.label_full, self , modBox, eqType = 'ineq')
      eq_max.addSummandSumOf(var_K1, 1)
      eq_max.addSummandSumOf(var_K0, 1)
      eq_max.addRightSide(self.noOfCorrections)  # Maximum  
    return eq

    
  def addShareToGlobals(self,globalComp:flixStructure.cGlobal,modBox) :       
    
    # einzelne Stellen korrigierbar machen (aber mit Kosten)
    if (self.percentageOfPeriodFreedom > 0) & (self.costsOfPeriodFreedom!= 0):
      for var_K in self.var_K_list:  
        # todo: Krücke, weil muss eigentlich sowas wie Strafkosten sein!!!     
        globalComp.objective.addSummandSumOf(var_K, self.costsOfPeriodFreedom)



