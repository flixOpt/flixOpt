# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 10:23:49 2021
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""
# -*- coding: utf-8 -*-

"""
Modul zur aggregierten Berechnung eines Energiesystemmodells.
"""

# Paket-Importe
# from component import Storage, Trading, Converter
from datetime import datetime
import copy
import time
from typing import Optional, List, Dict
import warnings

import pandas as pd
import numpy as np
import tsam.timeseriesaggregation as tsam
import pyomo.environ as pyo
import pyomo.opt as opt
from pyomo.util.infeasible import log_infeasible_constraints
import yaml

from flixOpt.flixBasics import Skalar, TimeSeries

warnings.filterwarnings("ignore", category=DeprecationWarning)




class Aggregation:
    """
    aggregation organizing class
    """

    def __init__(self,
                 name: str,
                 timeseries: pd.DataFrame,
                 hours_per_time_step: Skalar,
                 hours_per_period: Skalar,
                 hasTSA=False,  # TODO: Remove unused parameter
                 nr_of_typical_periods: int = 8,
                 use_extreme_periods: bool = True,
                 weights: Optional[dict] = None,
                 addPeakMax: Optional[List[TimeSeries]] =None,
                 addPeakMin: Optional[List[TimeSeries]] = None
                 ):

        """ 
        Konstruktor für die Klasse EnergySystemModel
        """

        self.results = None
        self.aggregation: Optional[tsam.TimeSeriesAggregation] = None

        self.name = name
        self.timeseries = copy.deepcopy(timeseries)
        self.hours_per_time_step = hours_per_time_step
        self.hours_per_period = hours_per_period
        self.hasTSA = hasTSA
        self.nr_of_typical_periods = nr_of_typical_periods
        self.use_extreme_periods = use_extreme_periods
        self.weights = weights
        self.addPeakMax = addPeakMax
        self.addPeakMin = addPeakMin

        # Wenn Extremperioden eingebunden werden sollen, nutze die Methode 'new_cluster_center' aus tsam
        self.extreme_period_method = 'new_cluster_center' if self.use_extreme_periods else 'None'
        if self.use_extreme_periods and not (self.addPeakMax or self.addPeakMin):   # Check
            raise Exception('addPeakMax or addPeakMin timeseries given if useExtremValues=True!')

        # Initiales Setzen von Zeitreiheninformationen; werden überschrieben, falls Zeitreihenaggregation
        self.nr_of_time_steps = len(self.timeseries.index)
        self.nr_of_time_steps_per_period = self.nr_of_time_steps   # TODO: Remove unused parameter
        self.total_periods = 1      # TODO: Remove unused parameter

        # Timeseries Index anpassen, damit gesamter Betrachtungszeitraum als eine lange Periode + entsprechender Zeitschrittanzahl interpretiert wird
        self.original_timeseries_index = self.timeseries.index  # ursprünglichen Index in Array speichern für späteres Speichern
        period_index = [0] * self.nr_of_time_steps
        step_index = list(range(self.nr_of_time_steps))
        self.timeseries.index = pd.MultiIndex.from_arrays([period_index, step_index],
                                                          names=['Period', 'TimeStep'])

        # Setzen der Zeitreihendaten für Modell
        # werden später überschrieben, falls Zeitreihenaggregation
        self.typical_periods = [0]
        self.periods, self.periods_order, self.period_occurrences = [0], [0], [1]
        self.total_time_steps = list(range(self.nr_of_time_steps))  # gesamte Anzahl an Zeitschritten
        self.time_steps_per_period = list(range(self.nr_of_time_steps))  # Zeitschritte pro Periode, ohne ZRA = gesamte Anzahl
        self.inter_period_time_steps = list(range(int(len(self.total_time_steps) / len(self.time_steps_per_period)) + 1))

    def cluster(self):

        """
        Durchführung der Zeitreihenaggregation
        """

        tClusterStart = time.time()

        # Neu berechnen der nr_of_time_steps_per_period
        self.nr_of_time_steps_per_period = int(self.hours_per_period / self.hours_per_time_step)

        # Erstellen des aggregation objects
        self.aggregation = tsam.TimeSeriesAggregation(self.timeseries,
                                                      noTypicalPeriods=self.nr_of_typical_periods,
                                                      hoursPerPeriod=self.hours_per_period,
                                                      resolution=self.hours_per_time_step,
                                                      clusterMethod='k_means',
                                                      extremePeriodMethod=self.extreme_period_method,
                                                      weightDict=self.weights,
                                                      addPeakMax=self.addPeakMax,
                                                      addPeakMin=self.addPeakMin
                                                      )

        self.aggregation.createTypicalPeriods()   # Ausführen der Aggregation/Clustering
        predictedPeriods = self.aggregation.predictOriginalData()   # Hochrechnen der aggregierten Daten, um sie später zu speichern

        # ERGEBNISSE:
        self.aggregated_timeseries = predictedPeriods
        self.aggregated_timeseries_t = self.aggregated_timeseries.set_index(self.original_timeseries_index,
                                                                            inplace=False)  # neue DF erstellen

        self.indexVectorsOfClusters = self.getIndexVectorsOfClusters()
        self.printDescriptionOfClusters()

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
        stepIdx = np.array(self.aggregation.stepIdx)  # nur Umwandlen in array
        for period in range(len(self.aggregation.clusterOrder)):
            clusterNr = self.aggregation.clusterOrder[period]

            periodStartIndex = period * period_len
            periodEndIndex = min(periodStartIndex + period_len - 1,
                                 len(self.timeseries) - 1)  # Beachtet auch letzte Periode
            indexVector = np.array(range(periodStartIndex, periodEndIndex + 1))

            indexVectorsOfClusters[clusterNr].append(indexVector)

        return indexVectorsOfClusters

    def printDescriptionOfClusters(self):
        print('#########################')
        print('###### Clustering #######')
        print('periods_order:')
        print(self.aggregation.clusterOrder)
        print('clusterPeriodNoOccur:')
        print(self.aggregation.clusterPeriodNoOccur)
        print('indexVectorsOfClusters:')
        aVisual = {}
        for cluster in self.indexVectorsOfClusters.keys():
            aVisual[cluster] = [str(indexVector[0]) + '...' + str(indexVector[-1]) for indexVector in
                                self.indexVectorsOfClusters[cluster]]
        print(aVisual)
        print('########################')

        if self.use_extreme_periods:
            print('extremePeriods:')
            # Zeitreihe rauslöschen:
            extremePeriods = self.aggregation.extremePeriods.copy()
            for key, val in extremePeriods.items():
                del (extremePeriods[key]['profile'])
            print(extremePeriods)
            print('########################')

    def saveResults(self, addTimeStamp=True):
        """
        Speichern
        """

        timestamp = datetime.now()
        timestring = timestamp.strftime('%Y-%m-%d')

        # Rahmenbedingungen und Ergebnisse speichern
        periodsOrderString = ','.join(map(str, self.periods_order))

        # Anzahl Perioden mit ggfs. Extremperioden
        noPer = self.nr_of_typical_periods
        if self.use_extreme_periods:
            noPer += 4

        parameterDict = {
            'date': timestring,
            'name': self.name,
            'start': str(self.original_timeseries_index[0]),
            'end': str(self.original_timeseries_index[-1]),
            'hours_per_time_step': self.hours_per_time_step,
            'use_extreme_periods': self.use_extreme_periods,
            'hasTSA': self.hasTSA,
            'hours_per_period': self.hours_per_period,
            'nr_of_typical_periods': noPer,
            'periods_order': periodsOrderString,
            #   'periodsOrder_raw': list(self.periods_order),
            'cluster time': self.tCluster,

            # 'best bound': float(str(self.results['Problem']).split('\n')[2].split(' ')[4]),
            # 'best objective': float(str(self.results['Problem']).split('\n')[3].split(' ')[4])
        }

        for key, value in {'a': 1, 'b': 2, 'c': 3}.items():
            parameterDict[key] = value

        filename = self.name.replace(" ", "") + '_Parameters.yaml'
        if addTimeStamp: filename = timestring + '_' + filename
        yamlPath = './results/Parameters/' + filename
        with open(yamlPath, 'w') as f:
            yaml.dump(parameterDict, f)

    def saveAgg(self, addTimeStamp=True):
        """
        Speichern der aggregierten Zeitreihen.
        """
        timestamp = datetime.now()
        timestring = timestamp.strftime('%Y-%m-%d')

        filename = self.name.replace(" ", "") + '_Timeseries.csv'
        if addTimeStamp: filename = timestring + '_' + filename

        pathAgg = './results/aggTimeseries/'
        self.aggregated_timeseries.to_csv(pathAgg + filename)


from flixOpt import flixStructure
from flixOpt import flixComps
from flixOpt.basicModeling import *


# ModelingElement mit Zusatz-Glg. und Variablen für aggregierte Berechnung
class cAggregationModeling(flixStructure.Element):
    def __init__(self, label, system, indexVectorsOfClusters, fixStorageFlows=True, fixBinaryVarsOnly=True,
                 listOfElementsToClusterize=None, percentageOfPeriodFreedom=0, costsOfPeriodFreedom=0, **kwargs):
        '''
        Modeling-Element for "index-equating"-equations


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
        listOfElementsToClusterize : TYPE, optional
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
        es: flixStructure.System
        self.system = system
        self.indexVectorsOfClusters = indexVectorsOfClusters
        self.fixStorageFlows = fixStorageFlows
        self.fixBinaryVarsOnly = fixBinaryVarsOnly
        self.listOfElementsToClusterize = listOfElementsToClusterize

        self.percentageOfPeriodFreedom = percentageOfPeriodFreedom
        self.costsOfPeriodFreedom = costsOfPeriodFreedom

        super().__init__(label, **kwargs)
        # invest_parameters to attributes:

        self.var_K_list = []

        # self.sub_elements.append(self.featureOn)

    def finalize(self):
        super().finalize()

    def declare_vars_and_eqs(self, system_model: flixStructure.SystemModel):
        super().declare_vars_and_eqs(system_model)

    def do_modeling(self, system_model: flixStructure.SystemModel, time_indices: Union[list[int], range]):

        if self.listOfElementsToClusterize is None:
            # Alle:
            compSet = set(self.system.components)
            flowSet = self.system.flows
        else:
            # Ausgewählte:
            compSet = set(self.listOfElementsToClusterize)
            flowSet = {flow for flow in self.system.flows if flow.comp in self.listOfElementsToClusterize}

        flow: flixStructure.Flow

        # todo: hier anstelle alle Elemente durchgehen, nicht nur flows und comps:
        for element in flowSet | compSet:
            # Wenn StorageFlows nicht gefixt werden sollen und flow ein storage-Flow ist:
            if (not self.fixStorageFlows) and hasattr(element, 'comp') and (isinstance(element.comp, flixComps.Storage)):
                pass  # flow hier nicht fixen!
            else:
                # On-Variablen:
                if element.model.var_on is not None:
                    aVar = element.model.var_on
                    aEq = self.getEqForLinkedIndexe(aVar, system_model, fixFirstIndexOfPeriod=True)
                    # SwitchOn-Variablen:
                if element.model.var_switchOn is not None:
                    aVar = element.model.var_switchOn
                    # --> hier ersten Index weglassen:
                    aEq = self.getEqForLinkedIndexe(aVar, system_model, fixFirstIndexOfPeriod=False)
                if element.model.var_switchOff is not None:
                    aVar = element.model.var_switchOff
                    # --> hier ersten Index weglassen:
                    aEq = self.getEqForLinkedIndexe(aVar, system_model, fixFirstIndexOfPeriod=False)

                    # todo: nicht schön! Zugriff muss über alle cTSVariablen erfolgen!
                # Nicht-Binär-Variablen:
                if not self.fixBinaryVarsOnly:
                    # Value-Variablen:
                    if hasattr(element.model, 'var_val'):
                        aVar = element.model.var_val
                        aEq = self.getEqForLinkedIndexe(aVar, system_model, fixFirstIndexOfPeriod=True)

    def getEqForLinkedIndexe(self, aVar, system_model, fixFirstIndexOfPeriod):
        aVar: Variable
        # todo!: idx_var1/2 wird jedes mal gemacht! nicht schick

        # Gleichung:
        # eq1: x(p1,t) - x(p3,t) = 0 # wobei p1 und p3 im gleichen Cluster sind und t = 0..N_p
        idx_var1 = np.array([])
        idx_var2 = np.array([])
        for clusterNr in self.indexVectorsOfClusters.keys():
            listOfIndexVectors = self.indexVectorsOfClusters[clusterNr]
            # alle Indexvektor-Tupel durchgehen:
            for i in range(len(listOfIndexVectors) - 1):  # ! Nur wenn cluster mehr als eine Periode enthält:
                # Falls eine Periode nicht ganz voll (eigl. nur bei letzter Periode möglich)
                v1 = listOfIndexVectors[0]
                v2 = listOfIndexVectors[i + 1]
                if not fixFirstIndexOfPeriod:
                    v1 = v1[1:]  # erstes Element immer weglassen
                    v2 = v2[1:]
                minLen = min(len(v1), len(v2))
                idx_var1 = np.append(idx_var1, v1[:minLen])
                idx_var2 = np.append(idx_var2, v2[:minLen])

        eq = flixStructure.Equation('equalIdx_' + aVar.label_full, self, system_model, eqType='eq')
        eq.add_summand(aVar, 1, indices_of_variable=idx_var1)
        eq.add_summand(aVar, -1, indices_of_variable=idx_var2)

        # Korrektur: (bisher nur für Binärvariablen:)
        if aVar.is_binary and self.percentageOfPeriodFreedom > 0:
            # correction-vars (so viele wie Indexe in eq:)
            var_K1 = Variable('Korr1_' + aVar.label_full.replace('.', '_'), eq.nr_of_single_equations, self, system_model,
                              is_binary=True)
            var_K0 = Variable('Korr0_' + aVar.label_full.replace('.', '_'), eq.nr_of_single_equations, self, system_model,
                              is_binary=True)
            # equation extends ...
            # --> On(p3) can be 0/1 independent of On(p1,t)!
            # eq1: On(p1,t) - On(p3,t) + K1(p3,t) - K0(p3,t) = 0
            # --> correction On(p3) can be:
            #  On(p1,t) = 1 -> On(p3) can be 0 -> K0=1 (,K1=0)
            #  On(p1,t) = 0 -> On(p3) can be 1 -> K1=1 (,K0=1)
            eq.add_summand(var_K1, +1)
            eq.add_summand(var_K0, -1)
            self.var_K_list.append(var_K1)
            self.var_K_list.append(var_K0)

            # interlock var_K1 and var_K2:
            # eq: var_K0(t)+var_K1(t) <= 1.1
            eq_lock = flixStructure.Equation('lock_K0andK1' + aVar.label_full, self, system_model, eqType='ineq')
            eq_lock.add_summand(var_K0, 1)
            eq_lock.add_summand(var_K1, 1)
            eq_lock.add_constant(1.1)

            # Begrenzung der Korrektur-Anzahl:
            # eq: sum(K) <= n_Corr_max
            self.noOfCorrections = round(self.percentageOfPeriodFreedom / 100 * var_K1.length)
            eq_max = flixStructure.Equation('maxNoOfCorrections_' + aVar.label_full, self, system_model, eqType='ineq')
            eq_max.add_summand(var_K1, 1, as_sum=True)
            eq_max.add_summand(var_K0, 1, as_sum=True)
            eq_max.add_constant(self.noOfCorrections)  # Maximum
        return eq

    def add_share_to_globals(self, globalComp: flixStructure.Global, system_model):

        # einzelne Stellen korrigierbar machen (aber mit Kosten)
        if (self.percentageOfPeriodFreedom > 0) & (self.costsOfPeriodFreedom != 0):
            for var_K in self.var_K_list:
                # todo: Krücke, weil muss eigentlich sowas wie Strafkosten sein!!!
                globalComp.objective.add_summand(var_K, self.costsOfPeriodFreedom, as_sum=True)
