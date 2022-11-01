# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:19:17 2022

@author: Panitz
"""
# TODO:
    #  Effektbeispiel mit Anzahl (z.b. Kesselanzahl < 3) 

useRohr = True
# useRohr = False
useAdditionalSink1 = True
# useAdditionalSink1 = False



### Inputs: ####
# Solver-Inputs:
displaySolverOutput = True  # ausführlicher Solver-Output.
gapFrac = 0.0001
timelimit = 3600

solver_name = 'glpk'
# solver_name = 'gurobi'
# solver_name    = 'cbc'
nrOfThreads    = 1

### Durchführungs-Optionen: ###
# doSegmentedCalc = True
doSegmentedCalc  = False
checkPenalty    = False  
excessCosts = None
excessCosts = 1e5 # default vlaue
################

import matplotlib.pyplot as plt
import numpy as np
import datetime

from flixStructure import *
from flixComps    import *


####################### kleine Daten zum Test ###############################

nrOfTimeSteps = 4
sink1 = [0.,   0.,  0., 100 , 40 , 40, 40, 40, 40][:nrOfTimeSteps]
sink2 = [10., 20., 30., 0   , 40 , 40, 40, 40, 40][:nrOfTimeSteps]
  
# todo: ggf. Umstellung auf numpy: aTimeSeries = datetime.datetime(2020, 1,1) +  np.arange(len(Q_th_Last)) * datetime.timedelta(hours=1)
aTimeSeries = datetime.datetime(2020, 1,1) +  np.arange(len(sink1)) * datetime.timedelta(hours=1)
aTimeSeries = aTimeSeries.astype('datetime64')

    
# # DAtenreihen kürzen:
# source1 = P_el_Last[0:nrOfTimeSteps]
# sink1 = Q_th_Last[0:nrOfTimeSteps]
# source2      = p_el     [0:nrOfTimeSteps]
# aTimeSeries = aTimeSeries[0:nrOfTimeSteps]
        

##########################################################################

print('#######################################################################')
print('################### start of modeling #################################')

# Bus-Definition:
#                 Typ         Name              
heat1 = cBus('heat'        ,'heat1'     , excessCostsPerFlowHour = excessCosts);
heat2 = cBus('heat'        ,'heat2'     , excessCostsPerFlowHour = excessCosts);

# Effect-Definition:
costs = cEffectType('costs','€'      , 'Kosten', isStandard = True, isObjective = True)


aSink1   = cSink   ('Sink1'  ,sink   = cFlow('Q_th' , bus = heat1, nominal_val = 1, val_rel = sink1))
aSink2   = cSink   ('Sink2'  ,sink   = cFlow('Q_th' , bus = heat2, nominal_val = 1, val_rel = sink2))
aSource1 = cSource ('Source1',source = cFlow('Q_th' , bus = heat1, nominal_val = 60, costsPerFlowHour = -1))
aSource2 = cSource ('Source2',source = cFlow('Q_th' , bus = heat2, nominal_val = 60, costsPerFlowHour = -1)) # doppelt so teuer


loss_abs = 1
# loss_abs = 0
loss_rel = 0.1
# loss_rel = 0

invest1 = cInvestArgs(fixCosts=10,
                      investmentSize_is_fixed=False,
                      investment_is_optional=True,                      
                      max_investmentSize=1000 ,
                      specificCosts=1)

# only for getting realizing investSize-Variable:
invest2 = cInvestArgs(fixCosts=0,
                      investmentSize_is_fixed=False,
                      investment_is_optional=True,                      
                      max_investmentSize=1000 ,
                      specificCosts=0
                      )

aTransporter = cTransportation('Rohr', 
                                in1  = cFlow('in1',  bus=heat1, investArgs=invest1, nominal_val = None, min_rel = 0.1 ),
                                out1 = cFlow('out1', bus=heat2),
                                loss_abs = loss_abs,
                                loss_rel = loss_rel,
                                in2  = cFlow('in2',  bus=heat2, investArgs=invest2, nominal_val = None, min_rel = 0.1 ),
                                out2 = cFlow('out2', bus=heat1),
                                )

# Built energysystem:
es = cEnergySystem(aTimeSeries, dt_last=None)
# es.addComponents(aGaskessel,aWaermeLast,aGasTarif)#,aGaskessel2)
es.addEffects(costs)
es.addComponents(aSink2, aSource1, aSource2)
if useAdditionalSink1 : es.addComponents(aSink1)
if useRohr : es.addComponents(aTransporter)

chosenEsTimeIndexe = None
# chosenEsTimeIndexe = [1,3,5]

## modeling "full":
aCalc = cCalculation('Sim1', es, 'pyomo', chosenEsTimeIndexe)
aCalc.doModelingAsOneSegment()

# PRINT Model-Charactaricstics:
es.printModel()
es.printVariables()
es.printEquations()

solverProps = {'gapFrac': gapFrac, 
               'timelimit': timelimit,
               'solver': solver_name, 
               'displaySolverOutput' : displaySolverOutput,
               }
if solver_name == 'gurobi': solverProps['threads'] = nrOfThreads

## calculation "full":

aCalc.solve(solverProps, nameSuffix = '_' + solver_name)

aCalc.results_struct

print('source1 :' + str(aCalc.results_struct.Source1.Q_th.val))
print('source2 :' + str(aCalc.results_struct.Source2.Q_th.val))
# print(aCalc.results_struct.Sink2.Q_th.val)

if useRohr :
    print('rohr_in :' + str(aCalc.results_struct.Rohr.in1.val))
    print('rohr_out:' + str(aCalc.results_struct.Rohr.out1.val))

print(aCalc.infos['modboxes']['info'][0]['main_results']['Invest-Decisions'])

## calculation "segmented":
