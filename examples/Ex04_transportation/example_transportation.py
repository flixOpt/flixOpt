# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:19:17 2022
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
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

# solver_name = 'glpk'
# solver_name = 'gurobi'
solver_name    = 'cbc'
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

from flixOpt.flixStructure import *
from flixOpt.flixComps import *

####################### kleine Daten zum Test ###############################

nrOfTimeSteps = 4
sink1 = [0.,   0.,  0., 100 , 40 , 40, 40, 40, 40][:nrOfTimeSteps]
sink2 = [10., 20., 30., 0   , 40 , 40, 40, 40, 40][:nrOfTimeSteps]
  
# todo: ggf. Umstellung auf numpy: aTimeSeries = datetime.datetime(2020, 1,1) +  np.arange(length(Q_th_Last)) * datetime.timedelta(hours=1)
aTimeSeries = datetime.datetime(2020, 1,1) +  np.arange(len(sink1)) * datetime.timedelta(hours=1)
aTimeSeries = aTimeSeries.astype('datetime64')

    
# # DAtenreihen kürzen:
# source1 = P_el_Last[0:nrOfTimeSteps]
# sink1 = Q_th_Last[0:nrOfTimeSteps]
# source2      = p_el     [0:nrOfTimeSteps]
# aTimeSeries = aTimeSeries[0:nrOfTimeSteps]
# #########################################################################

print('#######################################################################')
print('################### start of modeling #################################')

# Bus-Definition:
#                 Typ         Name              
heat1 = Bus('heat', 'heat1', excessCostsPerFlowHour = excessCosts);
heat2 = Bus('heat', 'heat2', excessCostsPerFlowHour = excessCosts);

# Effect-Definition:
costs = Effect('costs', '€', 'Kosten', is_standard= True, is_objective= True)


aSink1   = Sink   ('Sink1', sink   = Flow('Q_th', bus = heat1, nominal_val = 1, val_rel = sink1))
aSink2   = Sink   ('Sink2', sink   = Flow('Q_th', bus = heat2, nominal_val = 1, val_rel = sink2))
aSource1 = Source ('Source1', source = Flow('Q_th', bus = heat1, nominal_val = 60, costsPerFlowHour = -1))
aSource2 = Source ('Source2', source = Flow('Q_th', bus = heat2, nominal_val = 60, costsPerFlowHour = -1)) # doppelt so teuer


loss_abs = 1
# loss_abs = 0
loss_rel = 0.1
# loss_rel = 0

invest1 = InvestParameters(fix_effects=10,
                           fixed_size=False,
                           optional=True,
                           maximum_size=1000,
                           specific_effects=1)

# only for getting realizing investSize-Variable:
invest2 = InvestParameters(fix_effects=0,
                           fixed_size=False,
                           optional=True,
                           maximum_size=1000,
                           specific_effects=0
                           )

aTransporter = Transportation('Rohr',
                              in1  = Flow('in1', bus=heat1, invest_parameters=invest1, nominal_val = None, min_rel = 0.1),
                              out1 = Flow('out1', bus=heat2),
                              loss_abs = loss_abs,
                              loss_rel = loss_rel,
                              in2  = Flow('in2', bus=heat2, invest_parameters=invest2, nominal_val = None, min_rel = 0.1),
                              out2 = Flow('out2', bus=heat1),
                              )

# Built energysystem:
system = System(aTimeSeries, last_time_step_hours=None)
# system.addComponents(aGaskessel,aWaermeLast,aGasTarif)#,aGaskessel2)
system.addEffects(costs)
system.addComponents(aSink2, aSource1, aSource2)
if useAdditionalSink1 : system.addComponents(aSink1)
if useRohr : system.addComponents(aTransporter)

time_indices = None
# time_indices = [1,3,5]

## modeling "full":
aCalc = Calculation('Sim1', system, 'pyomo', time_indices)
aCalc.doModelingAsOneSegment()

# PRINT Model-Charactaricstics:
system.printModel()
system.printVariables()
system.printEquations()

solverProps = {'mip_gap': gapFrac,
               'time_limit_seconds': timelimit,
               'solver': solver_name, 
               'solver_output_to_console' : displaySolverOutput,
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
