# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:19:17 2022

@author: Panitz
"""

### Inputs: ####
# Solver-Inputs:
displaySolverOutput = False # ausführlicher Solver-Output.
displaySolverOutput = True  # ausführlicher Solver-Output.
gapFrac = 0.0001
timelimit = 3600

solver_name = 'glpk' # warning, glpk quickly has numerical problems with binaries (big and epsilon)
# solver_name = 'gurobi'
# solver_name    = 'cbc'

################

import numpy as np
import datetime

from flixStructure import *
from flixComps    import *

####################### kleine Daten zum Test ###############################
Q_th_Last = [30., 0., 90., 110, 110 , 20, 20, 20, 20]
p_el      = [40., 40., 40., 40 , 40, 40, 40, 40, 40]
   
aTimeSeries = datetime.datetime(2020, 1,1) +  np.arange(len(Q_th_Last)) * datetime.timedelta(hours=1)
aTimeSeries = aTimeSeries.astype('datetime64')
  

print('#######################################################################')
print('################### start of modeling #################################')


# Bus-Definition:
Strom = cBus('el', 'Strom', excessCostsPerFlowHour=None)
Fernwaerme = cBus('heat', 'Fernwärme', excessCostsPerFlowHour=None)
Gas = cBus('fuel', 'Gas', excessCostsPerFlowHour=None)


# Effect-Definition:
costs = cEffectType('costs','€'      , 'Kosten', isStandard = True, isObjective = True)
CO2   = cEffectType('CO2'  ,'kg'     , 'CO2_e-Emissionen', specificShareToOtherEffects_operation = {costs: 0.2}, 
                    )
aGaskessel = cKessel('Kessel', eta = 0.5, #, switchOnCosts = 0
                    Q_th = cFlow(label   = 'Q_th', 
                                 bus = Fernwaerme, 
                                 nominal_val = 50, 
                                 min_rel = 5/50, 
                                 max_rel = 1, 
                                 ),       # maxGradient = 5),
                    Q_fu = cFlow(label   = 'Q_fu', bus = Gas)) 


aKWK  = cKWK('BHKW', eta_th = 0.5, eta_el = 0.4,
            P_el = cFlow('P_el',bus = Strom     , nominal_val = 60, min_rel = 5/60, ),
            Q_th = cFlow('Q_th',bus = Fernwaerme),
            Q_fu = cFlow('Q_fu',bus = Gas))

aSpeicher = cStorage('Speicher',
                     inFlow  = cFlow('Q_th_load', bus = Fernwaerme, nominal_val = 1e4),
                     outFlow = cFlow('Q_th_unload',bus = Fernwaerme, nominal_val = 1e4),
                     capacity_inFlowHours=30,
                     chargeState0_inFlowHours=0,
                     eta_load=0.9, eta_unload=1,
                     fracLossPerHour=0.08,
                     avoidInAndOutAtOnce=True
                     )
 
aWaermeLast       = cSink  ('Wärmelast',sink   = cFlow('Q_th_Last' , bus = Fernwaerme, nominal_val = 1, val_rel = Q_th_Last))

aGasTarif         = cSource('Gastarif' ,source = cFlow('Q_Gas'     , bus = Gas  , nominal_val = 1000, costsPerFlowHour= {costs: 0.04, CO2: 0.3}))

aStromEinspeisung = cSink  ('Einspeisung', sink=cFlow('P_el', bus = Strom, costsPerFlowHour = -1*np.array(p_el)))

# Built energysystem:
es = cEnergySystem(aTimeSeries, dt_last=None)
es.addEffects(costs, CO2)
es.addComponents(aGaskessel, aWaermeLast, aGasTarif)
es.addComponents(aStromEinspeisung)
es.addComponents(aKWK)
es.addComponents(aSpeicher)

chosenEsTimeIndexe = None
# chosenEsTimeIndexe = [1,3,5]

## modeling "full" ##
aCalc = cCalculation('Sim1', es, 'pyomo', chosenEsTimeIndexe)
aCalc.doModelingAsOneSegment()

# PRINT Model-Characteristics:
es.printModel()
es.printVariables()
es.printEquations()

solverProps = {'gapFrac': gapFrac, 
               'timelimit': timelimit,
               'solver': solver_name, 
               'displaySolverOutput' : displaySolverOutput,
               }

## calculation ##

aCalc.solve(solverProps, nameSuffix = '_' + solver_name)
# .. results are saved under /results/...
# these files are written:
# -> json-file with model- and solve-Informations!
# -> log-file
# -> data-file


##############################
## direct access to results ##
##############################

# example timeseries access:
print('Variante 1:')
print(aCalc.results['Kessel']['Q_th']['val'])
print('Variante 2:')
print(aCalc.results_struct.Kessel.Q_th.val)


# ####################
# # PostProcessing: ##
# ####################


# ##### loading ######
import flixPostprocessing as flixPost

nameOfCalc = aCalc.nameOfCalc
print(nameOfCalc)
calc1 = flixPost.flix_results(nameOfCalc)

# #### plotting ######

fig1 = calc1.plotInAndOuts('Fernwaerme',stacked=True)
fig1.savefig('test1')
fig2 = calc1.plotInAndOuts('Fernwaerme',stacked=True, plotAsPlotly = True)
fig2.show()
fig2.write_html('test2.html')
fig3 = calc1.plotInAndOuts('Strom',stacked=True, plotAsPlotly = True)
fig3.show()

# access to timeseries:
print('# access to timeseries:#')
print('way 1:')
print(calc1.results['Kessel']['Q_th']['val'])
print('way 2:')
print(calc1.results_struct.Kessel.Q_th.val)


## saving csv of special flows of Fernwaerme ##

calc1.to_csv('Fernwaerme', 'FW.csv')
    
