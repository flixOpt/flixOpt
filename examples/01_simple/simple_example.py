# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:19:17 2022
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, 
  Technische Universität Dresden
"""

import numpy as np
import datetime
from flixStructure import *
from flixComps    import *

### some Solver-Inputs: ###
displaySolverOutput = False # ausführlicher Solver-Output.
displaySolverOutput = True  # ausführlicher Solver-Output.
gapFrac = 0.0001 # solver-gap
timelimit = 3600
# choose the solver, you have installed:
solver_name = 'glpk' # warning, glpk quickly has numerical problems with binaries (big and epsilon)
# solver_name = 'gurobi'
# solver_name    = 'cbc'
solverProps = {'gapFrac': gapFrac, 
               'timelimit': timelimit,
               'solver': solver_name, 
               'displaySolverOutput' : displaySolverOutput,
               }

#####################
## some timeseries ##
Q_th_Last = [30., 0., 90., 110, 110 , 20, 20, 20, 20]
p_el      = [40., 40., 40., 40 , 40, 40, 40, 40, 40]
aTimeSeries = datetime.datetime(2020, 1,1) +  np.arange(len(Q_th_Last)) * datetime.timedelta(hours=1)
aTimeSeries = aTimeSeries.astype('datetime64')

print('#######################################################################')
print('################### start of modeling #################################')

# #####################
# ## Bus-Definition: ##
Strom = cBus('el', 'Strom', excessCostsPerFlowHour=None)
Fernwaerme = cBus('heat', 'Fernwärme', excessCostsPerFlowHour=None)
Gas = cBus('fuel', 'Gas', excessCostsPerFlowHour=None)

# ########################
# ## Effect-Definition: ##
costs = cEffectType('costs','€','Kosten', 
                    isStandard = True, 
                    isObjective = True)
CO2   = cEffectType('CO2','kg','CO2_e-Emissionen',
                    specificShareToOtherEffects_operation = {costs: 0.2})

# ###########################
# ## Component-Definition: ##

# # 1. heat supply units: #

aBoiler = cKessel('Boiler', eta = 0.5, 
                  Q_th = cFlow(label = 'Q_th', # thermal output flow
                               bus = Fernwaerme, 
                               nominal_val = 50, 
                               min_rel = 5/50, 
                               max_rel = 1, 
                               ),       # maxGradient = 5),
                  Q_fu = cFlow(label = 'Q_fu', # fuel input flow
                               bus = Gas)
                  ) 

aKWK  = cKWK('CHP_unit', eta_th = 0.5, eta_el = 0.4,
            P_el = cFlow('P_el',bus = Strom     , nominal_val = 60, min_rel = 5/60, ),
            Q_th = cFlow('Q_th',bus = Fernwaerme),
            Q_fu = cFlow('Q_fu',bus = Gas))

# # 2. storage #

aSpeicher = cStorage('Speicher',
                     inFlow  = cFlow('Q_th_load', bus = Fernwaerme, nominal_val = 1e4), # load-flow
                     outFlow = cFlow('Q_th_unload',bus = Fernwaerme, nominal_val = 1e4), # unload-flow
                     capacity_inFlowHours=30, # 30 kWh
                     chargeState0_inFlowHours=0, # empty storage at first time step
                     eta_load=0.9, eta_unload=1, 
                     fracLossPerHour=0.08,
                     avoidInAndOutAtOnce=True
                     )
 
# # 3. sinks and sources #

aWaermeLast       = cSink  ('Wärmelast',sink   = cFlow('Q_th_Last' , bus = Fernwaerme, nominal_val = 1, val_rel = Q_th_Last))

aGasTarif         = cSource('Gastarif' ,source = cFlow('Q_Gas'     , bus = Gas  , nominal_val = 1000, costsPerFlowHour= {costs: 0.04, CO2: 0.3}))

aStromEinspeisung = cSink  ('Einspeisung', sink=cFlow('P_el', bus = Strom, costsPerFlowHour = -1*np.array(p_el)))


# ######################################################
# ## Build energysystem - Registering of all elements ##
es = cEnergySystem(aTimeSeries, dt_last=None)
es.addEffects(costs, CO2)
es.addComponents(aBoiler, aWaermeLast, aGasTarif)
es.addComponents(aStromEinspeisung)
es.addComponents(aKWK)
es.addComponents(aSpeicher)

chosenEsTimeIndexe = None
# chosenEsTimeIndexe = [1,3,5]

# ## modeling the system ##

# 1. create a Calculation 
aCalc = cCalculation('Sim1', es, 'pyomo', chosenEsTimeIndexe)

# 2. modeling:
aCalc.doModelingAsOneSegment()

# 3. (optional) print Model-Characteristics:
es.printModel()
es.printVariables()
es.printEquations()


# #################
# ## calculation ##

aCalc.solve(solverProps, nameSuffix = '_' + solver_name)
# .. results are saved under /results/...
# these files are written:
# -> json-file with model- and solve-Informations!
# -> log-file
# -> data-file


##############################
## direct access to results ##
# (not recommended, use postProcessing instead, see next)

# example timeseries access:
print('Variante 1:')
print(aCalc.results['Boiler']['Q_th']['val'])
print('Variante 2:')
print(aCalc.results_struct.Kessel.Q_th.val)


# ####################
# # PostProcessing: ##
# ####################


# ##### loading results from output-files ######
import flixPostprocessing as flixPost

nameOfCalc = aCalc.nameOfCalc
print(nameOfCalc)
calc1 = flixPost.flix_results(nameOfCalc)

# ## plotting ##

fig1 = calc1.plotInAndOuts('Fernwaerme',stacked=True)
fig1.savefig('results/test1')
fig2 = calc1.plotInAndOuts('Fernwaerme',stacked=True, plotAsPlotly = True)
fig2.show()
fig2.write_html('results/test2.html')
fig3 = calc1.plotInAndOuts('Strom',stacked=True, plotAsPlotly = True)
fig3.show()

# ###########################
# ## access to timeseries: ##
print('# access to timeseries:#')
print('way 1:')
print(calc1.results['Boiler']['Q_th']['val'])
print('way 2:')
print(calc1.results_struct.Kessel.Q_th.val)

# ###############################################
# ## saving csv of special flows of bus "Fernwaerme" ##
calc1.to_csv('Fernwaerme', 'results/FW.csv')
    
