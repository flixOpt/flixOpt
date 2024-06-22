# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:19:17 2022
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, 
  Technische Universität Dresden
"""

import numpy as np
import datetime
from flixOpt.flixStructure import *
from flixOpt.flixComps    import *

#####################
## some timeseries ##
Q_th_Last = [30., 0., 20.] # kW; thermal load profile in
aTimeSeries = datetime.datetime(2020, 1,1) +  np.arange(len(Q_th_Last)) * datetime.timedelta(hours=1) # creating timeseries
aTimeSeries = aTimeSeries.astype('datetime64') # needed format for timeseries in flixOpt

print('#######################################################################')
print('################### start of modeling #################################')

# #####################
# ## Bus-Definition: ##
# define buses for the 3 used media:
Strom = Bus('el', 'Strom') # balancing node/bus of electricity
Fernwaerme = Bus('heat', 'Fernwärme') # balancing node/bus of heat
Gas = Bus('fuel', 'Gas') # balancing node/bus of gas

# ########################
# ## Effect-Definition: ##
costs = Effect('costs', '€', 'Kosten',  # name, unit, description
               isStandard = True,  # standard effect --> shorter input possible (without effect as a key)
               isObjective = True) # defining costs as objective of optimiziation

# ###########################
# ## Component-Definition: ##

aBoiler = Boiler('Boiler', eta = 0.5,  # name, efficiency factor
                 # defining the output-flow = thermal -flow
                 Q_th = Flow(label ='Q_th',  # name of flow
                             bus = Fernwaerme,  # define, where flow is linked to (here: Fernwaerme-Bus)
                             nominal_val = 50,  # kW; nominal_size of boiler
                             ),
                 # defining the input-flow = fuel-flow
                 Q_fu = Flow(label ='Q_fu',  # name of flow
                             bus = Gas)  # define, where flow is linked to (here: Gas-Bus)
                 )

# sink of heat load:
aWaermeLast = Sink('Wärmelast',
                   # defining input-flow:
                   sink   = Flow('Q_th_Last',  # name
                                 bus = Fernwaerme,  # linked to bus "Fernwaerme"
                                 nominal_val = 1,  # nominal_value
                                 val_rel = Q_th_Last)) # fixed profile
                                   # relative fixed values (timeseries) of the flow
                                   # value = val_rel * nominal_val
    
# source of gas:
aGasTarif = Source('Gastarif',
                   # defining output-flow:
                   source = Flow('Q_Gas',  # name
                                 bus = Gas,  # linked to bus "Gas"
                                 nominal_val = 1000,  # nominal size, i.e. 1000 kW maximum
                                 # defining effect-shares.
                                 #    Here not only "costs", but also CO2-emissions:
                                 costsPerFlowHour= 0.04)) # 0.04 €/kWh


# ######################################################
# ## Build energysystem - Registering of all elements ##

es = System(aTimeSeries, dt_last=None) # creating system, (duration of last timestep is like the one before)
es.addEffects(costs) # adding defined effects
es.addComponents(aBoiler, aWaermeLast, aGasTarif) # adding components

# choose used timeindexe:
chosenEsTimeIndexe = None # all timeindexe are used

# ## modeling the system ##

# 1. create a Calculation 
aCalc = cCalculation('Sim1', # name of calculation
                     es, # energysystem to calculate
                     'pyomo', # optimization modeling language (only "pyomo" implemented, yet)
                     chosenEsTimeIndexe) # used time steps

# 2. modeling:
aCalc.doModelingAsOneSegment() # mathematic modeling of system

# 3. (optional) print Model-Characteristics:
es.printModel() # string-output:network structure of model
es.printVariables() # string output: variables of model
es.printEquations() # string-output: equations of model


# #################
# ## calculation ##

### some Solver-Inputs: ###
displaySolverOutput = True  # ausführlicher Solver-Output.
gapFrac = 0.01 # solver-gap
timelimit = 3600 # seconds until solver abort
# choose the solver, you have installed:
# solver_name = 'glpk' # warning, glpk quickly has numerical problems with binaries (big and epsilon)
# solver_name = 'gurobi'
solver_name = 'cbc'
solverProps = {'gapFrac': gapFrac,
               'timelimit': timelimit,
               'solver': solver_name,
               'displaySolverOutput' : displaySolverOutput,
               }

aCalc.solve(solverProps, # some solver options
            nameSuffix = '_' + solver_name) # nameSuffix for the results
#  results are saved under /results/

# ##### loading results from output-files ######
import flixOpt.flixPostprocessing as flixPost

aCalc_post = flixPost.flix_results(aCalc.nameOfCalc)

