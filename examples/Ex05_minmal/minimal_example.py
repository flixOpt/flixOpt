# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:19:17 2022
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, 
  Technische Universität Dresden
"""

import numpy as np
import datetime
from flixOpt.structure import *
from flixOpt.components    import *

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
               is_standard= True,  # standard effect --> shorter input possible (without effect as a key)
               is_objective= True) # defining costs as objective of optimiziation

# ###########################
# ## Component-Definition: ##

aBoiler = Boiler('Boiler', eta = 0.5,  # name, efficiency factor
                 # defining the output-flow = thermal -flow
                 Q_th = Flow(label ='Q_th',  # name of flow
                             bus = Fernwaerme,  # define, where flow is linked to (here: Fernwaerme-Bus)
                             size=50,  # kW; nominal_size of boiler
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
                                 size=1,  # sizeue
                                 val_rel = Q_th_Last)) # fixed profile
                                   # relative fixed values (timeseries) of the flow
                                   # value = val_rel * size
    
# source of gas:
aGasTarif = Source('Gastarif',
                   # defining output-flow:
                   source = Flow('Q_Gas',  # name
                                 bus = Gas,  # linked to bus "Gas"
                                 size=1000,  # nominal size, i.e. 1000 kW maximum
                                 # defining effect-shares.
                                 #    Here not only "costs", but also CO2-emissions:
                                 effects_per_flow_hour= 0.04)) # 0.04 €/kWh


# ######################################################
# ## Build energysystem - Registering of all elements ##

flow_system = FlowSystem(aTimeSeries, last_time_step_hours=None) # creating flow_system, (duration of last timestep is like the one before)
flow_system.add_effects(costs) # adding defined effects
flow_system.add_components(aBoiler, aWaermeLast, aGasTarif) # adding components

# choose used timeindexe:
time_indices = None # all timeindexe are used

# ## modeling the flow_system ##

# 1. create a Calculation 
aCalc = Calculation('Sim1',  # name of calculation
                    flow_system,  # energysystem to calculate
                     'pyomo',  # optimization modeling language (only "pyomo" implemented, yet)
                    time_indices) # used time steps

# 2. modeling:
aCalc.do_modeling_as_one_segment() # mathematic modeling of flow_system

# 3. (optional) print Model-Characteristics:
flow_system.printModel() # string-output:network structure of model
flow_system.print_variables() # string output: variables of model
flow_system.print_equations() # string-output: equations of model


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
solverProps = {'mip_gap': gapFrac,
               'time_limit_seconds': timelimit,
               'solver_name': solver_name,
               'solver_output_to_console' : displaySolverOutput,
               }

aCalc.solve(solverProps)
#  results are saved under /results/

# ##### loading results from output-files ######
import flixOpt.flixPostprocessing as flixPost

aCalc_post = flixPost.flix_results(aCalc.name)

