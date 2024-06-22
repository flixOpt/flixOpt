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

### some Solver-Inputs: ###
displaySolverOutput = False # ausführlicher Solver-Output.
displaySolverOutput = True  # ausführlicher Solver-Output.
gapFrac = 0.0001 # solver-gap
timelimit = 3600 # seconds until solver abort
# choose the solver, you have installed:
# solver_name = 'glpk' # warning, glpk quickly has numerical problems with binaries (big and epsilon)
# solver_name = 'gurobi'
solver_name = 'highs'
solverProps = {'gapFrac': gapFrac,
               'timelimit': timelimit,
               'solver': solver_name, 
               'displaySolverOutput' : displaySolverOutput,
               }

#####################
## some timeseries ##
Q_th_Last = np.array([30., 0., 90., 110, 110 , 20, 20, 20, 20]) # kW; thermal load profile in
p_el = 1/1000*np.array([80., 80., 80., 80 , 80, 80, 80, 80, 80]) # €/kWh; feed_in tariff;
aTimeSeries = datetime.datetime(2020, 1,1) +  np.arange(len(Q_th_Last)) * datetime.timedelta(hours=1) # creating timeseries
aTimeSeries = aTimeSeries.astype('datetime64') # needed format for timeseries in flixOpt
max_emissions_per_hour = 1000 # kg per timestep

print('#######################################################################')
print('################### start of modeling #################################')

# #####################
# ## Bus-Definition: ##
# define buses for the 3 used media:
Strom = cBus('el', 'Strom') # balancing node/bus of electricity
Fernwaerme = cBus('heat', 'Fernwärme') # balancing node/bus of heat
Gas = cBus('fuel', 'Gas') # balancing node/bus of gas

# ########################
# ## Effect-Definition: ##
costs = cEffectType('costs','€','Kosten',  # name, unit, description
                    isStandard = True, # standard effect --> shorter input possible (without effect as a key)
                    isObjective = True) # defining costs as objective of optimiziation

CO2   = cEffectType('CO2','kg','CO2_e-Emissionen', # name, unit, description
                    specificShareToOtherEffects_operation = {costs: 0.2}, max_per_hour_operation = max_emissions_per_hour) # 0.2 €/kg; defining links between effects, here CO2-price 

# ###########################
# ## Component-Definition: ##

# # 1. heat supply units: #

# 1.a) defining a boiler
aBoiler = cKessel('Boiler', eta = 0.5, # name, efficiency factor
                  # defining the output-flow = thermal -flow
                  Q_th = cFlow(label = 'Q_th', # name of flow
                               bus = Fernwaerme, # define, where flow is linked to (here: Fernwaerme-Bus)
                               nominal_val = 50, # kW; nominal_size of boiler
                               min_rel = 5/50, # 10 % minimum load, i.e. 5 kW
                               max_rel = 1, # 100 % maximum load, i.e. 50 kW
                               ),    
                  # defining the input-flow = fuel-flow
                  Q_fu = cFlow(label = 'Q_fu', # name of flow
                               bus = Gas) # define, where flow is linked to (here: Gas-Bus)
                  ) 

# 2.b) defining a CHP unit:
aKWK  = cKWK('CHP_unit', eta_th = 0.5, eta_el = 0.4, # name, thermal efficiency, electric efficiency
             # defining flows:
             P_el = cFlow('P_el',bus = Strom, 
                          nominal_val = 60, # 60 kW_el
                          min_rel = 5/60, ), # 5 kW_el, min- and max-load (100%) are here defined through this electric flow
             Q_th = cFlow('Q_th',bus = Fernwaerme),
             Q_fu = cFlow('Q_fu',bus = Gas))

# # 2. storage #

aSpeicher = cStorage('Speicher',
                     inFlow  = cFlow('Q_th_load', bus = Fernwaerme, nominal_val = 1e4),  # load-flow, maximum load-power: 1e4 kW
                     outFlow = cFlow('Q_th_unload',bus = Fernwaerme, nominal_val = 1e4),  # unload-flow, maximum load-power: 1e4 kW
                     capacity_inFlowHours=30,  # 30 kWh; storage capacity
                     chargeState0_inFlowHours=0,  # empty storage at first time step
                     max_rel_chargeState = 1/100*np.array([80., 70., 80., 80 , 80, 80, 80, 80, 80, 80]),
                     eta_load=0.9, eta_unload=1,  #loading efficiency factor, unloading efficiency factor
                     fracLossPerHour=0.08,  # 8 %/h; 8 percent of storage loading level is lossed every hour
                     avoidInAndOutAtOnce=True,  # no parallel loading and unloading at one time
                     invest_parameters=InvestParameters(fixCosts=20,
                                                        investmentSize_is_fixed=True,
                                                        investment_is_optional=False)
                     )
 
# # 3. sinks and sources #

# sink of heat load:
aWaermeLast = cSink('Wärmelast',
                    # defining input-flow:
                    sink   = cFlow('Q_th_Last', # name
                                   bus = Fernwaerme, # linked to bus "Fernwaerme"
                                   nominal_val = 1, # nominal_value
                                   val_rel = Q_th_Last)) # fixed profile
                                   # relative fixed values (timeseries) of the flow
                                   # value = val_rel * nominal_val
    
# source of gas:
aGasTarif = cSource('Gastarif' ,
                    # defining output-flow:
                    source = cFlow('Q_Gas', # name
                                   bus = Gas, # linked to bus "Gas"
                                   nominal_val = 1000, # nominal size, i.e. 1000 kW maximum
                                   # defining effect-shares. 
                                   #    Here not only "costs", but also CO2-emissions:
                                   costsPerFlowHour= {costs: 0.04, CO2: 0.3})) # 0.04 €/kWh, 0.3 kg_CO2/kWh

# sink of electricity feed-in:
aStromEinspeisung = cSink('Einspeisung', 
                          # defining input-flow:
                          sink=cFlow('P_el', # name
                                     bus = Strom, # linked to bus "Strom"
                                     costsPerFlowHour = -1*p_el)) # gains (negative costs) per kWh


# ######################################################
# ## Build energysystem - Registering of all elements ##

es = cEnergySystem(aTimeSeries, dt_last=None) # creating system, (duration of last timestep is like the one before)
es.addComponents(aSpeicher) # adding components
es.addEffects(costs, CO2) # adding defined effects
es.addComponents(aBoiler, aWaermeLast, aGasTarif) # adding components
es.addComponents(aStromEinspeisung) # adding components
es.addComponents(aKWK) # adding components


# choose used timeindexe:
chosenEsTimeIndexe = None # all timeindexe are used
# chosenEsTimeIndexe = [1,3,5] # only a subset shall be used

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

aCalc.solve(solverProps, # some solver options
            nameSuffix = '_' + solver_name) # nameSuffix for the results
# .. results are saved under /results/
# these files are written:
# -> json-file with model- and solve-Informations!
# -> log-file
# -> data-file


# ####################
# # PostProcessing: ##
# ####################


# ##### loading results from output-files ######
import flixOpt.flixPostprocessing as flixPost

nameOfCalc = aCalc.nameOfCalc
print(nameOfCalc)
# loading results, creating postprocessing Object:
aCalc_post = flixPost.flix_results(nameOfCalc) 

# ## plotting ##
# plotting all in- and out-flows of bus "Fernwaerme":
fig1 = aCalc_post.plotInAndOuts('Fernwaerme',stacked=True)
fig1.savefig('results/test1')
fig2 = aCalc_post.plotInAndOuts('Fernwaerme',stacked=True, plotAsPlotly = True)
fig2.show()
fig2.write_html('results/test2.html')
fig3 = aCalc_post.plotInAndOuts('Strom',stacked=True, plotAsPlotly = True)
fig3.show()
fig4 = aCalc_post.plotShares('Fernwaerme',plotAsPlotly=True)
fig4.show()


##############################
# ## access to timeseries: ##

# 1. direct access:
# (not recommended, better use postProcessing instead, see next)
print('# direct access:')
print('way 1:')
print(aCalc.results['Boiler']['Q_th']['val']) # access through dict
print('way 2:')
print(aCalc.results_struct.Boiler.Q_th.val) # access matlab-struct like
print('way 3:')
print(aBoiler.Q_th.mod.var_val.getResult()) # access directly through component/flow-variables
#    (warning: there are only temporarily the results of the last executed solve-command of the energy-system)

# 2. post-processing access:
print('# access to timeseries:#')
print('way 1:')
print(aCalc_post.results['Boiler']['Q_th']['val']) # access through dict
print('way 2:')
print(aCalc_post.results_struct.Boiler.Q_th.val) # access matlab-struct like
print('way 3:')
# find flow:
aFlow_post = aCalc_post.getFlowsOf('Fernwaerme','Boiler')[0][0] # getting flow
print(aFlow_post.results['val']) # access through cFlow_post object

# ###############################################
# ## saving csv of special flows of bus "Fernwaerme" ##
aCalc_post.to_csv('Fernwaerme', 'results/FW.csv')
    
