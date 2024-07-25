# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:19:17 2022
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import datetime

import numpy as np

from flixOpt.calculation import FullCalculation
from flixOpt.components import Boiler, CHP, Storage, Sink, Source, LinearTransformer
from flixOpt.elements import Bus, Flow, Effect
from flixOpt.flixBasicsPublic import InvestParameters
from flixOpt.flow_system import FlowSystem

# ## Solver-Inputs:##
displaySolverOutput = False  # ausführlicher Solver-Output.
displaySolverOutput = True  # ausführlicher Solver-Output.
gapFrac = 0.0001
timelimit = 3600

# solver_name = 'glpk' # warning, glpk quickly has numerical problems with big and epsilon
# solver_name = 'gurobi'
solver_name = 'highs'
nrOfThreads = 1

# ## calculation-options - you can vary! ###
useCHPwithLinearSegments = False
# useCHPwithLinearSegments = True
checkPenalty = False
excessCosts = None
excessCosts = 1e5  # default value
################


# ## timeseries ##
P_el_Last = [70., 80., 90., 90, 90, 90, 90, 90, 90]
if checkPenalty:
    Q_th_Last = [30., 0., 90., 110, 2000, 20, 20, 20, 20]
else:
    Q_th_Last = [30., 0., 90., 110, 110, 20, 20, 20, 20]

p_el = [40., 40., 40., 40, 40, 40, 40, 40, 40]

aTimeSeries = datetime.datetime(2020, 1, 1) + np.arange(len(Q_th_Last)) * datetime.timedelta(hours=1)
aTimeSeries = aTimeSeries.astype('datetime64')

##########################################################################

print('#######################################################################')
print('################### start of modeling #################################')

# ## Bus-Definition: ##
#######################
#                 Typ         Name              
Strom = Bus('el', 'Strom', excess_effects_per_flow_hour=excessCosts)
Fernwaerme = Bus('heat', 'Fernwärme', excess_effects_per_flow_hour=excessCosts)
Gas = Bus('fuel', 'Gas', excess_effects_per_flow_hour=excessCosts)

# Effect-Definition:
costs = Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True)
CO2 = Effect('CO2', 'kg', 'CO2_e-Emissionen', specific_share_to_other_effects_operation={costs: 0.2}, )
PE = Effect('PE', 'kWh_PE', 'Primärenergie', maximum_total=3.5e3)

################################
# ## definition of components ##
################################

# 1. definition of boiler #
# 1. a) investment-options:
invest_Gaskessel = InvestParameters(fix_effects=1000,  # 1000 € investment costs
                                    fixed_size=True,  # fix nominal size
                                    optional=False,  # forced investment
                                    specific_effects={costs: 10, PE: 2},  # specific costs: 10 €/kW; 2 kWh_PE/kW
                                    )
# invest_Gaskessel = None #
# 1. b) boiler itself:
aGaskessel = Boiler('Kessel', eta=0.5,  # efficiency ratio
                    effects_per_running_hour={costs: 0, CO2: 1000},  # 1000 kg_CO2/h (just for testing)
                    # defining flows:
                    Q_th=Flow(label='Q_th',  # name
                              bus=Fernwaerme,  # linked bus
                              size=50,  # 50 kW_th nominal size
                              load_factor_max=1.0,  # maximal mean power 50 kW
                              load_factor_min=0.1,  # minimal mean power 5 kW
                              min_rel=5 / 50,  # 10 % part load
                              max_rel=1,  # 50 kW
                              on_hours_total_min=0,  # minimum of working hours
                              on_hours_total_max=1000,  # maximum of working hours
                              on_hours_max=10,  # maximum of working hours in one step
                              off_hours_max=10,  # maximum of off hours in one step
                              # on_hours_min = 2, # minimum on hours in one step
                              # off_hours_min = 4, # minimum off hours in one step
                              switch_on_effects=0.01,  # € per start
                              switch_on_total_max=1000,  # max nr of starts
                              values_before_begin=[50],  # 50 kW is value before start
                              invest_parameters=invest_Gaskessel,  # see above
                              flow_hours_total_max=1e6,  # kWh, overall maximum "flow-work"
                              ),
                    Q_fu=Flow(label='Q_fu',  # name
                              bus=Gas,  # linked bus
                              size=200,  # kW
                              min_rel=0,
                              max_rel=1))

# 2. defining of CHP-unit:
aKWK = CHP('BHKW2', eta_th=0.5, eta_el=0.4, switch_on_effects=0.01,
           P_el=Flow('P_el', bus=Strom, size=60, min_rel=5 / 60),
           Q_th=Flow('Q_th', bus=Fernwaerme, size=1e3),
           Q_fu=Flow('Q_fu', bus=Gas, size=1e3), on_values_before_begin=[1])

# 3. defining a alternative CHP-unit with linear segments :
# defining flows:
#   (explicitly outside, because variables 'P_el', 'Q_th', 'Q_fu' must be picked 
#    in segment definition)
P_el = Flow('P_el', bus=Strom, size=60, max_rel=55)
Q_th = Flow('Q_th', bus=Fernwaerme)
Q_fu = Flow('Q_fu', bus=Gas)
# linear segments (eta-definitions than become useless!):
segmentsOfFlows = ({P_el: [5, 30, 40, 60],  # elements an be list (timeseries)
                    Q_th: [6, 35, 45, 100],
                    Q_fu: [12, 70, 90, 200]})

aKWK2 = LinearTransformer('BHKW2', inputs=[Q_fu], outputs=[P_el, Q_th], segmentsOfFlows=segmentsOfFlows,
                          switch_on_effects=0.01, on_values_before_begin=[1])

# 4. definition of storage:
# 4.a) investment options:


# linear segments of costs: [start1, end1, start2, end2, ...]
costsInvestsizeSegments = [[5, 25, 25, 100],  # kW
                           {costs: [50, 250, 250, 800],  # €
                            PE: [5, 25, 25, 100]  # kWh_PE
                            }]
# Anmerkung: points also realizable, through same start- end endpoint, i.g. [4,4]


# # alternative input only for standard-effect:
# costsInvestsizeSegments = [[5,25,25,100], #kW
#                             [50,250,250,800],#€ (standard-effect)
#                           ]

invest_Speicher = InvestParameters(fix_effects=0,  # no fix costs
                                   fixed_size=False,  # variable size
                                   effects_in_segments=costsInvestsizeSegments,  # see above
                                   optional=False,  # forced invest
                                   specific_effects={costs: 0.01, CO2: 0.01},  # €/kWh; kg_CO2/kWh
                                   minimum_size=0, maximum_size=1000)  # optimizing between 0...1000 kWh

# 4.b) storage itself:
aSpeicher = Storage('Speicher', # defining flows:
                    inFlow=Flow('Q_th_load', bus=Fernwaerme, size=1e4),
                    outFlow=Flow('Q_th_unload', bus=Fernwaerme, size=1e4), capacity_inFlowHours=None,
                    # None, because invest-size is variable
                    chargeState0_inFlowHours=0,  # empty storage at beginning
                    # charge_state_end_min = 3, # min charge state and end
                    charge_state_end_max=10,  # max charge state and end
                    eta_load=0.9, eta_unload=1,  # efficiency of (un)-loading
                    fracLossPerHour=0.08,  # loss of storage per time
                    avoidInAndOutAtOnce=True,  # no parallel loading and unloading
                    invest_parameters=invest_Speicher)  # see above

# 5. definition of sinks and sources:
# 5.a) heat load profile:    
aWaermeLast = Sink('Wärmelast', sink=Flow('Q_th_Last',  # name
                                          bus=Fernwaerme,  # linked bus
                                          size=1, min_rel=0, fixed_relative_value=Q_th_Last))  # fixed values fixed_relative_value * size
# 5.b) gas tarif:
aGasTarif = Source('Gastarif', source=Flow('Q_Gas', bus=Gas,  # linked bus
                                           size=1000,  # defining nominal size
                                           effects_per_flow_hour={costs: 0.04, CO2: 0.3}))
# 5.c) feed-in of electricity:
aStromEinspeisung = Sink('Einspeisung', sink=Flow('P_el', bus=Strom,  # linked bus
                                                  effects_per_flow_hour=-1 * np.array(p_el)))  # feed-in tariff

##########################
# ## Build energysystem ##
##########################

flow_system = FlowSystem(aTimeSeries, last_time_step_hours=None)  # creating FlowSystem

flow_system.add_effects(costs, CO2, PE)  # adding effects
flow_system.add_components(aGaskessel, aWaermeLast, aGasTarif)  # adding components
flow_system.add_components(aStromEinspeisung)  # adding components

if useCHPwithLinearSegments:
    flow_system.add_components(aKWK2)  # adding components
else:
    flow_system.add_components(aKWK)  # adding components

flow_system.add_components(aSpeicher)  # adding components

################################
# ## modeling and calculation ##
################################

time_indices = None
# time_indices = [1,3,5]

# ## modeling "full" calculation:
aCalc = FullCalculation('Sim1', flow_system, 'pyomo', time_indices)
aCalc.do_modeling()

# print Model-Charactaricstics:
flow_system.printModel()
flow_system.print_variables()
flow_system.print_equations()

solverProps = {'mip_gap': gapFrac, 'time_limit_seconds': timelimit, 'solver_name': solver_name,
               'solver_output_to_console': displaySolverOutput, }
if solver_name == 'gurobi': solverProps['threads'] = nrOfThreads

# ## solving calculation ##

aCalc.solve(solverProps)

# -> for analysis of results, see separate postprocessing-script!
