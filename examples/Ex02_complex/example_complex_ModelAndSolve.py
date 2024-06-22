# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:19:17 2022
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import numpy as np
import datetime
from flixOpt.flixStructure import *
from flixOpt.flixComps import *
from flixOpt.flixBasicsPublic import *

# ## Solver-Inputs:##
displaySolverOutput = False # ausführlicher Solver-Output.
displaySolverOutput = True  # ausführlicher Solver-Output.
gapFrac = 0.0001
timelimit = 3600

# solver_name = 'glpk' # warning, glpk quickly has numerical problems with big and epsilon
# solver_name = 'gurobi'
solver_name    = 'cbc'
nrOfThreads    = 1

# ## calculation-options - you can vary! ###
useCHPwithLinearSegments = False
# useCHPwithLinearSegments = True
checkPenalty    = False  
excessCosts = None
excessCosts = 1e5 # default value
################


# ## timeseries ##
P_el_Last = [70., 80., 90., 90 , 90 , 90, 90, 90, 90]
if checkPenalty :
    Q_th_Last = [30., 0., 90., 110, 2000 , 20, 20, 20, 20]
else :
    Q_th_Last = [30., 0., 90., 110, 110 , 20, 20, 20, 20]

p_el      = [40., 40., 40., 40 , 40, 40, 40, 40, 40]
   
aTimeSeries = datetime.datetime(2020, 1,1) +  np.arange(len(Q_th_Last)) * datetime.timedelta(hours=1)
aTimeSeries = aTimeSeries.astype('datetime64')     

##########################################################################

print('#######################################################################')
print('################### start of modeling #################################')


# ## Bus-Definition: ##
#######################
#                 Typ         Name              
Strom = cBus('el', 'Strom', excessCostsPerFlowHour=excessCosts)
Fernwaerme = cBus('heat', 'Fernwärme', excessCostsPerFlowHour=excessCosts)
Gas = cBus('fuel', 'Gas', excessCostsPerFlowHour=excessCosts)


# Effect-Definition:
costs = cEffectType('costs','€'      , 'Kosten', isStandard = True, isObjective = True)
CO2   = cEffectType('CO2'  ,'kg'     , 'CO2_e-Emissionen', 
                    specificShareToOtherEffects_operation = {costs: 0.2}, 
                    )
PE    = cEffectType('PE'   ,'kWh_PE' , 'Primärenergie', max_Sum = 3.5e3  )

################################
# ## definition of components ##
################################

# 1. definition of boiler #
# 1. a) investment-options:
invest_Gaskessel = InvestParameters(fixCosts = 1000,  # 1000 € investment costs
                                    investmentSize_is_fixed = True,  # fix nominal size
                                    investment_is_optional=False,  # forced investment
                                    specificCosts= {costs:10, PE:2},  # specific costs: 10 €/kW; 2 kWh_PE/kW
                                    )
# invest_Gaskessel = None #
# 1. b) boiler itself:
aGaskessel = Boiler('Kessel',
                    eta = 0.5,  # efficiency ratio
                    costsPerRunningHour = {costs:0, CO2:1000},  # 1000 kg_CO2/h (just for testing)
                    # defining flows:
                    Q_th = cFlow(label   = 'Q_th',  # name
                                  bus = Fernwaerme,  # linked bus
                                  nominal_val = 50,  # 50 kW_th nominal size
                                  loadFactor_max = 1.0,  # maximal mean power 50 kW
                                  loadFactor_min = 0.1,  # minimal mean power 5 kW
                                  min_rel = 5/50,  # 10 % part load
                                  max_rel = 1,  # 50 kW
                                  onHoursSum_min = 0,  # minimum of working hours
                                  onHoursSum_max = 1000,  # maximum of working hours
                                  onHours_max = 10,  # maximum of working hours in one step
                                  offHours_max = 10,  # maximum of off hours in one step
                                  # onHours_min = 2, # minimum on hours in one step
                                  # offHours_min = 4, # minimum off hours in one step
                                  switchOnCosts = 0.01,  # € per start
                                  switchOn_maxNr = 1000,  # max nr of starts
                                  valuesBeforeBegin=[50],  # 50 kW is value before start
                                  invest_parameters= invest_Gaskessel,  # see above
                                  sumFlowHours_max = 1e6,  # kWh, overall maximum "flow-work"
                                  ),
                    Q_fu = cFlow(label = 'Q_fu', # name
                                  bus = Gas, # linked bus
                                  nominal_val = 200, # kW
                                  min_rel = 0, 
                                  max_rel = 1)) 

# 2. defining of CHP-unit:
aKWK  = CHP('BHKW2', eta_th = 0.5, eta_el = 0.4, switchOnCosts =  0.01,
            P_el = cFlow('P_el',bus = Strom     , nominal_val = 60, min_rel = 5/60, ),
            Q_th = cFlow('Q_th',bus = Fernwaerme, nominal_val = 1e3),
            Q_fu = cFlow('Q_fu',bus = Gas, nominal_val = 1e3), on_valuesBeforeBegin = [1])


# 3. defining a alternative CHP-unit with linear segments :
# defining flows:
#   (explicitly outside, because variables 'P_el', 'Q_th', 'Q_fu' must be picked 
#    in segment definition)
P_el = cFlow('P_el', bus=Strom, nominal_val=60, max_rel=55)
Q_th = cFlow('Q_th', bus=Fernwaerme)
Q_fu = cFlow('Q_fu', bus=Gas)
# linear segments (eta-definitions than become useless!):
segmentsOfFlows = ({P_el: [5  ,30, 40,60 ], # elements an be list (timeseries)
                   Q_th: [6  ,35, 45,100], 
                   Q_fu: [12 ,70, 90,200]})

aKWK2 = LinearTransformer('BHKW2', inputs = [Q_fu], outputs = [P_el, Q_th], segmentsOfFlows = segmentsOfFlows, switchOnCosts = 0.01, on_valuesBeforeBegin = [1])



# 4. definition of storage:
# 4.a) investment options:
    

# linear segments of costs: [start1, end1, start2, end2, ...]
costsInvestsizeSegments = [[5,25,25,100], #kW
                           {costs:[50,250,250,800],#€
                            PE:[5,25,25,100] #kWh_PE
                            }
                           ]
# Anmerkung: points also realizable, through same start- end endpoint, i.g. [4,4]


# # alternative input only for standard-effect:
# costsInvestsizeSegments = [[5,25,25,100], #kW
#                             [50,250,250,800],#€ (standard-effect)
#                           ]

invest_Speicher = InvestParameters(fixCosts = 0,  # no fix costs
                                   investmentSize_is_fixed = False,  # variable size
                                   costsInInvestsizeSegments = costsInvestsizeSegments,  # see above
                                   investment_is_optional=False,  # forced invest
                                   specificCosts= {costs: 0.01, CO2: 0.01},  # €/kWh; kg_CO2/kWh
                                   min_investmentSize=0, max_investmentSize=1000) # optimizing between 0...1000 kWh

# 4.b) storage itself:
aSpeicher = Storage('Speicher',
                    # defining flows:
                    inFlow  = cFlow('Q_th_load', bus = Fernwaerme, nominal_val = 1e4),
                    outFlow = cFlow('Q_th_unload',bus = Fernwaerme, nominal_val = 1e4),
                    capacity_inFlowHours=None,  # None, because invest-size is variable
                    chargeState0_inFlowHours=0,  # empty storage at beginning
                    # charge_state_end_min = 3, # min charge state and end
                    charge_state_end_max=10,  # max charge state and end
                    eta_load=0.9, eta_unload=1,  # efficiency of (un)-loading
                    fracLossPerHour=0.08,  # loss of storage per time
                    avoidInAndOutAtOnce=True,  # no parallel loading and unloading
                    invest_parameters=invest_Speicher) # see above
 

# 5. definition of sinks and sources:
# 5.a) heat load profile:    
aWaermeLast = cSink('Wärmelast',
                    sink = cFlow('Q_th_Last', # name
                                 bus = Fernwaerme, # linked bus
                                 nominal_val = 1, 
                                 min_rel = 0,
                                 val_rel = Q_th_Last)) # fixed values val_rel * nominal_val
# 5.b) gas tarif:
aGasTarif = cSource('Gastarif',
                    source = cFlow('Q_Gas',
                                   bus = Gas, # linked bus
                                   nominal_val = 1000, # defining nominal size
                                   costsPerFlowHour= {costs: 0.04, CO2: 0.3})) 
# 5.c) feed-in of electricity:
aStromEinspeisung = cSink('Einspeisung',
                          sink = cFlow('P_el',
                                       bus = Strom, # linked bus
                                       costsPerFlowHour = -1*np.array(p_el)))# feed-in tariff


##########################
# ## Build energysystem ##
##########################

es = cEnergySystem(aTimeSeries, dt_last=None) # creating System

es.addEffects(costs, CO2, PE) # adding effects
es.addComponents(aGaskessel, aWaermeLast, aGasTarif) # adding components
es.addComponents(aStromEinspeisung) # adding components

if useCHPwithLinearSegments:
    es.addComponents(aKWK2) # adding components
else:
    es.addComponents(aKWK) # adding components
    
es.addComponents(aSpeicher) # adding components

################################
# ## modeling and calculation ##
################################

chosenEsTimeIndexe = None
# chosenEsTimeIndexe = [1,3,5]

# ## modeling "full" calculation:
aCalc = cCalculation('Sim1', es, 'pyomo', chosenEsTimeIndexe)
aCalc.doModelingAsOneSegment()

# print Model-Charactaricstics:
es.printModel()
es.printVariables()
es.printEquations()

solverProps = {'gapFrac': gapFrac, 
               'timelimit': timelimit,
               'solver': solver_name, 
               'displaySolverOutput' : displaySolverOutput,
               }
if solver_name == 'gurobi': solverProps['threads'] = nrOfThreads

# ## solving calculation ##

aCalc.solve(solverProps, nameSuffix = '_' + solver_name)

# -> for analysis of results, see separate postprocessing-script!