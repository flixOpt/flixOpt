# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:19:17 2022
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""
import numpy as np
import datetime
from flixStructure import *
from flixComps import *
from flixBasicsPublic import *

# ## Solver-Inputs:##
displaySolverOutput = False # ausführlicher Solver-Output.
displaySolverOutput = True  # ausführlicher Solver-Output.
gapFrac = 0.0001
timelimit = 3600

solver_name = 'glpk' # warning, glpk quickly has numerical problems with big and epsilon
# solver_name = 'gurobi'
# solver_name    = 'cbc'
nrOfThreads    = 1

# ## calculation-options - you can vary! ###
useCHPwithLinearSegments = False
# useCHPwithLinearSegments = True
checkPenalty    = False  
excessCosts = None
# excessCosts = 1e5 # default value
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


# Bus-Definition:
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

# definition of components:
invest_Gaskessel = cInvestArgs(fixCosts = 1000,
                               investmentSize_is_fixed = True,
                               investment_is_optional=False,
                               specificCosts= {costs:10, PE:2},
                               min_investmentSize=0,
                               max_investmentSize=1000)
# invest_Gaskessel = None #
aGaskessel = cKessel('Kessel', eta = 0.5, costsPerRunningHour = {costs:0, CO2:1000},#, switchOnCosts = 0
                    Q_th = cFlow(label   = 'Q_th', 
                                 bus = Fernwaerme, 
                                 nominal_val = 50, 
                                 loadFactor_max = 1.0, 
                                 loadFactor_min = 0.1,
                                 min_rel = 5/50, 
                                 max_rel = 1, 
                                 iCanSwitchOff = True, 
                                 onHoursSum_min = 0, 
                                 onHoursSum_max = 1000,
                                 onHours_max = 10,
                                 offHours_max = 10,
                                 switchOnCosts = 0.01, 
                                 switchOn_maxNr = 1000, 
                                 valuesBeforeBegin=[50], 
                                 investArgs = invest_Gaskessel,
                                 sumFlowHours_max = 1e6,
                                 ),
                    Q_fu = cFlow(label   = 'Q_fu', bus = Gas       , nominal_val = 200, min_rel = 0   , max_rel = 1)) 


aKWK  = cKWK('BHKW2', eta_th = 0.5, eta_el = 0.4, switchOnCosts =  0.01,
            P_el = cFlow('P_el',bus = Strom     , nominal_val = 60, min_rel = 5/60, ),
            Q_th = cFlow('Q_th',bus = Fernwaerme, nominal_val = 1e3),
            Q_fu = cFlow('Q_fu',bus = Gas, nominal_val = 1e3),on_valuesBeforeBegin = [1])




# alternative CHP with linear segments :
P_el = cFlow('P_el', bus=Strom, nominal_val=60, max_rel=55)
Q_th = cFlow('Q_th', bus=Fernwaerme)
Q_fu = cFlow('Q_fu', bus=Gas)
# linear segments (eta-definitions than become useless!):
segmentsOfFlows = ({P_el: [5  ,30, 40,60 ], # elements an be list (timeseries)
                   Q_th: [6  ,35, 45,100], 
                   Q_fu: [12 ,70, 90,200]})

aKWK2 = cBaseLinearTransformer('BHKW2', inputs = [Q_fu], outputs = [P_el, Q_th], segmentsOfFlows = segmentsOfFlows, switchOnCosts = 0.01, on_valuesBeforeBegin = [1])




# Anmerkung: Punkte über Segmentstart = Segmentende realisierbar, d.h. z.B. [4,4]
# segmentierte Kosten:
costsInvestsizeSegments = [[5,25,25,100], #kW
                           {costs:[50,250,250,800],#€
                            PE:[5,25,25,100] #kWh_PE
                            }
                           ]

# # alternative input only for standard-effect:
# costsInvestsizeSegments = [[5,25,25,100], #kW
#                             [50,250,250,800],#€
#                           ]

invest_Speicher = cInvestArgs(fixCosts = 0, 
                              investmentSize_is_fixed = False, 
                              costsInInvestsizeSegments = costsInvestsizeSegments, 
                              investment_is_optional=False, 
                              specificCosts= {costs: 0.01, CO2: 0.01}, 
                              min_investmentSize=0, max_investmentSize=1000)


aSpeicher = cStorage('Speicher',
                     inFlow  = cFlow('Q_th_load', bus = Fernwaerme, nominal_val = 1e4),
                     outFlow = cFlow('Q_th_unload',bus = Fernwaerme, nominal_val = 1e4),
                     # capacity_inFlowHours = 30,
                     capacity_inFlowHours=None,
                     chargeState0_inFlowHours=0,
                     # charge_state_end_min = 3,
                     charge_state_end_max=10,
                     eta_load=0.9, eta_unload=1,
                     fracLossPerHour=0.08,
                     avoidInAndOutAtOnce=True,
                     investArgs=invest_Speicher)
 
aWaermeLast       = cSink  ('Wärmelast',sink   = cFlow('Q_th_Last' , bus = Fernwaerme, nominal_val = 1, min_rel = 0, val_rel = Q_th_Last))

aGasTarif         = cSource('Gastarif' ,source = cFlow('Q_Gas'     , bus = Gas  , nominal_val = 1000, costsPerFlowHour= {costs: 0.04, CO2: 0.3}))

aStromEinspeisung = cSink  ('Einspeisung'    ,sink   = cFlow('P_el'      , bus = Strom, costsPerFlowHour = -1*np.array(p_el)))

# ## Build energysystem ##
es = cEnergySystem(aTimeSeries, dt_last=None)
es.addEffects(costs, CO2, PE)
es.addComponents(aGaskessel, aWaermeLast, aGasTarif)
es.addComponents(aStromEinspeisung)

if useCHPwithLinearSegments:
    es.addComponents(aKWK2)
else:
    es.addComponents(aKWK)
    
es.addComponents(aSpeicher)

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

# -> for analysis of results, see postprocessing-script!