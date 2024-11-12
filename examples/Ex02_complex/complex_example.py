# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:19:17 2022
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import numpy as np
import pandas as pd

import flixOpt as fx

# Some options to experiment with
check_penalty = False
excess_penalty = 1e5
use_chp_with_segments = False
time_indices = None  # You can solve the whole calculation or just a custom fraction of time steps [0, 1, 2, 3]

electricity_demand = np.array([70., 80., 90., 90, 90, 90, 90, 90, 90])
heat_demand = np.array([30., 0., 90., 110, 2000, 20, 20, 20, 20]) if check_penalty else np.array([30., 0., 90., 110, 110, 20, 20, 20, 20])
electricity_price = np.array([40., 40., 40., 40, 40, 40, 40, 40, 40])

time_series = fx.create_datetime_array('2020-01-01', len(heat_demand), freq='h')

# ## Bus-Definition: ##
Strom = fx.Bus('Strom', excess_penalty_per_flow_hour=excess_penalty)
Fernwaerme = fx.Bus('Fernwärme', excess_penalty_per_flow_hour=excess_penalty)
Gas = fx.Bus('Gas', excess_penalty_per_flow_hour=excess_penalty)

# Effect-Definition:
Costs = fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True)
CO2 = fx.Effect('CO2', 'kg', 'CO2_e-Emissionen', specific_share_to_other_effects_operation={Costs: 0.2})
PE = fx.Effect('PE', 'kWh_PE', 'Primärenergie', maximum_total=3.5e3)

################################
# ## definition of components ##
################################

# 1. definition of boiler #
Gaskessel = fx.linear_converters.Boiler(
    'Kessel', eta=0.5,  # efficiency ratio
    on_off_parameters=fx.OnOffParameters(effects_per_running_hour={Costs: 0, CO2: 1000}),  # 1000 kg_CO2/h (just for testing)
    # defining flows:
    Q_th=fx.Flow(label='Q_th',  # name
                 bus=Fernwaerme,  # linked bus
                 size=fx.InvestParameters(fix_effects=1000,  # 1000 € investment costs
                                          fixed_size=50,  # fixed size
                                          optional=False,  # forced investment
                                          specific_effects={Costs: 10, PE: 2}),  # specific costs: 10 €/kW; 2 kWh_PE/kW
                 load_factor_max=1.0,  # maximal mean power 50 kW
                 load_factor_min=0.1,  # minimal mean power 5 kW
                 relative_minimum=5 / 50,  # 10 % part load
                 relative_maximum=1,  # 50 kW
                 previous_flow_rate=50,  # 50 kW is value before start
                 flow_hours_total_max=1e6,  # kWh, overall maximum "flow-work"
                 can_be_off=fx.OnOffParameters(
                     on_hours_total_min=0,  # minimum of working hours
                     on_hours_total_max=1000,  # maximum of working hours
                     consecutive_on_hours_max=10,  # maximum of working hours in one step
                     consecutive_off_hours_max=10,  # maximum of off hours in one step
                     # consecutive_on_hours_min = 2, # minimum on hours in one step
                     # consecutive_off_hours_min = 4, # minimum off hours in one step
                     effects_per_switch_on=0.01,  # € per start
                     switch_on_total_max=1000),  # max nr of starts
                 ),
    Q_fu=fx.Flow(label='Q_fu',  # name
                 bus=Gas,  # linked bus
                 size=200,  # kW
                 relative_minimum=0,
                 relative_maximum=1))

# 2. defining of CHP-unit:
aKWK = fx.linear_converters.CHP(
    'BHKW2', eta_th=0.5, eta_el=0.4, on_off_parameters=fx.OnOffParameters(effects_per_switch_on=0.01),
    P_el=fx.Flow('P_el', bus=Strom, size=60, relative_minimum=5 / 60),
    Q_th=fx.Flow('Q_th', bus=Fernwaerme, size=1e3),
    Q_fu=fx.Flow('Q_fu', bus=Gas, size=1e3, previous_flow_rate=20))  # The CHP was ON previously

# 3. defining a alternative CHP-unit with linear segments :
# (Flows mst be defined before the segmented_conversion_factors can be defined)
P_el = fx.Flow('P_el', bus=Strom, size=60, previous_flow_rate=20)
Q_th = fx.Flow('Q_th', bus=Fernwaerme)
Q_fu = fx.Flow('Q_fu', bus=Gas)
segmented_conversion_factors = ({P_el: [(5, 30), (40, 60)],  # elements can also be a time series (can change over time)
                                 Q_th: [(6, 35), (45, 100)],
                                 Q_fu: [(12, 70), (90, 200)]})

aKWK2 = fx.LinearConverter('BHKW2', inputs=[Q_fu], outputs=[P_el, Q_th], segmented_conversion_factors=segmented_conversion_factors,
                           on_off_parameters=fx.OnOffParameters(effects_per_switch_on=0.01))

# 4. definition of storage:
# segmented_investment_effects: [start1, end1, start2, end2, ...]  If start and end are equal, then its a point
segmented_investment_effects = ([(5, 25), (25, 100)],  # size
                                {Costs: [(50, 250), (250, 800)],  # €
                                 PE: [(5, 25), (25, 100)]  # kWh_PE
                                 })

aSpeicher = fx.Storage('Speicher',  # defining flows:
                       charging=fx.Flow('Q_th_load', bus=Fernwaerme, size=1e4),
                       discharging=fx.Flow('Q_th_unload', bus=Fernwaerme, size=1e4),
                       capacity_in_flow_hours=fx.InvestParameters(
                           fix_effects=0,  # no fix costs
                           fixed_size=None,  # variable size
                           effects_in_segments=segmented_investment_effects,  # see above
                           optional=False,  # forced invest
                           specific_effects={Costs: 0.01, CO2: 0.01},  # €/kWh; kg_CO2/kWh
                           minimum_size=0, maximum_size=1000),  # optimizing between 0...1000 kWh
                       initial_charge_state=0,  # empty storage at beginning
                       # minimal_final_charge_state = 3, # min charge state and end
                       maximal_final_charge_state=10,  # max charge state and end
                       eta_charge=0.9, eta_discharge=1,  # efficiency of (un)-loading
                       relative_loss_per_hour=0.08,  # loss of storage per time
                       prevent_simultaneous_charge_and_discharge=True)  # no parallel loading and unloading

# 5. definition of sinks and sources:
# 5.a) heat load profile:    
Waermelast = fx.Sink('Wärmelast', sink=fx.Flow('Q_th_Last',  # name
                                               bus=Fernwaerme,  # linked bus
                                               size=1, relative_maximum=max(heat_demand), fixed_relative_value=heat_demand))  # fixed values fixed_relative_value * size
# 5.b) gas tarif:
Gasbezug = fx.Source('Gastarif', source=fx.Flow('Q_Gas', bus=Gas,  # linked bus
                                                size=1000,  # defining nominal size
                                                effects_per_flow_hour={Costs: 0.04, CO2: 0.3}))
# 5.c) feed-in of electricity:
Stromverkauf = fx.Sink('Einspeisung', sink=fx.Flow('P_el', bus=Strom,  # linked bus
                                                   effects_per_flow_hour=-1 * electricity_price))  # feed-in tariff

# ## Build FlowSystem ##
# Choose which of the above created Elements you want to add to The final FlowSystem. Not added Elements are not part of the Model!

flow_system = fx.FlowSystem(time_series, last_time_step_hours=None)  # creating FlowSystem

flow_system.add_elements(Costs, CO2, PE, Gaskessel, Waermelast, Gasbezug, Stromverkauf, aSpeicher)
flow_system.add_elements(aKWK2) if use_chp_with_segments else flow_system.add_components(aKWK)

print(flow_system)  # Get a string representation of the FlowSystem

calculation = fx.FullCalculation('Sim1', flow_system, 'pyomo', time_indices)
calculation.do_modeling()

# Show variables as str (else, you can find them in the results.yaml file
print(calculation.system_model.description_of_equations())
print(calculation.system_model.description_of_variables())


calculation.solve(fx.solvers.HighsSolver(mip_gap=0.005, time_limit_seconds=30),  # Specify which solver you want to use and specify parameters
                  save_results='results')  # If and where to save results


# You can analyze results directly. But it's better to save them to a file and start from there,
# letting you continue at any time
# See complex_example_evaluation.py
used_time_series = time_series[time_indices] if time_indices else time_series
# Analyze results directly
fig = fx.plotting.with_plotly(data=pd.DataFrame(Gaskessel.Q_th.model.flow_rate.result, index=used_time_series),
                              mode='bar', show=True)


