# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:19:17 2022
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, 
  Technische Universität Dresden
"""

import numpy as np

import flixOpt as fx

# Creating Time Series Data
heat_demand_per_h = np.array([30., 0., 90., 110, 110, 20, 20, 20, 20])
power_prices = 1 / 1000 * np.array([80., 80., 80., 80, 80, 80, 80, 80, 80])

time_series = fx.create_datetime_array('2020-01-01', len(heat_demand_per_h))

# define buses for the 3 used medias:
Strom, Fernwaerme, Gas = fx.Bus(label='Strom'), fx.Bus(label='Fernwärme'), fx.Bus(label='Gas')  # balancing nodes

costs = fx.Effect(label='costs', unit='€', description='Kosten',
                  is_standard=True,  # standard effect --> effect values not passed as a dict are treated as costs
                  is_objective=True)  # costs are the objective of optimization (get minimized

CO2 = fx.Effect(label='CO2', unit='kg', description='CO2_e-Emissionen',
                specific_share_to_other_effects_operation={costs: 0.2},
                # unit of CO2 in operation has an effect of 0.2 units of costs
                maximum_operation_per_hour=1000)  # Maximum CO2 per hour

boiler = fx.linear_converters.Boiler(label='Boiler', eta=0.5,
                                     Q_th=fx.Flow(label='Q_th',
                                                  bus=Fernwaerme,
                                                  size=50,
                                                  relative_minimum=0.1,
                                                  relative_maximum=1),
                                     Q_fu=fx.Flow(label='Q_fu',
                                                  bus=Gas))

chp = fx.linear_converters.CHP(label='CHP', eta_th=0.5, eta_el=0.4,
                               P_el=fx.Flow('P_el', bus=Strom,
                                            size=60,  # 60 kW_el
                                            relative_minimum=5/60),  # minimum load of 5 kW_el
                               Q_th=fx.Flow('Q_th', bus=Fernwaerme),
                               Q_fu=fx.Flow('Q_fu', bus=Gas))

storage = fx.Storage(label='Storage',
                     charging=fx.Flow('Q_th_load', bus=Fernwaerme, size=1000),
                     discharging=fx.Flow('Q_th_unload', bus=Fernwaerme, size=1000),
                     capacity_in_flow_hours=fx.InvestParameters(fix_effects=20,
                                                                fixed_size=30,
                                                                optional=False),
                     initial_charge_state=0,  # empty storage at first time step
                     relative_maximum_charge_state=1 / 100 * np.array([80, 70, 80, 80, 80, 80, 80, 80, 80, 80]),
                     eta_charge=0.9, eta_discharge=1,  # loading efficiency factor, unloading efficiency factor
                     relative_loss_per_hour=0.08,  # 8 %/h; 8 percent of storage loading level is lost every hour
                     prevent_simultaneous_charge_and_discharge=True,  # no parallel loading and unloading at one time
                     )

heat_sink = fx.Sink(label='Heat Demand',
                    sink=fx.Flow(label='Q_th_Last',
                                 bus=Fernwaerme,
                                 size=1,
                                 relative_maximum=max(heat_demand_per_h),
                                 fixed_relative_value=heat_demand_per_h))  # fixed profile

# source of gas:
gas_source = fx.Source(label='Gastarif',
                       source=fx.Flow(label='Q_Gas',
                                      bus=Gas,
                                      size=1000,
                                      effects_per_flow_hour={costs: 0.04, CO2: 0.3}))  # 0.04 €/kWh, 0.3 kg_CO2/kWh

power_sink = fx.Sink(label='Einspeisung',
                     sink=fx.Flow(label='P_el',
                                  bus=Strom,
                                  effects_per_flow_hour=-1 * power_prices))

flow_system = fx.FlowSystem(time_series=time_series)
flow_system.add_elements(costs, CO2, boiler, storage, chp, heat_sink, gas_source, power_sink)

flow_system.visualize_network()  # Get a visual representation of the flow system. Useful for validation.

calculation = fx.FullCalculation(name='Sim1',  # name of calculation
                                 flow_system=flow_system,  # flow_system to calculate
                                 modeling_language='pyomo')  # optimization modeling language (only "pyomo" implemented, yet)
calculation.do_modeling()  # Translating the Model to be solvable
calculation.solve(fx.solvers.HighsSolver(), save_results=True)


# Reloading results from file. Can be done at any time later
results = fx.results.CalculationResults(calculation.name, 'results')

# Plotting results
results.plot_operation('Fernwärme', 'area')
results.plot_operation('Fernwärme', 'bar')
results.plot_operation('Fernwärme', 'line')
results.plot_flow_rate('CHP__Q_th', 'line')
results.plot_flow_rate('CHP__Q_th', 'heatmap')
results.to_dataframe('Storage')
print(results.all_results)




