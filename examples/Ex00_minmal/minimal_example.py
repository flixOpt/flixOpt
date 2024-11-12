# -*- coding: utf-8 -*-
"""
Energy System Optimization Example
Developed by Felix Panitz and Peter Stange
Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import numpy as np

import flixOpt as fx

# --- Thermal Load Profile ---
thermal_load_profile = np.array([30., 0., 20.])  # Load in f.ex. kW
datetime_series = fx.create_datetime_array('2020-01-01', 3, 'h')

# --- Define Buses ---
electricity_bus = fx.Bus('Electricity')
heat_bus = fx.Bus('District Heating')
fuel_bus = fx.Bus('Natural Gas')

# --- Define Objective Effect (Cost) ---
cost_effect = fx.Effect('costs', '€', 'Cost', is_standard=True, is_objective=True)

# --- Define Components ---
# Boiler with thermal output linked to heat bus and fuel input linked to fuel bus
boiler = fx.linear_converters.Boiler('Boiler', eta=0.5,
    Q_th=fx.Flow(label='Thermal Output', bus=heat_bus, size=50),
    Q_fu=fx.Flow(label='Fuel Input', bus=fuel_bus))

# Heat load (sink) with fixed load profile linked to heat bus
heat_load = fx.Sink('Heat Demand',
    sink=fx.Flow(label='Thermal Load', bus=heat_bus, size=1, relative_maximum=max(thermal_load_profile),
                 fixed_relative_value=thermal_load_profile))

# Gas source with cost effect
gas_source = fx.Source('Natural Gas Tariff',
    source=fx.Flow(label='Gas Flow', bus=fuel_bus, size=1000, effects_per_flow_hour=0.04))  # 0.04 €/kWh

# --- Build Energy System ---
flow_system = fx.FlowSystem(datetime_series)
flow_system.add_elements(cost_effect, boiler, heat_load, gas_source)

# --- Define and Run Calculation ---
calculation = fx.FullCalculation('Simulation1', flow_system)
calculation.do_modeling()

# Solve and save results
calculation.solve(fx.solvers.HighsSolver(), save_results=True)

results = fx.results.CalculationResults(calculation.name, folder='results')
results.plot_operation('District Heating', 'area')

print(calculation.results())
print(f'Look into .yaml and .json file for results')

