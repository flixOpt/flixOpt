# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:26:10 2020
Developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import datetime
import os
import pathlib
import time
from typing import Union, List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import flixOpt as fx
from dirty_test_dir.firsts_tests_new import calculation

# Solver Configuration
gap_frac = 0.0005
solver_name = 'cbc'  # Options: 'cbc', 'gurobi', 'glpk'
solver_props = {
    'mip_gap': gap_frac,
    'solver_name': solver_name,
    'solver_output_to_console': True,
    'threads': 16
}

# Calculation Types
do_full_calc = True
do_segmented_calc = True
do_aggregated_calc = True

# Segmented Properties
nr_of_used_steps = 96 * 1
segment_len = nr_of_used_steps + 1 * 96

# Aggregated Properties
period_length_in_hours = 6
nr_of_typical_periods = 4
use_extreme_values = True
fix_binary_vars_only = False
fix_storage_flows = True
percentage_of_period_freedom = 0
costs_of_period_freedom = 0

# #########################################################################
# ######################  Data Import  ####################################

# Define file path
filename = os.path.join(os.path.dirname(__file__), "Zeitreihen2020.csv")

# Read and preprocess data
ts_raw = pd.read_csv(filename, index_col=0)
ts_raw = ts_raw.sort_index()

# Define the time range (edit as needed)
# Example: ts = ts_raw['2020-01-01':'2020-12-31 23:45:00']
ts = ts_raw['2020-01-01':'2020-01-15 23:45:00']

# Convert index to datetime
ts.set_index(pd.to_datetime(ts.index), inplace=True)

# Apply data conversion to native types
data = convert_numeric_lists_to_arrays(ts.to_dict())

# Convert back to DataFrame
data = pd.DataFrame.from_dict(data)

# Handle any remaining conversion if necessary
data = data.applymap(convert_to_native_types)

time_zero = time.time()

# Select a subset of the data (edit as needed)
# Example: data_sub = data['2020-01-01':'2020-01-07 23:45:00']
data_sub = data['2020-01-01':'2020-01-15 23:45:00']

# Time Index
a_time_index = data_sub.index.to_pydatetime()  # Convert to datetime objects

# ################ Bemerkungen: #################
# Access specific columns
P_el_Last = data_sub['P_Netz/MW']
Q_th_Last = data_sub['Q_Netz/MW']
p_el = data_sub['Strompr.€/MWh']
gP = data_sub['Gaspr.€/MWh']

# Define price bounds
HG_EK_min = 0
HG_EK_max = 100000
HG_VK_min = -100000
HG_VK_max = 0

# Create TimeSeriesData objects
TS_Q_th_Last = fx.TimeSeriesData(Q_th_Last)
TS_P_el_Last = fx.TimeSeriesData(P_el_Last, agg_weight=0.7)

# Aggregated TimeSeries
nr_of_periods = len(P_el_Last)
a_time_series = datetime.datetime(2020, 1, 1) + np.arange(nr_of_periods) * np.timedelta64(15, 'm')

# ##########################################################################
# ######################  Model Setup  ####################################

print('#######################################################################')
print('################### Start of Modeling #################################')

# Bus Definitions
excess_costs = 1e5  # or set to None if not needed
Strom = fx.Bus('el', 'Strom', excess_effects_per_flow_hour=excess_costs)
Fernwaerme = fx.Bus('heat', 'Fernwärme', excess_effects_per_flow_hour=excess_costs)
Gas = fx.Bus('fuel', 'Gas', excess_effects_per_flow_hour=excess_costs)
Kohle = fx.Bus('fuel', 'Kohle', excess_effects_per_flow_hour=excess_costs)

# Effects
costs = fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True)
CO2 = fx.Effect('CO2', 'kg', 'CO2_e-Emissionen')
PE = fx.Effect('PE', 'kWh_PE', 'Primärenergie')

# Component Definitions

# 1. Boiler
a_gaskessel = Boiler(
    'Kessel',
    eta=0.85,
    Q_th=fx.Flow(label='Q_th', bus=Fernwaerme),
    Q_fu=fx.Flow(
        label='Q_fu',
        bus=Gas,
        size=95,
        relative_minimum=12 / 95,
        can_switch_off=True,
        effects_per_switch_on=1000,
        values_before_begin=[0]
    )
)

# 2. CHP
a_kwk = CHP(
    'BHKW2',
    eta_th=0.58,
    eta_el=0.22,
    effects_per_switch_on=24000,
    P_el=fx.Flow('P_el', bus=Strom),
    Q_th=fx.Flow('Q_th', bus=Fernwaerme),
    Q_fu=fx.Flow(
        'Q_fu',
        bus=Kohle,
        size=288,
        relative_minimum=87 / 288
    ),
    on_values_before_begin=[0]
)

# 3. Storage
a_speicher = fx.Storage(
    'Speicher',
    charging=fx.Flow('Q_th_load', size=137, bus=Fernwaerme),
    discharging=fx.Flow('Q_th_unload', size=158, bus=Fernwaerme),
    capacity_in_flow_hours=684,
    initial_charge_state=137,
    minimal_final_charge_state=137,
    maximal_final_charge_state=158,
    eta_load=1,
    eta_unload=1,
    relative_loss_per_hour=0.001,
    prevent_simultaneous_charge_and_discharge=True
)

# 4. Sinks and Sources

# Heat Load Profile
a_waermelast = fx.Sink(
    'Wärmelast',
    sink=fx.Flow(
        'Q_th_Last',
        bus=Fernwaerme,
        size=1,
        fixed_relative_value=TS_Q_th_Last
    )
)

# Electricity Feed-in
a_strom_last = fx.Sink(
    'Stromlast',
    sink=fx.Flow(
        'P_el_Last',
        bus=Strom,
        size=1,
        fixed_relative_value=TS_P_el_Last
    )
)

# Gas Tariff
a_gas_tarif = fx.Source(
    'Gastarif',
    source=fx.Flow(
        'Q_Gas',
        bus=Gas,
        size=1000,
        effects_per_flow_hour={costs: gP, CO2: 0.3}
    )
)

# Coal Tariff
a_kohle_tarif = fx.Source(
    'Kohletarif',
    source=fx.Flow(
        'Q_Kohle',
        bus=Kohle,
        size=1000,
        effects_per_flow_hour={costs: 4.6, CO2: 0.3}
    )
)

# Electricity Tariff and Feed-in
a_strom_einspeisung = fx.Sink(
    'Einspeisung',
    sink=fx.Flow(
        'P_el',
        bus=Strom,
        size=1000,
        effects_per_flow_hour=TS_P_el_Last
    )
)

a_strom_tarif = fx.Source(
    'Stromtarif',
    source=fx.Flow(
        'P_el',
        bus=Strom,
        size=1000,
        effects_per_flow_hour={costs: TS_P_el_Last, CO2: 0.3}
    )
)

# Aggregated TimeSeries Data
p_feed_in = fx.TimeSeriesData(-(p_el - 0.5), agg_group='p_el')
p_sell = fx.TimeSeriesData(p_el + 0.5, agg_group='p_el')

a_strom_einspeisung = fx.Sink(
    'Einspeisung',
    sink=fx.Flow(
        'P_el',
        bus=Strom,
        size=1000,
        effects_per_flow_hour=p_feed_in
    )
)

a_strom_tarif = fx.Source(
    'Stromtarif',
    source=fx.Flow(
        'P_el',
        bus=Strom,
        size=1000,
        effects_per_flow_hour={costs: p_sell, CO2: 0.3}
    )
)

# ##########################################################################
# ######################  Flow System Setup  ###############################

# Initialize FlowSystem
flow_system = fx.FlowSystem(a_time_series, last_time_step_hours=None)

# Add Effects
flow_system.add_effects(costs, CO2, PE)

# Add Components
flow_system.add_components(
    a_gaskessel,
    a_waermelast,
    a_strom_last,
    a_gas_tarif,
    a_kohle_tarif,
    a_strom_einspeisung,
    a_strom_tarif,
    a_kwk,
    a_speicher
)

# ##########################################################################
# ######################  Calculations  ####################################

# Initialize Calculation Objects
calc_full = None
calc_segs = None
calc_agg = None
list_of_calcs = []

# Full Calculation
if do_full_calc:
    calculation = fx.FullCalculation('fullModel', flow_system, 'pyomo', None)
    calculation.do_modeling()
    calculation.solve(fx.solvers.HighsSolver)
    list_of_calcs.append(calculation)

# Segmented Calculation
if do_segmented_calc:
    calculation = fx.SegmentedCalculation('segModel', flow_system, 'pyomo', None)
    calculation.do_modeling_and_solve(fx.solvers.HighsSolver)
    list_of_calcs.append(calculation)

# Aggregated Calculation
if do_aggregated_calc:
    calc_agg = fx.AggregatedCalculation('aggModel', flow_system, 'pyomo')
    calc_agg.do_modeling(
        period_length_in_hours,
        nr_of_typical_periods,
        use_extreme_values,
        fix_storage_flows,
        fix_binary_vars_only,
        percentage_of_period_freedom=percentage_of_period_freedom,
        costs_of_period_freedom=costs_of_period_freedom,
        addPeakMax=[TS_Q_th_Last],
        addPeakMin=[TS_P_el_Last, TS_Q_th_Last]
    )
    calculation.solve(fx.solvers.HighsSolver)
    list_of_calcs.append(calculation)


# Segment Plot
if calc_segs and calc_full:
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(
        calc_segs.time_series_with_end,
        calc_segs.results_struct.Speicher.charge_state,
        '-',
        label='chargeState (complete)'
    )
    for system_model in calc_segs.system_models:
        plt.plot(
            system_model.time_series_with_end,
            system_model.results_struct.Speicher.charge_state,
            ':',
            label='chargeState'
        )
    plt.legend()
    plt.grid()
    plt.show()

    # Q_th_BHKW Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for system_model in calc_segs.system_models:
        plt.plot(system_model.time_series, system_model.results_struct.BHKW2.Q_th.val, label='Q_th_BHKW')
    plt.plot(calc_full.time_series, calc_full.results_struct.BHKW2.Q_th.val, label='Q_th_BHKW')
    plt.legend()
    plt.grid()
    plt.show()

    # Costs Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for system_model in calc_segs.system_models:
        plt.plot(
            system_model.time_series,
            system_model.results_struct.global_comp.costs.operation.sum_TS,
            label='costs'
        )
    plt.plot(
        calc_full.time_series,
        calc_full.results_struct.global_comp.costs.operation.sum_TS,
        ':',
        label='costs (full)'
    )
    plt.legend()
    plt.grid()
    plt.show()

# Penalty Variables
print('######### Sum Korr_... (wenn vorhanden) #########')
if calc_agg:
    aggregation_element = list(calc_agg.flow_system.other_elements)[0]
    for var in aggregation_element.model.variables:
        print(f"{var.label_full}: {sum(var.result())}")

# Time Durations
print('')
print('############ Time Durations #####################')
for a_result in list_of_calcs:
    print(f"{a_result.label}:")
    a_result.listOfModbox[0].describe()
    # print(a_result.infos)
    # print(a_result.duration)
    print(f"costs: {a_result.results_struct.global_comp.costs.all.sum}")

# ##########################################################################
# ######################  Post Processing  ################################

import flixOpt.postprocessing as flix_post

list_of_results = []

if do_full_calc:
    full = flix_post.flix_results(calc_full.label)
    list_of_results.append(full)
    costs = full.results_struct.global_comp.costs.all.sum

if do_aggregated_calc:
    agg = flix_post.flix_results(calc_agg.label)
    list_of_results.append(agg)
    costs = agg.results_struct.global_comp.costs.all.sum

if do_segmented_calc:
    seg = flix_post.flix_results(calc_segs.label)
    list_of_results.append(seg)
    costs = seg.results_struct.global_comp.costs.all.sum

# ###### Plotting #######

# Overview Plot
def uebersichts_plot(a_calc):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title(a_calc.name)

    plot_flow(a_calc, a_calc.results_struct.BHKW2.P_el.val, 'P_el')
    plot_flow(a_calc, a_calc.results_struct.BHKW2.Q_th.val, 'Q_th_BHKW')
    plot_flow(a_calc, a_calc.results_struct.Kessel.Q_th.val, 'Q_th_Kessel')

    plot_on(a_calc, a_calc.results_struct.Kessel.Q_th, 'Q_th_Kessel', -5)
    plot_on(a_calc, a_calc.results_struct.Kessel, 'Kessel', -10)
    plot_on(a_calc, a_calc.results_struct.BHKW2, 'BHKW ', -15)

    plot_flow(a_calc, a_calc.results_struct.Waermelast.Q_th_Last.val, 'Q_th_Last')

    plt.plot(
        a_calc.time_series,
        a_calc.results_struct.global_comp.costs.operation.sum_TS,
        '--',
        label='costs (operating)'
    )

    if hasattr(a_calc.results_struct, 'Speicher'):
        plt.step(
            a_calc.time_series,
            a_calc.results_struct.Speicher.Q_th_unload.val,
            where='post',
            label='Speicher_unload'
        )
        plt.step(
            a_calc.time_series,
            a_calc.results_struct.Speicher.Q_th_load.val,
            where='post',
            label='Speicher_load'
        )
        plt.plot(
            a_calc.time_series_with_end,
            a_calc.results_struct.Speicher.charge_state,
            label='charge_state'
        )

    plt.grid(axis='y')
    plt.legend(loc='center left', bbox_to_anchor=(1., 0.5))
    plt.show()

# Generate Plots
for a_result in list_of_results:
    uebersichts_plot(a_result)

# Component-Specific Plots
for a_result in list_of_results:
    # Ensure type hint for better IDE support
    a_result: flix_post.flix_results
    a_result.plot_in_and_outs('Fernwaerme', stacked=True)

# Penalty Costs
print('## Penalty: ##')
for a_result in list_of_results:
    print(f"Kosten penalty Sim1: {sum(a_result.results_struct.global_comp.penalty.sum_TS)}")

# Load YAML File
with open(agg.filename_infos, 'rb') as f:
    infos = yaml.safe_load(f)

# Periods Order of Aggregated Calculation
print(f"periods_order of aggregated calc: {infos['calculation']['aggregatedProps']['periods_order']}")
