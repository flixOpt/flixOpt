# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:26:10 2020
Developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import flixOpt as fx
from flixOpt import OnOffParameters

# Calculation Types
full, segmented, aggregated = True, True, True

# Segmented Properties
segment_length, overlap_length = 96, 1

# Aggregated Properties
aggregation_parameters = fx.AggregationParameters(
    hours_per_period=6,
    nr_of_periods=4,
    fix_storage_flows=True,
    aggregate_data_and_fix_non_binary_vars=True,
    percentage_of_period_freedom=0,
    penalty_of_period_freedom=0)
keep_extreme_periods = True


# ######################  Data Import  ####################################
data_import = pd.read_csv(pathlib.Path('Zeitreihen2020.csv'), index_col=0).sort_index()
filtered_data = data_import['2020-01-01':'2020-01-15 23:45:00']  # Only use data from specific dates
# filtered_data = data_import[0:500]  # Alternatively filter by index

filtered_data = filtered_data.set_index(pd.to_datetime(filtered_data.index))  # Convert index to datetime
datetime_series = np.array(filtered_data.index).astype('datetime64')  # Extract the datetime for the flowSystem

# Access specific columns
electricity_demand = filtered_data['P_Netz/MW']
heat_demand = filtered_data['Q_Netz/MW']
electricity_price = filtered_data['Strompr.€/MWh']
gas_price = filtered_data['Gaspr.€/MWh']

# Define price bounds
HG_EK_min = 0
HG_EK_max = 100000
HG_VK_min = -100000
HG_VK_max = 0

# Create TimeSeriesData objects, to customize how the Aggregation is performed
TS_heat_demand = fx.TimeSeriesData(heat_demand)
TS_electricity_demand = fx.TimeSeriesData(electricity_demand, agg_weight=0.7)
TS_electricity_price_sell = fx.TimeSeriesData(-(electricity_demand - 0.5), agg_group='p_el')
TS_electricity_price_buy = fx.TimeSeriesData(electricity_price + 0.5, agg_group='p_el')


# Bus Definitions
excess_penalty = 1e5  # or set to None if not needed
Strom = fx.Bus('Strom', excess_penalty_per_flow_hour=excess_penalty)
Fernwaerme = fx.Bus('Fernwärme', excess_penalty_per_flow_hour=excess_penalty)
Gas = fx.Bus('Gas', excess_penalty_per_flow_hour=excess_penalty)
Kohle = fx.Bus('Kohle', excess_penalty_per_flow_hour=excess_penalty)

# Effects
costs = fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True)
CO2 = fx.Effect('CO2', 'kg', 'CO2_e-Emissionen')
PE = fx.Effect('PE', 'kWh_PE', 'Primärenergie')

# Component Definitions

# 1. Boiler
a_gaskessel = fx.linear_converters.Boiler(
    'Kessel',
    eta=0.85,
    Q_th=fx.Flow(label='Q_th', bus=Fernwaerme),
    Q_fu=fx.Flow(
        label='Q_fu',
        bus=Gas,
        size=95,
        relative_minimum=12 / 95,
        previous_flow_rate=20,  # Neededto determin wether the FLow was on
        can_be_off=OnOffParameters(effects_per_switch_on=1000)))

# 2. CHP
a_kwk = fx.linear_converters.CHP(
    'BHKW2',
    eta_th=0.58,
    eta_el=0.22,
    on_off_parameters=fx.OnOffParameters(effects_per_switch_on=24000),
    P_el=fx.Flow('P_el', bus=Strom),
    Q_th=fx.Flow('Q_th', bus=Fernwaerme),
    Q_fu=fx.Flow(
        'Q_fu',
        bus=Kohle,
        size=288,
        relative_minimum=87 / 288,
        previous_flow_rate=100))

# 3. Storage
a_speicher = fx.Storage(
    'Speicher',
    charging=fx.Flow('Q_th_load', size=137, bus=Fernwaerme),
    discharging=fx.Flow('Q_th_unload', size=158, bus=Fernwaerme),
    capacity_in_flow_hours=684,
    initial_charge_state=137,
    minimal_final_charge_state=137,
    maximal_final_charge_state=158,
    eta_charge=1,
    eta_discharge=1,
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
        fixed_relative_value=TS_heat_demand
    )
)

# Electricity Feed-in
a_strom_last = fx.Sink(
    'Stromlast',
    sink=fx.Flow(
        'P_el_Last',
        bus=Strom,
        size=1,
        fixed_relative_value=TS_electricity_demand
    )
)

# Gas Tariff
a_gas_tarif = fx.Source(
    'Gastarif',
    source=fx.Flow(
        'Q_Gas',
        bus=Gas,
        size=1000,
        effects_per_flow_hour={costs: gas_price, CO2: 0.3}
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
        effects_per_flow_hour=TS_electricity_price_sell
    )
)

a_strom_tarif = fx.Source(
    'Stromtarif',
    source=fx.Flow(
        'P_el',
        bus=Strom,
        size=1000,
        effects_per_flow_hour={costs: TS_electricity_price_buy, CO2: 0.3}
    )
)

# ######################  Flow System Setup  ###############################
flow_system = fx.FlowSystem(datetime_series, last_time_step_hours=None)

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

# ######################  Calculations  ####################################

list_of_calculations = []
# Full Calculation
if full:
    calculation = fx.FullCalculation('fullModel', flow_system, 'pyomo', None)
    calculation.do_modeling()
    calculation.solve(fx.solvers.HighsSolver())
    list_of_calculations.append(calculation)

# Segmented Calculation
if segmented:
    calculation = fx.SegmentedCalculation('segModel', flow_system, segment_length, overlap_length)
    calculation.do_modeling_and_solve(fx.solvers.HighsSolver())
    list_of_calculations.append(calculation)

# Aggregated Calculation
if aggregated:
    if keep_extreme_periods:
        aggregation_parameters.time_series_for_high_peaks = [TS_heat_demand],
        aggregation_parameters.time_series_for_low_peaks = [TS_electricity_demand, TS_heat_demand]
    calculation = fx.AggregatedCalculation(
        'aggModel', flow_system,
        aggregation_parameters= aggregation_parameters)
    calculation.do_modeling()
    calculation.solve(fx.solvers.HighsSolver())
    list_of_calculations.append(calculation)


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
