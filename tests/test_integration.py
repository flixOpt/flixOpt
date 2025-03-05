import datetime
import os
from typing import Any, Dict, List, Literal

import flixOpt as fx
import numpy as np
import pandas as pd
import pytest

from .conftest import assert_almost_equal_numeric, get_solver, basic_flow_system, simple_flow_system


class TestFlowSystem:
    def test_simple_flow_system(self, simple_flow_system):
        """
        Test the effects of the simple energy system model
        """
        effects = simple_flow_system.flow_system.effects

        # Cost assertions
        assert_almost_equal_numeric(
            effects['costs'].model.total.solution.item(),
            81.88394666666667,
            'costs doesnt match expected value'
        )

        # CO2 assertions
        assert_almost_equal_numeric(
            effects['CO2'].model.total.solution.item(),
            255.09184,
            'CO2 doesnt match expected value'
        )

    def test_model_components(self, simple_flow_system):
        """
        Test the component flows of the simple energy system model
        """
        comps = simple_flow_system.flow_system.components

        # Boiler assertions
        assert_almost_equal_numeric(
            comps['Boiler'].Q_th.model.flow_rate.solution.values,
            [0, 0, 0, 28.4864, 35, 0, 0, 0, 0],
            'Q_th doesnt match expected value',
        )

        # CHP unit assertions
        assert_almost_equal_numeric(
            comps['CHP_unit'].Q_th.model.flow_rate.solution.values,
            [30.0, 26.66666667, 75.0, 75.0, 75.0, 20.0, 20.0, 20.0, 20.0],
            'Q_th doesnt match expected value',
        )

    def test_results_persistence(self, simple_flow_system):
        """
        Test saving and loading results
        """
        # Save results to file
        simple_flow_system.results.to_file()

        # Load results from file
        results = fx.results.CalculationResults.from_file(
            simple_flow_system.folder,
            simple_flow_system.name
        )

        # Verify key variables from loaded results
        assert_almost_equal_numeric(
            results.model.variables['costs|total'].solution.values,
            81.88394666666667,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            results.model.variables['CO2|total'].solution.values,
            255.09184,
            'CO2 doesnt match expected value'
        )


class TestComponents:
    def test_transmission_basic(self, basic_flow_system):
        """Test basic transmission functionality"""
        flow_system = basic_flow_system
        flow_system.add_elements(fx.Bus('Wärme lokal'))

        boiler = fx.linear_converters.Boiler(
            'Boiler',
            eta=0.5,
            Q_th=fx.Flow('Q_th', bus='Wärme lokal'),
            Q_fu=fx.Flow('Q_fu', bus='Gas')
        )

        transmission = fx.Transmission(
            'Rohr',
            relative_losses=0.2,
            absolute_losses=20,
            in1=fx.Flow('Rohr1', 'Wärme lokal', size=fx.InvestParameters(specific_effects=5, maximum_size=1e6)),
            out1=fx.Flow('Rohr2', 'Fernwärme', size=1000),
        )

        flow_system.add_elements(transmission, boiler)
        calculation = fx.FullCalculation('Test_Sim', flow_system)
        calculation.do_modeling()
        calculation.solve(get_solver())

        # Assertions
        assert_almost_equal_numeric(
            transmission.in1.model.on_off.on.solution.values,
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            'On does not work properly'
        )

        assert_almost_equal_numeric(
            transmission.in1.model.flow_rate.solution.values * 0.8 - 20,
            transmission.out1.model.flow_rate.solution.values,
            'Losses are not computed correctly',
        )

    def test_transmission_advanced(self, basic_flow_system):
        """Test advanced transmission functionality"""
        flow_system = basic_flow_system
        flow_system.add_elements(fx.Bus('Wärme lokal'))

        boiler = fx.linear_converters.Boiler(
            'Boiler_Standard',
            eta=0.9,
            Q_th=fx.Flow(
                'Q_th', bus='Fernwärme', relative_maximum=np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
            ),
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
        )

        boiler2 = fx.linear_converters.Boiler(
            'Boiler_backup', eta=0.4, Q_th=fx.Flow('Q_th', bus='Wärme lokal'), Q_fu=fx.Flow('Q_fu', bus='Gas')
        )

        last2 = fx.Sink(
            'Wärmelast2',
            sink=fx.Flow(
                'Q_th_Last',
                bus='Wärme lokal',
                size=1,
                fixed_relative_profile=flow_system.components['Wärmelast'].sink.fixed_relative_profile * np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            ),
        )

        transmission = fx.Transmission(
            'Rohr',
            relative_losses=0.2,
            absolute_losses=20,
            in1=fx.Flow('Rohr1a', bus='Wärme lokal', size=fx.InvestParameters(specific_effects=5, maximum_size=1000)),
            out1=fx.Flow('Rohr1b', 'Fernwärme', size=1000),
            in2=fx.Flow('Rohr2a', 'Fernwärme', size=1000),
            out2=fx.Flow('Rohr2b', bus='Wärme lokal', size=1000),
        )

        flow_system.add_elements(transmission, boiler, boiler2, last2)
        calculation = fx.FullCalculation('Test_Transmission', flow_system)
        calculation.do_modeling()
        calculation.solve(get_solver())

        # Assertions
        assert_almost_equal_numeric(
            transmission.in1.model.on_off.on.solution.values,
            np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
            'On does not work properly'
        )

        assert_almost_equal_numeric(
            calculation.results.model.variables['Rohr(Rohr1b)|flow_rate'].solution.values,
            transmission.out1.model.flow_rate.solution.values,
            'Flow rate of Rohr__Rohr1b is not correct',
        )

        assert_almost_equal_numeric(
            transmission.in1.model.flow_rate.solution.values * 0.8
            - np.array([20 if val > 0.1 else 0 for val in transmission.in1.model.flow_rate.solution.values]),
            transmission.out1.model.flow_rate.solution.values,
            'Losses are not computed correctly',
        )

        assert_almost_equal_numeric(
            transmission.in1.model._investment.size.solution.item(),
            transmission.in2.model._investment.size.solution.item(),
            'The Investments are not equated correctly',
        )


class TestComplex:

    def test_basic_flow_system(self, flow_system_base):
        flow_system = flow_system_base
        calculation = fx.FullCalculation('Test_Complex-Basic', flow_system)
        calculation.do_modeling()
        calculation.solve(get_solver())

        # Assertions
        assert_almost_equal_numeric(
            calculation.results.model['costs|total'].solution.item(),
            -11597.873624489237, 'costs doesnt match expected value'
        )

        assert_almost_equal_numeric(
            calculation.results.model['costs(operation)|total_per_timestep'].solution.values,
            [
                -2.38500000e03,
                -2.21681333e03,
                -2.38500000e03,
                -2.17599000e03,
                -2.35107029e03,
                -2.38500000e03,
                0.00000000e00,
                -1.68897826e-10,
                -2.16914486e-12,
            ],
            'costs doesnt match expected value',
        )

        assert_almost_equal_numeric(
            sum(calculation.results.model['CO2(operation)->costs(operation)'].solution.values),
            258.63729669618675,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            sum(calculation.results.model['Kessel(Q_th)->costs(operation)'].solution.values),
            0.01,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            sum(calculation.results.model['Kessel->costs(operation)'].solution.values),
            -0.0,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            sum(calculation.results.model['Gastarif(Q_Gas)->costs(operation)'].solution.values),
            39.09153113079115,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            sum(calculation.results.model['Einspeisung(P_el)->costs(operation)'].solution.values),
            -14196.61245231646,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            sum(calculation.results.model['KWK->costs(operation)'].solution.values),
            0.0,
            'costs doesnt match expected value',
        )

        assert_almost_equal_numeric(
            calculation.results.model['Kessel(Q_th)->costs(invest)'].solution.values,
            1000 + 500,
            'costs doesnt match expected value',
        )

        assert_almost_equal_numeric(
            calculation.results.model['Speicher->costs(invest)'].solution.values,
            800 + 1,
            'costs doesnt match expected value',
        )

        assert_almost_equal_numeric(
            calculation.results.model['CO2(operation)|total'].solution.values, 1293.1864834809337,
            'CO2 doesnt match expected value'
        )
        assert_almost_equal_numeric(
            calculation.results.model['CO2(invest)|total'].solution.values, 0.9999999999999994,
            'CO2 doesnt match expected value'
        )
        assert_almost_equal_numeric(
            calculation.results.model['Kessel(Q_th)|flow_rate'].solution.values,
            [0, 0, 0, 45, 0, 0, 0, 0, 0],
            'Kessel doesnt match expected value',
        )

        assert_almost_equal_numeric(
            calculation.results.model['KWK(Q_th)|flow_rate'].solution.values,
            [
                7.50000000e01,
                6.97111111e01,
                7.50000000e01,
                7.50000000e01,
                7.39330280e01,
                7.50000000e01,
                0.00000000e00,
                3.12638804e-14,
                3.83693077e-14,
            ],
            'KWK Q_th doesnt match expected value',
        )
        assert_almost_equal_numeric(
            calculation.results.model['KWK(P_el)|flow_rate'].solution.values,
            [
                6.00000000e01,
                5.57688889e01,
                6.00000000e01,
                6.00000000e01,
                5.91464224e01,
                6.00000000e01,
                0.00000000e00,
                2.50111043e-14,
                3.06954462e-14,
            ],
            'KWK P_el doesnt match expected value',
        )

        assert_almost_equal_numeric(
            calculation.results.model['Speicher|netto_discharge'].solution.values,
            [-45.0, -69.71111111, 15.0, -10.0, 36.06697198, -55.0, 20.0, 20.0, 20.0],
            'Speicher nettoFlow doesnt match expected value',
        )
        assert_almost_equal_numeric(
            calculation.results.model['Speicher|charge_state'].solution.values,
            [0.0, 40.5, 100.0, 77.0, 79.84, 37.38582802, 83.89496178, 57.18336484, 32.60869565, 10.0],
            'Speicher nettoFlow doesnt match expected value',
        )

        assert_almost_equal_numeric(
            calculation.results.model['Speicher|SegmentedShares|costs'].solution.values,
            800,
            'Speicher investCosts_segmented_costs doesnt match expected value',
        )

    def test_segments_of_flows(self, flow_system_segments_of_flows):
        flow_system = flow_system_segments_of_flows
        calculation = fx.FullCalculation('Test_Complex-Segments', flow_system)
        calculation.do_modeling()
        calculation.solve(get_solver())
        effects = calculation.flow_system.effects
        comps = calculation.flow_system.components

        # Compare expected values with actual values
        assert_almost_equal_numeric(
            effects['costs'].model.total.solution.item(), -10710.997365760755, 'costs doesnt match expected value'
        )
        assert_almost_equal_numeric(
            effects['CO2'].model.total.solution.item(), 1278.7939026086956, 'CO2 doesnt match expected value'
        )
        assert_almost_equal_numeric(
            comps['Kessel'].Q_th.model.flow_rate.solution.values,
            [0, 0, 0, 45, 0, 0, 0, 0, 0],
            'Kessel doesnt match expected value',
        )
        kwk_flows = {flow.label: flow for flow in comps['KWK'].inputs + comps['KWK'].outputs}
        assert_almost_equal_numeric(
            kwk_flows['Q_th'].model.flow_rate.solution.values,
            [45.0, 45.0, 64.5962087, 100.0, 61.3136, 45.0, 45.0, 12.86469565, 0.0],
            'KWK Q_th doesnt match expected value',
        )
        assert_almost_equal_numeric(
            kwk_flows['P_el'].model.flow_rate.solution.values,
            [40.0, 40.0, 47.12589407, 60.0, 45.93221818, 40.0, 40.0, 10.91784108, -0.0],
            'KWK P_el doesnt match expected value',
        )

        assert_almost_equal_numeric(
            comps['Speicher'].model.netto_discharge.solution.values,
            [-15.0, -45.0, 25.4037913, -35.0, 48.6864, -25.0, -25.0, 7.13530435, 20.0],
            'Speicher nettoFlow doesnt match expected value',
        )

        assert_almost_equal_numeric(
            comps['Speicher'].model.variables['Speicher|SegmentedShares|costs'].solution.values,
            454.74666666666667,
            'Speicher investCosts_segmented_costs doesnt match expected value',
        )


# Advanced Modeling Types Test
class TestModelingTypes:
    @pytest.fixture(params=['full', 'segmented', 'aggregated'])
    def modeling_calculation(self, request):
        """
        Fixture to run calculations with different modeling types
        """
        # Load data
        filename = os.path.join(os.path.dirname(__file__), 'ressources', 'Zeitreihen2020.csv')
        ts_raw = pd.read_csv(filename, index_col=0).sort_index()
        data = ts_raw['2020-01-01 00:00:00':'2020-12-31 23:45:00']['2020-01-01':'2020-01-03 23:45:00']

        # Extract data columns
        P_el_Last = data['P_Netz/MW'].values
        Q_th_Last = data['Q_Netz/MW'].values
        p_el = data['Strompr.€/MWh'].values
        gP = data['Gaspr.€/MWh'].values
        timesteps = pd.DatetimeIndex(data.index)

        # Create effects
        costs = fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True)
        CO2 = fx.Effect('CO2', 'kg', 'CO2_e-Emissionen')
        PE = fx.Effect('PE', 'kWh_PE', 'Primärenergie')

        # Create components (similar to original implementation)
        boiler = fx.linear_converters.Boiler(
            'Kessel',
            eta=0.85,
            Q_th=fx.Flow(label='Q_th', bus='Fernwärme'),
            Q_fu=fx.Flow(
                label='Q_fu',
                bus='Gas',
                size=95,
                relative_minimum=12 / 95,
                previous_flow_rate=0,
                on_off_parameters=fx.OnOffParameters(effects_per_switch_on=1000),
            ),
        )
        chp = fx.linear_converters.CHP(
            'BHKW2',
            eta_th=0.58,
            eta_el=0.22,
            on_off_parameters=fx.OnOffParameters(effects_per_switch_on=24000),
            P_el=fx.Flow('P_el', bus='Strom'),
            Q_th=fx.Flow('Q_th', bus='Fernwärme'),
            Q_fu=fx.Flow('Q_fu', bus='Kohle', size=288, relative_minimum=87 / 288),
        )
        storage = fx.Storage(
            'Speicher',
            charging=fx.Flow('Q_th_load', size=137, bus='Fernwärme'),
            discharging=fx.Flow('Q_th_unload', size=158, bus='Fernwärme'),
            capacity_in_flow_hours=684,
            initial_charge_state=137,
            minimal_final_charge_state=137,
            maximal_final_charge_state=158,
            eta_charge=1,
            eta_discharge=1,
            relative_loss_per_hour=0.001,
            prevent_simultaneous_charge_and_discharge=True,
        )

        TS_Q_th_Last, TS_P_el_Last = fx.TimeSeriesData(Q_th_Last), fx.TimeSeriesData(P_el_Last, agg_weight=0.7)
        heat_load, electricity_load = (
            fx.Sink(
                'Wärmelast', sink=fx.Flow('Q_th_Last', bus='Fernwärme', size=1, fixed_relative_profile=TS_Q_th_Last)
            ),
            fx.Sink('Stromlast', sink=fx.Flow('P_el_Last', bus='Strom', size=1, fixed_relative_profile=TS_P_el_Last)),
        )
        coal_tariff, gas_tariff = (
            fx.Source(
                'Kohletarif',
                source=fx.Flow('Q_Kohle', bus='Kohle', size=1000, effects_per_flow_hour={costs.label: 4.6, CO2.label: 0.3}),
            ),
            fx.Source(
                'Gastarif', source=fx.Flow('Q_Gas', bus='Gas', size=1000, effects_per_flow_hour={costs.label: gP, CO2.label: 0.3})
            ),
        )

        p_feed_in, p_sell = (
            fx.TimeSeriesData(-(p_el - 0.5), agg_group='p_el'),
            fx.TimeSeriesData(p_el + 0.5, agg_group='p_el'),
        )
        electricity_feed_in, electricity_tariff = (
            fx.Sink('Einspeisung', sink=fx.Flow('P_el', bus='Strom', size=1000, effects_per_flow_hour=p_feed_in)),
            fx.Source(
                'Stromtarif',
                source=fx.Flow('P_el', bus='Strom', size=1000, effects_per_flow_hour={costs.label: p_sell, CO2.label: 0.3}),
            ),
        )

        # Create flow system
        flow_system = fx.FlowSystem(timesteps)
        flow_system.add_elements(costs, CO2, PE)
        flow_system.add_elements(
            boiler, heat_load, electricity_load, gas_tariff, coal_tariff, electricity_feed_in, electricity_tariff,
            chp, storage
        )
        flow_system.add_elements(
            fx.Bus('Strom'), fx.Bus('Fernwärme'),
            fx.Bus('Gas'), fx.Bus('Kohle')
        )

        # Create calculation based on modeling type
        modeling_type = request.param
        if modeling_type == 'full':
            calc = fx.FullCalculation('fullModel', flow_system)
            calc.do_modeling()
            calc.solve(get_solver())
        elif modeling_type == 'segmented':
            calc = fx.SegmentedCalculation('segModel', flow_system, timesteps_per_segment=96, overlap_timesteps=1)
            calc.do_modeling_and_solve(get_solver())
        elif modeling_type == 'aggregated':
            calc = fx.AggregatedCalculation(
                'aggModel',
                flow_system,
                fx.AggregationParameters(
                    hours_per_period=6,
                    nr_of_periods=4,
                    fix_storage_flows=False,
                    aggregate_data_and_fix_non_binary_vars=True,
                    percentage_of_period_freedom=0,
                    penalty_of_period_freedom=0,
                    time_series_for_low_peaks=[TS_P_el_Last, TS_Q_th_Last],
                    time_series_for_high_peaks=[TS_Q_th_Last],
                ),
            )
            calc.do_modeling()
            calc.solve(get_solver())

        return calc, modeling_type

    def test_modeling_types_costs(self, modeling_calculation):
        """
        Test total costs for different modeling types
        """
        calc, modeling_type = modeling_calculation

        expected_costs = {
            'full': 343613,
            'segmented': 343613,  # Approximate value
            'aggregated': 342967.0
        }

        if modeling_type in ['full', 'aggregated']:
            assert_almost_equal_numeric(
                calc.results.model['costs|total'].solution.item(),
                expected_costs[modeling_type],
                f'Costs do not match for {modeling_type} modeling type'
            )
        else:
            assert_almost_equal_numeric(
                sum(calc.results.solution_without_overlap('costs(operation)|total_per_timestep')),
                expected_costs[modeling_type],
                f'Costs do not match for {modeling_type} modeling type'
            )

if __name__ == '__main__':
    pytest.main(['-v'])
