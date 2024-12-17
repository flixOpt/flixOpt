import unittest

import numpy as np
from numpy.testing import assert_allclose

import flixOpt as fx

np.random.seed(45)

class Data:
    def __init__(self, length: int):
        self.length = length

        self.thermal_demand = np.arange(0, 30, 10)
        self.electricity_demand = np.arange(1, 10.1, 1)

        self.thermal_demand = self._adjust_length(self.thermal_demand, length)
        self.electricity_demand = self._adjust_length(self.electricity_demand, length)

    def _adjust_length(self, array, new_length: int):
        if len(array) >= new_length:
            return array[:new_length]
        else:
            repeats = (new_length + len(array) - 1) // len(array)  # Calculate how many times to repeat
            extended_array = np.tile(array, repeats)  # Repeat the array
            return extended_array[:new_length]  # Truncate to exact length


class BaseTest(unittest.TestCase):
    def setUp(self):
        fx.setup_logging("DEBUG", 'flixOpt_testing.log')
        self.mip_gap = 0.0001
        self.datetime_array = fx.create_datetime_array('2020-01-01', 5, 'h')

    def create_model(self, datetime_array: np.ndarray[np.datetime64]) -> fx.FlowSystem:
        self.flow_system = fx.FlowSystem(datetime_array)
        self.buses = [fx.Bus('Fernwärme', excess_penalty_per_flow_hour=None), fx.Bus('Gas', excess_penalty_per_flow_hour=None)]
        self.flow_system.add_elements(
                                      fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True))
        data = Data(len(datetime_array))
        self.flow_system.add_elements(
            fx.Sink(label='Wärmelast', sink=fx.Flow(label='Wärme', bus=self.get_element('Fernwärme'),
                                                    fixed_relative_profile=data.thermal_demand, size=1)),
            fx.Source(label='Gastarif', source=fx.Flow(label='Gas', bus=self.get_element('Gas'),
                                                  effects_per_flow_hour=1)),
        )
        return self.flow_system

    def solve_and_load(self, flow_system: fx.FlowSystem) -> fx.results.CalculationResults:
        calculation = fx.FullCalculation('Calculation', flow_system)
        calculation.do_modeling()
        calculation.solve(self.solver(), True)
        results = fx.results.CalculationResults('Calculation', 'results')
        return results

    def get_element(self, label: str):
        elements = {element.label_full: element
                    for element in set(self.flow_system.effect_collection.effects + self.flow_system.components +
                    list(self.flow_system.all_buses) + self.buses)}
        return elements[label]

    def solver(self):
        return fx.solvers.HighsSolver(mip_gap=self.mip_gap, time_limit_seconds=3600, solver_output_to_console=False)


class TestMinimal(BaseTest):
    def create_model(self, datetime_array: np.ndarray[np.datetime64]) -> fx.FlowSystem:
        super().create_model(datetime_array)
        self.flow_system.add_elements(
            fx.linear_converters.Boiler('Boiler', 0.5,
                                        Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                                        Q_th=fx.Flow('Q_th', bus=self.get_element('Fernwärme')))
        )
        return self.flow_system

    def test_01_solve_and_load(self):
        flow_system = self.create_model(self.datetime_array)
        results = self.solve_and_load(flow_system)

    def test_02_results(self):
        flow_system = self.create_model(self.datetime_array)
        results = self.solve_and_load(flow_system)

        assert_allclose(results.effect_results['costs'].all_results['all']['all_sum'],
            80, rtol=self.mip_gap, atol=1e-10)

        assert_allclose(results.component_results['Boiler'].all_results['Q_th']['flow_rate'],
                        [-0., 10., 20., -0., 10.],
                        rtol=self.mip_gap, atol=1e-10)

        assert_allclose(results.effect_results['costs'].all_results['operation']['operation_sum_TS'],
                        [-0., 20., 40., -0., 20.],
                        rtol=self.mip_gap, atol=1e-10)

        assert_allclose(results.effect_results['costs'].all_results['operation']['Shares']['Gastarif__Gas__effects_per_flow_hour'],
                        [-0., 20., 40., -0., 20.],
                        rtol=self.mip_gap, atol=1e-10)

class TestInvestment(BaseTest):

    def test_01_fixed_size(self):
        self.flow_system = self.create_model(self.datetime_array)
        self.flow_system.add_elements(fx.linear_converters.Boiler(
                'Boiler', 0.5,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow('Q_th', bus=self.get_element('Fernwärme'),
                             size= fx.InvestParameters(fixed_size=1000, fix_effects=10, specific_effects=1))))

        self.solve_and_load(self.flow_system)
        boiler = self.get_element('Boiler')
        costs = self.get_element('costs')
        assert_allclose(costs.model.all.sum.result, 80 + 1000*1 + 10, rtol=self.mip_gap, atol=1e-10,
                        err_msg='The total costs does not have the right value')
        assert_allclose(boiler.model.all_variables['Boiler__Q_th__Investment_size'].result,
                        1000, rtol=self.mip_gap, atol=1e-10,
                        err_msg='"Boiler__Q_th__Investment_size" does not have the right value')
        assert_allclose(boiler.model.all_variables['Boiler__Q_th__Investment_isInvested'].result,
                        1, rtol=self.mip_gap, atol=1e-10,
                        err_msg='"Boiler__Q_th__Investment_size" does not have the right value')

    def test_02_optimize_size(self):
        self.flow_system = self.create_model(self.datetime_array)
        self.flow_system.add_elements(fx.linear_converters.Boiler(
            'Boiler', 0.5,
            Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
            Q_th=fx.Flow('Q_th', bus=self.get_element('Fernwärme'),
                         size=fx.InvestParameters(fix_effects=10, specific_effects=1))))

        self.solve_and_load(self.flow_system)
        boiler = self.get_element('Boiler')
        costs = self.get_element('costs')
        assert_allclose(costs.model.all.sum.result, 80 + 20 * 1 + 10, rtol=self.mip_gap, atol=1e-10,
                        err_msg='The total costs does not have the right value')
        assert_allclose(boiler.Q_th.model._investment.size.result,
                        20, rtol=self.mip_gap, atol=1e-10,
                        err_msg='"Boiler__Q_th__Investment_size" does not have the right value')
        assert_allclose(boiler.Q_th.model._investment.is_invested.result,
                        1, rtol=self.mip_gap, atol=1e-10,
                        err_msg='"Boiler__Q_th__IsInvested" does not have the right value')

    def test_03_restrict_size(self):
        self.flow_system = self.create_model(self.datetime_array)
        self.flow_system.add_elements(fx.linear_converters.Boiler(
            'Boiler', 0.5,
            Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
            Q_th=fx.Flow('Q_th', bus=self.get_element('Fernwärme'),
                         size=fx.InvestParameters(minimum_size=40, fix_effects=10, specific_effects=1))))

        self.solve_and_load(self.flow_system)
        boiler = self.get_element('Boiler')
        costs = self.get_element('costs')
        assert_allclose(costs.model.all.sum.result, 80 + 40 * 1 + 10, rtol=self.mip_gap, atol=1e-10,
                        err_msg='The total costs does not have the right value')
        assert_allclose(boiler.Q_th.model._investment.size.result,
                        40, rtol=self.mip_gap, atol=1e-10,
                        err_msg='"Boiler__Q_th__Investment_size" does not have the right value')
        assert_allclose(boiler.Q_th.model._investment.is_invested.result,
                        1, rtol=self.mip_gap, atol=1e-10,
                        err_msg='"Boiler__Q_th__IsInvested" does not have the right value')

    def test_04_optional_invest(self):
        self.flow_system = self.create_model(self.datetime_array)
        self.flow_system.add_elements(fx.linear_converters.Boiler(
            'Boiler', 0.5,
            Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
            Q_th=fx.Flow('Q_th', bus=self.get_element('Fernwärme'),
                         size=fx.InvestParameters(optional=True, minimum_size=40, fix_effects=10, specific_effects=1))),
            fx.linear_converters.Boiler(
                'Boiler_optional', 0.5,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow('Q_th', bus=self.get_element('Fernwärme'),
                             size=fx.InvestParameters(optional=True, minimum_size=50, fix_effects=10, specific_effects=1))))

        self.solve_and_load(self.flow_system)
        boiler = self.get_element('Boiler')
        boiler_optional = self.get_element('Boiler_optional')
        costs = self.get_element('costs')
        assert_allclose(costs.model.all.sum.result, 80 + 40 * 1 + 10, rtol=self.mip_gap, atol=1e-10,
                        err_msg='The total costs does not have the right value')
        assert_allclose(boiler.Q_th.model._investment.size.result,
                        40, rtol=self.mip_gap, atol=1e-10,
                        err_msg='"Boiler__Q_th__Investment_size" does not have the right value')
        assert_allclose(boiler.Q_th.model._investment.is_invested.result,
                        1, rtol=self.mip_gap, atol=1e-10,
                        err_msg='"Boiler__Q_th__IsInvested" does not have the right value')

        assert_allclose(boiler_optional.Q_th.model._investment.size.result,
                        0, rtol=self.mip_gap, atol=1e-10,
                        err_msg='"Boiler__Q_th__Investment_size" does not have the right value')
        assert_allclose(boiler_optional.Q_th.model._investment.is_invested.result,
                        0, rtol=self.mip_gap, atol=1e-10,
                        err_msg='"Boiler__Q_th__IsInvested" does not have the right value')


if __name__ == '__main__':
    unittest.main()