from typing import Dict, List, Optional, Union

import pytest
from conftest import assert_almost_equal_numeric, flow_system_base, flow_system_segments_of_flows, simple_flow_system, flow_system_long

import flixOpt as fx


@pytest.fixture(params=[flow_system_base, flow_system_segments_of_flows, simple_flow_system, flow_system_long])
def flow_system(request):
    fs = request.getfixturevalue(request.param.__name__)
    if isinstance(fs, fx.FlowSystem):
        return fs
    else:
        return fs[0]


def test_flow_system_io(flow_system):
    calculation_0 = fx.FullCalculation('IO', flow_system=flow_system)
    calculation_0.do_modeling()
    calculation_0.solve(fx.solvers.HighsSolver(mip_gap=0.001, time_limit_seconds=30))

    calculation_0.save_results(save_flow_system=True, compression=5)
    flow_system_1 = fx.FlowSystem.from_netcdf(f'results/{calculation_0.name}_flowsystem.nc')

    calculation_1 = fx.FullCalculation('Loaded_IO', flow_system=flow_system_1)
    calculation_1.do_modeling()
    calculation_1.solve(fx.solvers.HighsSolver(mip_gap=0.001, time_limit_seconds=30))

    assert_almost_equal_numeric(calculation_0.results.model.objective.value,
                                calculation_1.results.model.objective.value,
                                'objective of loaded flow_system doesnt match the original')

    assert_almost_equal_numeric(
        calculation_0.results.model.variables['costs|total'].solution.values,
        calculation_1.results.model.variables['costs|total'].solution.values,
        'costs doesnt match expected value',
    )

if __name__ == '__main__':
    pytest.main(['-v', '--disable-warnings'])
