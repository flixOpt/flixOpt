from typing import Dict, List, Optional, Union

import pytest

import flixOpt as fx

from .conftest import (
    assert_almost_equal_numeric,
    flow_system_base,
    flow_system_long,
    flow_system_segments_of_flows,
    simple_flow_system,
)


@pytest.fixture(params=[flow_system_base, flow_system_segments_of_flows, simple_flow_system, flow_system_long])
def flow_system(request):
    fs = request.getfixturevalue(request.param.__name__)
    if isinstance(fs, fx.FlowSystem):
        return fs
    else:
        return fs[0]


def test_flow_system_file_io(flow_system, highs_solver):
    calculation_0 = fx.FullCalculation('IO', flow_system=flow_system)
    calculation_0.do_modeling()
    calculation_0.solve(highs_solver)

    calculation_0.save_results(save_flow_system=True, compression=5)
    flow_system_1 = fx.FlowSystem.from_netcdf(f'results/{calculation_0.name}_flowsystem.nc')

    calculation_1 = fx.FullCalculation('Loaded_IO', flow_system=flow_system_1)
    calculation_1.do_modeling()
    calculation_1.solve(highs_solver)

    assert_almost_equal_numeric(calculation_0.results.model.objective.value,
                                calculation_1.results.model.objective.value,
                                'objective of loaded flow_system doesnt match the original')

    assert_almost_equal_numeric(
        calculation_0.results.model.variables['costs|total'].solution.values,
        calculation_1.results.model.variables['costs|total'].solution.values,
        'costs doesnt match expected value',
    )


def test_flow_system_io(flow_system):
    di = flow_system.as_dict()
    _ = fx.FlowSystem.from_dict(di)

    ds = flow_system.as_dataset()
    _ = fx.FlowSystem.from_dataset(ds)

    print(flow_system)
    flow_system.__repr__()
    flow_system.__str__()


if __name__ == '__main__':
    pytest.main(['-v', '--disable-warnings'])
