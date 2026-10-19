"""Tests for the solver configuration classes."""

import pytest

import flixopt as fx


def test_highs_threads():
    solver = fx.solvers.HighsSolver(threads=2)
    assert solver.threads == 2
    assert solver.options['threads'] == 2


def test_xpress_threads():
    solver = fx.solvers.XpressSolver(threads=2)
    assert solver.threads == 2
    assert solver.options['THREADS'] == 2


@pytest.mark.parametrize('solver_cls', [fx.solvers.HighsSolver, fx.solvers.XpressSolver])
def test_threads_omitted_by_default(solver_cls):
    assert all(value is not None for value in solver_cls().options.values())
    assert 'threads' not in solver_cls().options
    assert 'THREADS' not in solver_cls().options


@pytest.mark.parametrize('solver_cls', [fx.solvers.GurobiSolver, fx.solvers.HighsSolver, fx.solvers.XpressSolver])
def test_positional_args(solver_cls):
    solver = solver_cls(0.01, 60)
    assert solver.mip_gap == 0.01
    assert solver.time_limit_seconds == 60


def test_xpress_options():
    solver = fx.solvers.XpressSolver(
        mip_gap=0.05, time_limit_seconds=30, log_to_console=False, extra_options={'PRESOLVE': 0}
    )
    assert solver.options == {'MIPRELSTOP': 0.05, 'TIMELIMIT': 30, 'OUTPUTLOG': 0, 'PRESOLVE': 0}
