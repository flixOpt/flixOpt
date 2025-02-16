import pytest
import pandas as pd
import xarray as xr
import linopy

from flixOpt.core import TimeSeries  # Adjust import based on your module structure

# Helper function to create a test TimeSeries object
def create_test_timeseries():
    data = xr.DataArray([10, 20, 30], coords={'time': pd.date_range('2023-01-01', periods=3)})
    return TimeSeries(data, 'Name')

# Test initialization
def test_initialization():
    ts = create_test_timeseries()
    assert isinstance(ts, TimeSeries)
    assert isinstance(ts.stored_data, xr.DataArray)
    assert ts.stored_data.equals(xr.DataArray([10, 20, 30], coords={'time': pd.date_range('2023-01-01', periods=3)}))

# Test active_timesteps property setter and getter
def test_active_timesteps_setter_getter():
    ts = create_test_timeseries()
    new_index = pd.date_range('2023-01-02', periods=2)
    ts.active_timesteps = new_index
    assert ts.active_timesteps.equals(new_index)
    assert ts.active_data.equals(ts.stored_data.sel(time=new_index))

# Test invalid active_timesteps assignment
def test_invalid_active_timesteps():
    ts = create_test_timeseries()
    with pytest.raises(TypeError):
        ts.active_timesteps = "invalid_index"

# Test restoring data
def test_restore_data():
    ts = create_test_timeseries()
    ts.active_timesteps = pd.date_range('2023-01-02', periods=2)
    ts.restore_data()
    assert ts.active_timesteps.equals(ts.stored_data.indexes['time'])
    assert ts.active_data.equals(ts.stored_data)

    ts = create_test_timeseries()
    old_data = ts.stored_data
    new_data = xr.DataArray([1,2], coords=(pd.date_range("2023-01-02", periods=2, name='time'),))
    ts.stored_data = new_data
    assert ts.active_data.equals(new_data)

    ts.restore_data()  # Restore original data

    assert ts.active_timesteps.equals(old_data.indexes['time'])  # Ensure active_timesteps is reset to full index

    assert ts.active_data.equals(old_data)  # Ensure active_data matches stored_data

# Test arithmetic operations
def test_arithmetic_operations():
    ts1 = create_test_timeseries()
    ts2 = create_test_timeseries()

    # Test addition
    result = ts1 + ts2
    expected = ts1.active_data + ts2.active_data
    xr.testing.assert_equal(result, expected)

    # Test subtraction
    result = ts1 - ts2
    expected = ts1.active_data - ts2.active_data
    xr.testing.assert_equal(result, expected)

    # Test multiplication
    result = ts1 * ts2
    expected = ts1.active_data * ts2.active_data
    xr.testing.assert_equal(result, expected)

    # Test division
    result = ts1 / ts2
    expected = ts1.active_data / ts2.active_data
    xr.testing.assert_equal(result, expected)


# Test setting stored_data
def test_stored_data_setter():
    ts = create_test_timeseries()
    old_data = ts.stored_data
    new_data = xr.DataArray([40, 50, 60], coords={'time': pd.date_range('2023-01-01', periods=3)})
    ts.stored_data = new_data
    assert ts.stored_data.equals(new_data)
    assert ts.active_data.equals(new_data)
    assert ts._backup.equals(old_data)

# Test active_data direct modification prevention
def test_prevent_active_data_modification():
    ts = create_test_timeseries()
    with pytest.raises(AttributeError):
        ts.active_data = object()

# Test active_data default behavior
def test_active_data_default():
    ts = create_test_timeseries()
    ts.active_timesteps = None  # Should default to the full stored_data
    assert ts.active_data.equals(ts.stored_data)


# Test arithmetic operations with xarray.DataArray
def test_arithmetic_operations_xarray():
    time_idx = pd.date_range('2020-01-01', periods=3, freq='d', name='time')
    periods = pd.Index([2020, 2030], name='period')

    arithmetric_operations(
        xr.DataArray([10, 20, 30], coords=(time_idx,)),
        TimeSeries(xr.DataArray([10, 20, 30], coords=(time_idx,)), 'Name')
    )

    arithmetric_operations(
        xr.DataArray([[10, 20, 30], [1,2,3]], coords={'period': periods, 'time': time_idx}),
        TimeSeries(xr.DataArray([[10, 20, 30], [1,2,3]], coords={'period': periods, 'time': time_idx}),'Name')
    )

def arithmetric_operations(data1: xr.DataArray, ts1: TimeSeries):
    xr.testing.assert_equal(ts1 + data1, data1 + ts1, check_dim_order=True)
    xr.testing.assert_equal(ts1 - data1, data1 - ts1, check_dim_order=True)
    xr.testing.assert_equal(ts1 * data1, data1 * ts1, check_dim_order=True)
    xr.testing.assert_equal(ts1 / data1, data1 / ts1, check_dim_order=True)
    xr.testing.assert_equal(data1 + ts1.active_data, data1 + ts1, check_dim_order=True)
    xr.testing.assert_equal(data1 - ts1.active_data, data1 - ts1, check_dim_order=True)
    xr.testing.assert_equal(data1 * ts1.active_data, data1 * ts1, check_dim_order=True)
    xr.testing.assert_equal(data1 / ts1.active_data, data1 / ts1, check_dim_order=True)


def test_operations_with_linopy():
    index = pd.date_range("2023-01-01", periods=3, name="time")
    period = pd.Index([2020, 2030], name="period")

    m = linopy.Model()
    var1 = m.add_variables(coords=(period, index))
    timeseries1 = TimeSeries(xr.DataArray([[10, 20, 30], [1,2,3]], coords={'period': period, 'time':index}),'Name')
    expr = timeseries1 * var1
    expr + timeseries1
    (expr + timeseries1) / timeseries1
    expr = var1 * timeseries1

    con = m.add_constraints((expr * timeseries1)  <= 10)


if __name__ == "__main__":
    pytest.main()
