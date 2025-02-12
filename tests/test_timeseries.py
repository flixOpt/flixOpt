import pytest
import pandas as pd
import xarray as xr

from flixOpt.core import TimeSeries  # Adjust import based on your module structure

# Helper function to create a test TimeSeries object
def create_test_timeseries():
    data = pd.Series([10, 20, 30], index=pd.date_range('2023-01-01', periods=3))
    return TimeSeries(data)

# Test initialization
def test_initialization():
    ts = create_test_timeseries()
    assert isinstance(ts, TimeSeries)
    assert isinstance(ts.stored_data, pd.Series)
    assert ts.stored_data.equals(pd.Series([10, 20, 30], index=pd.date_range('2023-01-01', periods=3)))

# Test active_index property setter and getter
def test_active_index_setter_getter():
    ts = create_test_timeseries()
    new_index = pd.date_range('2023-01-02', periods=2)
    ts.active_index = new_index
    assert ts.active_index.equals(new_index)
    assert ts.active_data.equals(ts.stored_data.loc[new_index])

# Test invalid active_index assignment
def test_invalid_active_index():
    ts = create_test_timeseries()
    with pytest.raises(TypeError):
        ts.active_index = "invalid_index"

# Test restoring data
def test_restore_data():
    ts = create_test_timeseries()
    ts.active_index = pd.date_range('2023-01-02', periods=2)
    ts.restore_data()
    assert ts.active_index.equals(ts.stored_data.index)
    assert ts.active_data.equals(ts.stored_data)

    ts = create_test_timeseries()
    old_data = ts.stored_data
    new_data = pd.Series([1,2], pd.date_range("2023-01-02", periods=2))
    ts.stored_data = new_data
    assert ts.active_data.equals(new_data)

    ts.restore_data()  # Restore original data

    assert ts.active_index.equals(old_data.index)  # Ensure active_index is reset to full index

    assert ts.active_data.equals(old_data)  # Ensure active_data matches stored_data

# Test arithmetic operations
def test_arithmetic_operations():
    ts1 = create_test_timeseries()
    ts2 = create_test_timeseries()

    # Test addition
    result = ts1 + ts2
    expected = ts1.active_data + ts2.active_data
    pd.testing.assert_series_equal(result, expected)

    # Test subtraction
    result = ts1 - ts2
    expected = ts1.active_data - ts2.active_data
    pd.testing.assert_series_equal(result, expected)

    # Test multiplication
    result = ts1 * ts2
    expected = ts1.active_data * ts2.active_data
    pd.testing.assert_series_equal(result, expected)

    # Test division
    result = ts1 / ts2
    expected = ts1.active_data / ts2.active_data
    pd.testing.assert_series_equal(result, expected)

    # Test floordiv
    result = ts1 // ts2
    expected = ts1.active_data // ts2.active_data
    pd.testing.assert_series_equal(result, expected)

    # Test exponentiation
    result = ts1 ** ts2
    expected = ts1.active_data ** ts2.active_data
    pd.testing.assert_series_equal(result, expected)

# Test setting stored_data
def test_stored_data_setter():
    ts = create_test_timeseries()
    old_data = ts.stored_data
    new_data = pd.Series([40, 50, 60], index=pd.date_range('2023-01-01', periods=3))
    ts.stored_data = new_data
    assert ts.stored_data.equals(new_data)
    assert ts.active_data.equals(new_data)
    assert ts._backup.equals(old_data)

# Test active_data direct modification prevention
def test_prevent_active_data_modification():
    ts = create_test_timeseries()
    with pytest.raises(AttributeError):
        ts.active_data = pd.Series([1, 2, 3], index=pd.date_range('2023-01-01', periods=3))

# Test loc and iloc properties
def test_loc_iloc_properties():
    ts = create_test_timeseries()
    ts.active_index = pd.date_range('2023-01-01', periods=3)
    assert ts.loc['2023-01-02'] == 20
    assert ts.iloc[1] == 20

# Test active_data default behavior
def test_active_data_default():
    ts = create_test_timeseries()
    ts.active_index = None  # Should default to the full stored_data
    assert ts.active_data.equals(ts.stored_data)


# Test arithmetic operations with xarray.DataArray
def test_arithmetic_operations_xarray():
    time_idx = pd.date_range('2020-01-01', periods=3, freq='d', name='time')
    periods = pd.Index([2020, 2030], name='period')

    arithmetric_operations(
        xr.DataArray([10, 20, 30], coords=(time_idx,)),
        TimeSeries(pd.Series([10, 20, 30], index=time_idx))
    )

    arithmetric_operations(
        xr.DataArray([[10, 20, 30], [1,2,3]], coords=(periods, time_idx)),
        TimeSeries(pd.Series([10, 20, 30, 1, 2, 3], index=pd.MultiIndex.from_product([periods, time_idx])))
    )

def arithmetric_operations(data1: xr.DataArray, ts1: TimeSeries):
    xr.testing.assert_equal(ts1 + data1, data1 + ts1, check_dim_order=True)
    xr.testing.assert_equal(ts1 - data1, data1 - ts1, check_dim_order=True)
    xr.testing.assert_equal(ts1 * data1, data1 * ts1, check_dim_order=True)
    xr.testing.assert_equal(ts1 / data1, data1 / ts1, check_dim_order=True)
    if data1.ndim > 1:
        ts1_active = ts1.active_data.to_xarray()
    else:
        ts1_active = ts1.active_data
    xr.testing.assert_equal(data1 + ts1_active, data1 + ts1, check_dim_order=True)
    xr.testing.assert_equal(data1 - ts1_active, data1 - ts1, check_dim_order=True)
    xr.testing.assert_equal(data1 * ts1_active, data1 * ts1, check_dim_order=True)
    xr.testing.assert_equal(data1 / ts1_active, data1 / ts1, check_dim_order=True)



if __name__ == "__main__":
    pytest.main()
