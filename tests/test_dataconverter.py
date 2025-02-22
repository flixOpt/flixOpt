import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixOpt.core import DataConverter  # Adjust this import to match your project structure


@pytest.fixture
def sample_time_index(request):
    return pd.date_range("2024-01-01", periods=5, freq="D")

@pytest.fixture
def sample_period_index(request):
    return pd.Index(["A", "B", "C"])


def test_scalar_conversion(sample_time_index, sample_period_index):
    # Test scalar conversion without periods
    result = DataConverter.as_dataarray(42, sample_time_index)
    assert isinstance(result, xr.DataArray)
    assert result.shape == (len(sample_time_index),)
    assert np.all(result.values == 42)

    # Test scalar conversion with periods
    result = DataConverter.as_dataarray(42, sample_time_index, sample_period_index)
    assert result.shape == (len(sample_period_index), len(sample_time_index))
    assert np.all(result.values == 42)


def test_series_conversion(sample_time_index, sample_period_index):
    series = pd.Series([1, 2, 3, 4, 5], index=sample_time_index)

    # Test Series conversion without periods
    result = DataConverter.as_dataarray(series, sample_time_index)
    assert isinstance(result, xr.DataArray)
    assert result.shape == (5,)
    assert np.array_equal(result.values, series.values)

    # Test Series conversion with periods (should expand)
    result = DataConverter.as_dataarray(series, sample_time_index, sample_period_index)
    assert result.shape == (3, 5)
    assert np.all(result.values[:, 0] == 1)  # Ensure expansion
    assert np.all(result.isel(period=0).values == series.values)


def test_dataframe_conversion(sample_time_index, sample_period_index):
    df = pd.DataFrame(
        np.arange(15).reshape(5, 3),
        index=sample_time_index,
        columns=sample_period_index,
    )

    # Test DataFrame conversion
    result = DataConverter.as_dataarray(df, sample_time_index, sample_period_index)
    assert isinstance(result, xr.DataArray)
    assert result.shape == (3, 5)
    assert np.array_equal(result.values.T, df.values)


def test_dataframe_single_column_expansion(sample_time_index, sample_period_index):
    df = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5]},
        index=sample_time_index
    )

    # Test expansion
    result = DataConverter.as_dataarray(df, sample_time_index, sample_period_index)
    assert result.shape == (3, 5)
    assert np.all(result.values[:, 0] == 1)
    assert np.all(result.isel(period=0).values == df.values.flatten())


def test_ndarray_conversion(sample_time_index, sample_period_index):
    # Test 1D array conversion (should expand into each period)
    arr_1d = np.array([1, 2, 3, 4, 5])
    result = DataConverter.as_dataarray(arr_1d, sample_time_index, sample_period_index)
    assert result.shape == (3, 5)

    # Test 1D array conversion (should expand into each timestep)
    arr_1d_period = np.array([1, 2, 3])
    result = DataConverter.as_dataarray(arr_1d_period, sample_time_index, sample_period_index)
    assert result.shape == (3, 5)

    # Test 2D array conversion
    arr_2d = np.random.rand(3, 5)
    result = DataConverter.as_dataarray(arr_2d, sample_time_index, sample_period_index)
    assert result.shape == (3, 5)


def test_invalid_inputs(sample_time_index, sample_period_index):
    # Test invalid input type
    with pytest.raises(TypeError):
        DataConverter.as_dataarray("invalid_string", sample_time_index)

    # Test mismatched Series index
    mismatched_series = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("2025-01-01", periods=6, freq="D"))
    with pytest.raises(ValueError):
        DataConverter.as_dataarray(mismatched_series, sample_time_index)

    # Test mismatched DataFrame shape
    df_invalid = pd.DataFrame(np.random.rand(4, 2), index=sample_time_index[:4], columns=sample_period_index[:2])
    with pytest.raises(ValueError):
        DataConverter.as_dataarray(df_invalid, sample_time_index, sample_period_index)

    with pytest.raises(ValueError):
        # Test mismatched Shape. Array should be (3, 5)
        DataConverter.as_dataarray(np.random.rand(5, 3), sample_time_index, sample_period_index)


if __name__ == "__main__":
    pytest.main()
