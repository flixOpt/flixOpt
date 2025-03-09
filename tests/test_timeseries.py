import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixOpt.core import ConversionError, DataConverter, TimeSeries  # Adjust import path as needed


@pytest.fixture
def sample_time_index():
    """Create a sample time index with the required 'time' name."""
    return pd.date_range('2023-01-01', periods=5, freq='D', name='time')


@pytest.fixture
def simple_dataarray(sample_time_index):
    """Create a simple DataArray with time dimension."""
    return xr.DataArray([10, 20, 30, 40, 50], coords={'time': sample_time_index}, dims=['time'])


@pytest.fixture
def sample_timeseries(simple_dataarray):
    """Create a sample TimeSeries object."""
    return TimeSeries(simple_dataarray, name='Test Series')


class TestTimeSeries:
    """Test suite for TimeSeries class."""

    def test_initialization(self, simple_dataarray):
        """Test basic initialization of TimeSeries."""
        ts = TimeSeries(simple_dataarray, name='Test Series')

        # Check basic properties
        assert ts.name == 'Test Series'
        assert ts.aggregation_weight is None
        assert ts.aggregation_group is None

        # Check data initialization
        assert isinstance(ts.stored_data, xr.DataArray)
        assert ts.stored_data.equals(simple_dataarray)
        assert ts.active_data.equals(simple_dataarray)

        # Check backup was created
        assert ts._backup.equals(simple_dataarray)

        # Check active timesteps
        assert ts.active_timesteps.equals(simple_dataarray.indexes['time'])

    def test_initialization_with_aggregation_params(self, simple_dataarray):
        """Test initialization with aggregation parameters."""
        ts = TimeSeries(
            simple_dataarray,
            name='Weighted Series',
            aggregation_weight=0.5,
            aggregation_group='test_group'
        )

        assert ts.name == 'Weighted Series'
        assert ts.aggregation_weight == 0.5
        assert ts.aggregation_group == 'test_group'

    def test_initialization_validation(self, sample_time_index):
        """Test validation during initialization."""
        # Test missing time dimension
        invalid_data = xr.DataArray([1, 2, 3], dims=['invalid_dim'])
        with pytest.raises(ValueError, match='must have a "time" index'):
            TimeSeries(invalid_data, name='Invalid Series')

        # Test multi-dimensional data
        multi_dim_data = xr.DataArray(
            [[1, 2, 3], [4, 5, 6]],
            coords={'dim1': [0, 1], 'time': sample_time_index[:3]},
            dims=['dim1', 'time']
        )
        with pytest.raises(ValueError, match='dimensions of DataArray must be 1'):
            TimeSeries(multi_dim_data, name='Multi-dim Series')

    def test_active_timesteps_getter_setter(self, sample_timeseries, sample_time_index):
        """Test active_timesteps getter and setter."""
        # Initial state should use all timesteps
        assert sample_timeseries.active_timesteps.equals(sample_time_index)

        # Set to a subset
        subset_index = sample_time_index[1:3]
        sample_timeseries.active_timesteps = subset_index
        assert sample_timeseries.active_timesteps.equals(subset_index)

        # Active data should reflect the subset
        assert sample_timeseries.active_data.equals(
            sample_timeseries.stored_data.sel(time=subset_index)
        )

        # Reset to full index
        sample_timeseries.active_timesteps = None
        assert sample_timeseries.active_timesteps.equals(sample_time_index)

        # Test invalid type
        with pytest.raises(TypeError, match="must be a pandas Index"):
            sample_timeseries.active_timesteps = "invalid"

    def test_reset(self, sample_timeseries, sample_time_index):
        """Test reset method."""
        # Set to subset first
        subset_index = sample_time_index[1:3]
        sample_timeseries.active_timesteps = subset_index

        # Reset
        sample_timeseries.reset()

        # Should be back to full index
        assert sample_timeseries.active_timesteps.equals(sample_time_index)
        assert sample_timeseries.active_data.equals(sample_timeseries.stored_data)

    def test_restore_data(self, sample_timeseries, simple_dataarray):
        """Test restore_data method."""
        # Modify the stored data
        new_data = xr.DataArray(
            [1, 2, 3, 4, 5],
            coords={'time': sample_timeseries.active_timesteps},
            dims=['time']
        )

        # Store original data for comparison
        original_data = sample_timeseries.stored_data

        # Set new data
        sample_timeseries.stored_data = new_data
        assert sample_timeseries.stored_data.equals(new_data)

        # Restore from backup
        sample_timeseries.restore_data()

        # Should be back to original data
        assert sample_timeseries.stored_data.equals(original_data)
        assert sample_timeseries.active_data.equals(original_data)

    def test_stored_data_setter(self, sample_timeseries, sample_time_index):
        """Test stored_data setter with different data types."""
        # Test with a Series
        series_data = pd.Series([5, 6, 7, 8, 9], index=sample_time_index)
        sample_timeseries.stored_data = series_data
        assert np.array_equal(
            sample_timeseries.stored_data.values,
            series_data.values
        )

        # Test with a single-column DataFrame
        df_data = pd.DataFrame({'col1': [15, 16, 17, 18, 19]}, index=sample_time_index)
        sample_timeseries.stored_data = df_data
        assert np.array_equal(
            sample_timeseries.stored_data.values,
            df_data['col1'].values
        )

        # Test with a NumPy array
        array_data = np.array([25, 26, 27, 28, 29])
        sample_timeseries.stored_data = array_data
        assert np.array_equal(
            sample_timeseries.stored_data.values,
            array_data
        )

        # Test with a scalar
        sample_timeseries.stored_data = 42
        assert np.all(sample_timeseries.stored_data.values == 42)

        # Test with another DataArray
        another_dataarray = xr.DataArray(
            [30, 31, 32, 33, 34],
            coords={'time': sample_time_index},
            dims=['time']
        )
        sample_timeseries.stored_data = another_dataarray
        assert sample_timeseries.stored_data.equals(another_dataarray)

    def test_stored_data_setter_no_change(self, sample_timeseries):
        """Test stored_data setter when data doesn't change."""
        # Get current data
        current_data = sample_timeseries.stored_data
        current_backup = sample_timeseries._backup

        # Set the same data
        sample_timeseries.stored_data = current_data

        # Backup shouldn't change
        assert sample_timeseries._backup is current_backup  # Should be the same object

    def test_from_datasource(self, sample_time_index):
        """Test from_datasource class method."""
        # Test with scalar
        ts_scalar = TimeSeries.from_datasource(42, 'Scalar Series', sample_time_index)
        assert np.all(ts_scalar.stored_data.values == 42)

        # Test with Series
        series_data = pd.Series([1, 2, 3, 4, 5], index=sample_time_index)
        ts_series = TimeSeries.from_datasource(series_data, 'Series Data', sample_time_index)
        assert np.array_equal(ts_series.stored_data.values, series_data.values)

        # Test with aggregation parameters
        ts_with_agg = TimeSeries.from_datasource(
            series_data,
            'Aggregated Series',
            sample_time_index,
            aggregation_weight=0.7,
            aggregation_group='group1'
        )
        assert ts_with_agg.aggregation_weight == 0.7
        assert ts_with_agg.aggregation_group == 'group1'

    def test_to_json_from_json(self, sample_timeseries):
        """Test to_json and from_json methods."""
        # Test to_json (dictionary only)
        json_dict = sample_timeseries.to_json()
        assert json_dict['name'] == sample_timeseries.name
        assert 'data' in json_dict
        assert 'coords' in json_dict['data']
        assert 'time' in json_dict['data']['coords']

        # Test to_json with file saving
        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = Path(tmpdirname) / 'timeseries.json'
            sample_timeseries.to_json(filepath)
            assert filepath.exists()

            # Test from_json with file loading
            loaded_ts = TimeSeries.from_json(path=filepath)
            assert loaded_ts.name == sample_timeseries.name
            assert np.array_equal(
                loaded_ts.stored_data.values,
                sample_timeseries.stored_data.values
            )

        # Test from_json with dictionary
        loaded_ts_dict = TimeSeries.from_json(data=json_dict)
        assert loaded_ts_dict.name == sample_timeseries.name
        assert np.array_equal(
            loaded_ts_dict.stored_data.values,
            sample_timeseries.stored_data.values
        )

        # Test validation in from_json
        with pytest.raises(ValueError, match="Only one of path and data"):
            TimeSeries.from_json(data=json_dict, path='dummy.json')

    def test_all_equal(self, sample_time_index):
        """Test all_equal property."""
        # All equal values
        equal_data = xr.DataArray(
            [5, 5, 5, 5, 5],
            coords={'time': sample_time_index},
            dims=['time']
        )
        ts_equal = TimeSeries(equal_data, 'Equal Series')
        assert ts_equal.all_equal is True

        # Not all equal
        unequal_data = xr.DataArray(
            [5, 5, 6, 5, 5],
            coords={'time': sample_time_index},
            dims=['time']
        )
        ts_unequal = TimeSeries(unequal_data, 'Unequal Series')
        assert ts_unequal.all_equal is False

    def test_arithmetic_operations(self, sample_timeseries):
        """Test arithmetic operations."""
        # Create a second TimeSeries for testing
        data2 = xr.DataArray(
            [1, 2, 3, 4, 5],
            coords={'time': sample_timeseries.active_timesteps},
            dims=['time']
        )
        ts2 = TimeSeries(data2, 'Second Series')

        # Test operations between two TimeSeries objects
        assert np.array_equal((sample_timeseries + ts2).values, sample_timeseries.active_data.values + ts2.active_data.values)
        assert np.array_equal((sample_timeseries - ts2).values, sample_timeseries.active_data.values - ts2.active_data.values)
        assert np.array_equal((sample_timeseries * ts2).values, sample_timeseries.active_data.values * ts2.active_data.values)
        assert np.array_equal((sample_timeseries / ts2).values, sample_timeseries.active_data.values / ts2.active_data.values)

        # Test operations with DataArrays
        assert np.array_equal((sample_timeseries + data2).values, sample_timeseries.active_data.values + data2.values)
        assert np.array_equal((data2 + sample_timeseries).values, data2.values + sample_timeseries.active_data.values)

        # Test operations with scalars
        assert np.array_equal((sample_timeseries + 5).values, sample_timeseries.active_data.values + 5)
        assert np.array_equal((5 + sample_timeseries).values, 5 + sample_timeseries.active_data.values)

        # Test unary operations
        assert np.array_equal((-sample_timeseries).values, -sample_timeseries.active_data.values)
        assert np.array_equal((+sample_timeseries).values, +sample_timeseries.active_data.values)
        assert np.array_equal((abs(sample_timeseries)).values, abs(sample_timeseries.active_data.values))

    def test_comparison_operations(self, sample_time_index):
        """Test comparison operations."""
        data1 = xr.DataArray([10, 20, 30, 40, 50], coords={'time': sample_time_index}, dims=['time'])
        data2 = xr.DataArray([5, 10, 15, 20, 25], coords={'time': sample_time_index}, dims=['time'])

        ts1 = TimeSeries(data1, 'Series 1')
        ts2 = TimeSeries(data2, 'Series 2')

        # Test __gt__ method
        assert (ts1 > ts2) is True  # All values in ts1 are greater than ts2

        # Test with mixed values
        data3 = xr.DataArray([5, 25, 15, 45, 25], coords={'time': sample_time_index}, dims=['time'])
        ts3 = TimeSeries(data3, 'Series 3')

        assert (ts1 > ts3) is False  # Not all values in ts1 are greater than ts3

    def test_numpy_ufunc(self, sample_timeseries):
        """Test numpy ufunc compatibility."""
        # Test basic numpy functions
        assert np.array_equal(
            np.add(sample_timeseries, 5).values,
            np.add(sample_timeseries.active_data, 5).values
        )

        assert np.array_equal(
            np.multiply(sample_timeseries, 2).values,
            np.multiply(sample_timeseries.active_data, 2).values
        )

        # Test with two TimeSeries objects
        data2 = xr.DataArray(
            [1, 2, 3, 4, 5],
            coords={'time': sample_timeseries.active_timesteps},
            dims=['time']
        )
        ts2 = TimeSeries(data2, 'Second Series')

        assert np.array_equal(
            np.add(sample_timeseries, ts2).values,
            np.add(sample_timeseries.active_data, ts2.active_data).values
        )

    def test_sel_and_isel_properties(self, sample_timeseries):
        """Test sel and isel properties."""
        # Test that sel property works
        selected = sample_timeseries.sel(time=sample_timeseries.active_timesteps[0])
        assert selected.item() == sample_timeseries.active_data.values[0]

        # Test that isel property works
        indexed = sample_timeseries.isel(time=0)
        assert indexed.item() == sample_timeseries.active_data.values[0]
