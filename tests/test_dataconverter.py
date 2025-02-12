import numpy as np
import pandas as pd
import pytest
from flixOpt.core import DataConverter  # Update with actual module name


def test_as_series_scalar():
    """Test scalar input conversion."""
    index = pd.date_range("2023-01-01", periods=3)
    result = DataConverter.as_series(42, (index,))

    assert isinstance(result, pd.Series)
    assert (result == 42).all()
    assert result.index.equals(index)

def test_as_series_scalar_2dims():
    """Test scalar input conversion."""
    index = pd.date_range("2023-01-01", periods=3)
    period = pd.Index([2020, 2030])
    result = DataConverter.as_series(42, (period, index))

    assert isinstance(result, pd.Series)
    assert (result == 42).all()
    assert result.index.equals(pd.MultiIndex.from_product([period, index]))


def test_as_series_1d_array():
    """Test 1D NumPy array conversion."""
    index = pd.date_range("2023-01-01", periods=3)
    data = np.array([1, 2, 3])

    result = DataConverter.as_series(data, (index,))

    assert isinstance(result, pd.Series)
    assert (result.values == data).all()
    assert result.index.equals(index)

def test_as_series_1d_array_broadcast():
    """Test 1D NumPy array conversion."""
    index = pd.date_range("2023-01-01", periods=6)
    period = pd.Index([2020, 2030])
    data = np.array([1, 2, 3, 4, 5, 6])

    result = DataConverter.as_series(data, (period, index))

    assert isinstance(result, pd.Series)
    assert (result.values == np.tile(data, 2)).all()
    assert result.index.equals(pd.MultiIndex.from_product([period, index]))


def test_as_series_2d_array():
    """Test 2D NumPy array conversion."""
    index1 = pd.date_range("2023-01-01", periods=2)
    index2 = pd.Index(["A", "B", "C"])

    data = np.array([[1, 2, 3], [4, 5, 6]])
    result = DataConverter.as_series(data, (index1, index2))

    expected_index = pd.MultiIndex.from_product([index1, index2])
    assert isinstance(result, pd.Series)
    assert result.index.equals(expected_index)
    assert (result.values == data.ravel()).all()


def test_as_series_series_matching_index():
    """Test Pandas Series input with matching index."""
    index = pd.date_range("2023-01-01", periods=3)
    data = pd.Series([10, 20, 30], index=index)

    result = DataConverter.as_series(data, (index,))

    assert isinstance(result, pd.Series)
    assert result.equals(data)


def test_as_series_series_mismatching_index():
    """Test Pandas Series with a different index should raise an error."""
    index = pd.date_range("2023-01-01", periods=3)
    wrong_index = pd.date_range("2023-01-02", periods=3)
    data = pd.Series([10, 20, 30], index=wrong_index)

    with pytest.raises(ValueError, match="Series index does not match the provided index"):
        DataConverter.as_series(data, (index,))


def test_as_series_dataframe():
    """Test DataFrame conversion."""
    index1 = pd.date_range("2023-01-01", periods=2)
    index2 = pd.Index(["A", "B", "C"])

    data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=index1, columns=index2)
    result = DataConverter.as_series(data, (index2, index1))

    expected_index = pd.MultiIndex.from_product([index2, index1])
    expected_series = data.stack().swaplevel(0, 1).sort_index()

    assert isinstance(result, pd.Series)
    assert result.index.equals(expected_index)
    assert result.equals(expected_series)


def test_invalid_dims():
    """Test invalid dims input."""
    with pytest.raises(TypeError, match="dims must be a tuple of pandas Index objects"):
        DataConverter.as_series(10, ["not", "an", "index"])


def test_invalid_data_type():
    """Test invalid data type handling."""
    with pytest.raises(TypeError, match="Unsupported data type"):
        DataConverter.as_series({"a": 1}, (pd.Index([1, 2, 3]),))


def test_shape_mismatch():
    """Test shape mismatch between data and index."""
    index1 = pd.Index(["A", "B"])
    index2 = pd.Index(["X", "Y", "Z"])
    data = np.array([[1, 2], [3, 4]])  # Wrong shape

    with pytest.raises(ValueError, match="Shape of data"):
        DataConverter.as_series(data, (index1, index2))
