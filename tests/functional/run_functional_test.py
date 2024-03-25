import unittest
from tests.functional.modeling_types.Modeling_types import run_model as run_model_modeling_types
from tests.functional.misc import test_example_complex
from nose.tools import assert_almost_equal

class TestModelingTypes(unittest.TestCase):
    def test_full_calculation(self):
        costs = run_model_modeling_types(modeling_type="full")
        expected_cost = 343613
        assert_almost_equal(first=costs, second=expected_cost, places=-1,
                            msg=f"Full calculation cost mismatch: Expected {expected_cost}, got {costs}")

    def test_aggregated_calculation(self):
        costs = run_model_modeling_types(modeling_type="aggregated")
        expected_cost = 342967.0
        assert_almost_equal(first=costs, second=expected_cost, places=-3,
                            msg=f"Aggregated calculation cost mismatch: Expected {expected_cost}, got {costs}")

    def test_segmented_calculation(self):
        costs = run_model_modeling_types(modeling_type="segmented")
        expected_costs = [210569.29522477, 270926.44574398]
        expected_total_cost = sum(expected_costs)
        assert_almost_equal(first=sum(costs), second=expected_total_cost, places=-1,
                            msg=f"Segmented calculation cost mismatch: Expected {expected_total_cost}, got {sum(costs)}")

class TestMisc(unittest.TestCase):
    def test_example_complex(self):
        costs = test_example_complex.calc1.results_struct.globalComp.costs.all.sum
        expected_cost = -11597.874
        assert_almost_equal(first=costs, second=expected_cost, places=0,
                            msg=f"Segmented calculation cost mismatch: Expected {expected_cost}, got {costs}")

if __name__ == '__main__':
    unittest.main()
