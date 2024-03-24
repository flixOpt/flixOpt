import unittest
from tests.functional.modeling_types.Modeling_types import run_model
from nose.tools import assert_almost_equal

class TestModelingTypes(unittest.TestCase):
    def test_full_calculation(self):
        costs = run_model(modeling_type="full")
        expected_cost = 343613
        assert_almost_equal(first=costs, second=expected_cost, places=-1,
                            msg=f"Full calculation cost mismatch: Expected {expected_cost}, got {costs}")

    def test_aggregated_calculation(self):
        costs = run_model(modeling_type="aggregated")
        expected_cost = 342967.0
        assert_almost_equal(first=costs, second=expected_cost, places=-3,
                            msg=f"Aggregated calculation cost mismatch: Expected {expected_cost}, got {costs}")

    def test_segmented_calculation(self):
        costs = run_model(modeling_type="segmented")
        expected_costs = [210569.29522477, 270926.44574398]
        expected_total_cost = sum(expected_costs)
        assert_almost_equal(first=sum(costs), second=expected_total_cost, places=-1,
                            msg=f"Segmented calculation cost mismatch: Expected {expected_total_cost}, got {sum(costs)}")


if __name__ == '__main__':
    unittest.main()
