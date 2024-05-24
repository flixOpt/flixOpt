import unittest
from ressources.Modeling_types import run_model as run_model_modeling_types

class TestModelingTypes(unittest.TestCase):
    def assertCostAlmostEqual(self, actual, expected, places, msg):
        self.assertAlmostEqual(first=actual, second=expected, places=places, msg=msg)

    def test_full_calculation(self):
        """Test full calculation modeling type."""
        costs = run_model_modeling_types(modeling_type="full")
        expected_cost = 343613
        self.assertCostAlmostEqual(costs, expected_cost, places=-1,
                                   msg=f"Full calculation cost mismatch: Expected {expected_cost}, got {costs}")

    def test_aggregated_calculation(self):
        """Test aggregated calculation modeling type."""
        costs = run_model_modeling_types(modeling_type="aggregated")
        expected_cost = 342967.0
        self.assertCostAlmostEqual(costs, expected_cost, places=-2,
                                   msg=f"Aggregated calculation cost mismatch: Expected {expected_cost}, got {costs}")

    def test_segmented_calculation(self):
        """Test segmented calculation modeling type."""
        costs = run_model_modeling_types(modeling_type="segmented")
        expected_costs = 343613
        self.assertCostAlmostEqual(costs, expected_costs, places=-1,
                                   msg=f"Segmented calculation cost mismatch: Expected {expected_costs}, got {costs}")

class TestSimple(unittest.TestCase):
    def assertCostAlmostEqual(self, actual, expected, places, msg):
        self.assertAlmostEqual(first=actual, second=expected, places=places, msg=msg)

    def test_Ex01_simple_example(self):
        """Test simple example from Ex01."""
        from examples.Ex01_simple import simple_example
        costs = simple_example.aCalc_post.results_struct.costs.all.sum
        expected_cost = 81.883947
        self.assertCostAlmostEqual(costs, expected_cost, places=2,
                                   msg=f"Simple Example cost mismatch: Expected {expected_cost}, got {costs}")
        print("Simple Example successfully tested")

    def test_Ex02_complex_example(self):
        """Test complex example from Ex02."""
        from examples.Ex02_complex import example_complex_WithPostprocessing
        costs = example_complex_WithPostprocessing.calc1.results_struct.globalComp.costs.all.sum
        expected_cost = -11597.874
        self.assertCostAlmostEqual(costs, expected_cost, places=0,
                                   msg=f"Complex Example cost mismatch: Expected {expected_cost}, got {costs}")
        print("Complex Example successfully tested")

    def test_Ex03_full_seg_agg(self):
        """Test full, segmented, and aggregated modeling from Ex03."""
        from examples.Ex03_full_seg_agg import Model_and_solve

        costs_full = Model_and_solve.full.results_struct.costs.all.sum
        expected_cost_full = 343613
        self.assertCostAlmostEqual(costs_full, expected_cost_full, places=-1,
                                   msg=f"Full Modeling cost mismatch: Expected {expected_cost_full}, got {costs_full}")
        print("Full Modeling successfully tested")

        costs_agg = Model_and_solve.agg.results_struct.costs.all.sum
        expected_cost_agg = 342967.0
        self.assertCostAlmostEqual(costs_agg, expected_cost_agg, places=-2,
                                   msg=f"Aggregated Modeling cost mismatch: Expected {expected_cost_agg}, got {costs_agg}")
        print("Aggregated Modeling successfully tested")

        costs_seg = sum(Model_and_solve.seg.results_struct.costs.operation.sum_TS)
        expected_cost_seg = 343613
        self.assertCostAlmostEqual(costs_seg, expected_cost_seg, places=-1,
                                   msg=f"Segmented Modeling cost mismatch: Expected {expected_cost_seg}, got {costs_seg}")
        print("Segmented Modeling successfully tested")

    # TODO: Create Test for Transportation

    def test_Ex05_minimal(self):
        """Test minimal example from Ex05."""
        from examples.Ex05_minmal import minimal_example
        costs = minimal_example.aCalc_post.results_struct.costs.all.sum
        expected_cost = 4
        self.assertCostAlmostEqual(costs, expected_cost, places=1,
                                   msg=f"Minimal Example cost mismatch: Expected {expected_cost}, got {costs}")
        print("Minimal Example successfully tested")

if __name__ == '__main__':
    unittest.main()