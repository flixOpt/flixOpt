import unittest
from tests.integration.ressources.Modeling_types import run_model as run_model_modeling_types


class TestModelingTypes(unittest.TestCase):
    def test_full_calculation(self):
        try:
            costs = run_model_modeling_types(modeling_type="full")
            expected_cost = 343613
            self.assertAlmostEqual(first=costs, second=expected_cost, places=-1,
                                   msg=f"Full calculation cost mismatch: Expected {expected_cost}, got {costs}")
        except Exception as e:
            self.fail(f"Testing of Full Calculation failed with exception: {e}")

    def test_aggregated_calculation(self):
        try:
            costs = run_model_modeling_types(modeling_type="aggregated")
            expected_cost = 342967.0
            self.assertAlmostEqual(first=costs, second=expected_cost, places=-3,
                                   msg=f"Aggregated calculation cost mismatch: Expected {expected_cost}, got {costs}")
        except Exception as e:
            self.fail(f"Testing of Aggregated Calculation failed with exception: {e}")

    def test_segmented_calculation(self):
        try:
            costs = run_model_modeling_types(modeling_type="segmented")
            expected_costs = [210569.29522477, 270926.44574398]
            expected_total_cost = sum(expected_costs)
            self.assertAlmostEqual(first=sum(costs), second=expected_total_cost, places=-1,
                                   msg=f"Segmented calculation cost mismatch: Expected {expected_total_cost}, got {sum(costs)}")
        except Exception as e:
            self.fail(f"Testing of Segmented Calculation failed with exception: {e}")
class TestExamples(unittest.TestCase):

    def test_Ex01_simple_example(self):
        try:
            from examples.Ex01_simple import simple_example
            costs = simple_example.aCalc_post.results_struct.costs.all.sum
            expected_cost = 81.883947
            self.assertAlmostEqual(first=costs, second=expected_cost, places=2,
                                   msg=f"Simple Example cost mismatch: Expected {expected_cost}, got {costs}")
            print("Simple Example sucessfully tested")
        except Exception as e:
            self.fail(f"Test failed with exception: {e}")

    def test_Ex02_complex_example(self):
        try:
            from examples.Ex02_complex import example_complex_WithPostprocessing
            costs = example_complex_WithPostprocessing.calc1.results_struct.globalComp.costs.all.sum
            expected_cost = -11597.874
            self.assertAlmostEqual(first=costs, second=expected_cost, places=0,
                                   msg=f"Complex Example cost mismatch: Expected {expected_cost}, got {costs}")

            print("Complex Example sucessfully tested")
        except Exception as e:
            self.fail(f"Test failed with exception: {e}")

    def test_Ex03_full_seg_agg(self):
        try:
            from examples.Ex03_full_seg_agg import Model_and_solve
        except Exception as e:
            self.fail(f"Test failed with exception: {e}")
        try:
            costs_full = Model_and_solve.full.results_struct.costs.all.sum
            expected_cost_full= 342967.0
            self.assertAlmostEqual(first=costs_full, second=expected_cost_full, places=2,
                                   msg=f"Aggregated Modeling cost mismatch: Expected {expected_cost_full}, got {costs_full}")
            print("Full Modeling sucessfully tested")


            costs_agg = Model_and_solve.agg.results_struct.costs.all.sum
            expected_cost_agg = 342967.0
            self.assertAlmostEqual(first=costs_agg, second=expected_cost_agg, places=0,
                                   msg=f"Aggregated Modeling cost mismatch: Expected {expected_cost_agg}, got {costs_agg}")
            print("Aggregated Modeling sucessfully tested")


            costs_seg = Model_and_solve.agg.results_struct.costs.all.sum
            expected_cost_seg = [210569.29522477, 270926.44574398]
            self.assertAlmostEqual(first=sum(costs_seg), second=sum(expected_cost_seg), places=0,
                                   msg=f"Segmented Modeling cost mismatch: Expected {expected_cost_seg}, got {costs_seg}")
            print("Segmented Modeling sucessfully tested")

        except Exception as e:
            self.fail(f"Test failed with exception: {e}")


    #TODO: Create Test for Transportation


    def test_Ex05_minimal(self):
        try:
            from examples.Ex05_minmal import minimal_example
            costs = minimal_example.aCalc_post.results_struct.costs.all.sum
            expected_cost = 4
            self.assertAlmostEqual(first=costs, second=expected_cost, places=1,
                                   msg=f"Complex Example cost mismatch: Expected {expected_cost}, got {costs}")

            print("Complex Example sucessfully tested")
        except Exception as e:
            self.fail(f"Test failed with exception: {e}")







if __name__ == '__main__':
    unittest.main()
