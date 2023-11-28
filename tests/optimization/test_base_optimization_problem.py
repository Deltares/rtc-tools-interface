"""Tests for the base optimization problem class."""
import unittest

from rtctools_interface.optimization.base_optimization_problem import BaseOptimizationProblem
from tests.utils.get_test import get_test_data



class TestBaseOptimizationProblem(unittest.TestCase):
    """Test for the base optimization problem class."""
    def run_test(self, test):
        """Solve an optimization problem."""
        test_data = get_test_data(test, optimization=True)
        problem = BaseOptimizationProblem(
            goal_table_file=test_data["goals_file"],
            model_folder=test_data["model_folder"],
            model_name=test_data["model_name"],
            input_folder=test_data["model_input_folder"],
            output_folder=test_data["output_folder"],
        )
        problem.optimize()

    # TODO: use pytest instead to parametrise tests.
    def test_base_optimization_problem(self):
        """Solve several optimization problems."""
        for test in ["basic", "target_bounds_as_parameters", "target_bounds_as_timeseries"]:
            self.run_test(test)
