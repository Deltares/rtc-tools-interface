"""Tests for the base simulation problem class."""

import unittest

from rtctools_interface.simulation.base_simulation_problem import BaseSimulationProblem
from tests.utils.get_test import get_test_data


class TestBasSimulationProblem(unittest.TestCase):
    """Test for the base simulation problem class."""

    def run_test(self, test):
        """Solve an simulation problem."""
        test_data = get_test_data(test, optimization=False)
        problem = BaseSimulationProblem(
            model_folder=test_data["model_folder"],
            model_name=test_data["model_name"],
            input_folder=test_data["model_input_folder"],
            output_folder=test_data["output_folder"],
        )
        problem.simulate()

    # TODO: use pytest instead to parametrise tests.
    def test_base_simulation_problem(self):
        """Solve several simulation problems."""
        for test in ["basic"]:
            self.run_test(test)
