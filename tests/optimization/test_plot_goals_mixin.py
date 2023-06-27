import unittest

from rtctools_interface.optimization.base_optimization_problem import (
    BaseOptimizationProblem,
)
from rtctools_interface.optimization.plot_goals_mixin import PlotGoalsMixin

from .get_test import get_test_data


class BaseOptimizationProblemPlotting(PlotGoalsMixin, BaseOptimizationProblem):
    def __init__(
        self,
        plot_table_file,
        goal_table_file,
        **kwargs,
    ):
        self.plot_table_file = plot_table_file
        super().__init__(goal_table_file=goal_table_file, **kwargs)


class TestPlotGoalsMixin(unittest.TestCase):
    def run_test(self, test):
        test_data = get_test_data(test)
        problem = BaseOptimizationProblemPlotting(
            goal_table_file=test_data["goals_file"],
            plot_table_file=test_data["plot_table_file"],
            model_folder=test_data["model_folder"],
            model_name=test_data["model_name"],
            input_folder=test_data["model_input_folder"],
            output_folder=test_data["output_folder"],
        )
        problem.optimize()

    def test_plot_goals_mixin(self):
        for test in [
            "basic",
            "target_bounds_as_parameters",
            "target_bounds_as_timeseries",
        ]:
            self.run_test(test)
