"""Tests for goal-plotting functionalities."""
import unittest

from rtctools_interface.optimization.base_optimization_problem import (
    BaseOptimizationProblem,
)
from rtctools_interface.optimization.plot_mixin import PlotMixin

from .get_test import get_test_data


class BaseOptimizationProblemPlotting(PlotMixin, BaseOptimizationProblem):
    # Ignore too many ancestors, since the use of mixin classes is how rtc-tools is set up.
    # pylint: disable=too-many-ancestors
    """Optimization problem with plotting functionalities."""

    def __init__(
        self,
        plot_table_file,
        goal_table_file,
        **kwargs,
    ):
        self.plot_table_file = plot_table_file
        super().__init__(goal_table_file=goal_table_file, **kwargs)


class TestPlotMixin(unittest.TestCase):
    """Test for goal-plotting functionalities."""

    def run_test(self, test, plotting_library):
        """Solve an optimization problem."""
        test_data = get_test_data(test)
        problem = BaseOptimizationProblemPlotting(
            goal_table_file=test_data["goals_file"],
            plot_table_file=test_data["plot_table_file"],
            model_folder=test_data["model_folder"],
            model_name=test_data["model_name"],
            input_folder=test_data["model_input_folder"],
            output_folder=test_data["output_folder"],
            plotting_library=plotting_library,
        )
        problem.optimize()

    def test_plot_goals_mixin_matplotlib(self):
        """Solve several optimization problems."""
        for test in [
            "basic",
            "target_bounds_as_parameters",
            "target_bounds_as_timeseries",
        ]:
            self.run_test(test, plotting_library="matplotlib")

    def test_plot_goals_mixin_plotly(self):
        """Solve several optimization problems."""
        for test in [
            "basic",
            "target_bounds_as_parameters",
            "target_bounds_as_timeseries",
        ]:
            self.run_test(test, plotting_library="plotly")
