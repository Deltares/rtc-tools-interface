"""Tests for goal-plotting functionalities."""

import matplotlib

matplotlib.use("Agg")

import unittest

from rtctools_interface.simulation.base_simulation_problem import BaseSimulationProblem
from rtctools_interface.simulation.plot_mixin import PlotMixin
from tests.utils.get_test import get_test_data


class BaseSimulationProblemPlotting(PlotMixin, BaseSimulationProblem):
    # Ignore too many ancestors, since the use of mixin classes is
    #   how rtc-tools is set up.
    # Ignore abstract-method not implemented, as this is related how
    #   some todo's are setup in simulation mode.
    # pylint: disable=too-many-ancestors, abstract-method
    """Simulation problem with plotting functionalities."""

    def __init__(
        self,
        plot_table_file,
        **kwargs,
    ):
        self.plot_table_file = plot_table_file
        super().__init__(**kwargs)


class TestPlotMixin(unittest.TestCase):
    """Test for goal-plotting functionalities."""

    def run_test(self, test, plotting_library):
        """Solve an simulation problem."""
        test_data = get_test_data(test, optimization=False)
        problem = BaseSimulationProblemPlotting(
            plot_table_file=test_data["plot_table_file"],
            model_folder=test_data["model_folder"],
            model_name=test_data["model_name"],
            input_folder=test_data["model_input_folder"],
            output_folder=test_data["output_folder"],
            plotting_library=plotting_library,
        )
        problem.simulate()

    def test_plot_goals_mixin_plotly(self):
        """Solve several simulation problems."""
        for test in [
            "basic",
        ]:
            self.run_test(test, plotting_library="plotly")

    def test_plot_goals_mixin_matplotlib(self):
        """Solve several simulation problems."""
        for test in [
            "basic",
        ]:
            self.run_test(test, plotting_library="matplotlib")
