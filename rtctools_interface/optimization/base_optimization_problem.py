"""Module for a basic optimization problem."""
from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.goal_programming_mixin import GoalProgrammingMixin
from rtctools.optimization.modelica_mixin import ModelicaMixin

from rtctools_interface.optimization.goal_generator_mixin import GoalGeneratorMixin


class BaseOptimizationProblem(
    GoalGeneratorMixin,
    GoalProgrammingMixin,
    CSVMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    # Ignore too many ancestors, since the use of mixin classes is how rtc-tools is set up.
    # pylint: disable=too-many-ancestors
    """
    Basic optimization problem for a given state.

    :cvar goal_table_file:
        path to csv file containing a list of goals.
    """

    def __init__(
        self,
        goal_table_file=None,
        **kwargs,
    ):
        self.goal_table_file = goal_table_file
        super().__init__(**kwargs)
