"""Module for a basic optimization problem."""
import pandas as pd

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.goal_programming_mixin import (
    GoalProgrammingMixin
)
from rtctools.optimization.modelica_mixin import ModelicaMixin

from rtctools_interface.optimization.base_goal import BaseGoal
from rtctools_interface.optimization.read_goals import read_goals


class BaseOptimizationProblem(
    GoalProgrammingMixin,
    CSVMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem
):
    r"""
    Basic optimization goal for a given state.

    :cvar goals:
        csv file containing a list of goals.
    """

    def __init__(
        self,
        goals,
        **kwargs,
    ):
        self._goals = goals
        super().__init__(**kwargs)

    def _goal_data_to_goal(self, goal_data: pd.Series):
        """Convert a series with goal data to a BaseGoal."""
        return BaseGoal(
            optimization_problem=self,
            **goal_data.to_dict()
        )

    def path_goals(self):
        goal_df = read_goals(self._goals)
        goals = goal_df.apply(self._goal_data_to_goal, axis=1)
        return goals
