import logging
import os

import pandas as pd

from rtctools_interface.optimization.read_goals import read_goals
from rtctools_interface.optimization.base_goal import BaseGoal

logger = logging.getLogger("rtctools")


class GoalGeneratorMixin:
    """Add path goals as specified in the goal_table.

    By default, the mixin looks for the csv in the in the default input
    folder. One can also set the path to the goal_table_file manually
    with the `goal_table_file` class variable.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, "goal_table_file"):
            self.goal_table_file = os.path.join(self._input_folder, "goal_table.csv")

    def _goal_data_to_goal(self, goal_data: pd.Series):
        """Convert a series with goal data to a BaseGoal."""
        return BaseGoal(optimization_problem=self, **goal_data.to_dict())

    def path_goals(self):
        goal_df = read_goals(self.goal_table_file)
        goals = goal_df.apply(self._goal_data_to_goal, axis=1)
        return goals
