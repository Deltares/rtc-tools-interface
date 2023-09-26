"""Module for a basic optimization problem."""
import logging
import os

from rtctools_interface.optimization.base_goal import BaseGoal
from rtctools_interface.optimization.read_goals import read_goals

logger = logging.getLogger("rtctools")


class GoalGeneratorMixin:
    # TODO: remove pylint disable below once we have more public functions.
    # pylint: disable=too-few-public-methods
    """Add path goals as specified in the goal_table.

    By default, the mixin looks for the csv in the in the default input
    folder. One can also set the path to the goal_table_file manually
    with the `goal_table_file` class variable.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.goals_to_generate = kwargs.get("goals_to_generate", [])
        self.read_from = kwargs.get("read_goals_from", "csv_table")
        if not hasattr(self, "goal_table_file"):
            self.goal_table_file = os.path.join(self._input_folder, "goal_table.csv")

    def path_goals(self):
        """Return the list of path goals."""
        goals = super().path_goals()
        new_goals = read_goals(
            self.goal_table_file, path_goal=True, read_from=self.read_from, goals_to_generate=self.goals_to_generate
        )
        if new_goals:
            goals = goals + [BaseGoal(optimization_problem=self, **goal.__dict__) for goal in new_goals]
        return goals

    def goals(self):
        """Return the list of goals."""
        goals = super().goals()
        new_goals = read_goals(
            self.goal_table_file, path_goal=False, read_from=self.read_from, goals_to_generate=self.goals_to_generate
        )
        if new_goals:
            goals = goals + [BaseGoal(optimization_problem=self, **goal.__dict__) for goal in new_goals]
        return goals
