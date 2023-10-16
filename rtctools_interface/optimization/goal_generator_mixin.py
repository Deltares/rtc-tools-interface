"""Module for a basic optimization problem."""
import logging
import os

from rtctools_interface.optimization.base_goal import BaseGoal
from rtctools_interface.optimization.goal_performance_metrics import get_performance_metrics
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
        self.performance_metrics = {}

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

    def store_performance_metrics(self, label):
        """Calculate and store performance metrics."""
        results = self.extract_results()
        goal_generator_goals = read_goals(
            self.goal_table_file, path_goal=False, read_from=self.read_from, goals_to_generate=self.goals_to_generate
        ) + read_goals(
            self.goal_table_file, path_goal=True, read_from=self.read_from, goals_to_generate=self.goals_to_generate
        )
        performance_metrics_priority = {}
        for goal in goal_generator_goals:
            performance_metrics_priority[goal.goal_id] = get_performance_metrics(results, goal)

        self.performance_metrics[label] = performance_metrics_priority

    def priority_completed(self, priority):
        super().priority_completed(priority)
        self.store_performance_metrics(priority)

    def post(self):
        """Tasks after optimizing."""
        super().post()
        self.store_performance_metrics("final_results")
