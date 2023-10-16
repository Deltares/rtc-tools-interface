"""Module for a basic optimization problem."""
from pathlib import Path
from typing import Dict, Union
import logging
import os
import pandas as pd

from rtctools_interface.optimization.base_goal import BaseGoal
from rtctools_interface.optimization.goal_performance_metrics import get_performance_metrics
from rtctools_interface.optimization.helpers.statistics_mixin import StatisticsMixin
from rtctools_interface.optimization.read_goals import read_goals

logger = logging.getLogger("rtctools")


def write_performance_metrics(performance_metrics: Dict[str, pd.DataFrame], output_path: Union[str, Path]):
    """Write the performance metrics for each goal to a csv file."""
    output_path = Path(output_path) / "performance_metrics"
    output_path.mkdir(parents=True, exist_ok=True)
    for goal_id, performance_metric_table in performance_metrics.items():
        performance_metric_table.to_csv(output_path / f"{goal_id}.csv")


class GoalGeneratorMixin(StatisticsMixin):
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

        self._path_goals = read_goals(
            self.goal_table_file, path_goal=True, read_from=self.read_from, goals_to_generate=self.goals_to_generate
        )
        self._non_path_goals = read_goals(
            self.goal_table_file, path_goal=False, read_from=self.read_from, goals_to_generate=self.goals_to_generate
        )
        self._all_goals = self._path_goals + self._non_path_goals
        # A dataframe for each goal
        self.performance_metrics = {}
        for goal in self._all_goals:
            self.performance_metrics[goal.goal_id] = pd.DataFrame()

    def path_goals(self):
        """Return the list of path goals."""
        goals = super().path_goals()
        new_goals = self._path_goals
        if new_goals:
            goals = goals + [BaseGoal(optimization_problem=self, **goal.__dict__) for goal in new_goals]
        return goals

    def goals(self):
        """Return the list of goals."""
        goals = super().goals()
        new_goals = self._non_path_goals
        if new_goals:
            goals = goals + [BaseGoal(optimization_problem=self, **goal.__dict__) for goal in new_goals]
        return goals

    def store_performance_metrics(self, label):
        """Calculate and store performance metrics."""
        results = self.extract_results()
        goal_generator_goals = self._all_goals
        targets = self.collect_range_target_values(goal_generator_goals)
        for goal in goal_generator_goals:
            new_row = get_performance_metrics(results, goal, targets.get(goal.goal_id))
            if new_row is not None:
                new_row.rename(label, inplace=True)
                self.performance_metrics[goal.goal_id] = pd.concat(
                    [self.performance_metrics[goal.goal_id].T, new_row], axis=1
                ).T

    def priority_completed(self, priority):
        """Tasks after priority optimization."""
        super().priority_completed(priority)
        self.store_performance_metrics(priority)

    def post(self):
        """Tasks after all optimization steps."""
        super().post()
        self.store_performance_metrics("final_results")
        write_performance_metrics(self.performance_metrics, self._output_folder)
