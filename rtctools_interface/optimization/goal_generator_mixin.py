"""Module for a basic optimization problem."""

import logging
from pathlib import Path

import pandas as pd

from rtctools_interface.optimization.base_goal import BaseGoal
from rtctools_interface.optimization.goal_performance_metrics import get_performance_metrics
from rtctools_interface.optimization.helpers.statistics_mixin import StatisticsMixin
from rtctools_interface.utils.read_goals_mixin import ReadGoalsMixin

logger = logging.getLogger("rtctools")


def write_performance_metrics(
    performance_metrics: dict[str, pd.DataFrame], output_path: str | Path
):
    """Write the performance metrics for each goal to a csv file."""
    output_path = Path(output_path) / "performance_metrics"
    output_path.mkdir(parents=True, exist_ok=True)
    for goal_id, performance_metric_table in performance_metrics.items():
        performance_metric_table.to_csv(output_path / f"{goal_id}.csv")


class GoalGeneratorMixin(ReadGoalsMixin, StatisticsMixin):
    # TODO: remove pylint disable below once we have more public functions.
    # pylint: disable=too-few-public-methods
    """Add path goals as specified in the goal_table.

    By default, the mixin looks for the csv in the in the default input
    folder. One can also set the path to the goal_table_file manually
    with the `goal_table_file` class variable.
    """

    calculate_performance_metrics = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, "_all_goal_generator_goals"):
            goals_to_generate = kwargs.get("goals_to_generate", [])
            read_from = kwargs.get("read_goals_from", "csv_table")
            self.load_goals(read_from, goals_to_generate)
        if self.calculate_performance_metrics:
            # A dataframe for each goal defined by the goal generator
            self._performance_metrics = {}
            for goal in self._all_goal_generator_goals:
                self._performance_metrics[goal.goal_id] = pd.DataFrame()

    def path_goals(self):
        """Return the list of path goals."""
        goals = super().path_goals()
        new_goals = self._goal_generator_path_goals
        if new_goals:
            goals = goals + [
                BaseGoal(optimization_problem=self, **goal.__dict__) for goal in new_goals
            ]
        return goals

    def goals(self):
        """Return the list of goals."""
        goals = super().goals()
        new_goals = self._goal_generator_non_path_goals
        if new_goals:
            goals = goals + [
                BaseGoal(optimization_problem=self, **goal.__dict__) for goal in new_goals
            ]
        return goals

    def store_performance_metrics(self, label):
        """Calculate and store performance metrics."""
        results = self.extract_results()
        goal_generator_goals = self._all_goal_generator_goals
        all_base_goals = [
            goal for goal in self.goals() + self.path_goals() if isinstance(goal, BaseGoal)
        ]
        targets = self.collect_range_target_values(all_base_goals)
        for goal in goal_generator_goals:
            next_row = get_performance_metrics(results, goal, targets.get(str(goal.goal_id)))
            if next_row is not None:
                next_row.rename(label, inplace=True)
                self._performance_metrics[goal.goal_id] = pd.concat(
                    [self._performance_metrics[goal.goal_id].T, next_row], axis=1
                ).T

    def priority_completed(self, priority):
        """Tasks after priority optimization."""
        super().priority_completed(priority)
        if self.calculate_performance_metrics:
            self.store_performance_metrics(priority)

    def post(self):
        """Tasks after all optimization steps."""
        super().post()
        if self.calculate_performance_metrics:
            self.store_performance_metrics("final_results")
            write_performance_metrics(self._performance_metrics, self._output_folder)

    def get_performance_metrics(self):
        """Get the plot data and config from the current run."""
        return self._performance_metrics
