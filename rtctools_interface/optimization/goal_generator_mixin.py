"""Module for a basic optimization problem."""

import logging
from pathlib import Path

import pandas as pd

from rtctools_interface.optimization.base_goal import BaseGoal
from rtctools_interface.optimization.goal_performance_metrics import (
    get_custom_performance_metrics,
    get_performance_metrics,
)
from rtctools_interface.optimization.helpers.statistics_mixin import StatisticsMixin
from rtctools_interface.plotting.performance_metrics_plot_tools import (
    create_performance_metrics_dashboard,
)
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


def write_active_constraint_metrics(
    active_constraint_metrics: pd.DataFrame, output_path: str | Path
):
    """Write the active-constraint summary per priority to a csv file."""
    output_path = Path(output_path) / "performance_metrics"
    output_path.mkdir(parents=True, exist_ok=True)
    if not active_constraint_metrics.empty:
        active_constraint_metrics.to_csv(output_path / "active_constraint_metrics.csv")


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
            csv_list_separator = kwargs.get("goal_table_list_separator", ",")
            self.load_goals(read_from, goals_to_generate, csv_list_separator)
        if self.calculate_performance_metrics:
            # A dataframe for each goal defined by the goal generator
            self._performance_metrics = {}
            self._active_constraint_metrics = pd.DataFrame()
            self._performance_metrics_plot_file = None
            self._performance_metrics_plot_figures = {}
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
        goals = self.goals()
        path_goals = self.path_goals()
        all_base_goals = [goal for goal in goals + path_goals if isinstance(goal, BaseGoal)]
        targets = self.collect_range_target_values(all_base_goals)
        constraint_metrics = pd.Series(self.get_constraint_activity_metrics())
        constraint_metrics.rename(label, inplace=True)
        self._active_constraint_metrics = pd.concat(
            [self._active_constraint_metrics.T, constraint_metrics], axis=1
        ).T

        for goal in goal_generator_goals:
            next_row = get_performance_metrics(results, goal, targets.get(str(goal.goal_id)))
            if next_row is not None:
                next_row.rename(label, inplace=True)
                self._performance_metrics.setdefault(str(goal.goal_id), pd.DataFrame())
                self._performance_metrics[goal.goal_id] = pd.concat(
                    [self._performance_metrics[goal.goal_id].T, next_row], axis=1
                ).T

        custom_goals = [
            (goal, False, i) for i, goal in enumerate(goals) if not isinstance(goal, BaseGoal)
        ]
        custom_goals.extend(
            (goal, True, i) for i, goal in enumerate(path_goals) if not isinstance(goal, BaseGoal)
        )

        for goal, is_path_goal, goal_index in custom_goals:
            target_values = self.collect_target_values_for_goal(goal, is_path_goal=is_path_goal)
            evaluated_values = self.evaluate_goal_function(
                goal, ensemble_member=0, is_path_goal=is_path_goal
            )
            goal_id = self.get_performance_metric_id(
                goal, is_path_goal=is_path_goal, goal_index=goal_index
            )
            next_row = get_custom_performance_metrics(
                evaluated_values,
                None if target_values is None else target_values["target_min"],
                None if target_values is None else target_values["target_max"],
            )
            next_row.rename(label, inplace=True)
            self._performance_metrics.setdefault(goal_id, pd.DataFrame())
            self._performance_metrics[goal_id] = pd.concat(
                [self._performance_metrics[goal_id].T, next_row], axis=1
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
            write_active_constraint_metrics(self._active_constraint_metrics, self._output_folder)

    def get_performance_metrics(self):
        """Get the plot data and config from the current run."""
        return self._performance_metrics

    def get_active_constraint_metrics(self):
        """Get the active-constraint summary grouped by priority label."""
        return self._active_constraint_metrics

    def get_performance_metrics_with_plot(
        self,
        output_path: str | Path | None = None,
        file_name: str = "performance_metrics_dashboard.html",
    ):
        """
        Return performance metrics and also create an interactive HTML dashboard.

        Parameters
        ----------
        output_path : str | Path | None
            Folder where the HTML dashboard should be written.
            If None, it is written to <output_folder>/performance_metrics/.
        file_name : str
            Name of the HTML file.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Same object returned by get_performance_metrics().
        """
        performance_metrics = self.get_performance_metrics()

        if output_path is None:
            output_path = Path(self._output_folder) / "performance_metrics"
        else:
            output_path = Path(output_path)

        figures, html_path = create_performance_metrics_dashboard(
            performance_metrics,
            active_constraint_metrics=self.get_active_constraint_metrics(),
            output_folder=output_path,
            file_name=file_name,
        )

        self._performance_metrics_plot_file = html_path
        self._performance_metrics_plot_figures = figures
        return performance_metrics

    @property
    def performance_metrics_plot_file(self):
        """Path to the most recently generated performance metrics dashboard."""
        return self._performance_metrics_plot_file
