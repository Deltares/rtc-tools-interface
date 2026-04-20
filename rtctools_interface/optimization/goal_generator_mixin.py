"""Module for a basic optimization problem."""

import logging
from collections.abc import Iterable
from numbers import Real
from pathlib import Path
from tempfile import TemporaryDirectory

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


def write_shadow_price_metrics(shadow_price_metrics: pd.DataFrame, output_path: str | Path):
    """Write the shadow-price summary per priority to a csv file."""
    output_path = Path(output_path) / "performance_metrics"
    output_path.mkdir(parents=True, exist_ok=True)
    if not shadow_price_metrics.empty:
        shadow_price_metrics.to_csv(output_path / "shadow_price_metrics.csv")


def _format_percentage_for_file_name(percentage: float) -> str:
    """Format a percentage value for stable file names."""
    if float(percentage).is_integer():
        return f"{int(percentage)}pct"

    return f"{str(percentage).replace('.', 'p')}pct"


def _normalize_relaxation_percentages(
    relaxation_percentages: Iterable[Real] | Real | None,
) -> tuple[float, ...]:
    """Normalize user-provided relaxation percentages to unique floats."""
    if relaxation_percentages is None:
        return (1.0, 2.0, 5.0, 10.0)

    if isinstance(relaxation_percentages, Real):
        raw_percentages = [relaxation_percentages]
    else:
        raw_percentages = list(relaxation_percentages)

    if not raw_percentages:
        raise ValueError("relaxation_percentages must contain at least one percentage value.")

    normalized_percentages: list[float] = []
    for percentage in raw_percentages:
        if not isinstance(percentage, Real):
            raise TypeError(
                "Each value in relaxation_percentages must be a real number representing a "
                "percentage."
            )

        normalized_percentage = float(percentage)
        if pd.isna(normalized_percentage):
            raise ValueError("relaxation_percentages cannot contain NaN values.")
        if normalized_percentage < 0:
            raise ValueError("relaxation_percentages cannot contain negative values.")
        if normalized_percentage not in normalized_percentages:
            normalized_percentages.append(normalized_percentage)

    return tuple(normalized_percentages)


def write_finite_difference_rhs_sensitivity_analysis(
    sensitivity_analysis: dict[float, pd.DataFrame], output_path: str | Path
):
    """Write finite-difference RHS sensitivity matrices to csv files."""
    output_path = Path(output_path) / "sensitivity_analysis"
    output_path.mkdir(parents=True, exist_ok=True)
    for percentage, matrix in sensitivity_analysis.items():
        if matrix.empty:
            continue
        file_name = (
            f"finite_difference_rhs_sensitivity_{_format_percentage_for_file_name(percentage)}.csv"
        )
        matrix.to_csv(output_path / file_name)


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
        self._goal_generator_init_kwargs = dict(kwargs)
        super().__init__(**kwargs)
        self._priority_objective_values = {}
        self._last_completed_priority = None
        self._finite_difference_rhs_sensitivity_analysis = {}
        self._rhs_sensitivity_source_priority = None
        self._rhs_sensitivity_relaxation_fraction = None
        self._rhs_sensitivity_relaxation_applied = False
        if not hasattr(self, "_all_goal_generator_goals"):
            goals_to_generate = kwargs.get("goals_to_generate", [])
            read_from = kwargs.get("read_goals_from", "csv_table")
            csv_list_separator = kwargs.get("goal_table_list_separator", ",")
            self.load_goals(read_from, goals_to_generate, csv_list_separator)
        if self.calculate_performance_metrics:
            # A dataframe for each goal defined by the goal generator
            self._performance_metrics = {}
            self._active_constraint_metrics = pd.DataFrame()
            self._shadow_price_metrics = pd.DataFrame()
            self._shadow_price_warning_issued = False
            self._performance_metrics_plot_file = None
            self._performance_metrics_plot_figures = {}
            for goal in self._all_goal_generator_goals:
                self._performance_metrics[str(goal.goal_id)] = pd.DataFrame()

    def get_sensitivity_analysis_problem_kwargs(self) -> dict:
        """Return the keyword arguments required to recreate this problem instance."""
        kwargs = dict(self._goal_generator_init_kwargs)
        goal_table_file = getattr(self, "goal_table_file", None)
        if goal_table_file is not None:
            kwargs["goal_table_file"] = goal_table_file
        return kwargs

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

    def _warn_if_priority_metrics_are_missing(self):
        """Warn when only final results were stored, typically due to a skipped super() call."""
        if self._active_constraint_metrics.empty:
            return

        labels = set(self._active_constraint_metrics.index)
        if labels == {"final_results"}:
            logger.warning(
                "Only the 'final_results' row was collected for performance metrics. "
                "If your optimization problem overrides priority_completed(), make sure it calls "
                "super().priority_completed(priority)."
            )

    def store_performance_metrics(self, label, *, current_priority=None):
        """Calculate and store performance metrics."""
        results = self.extract_results()
        goal_generator_goals = self._all_goal_generator_goals
        goals = self.goals()
        path_goals = self.path_goals()
        all_base_goals = [goal for goal in goals + path_goals if isinstance(goal, BaseGoal)]
        targets = self.collect_range_target_values(all_base_goals)
        constraint_metrics = pd.Series(
            self.get_constraint_activity_metrics(current_priority=current_priority)
        )
        constraint_metrics.rename(label, inplace=True)
        self._active_constraint_metrics = pd.concat(
            [self._active_constraint_metrics.T, constraint_metrics], axis=1
        ).T
        shadow_price_metrics = self.get_shadow_price_metrics(current_priority=current_priority)
        shadow_price_row = pd.DataFrame([shadow_price_metrics], index=[label], dtype=float)
        self._shadow_price_metrics = pd.concat(
            [self._shadow_price_metrics, shadow_price_row], axis=0, sort=False
        )

        for goal in goal_generator_goals:
            goal_id_str = str(goal.goal_id)
            next_row = get_performance_metrics(results, goal, targets.get(goal_id_str))
            if next_row is not None:
                next_row.rename(label, inplace=True)
                self._performance_metrics.setdefault(goal_id_str, pd.DataFrame())
                self._performance_metrics[goal_id_str] = pd.concat(
                    [self._performance_metrics[goal_id_str].T, next_row], axis=1
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

    def priority_started(self, priority):
        """Tasks before a priority is solved."""
        super().priority_started(priority)
        if self._rhs_sensitivity_source_priority is None:
            return
        if self._rhs_sensitivity_relaxation_fraction is None:
            return
        if self._rhs_sensitivity_relaxation_applied:
            return
        if int(priority) <= int(self._rhs_sensitivity_source_priority):
            return

        self.relax_goal_constraints_for_priority(
            source_priority=int(self._rhs_sensitivity_source_priority),
            relaxation_fraction=float(self._rhs_sensitivity_relaxation_fraction),
        )
        self._rhs_sensitivity_relaxation_applied = True

    def priority_completed(self, priority):
        """Tasks after priority optimization."""
        super().priority_completed(priority)
        self._last_completed_priority = priority
        self._priority_objective_values[priority] = self.get_current_priority_objective_value()
        if self.calculate_performance_metrics:
            self.store_performance_metrics(priority, current_priority=priority)

    def post(self):
        """Tasks after all optimization steps."""
        super().post()
        if self._last_completed_priority is not None:
            self._priority_objective_values["final_results"] = self._priority_objective_values.get(
                self._last_completed_priority,
                self.get_current_priority_objective_value(),
            )
        if self.calculate_performance_metrics:
            self.store_performance_metrics(
                "final_results", current_priority=self._last_completed_priority
            )
            self._warn_if_priority_metrics_are_missing()
            write_performance_metrics(self._performance_metrics, self._output_folder)
            write_active_constraint_metrics(self._active_constraint_metrics, self._output_folder)
            write_shadow_price_metrics(self._shadow_price_metrics, self._output_folder)

    def get_performance_metrics(self):
        """Get the plot data and config from the current run."""
        return self._performance_metrics

    def get_active_constraint_metrics(self):
        """Get the active-constraint summary grouped by priority label."""
        return self._active_constraint_metrics

    def get_shadow_price_metrics(self, current_priority=None):
        """Get aggregated shadow prices for hard constraints from earlier priorities."""
        if current_priority is not None:
            return self.get_previous_priority_shadow_prices(current_priority=current_priority)
        return self._shadow_price_metrics

    def _run_relaxed_rhs_sensitivity_analysis(
        self, *, source_priority: int, relaxation_percentage: float
    ) -> dict[int, float]:
        """Re-run the optimization with relaxed constraints from one earlier priority."""
        problem_kwargs = self.get_sensitivity_analysis_problem_kwargs()
        with TemporaryDirectory(prefix="rtctools_interface_rhs_sensitivity_") as temp_dir:
            problem_kwargs["output_folder"] = temp_dir
            rerun_problem = self.__class__(**problem_kwargs)
            rerun_problem.calculate_performance_metrics = False
            rerun_problem._rhs_sensitivity_source_priority = int(source_priority)
            rerun_problem._rhs_sensitivity_relaxation_fraction = (
                float(relaxation_percentage) / 100.0
            )
            rerun_problem._rhs_sensitivity_relaxation_applied = False

            success = rerun_problem.optimize()
            if not success:
                raise RuntimeError(
                    "Finite-difference RHS sensitivity analysis failed while solving "
                    f"the relaxed problem for source priority {source_priority} at "
                    f"{relaxation_percentage}% relaxation."
                )

            return {
                int(priority): float(value)
                for priority, value in rerun_problem._priority_objective_values.items()
                if priority != "final_results" and int(priority) > int(source_priority)
            }

    def get_finite_difference_rhs_sensitivity_analysis(
        self,
        relaxation_percentages: Iterable[Real] | Real | None = None,
        output_path: str | Path | None = None,
    ) -> dict[float, pd.DataFrame]:
        """Return finite-difference RHS sensitivity matrices for mixed-integer problems."""
        relaxation_percentages = _normalize_relaxation_percentages(relaxation_percentages)

        if not self.is_mixed_integer_problem():
            logger.info(
                "Finite-difference RHS sensitivity analysis is only generated for mixed-integer "
                "problems. The current problem does not appear to contain discrete variables."
            )
            self._finite_difference_rhs_sensitivity_analysis = {}
            return {}

        if not self._priority_objective_values:
            raise RuntimeError(
                "Finite-difference RHS sensitivity analysis requires an optimized problem. "
                "Call optimize() before requesting the sensitivity matrices."
            )

        priorities = sorted(
            int(priority)
            for priority in self._priority_objective_values
            if priority != "final_results"
        )
        baseline_objectives = {
            int(priority): float(self._priority_objective_values[priority])
            for priority in priorities
        }

        results = {}
        columns = priorities[:-1]
        index = priorities + ["final_results"]
        final_priority = priorities[-1] if priorities else None

        for percentage in relaxation_percentages:
            matrix = pd.DataFrame(index=index, columns=columns, dtype=float)
            for source_priority in columns:
                relaxed_objectives = self._run_relaxed_rhs_sensitivity_analysis(
                    source_priority=source_priority,
                    relaxation_percentage=float(percentage),
                )
                for current_priority, relaxed_value in relaxed_objectives.items():
                    objective_improvement = baseline_objectives[current_priority] - relaxed_value
                    matrix.loc[current_priority, source_priority] = objective_improvement

                if final_priority is not None and final_priority in relaxed_objectives:
                    matrix.loc["final_results", source_priority] = matrix.loc[
                        final_priority, source_priority
                    ]

            results[float(percentage)] = matrix

        self._finite_difference_rhs_sensitivity_analysis = results
        if output_path is None:
            output_path = self._output_folder
        write_finite_difference_rhs_sensitivity_analysis(results, output_path)
        return results

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
            shadow_price_metrics=self.get_shadow_price_metrics(),
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
