"""Tests for performance metrics in GoalGeneratorMixin."""

import unittest
from pathlib import Path

from rtctools.optimization.goal_programming_mixin import Goal

from rtctools_interface.optimization.base_optimization_problem import BaseOptimizationProblem
from tests.utils.get_test import get_test_data


class WaterLevelRangeGoal(Goal):
    """Keep the water level inside a target band."""

    state = "x"
    function_range = (0.0, 15.0)
    target_min = 5.0
    target_max = 10.0
    priority = 10

    def function(self, optimization_problem, ensemble_member):
        del ensemble_member
        return optimization_problem.state("x")


class MinimizeUGoal(Goal):
    """Minimize the control signal directly as a path goal."""

    priority = 15
    function_nominal = 5.0
    order = 1

    def function(self, optimization_problem, ensemble_member):
        del ensemble_member
        return optimization_problem.state("u")


class MinimizeIntegralUGoal(Goal):
    """Minimize the integral of the control signal."""

    priority = 20
    function_nominal = 25.0
    order = 1

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.integral("u", ensemble_member=ensemble_member)


class CustomGoalOptimizationProblem(BaseOptimizationProblem):
    """Optimization problem with runtime-defined goals only."""

    def goals(self):
        return super().goals() + [MinimizeIntegralUGoal()]

    def path_goals(self):
        return super().path_goals() + [WaterLevelRangeGoal(), MinimizeUGoal()]


class TestGoalGeneratorMixin(unittest.TestCase):
    """Validate metric collection for goals not defined via a goal table."""

    def test_custom_goals_are_included_in_performance_metrics(self):
        test_data = get_test_data("basic", optimization=True)
        problem = CustomGoalOptimizationProblem(
            model_folder=test_data["model_folder"],
            model_name=test_data["model_name"],
            input_folder=test_data["model_input_folder"],
            output_folder=test_data["output_folder"],
        )

        problem.optimize()
        metrics = problem.get_performance_metrics()

        range_goal_id = "WaterLevelRangeGoal__x__path__priority_10__idx_0"
        smooth_goal_id = "MinimizeUGoal__path__priority_15__idx_1"
        integral_goal_id = "MinimizeIntegralUGoal__goal__priority_20__idx_0"

        self.assertEqual(set(metrics), {range_goal_id, smooth_goal_id, integral_goal_id})

        for goal_id in (range_goal_id, smooth_goal_id, integral_goal_id):
            self.assertFalse(metrics[goal_id].empty)
            self.assertIn("final_results", metrics[goal_id].index)
            self.assertIn("timeseries_sum", metrics[goal_id].columns)
            self.assertIn("timeseries_avg", metrics[goal_id].columns)

        self.assertIn(10, metrics[range_goal_id].index)
        self.assertIn("perc_below_target", metrics[range_goal_id].columns)
        self.assertIn("perc_above_target", metrics[range_goal_id].columns)
        self.assertIn("sum_below_target", metrics[range_goal_id].columns)
        self.assertIn("sum_above_target", metrics[range_goal_id].columns)

        self.assertEqual(
            metrics[integral_goal_id].loc["final_results", "mean_absolute_difference"], 0
        )
        self.assertEqual(metrics[integral_goal_id].loc["final_results", "max_difference"], 0)

    def test_get_performance_metrics_with_plot_writes_html(self):
        test_data = get_test_data("basic", optimization=True)
        problem = CustomGoalOptimizationProblem(
            model_folder=test_data["model_folder"],
            model_name=test_data["model_name"],
            input_folder=test_data["model_input_folder"],
            output_folder=test_data["output_folder"],
        )

        problem.optimize()
        metrics = problem.get_performance_metrics_with_plot()

        expected_html = (
            Path(test_data["output_folder"])
            / "performance_metrics"
            / "performance_metrics_dashboard.html"
        )

        self.assertTrue(expected_html.exists())
        self.assertEqual(metrics.keys(), problem.get_performance_metrics().keys())

        html = expected_html.read_text(encoding="utf-8")
        self.assertIn("Bar Charts", html)
        self.assertIn("Tables", html)
        self.assertIn("Performance Metrics Bar Chart", html)
        self.assertIn("Performance Metrics Tables", html)
        self.assertIn("Select all goals", html)
        self.assertIn("Unselect all goals", html)
        self.assertIn("All goals", html)
        self.assertNotIn("Heatmap view", html)
        self.assertNotIn("Metric-focused view", html)

