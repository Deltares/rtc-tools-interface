"""Tests for performance metrics in GoalGeneratorMixin."""

import unittest
from pathlib import Path
from unittest.mock import patch

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


class DiscreteFlagOptimizationProblem(CustomGoalOptimizationProblem):
    """Problem exposing one discrete variable for mixed-integer detection tests."""

    def variable_is_discrete(self, variable):
        return variable == "u" or super().variable_is_discrete(variable)


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
        active_constraint_metrics = problem.get_active_constraint_metrics()
        shadow_price_metrics = problem.get_shadow_price_metrics()

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
        self.assertNotIn("active_hard_constraints", metrics[range_goal_id].columns)
        self.assertNotIn("active_hard_constraints_fraction", metrics[range_goal_id].columns)
        self.assertNotIn("active_previous_priority_constraints", metrics[range_goal_id].columns)

        self.assertFalse(active_constraint_metrics.empty)
        self.assertIn(10, active_constraint_metrics.index)
        self.assertIn("final_results", active_constraint_metrics.index)
        self.assertEqual(
            set(active_constraint_metrics.columns),
            {
                "active_hard_constraints",
                "active_hard_constraints_fraction",
                "active_previous_priority_constraints",
            },
        )
        self.assertGreater(
            active_constraint_metrics.loc["final_results", "active_hard_constraints"], 0
        )
        self.assertGreater(
            active_constraint_metrics.loc["final_results", "active_hard_constraints_fraction"], 0
        )
        self.assertLessEqual(
            active_constraint_metrics.loc["final_results", "active_hard_constraints_fraction"], 1
        )
        self.assertGreater(
            active_constraint_metrics.loc["final_results", "active_previous_priority_constraints"],
            0,
        )

        self.assertFalse(shadow_price_metrics.empty)
        self.assertEqual(list(shadow_price_metrics.columns), [10, 15])
        self.assertIn(10, shadow_price_metrics.index)
        self.assertIn(15, shadow_price_metrics.index)
        self.assertIn(20, shadow_price_metrics.index)
        self.assertIn("final_results", shadow_price_metrics.index)
        self.assertTrue(shadow_price_metrics.loc[10].isna().all())
        self.assertEqual(int(shadow_price_metrics.loc[15].count()), 1)
        self.assertGreater(shadow_price_metrics.loc[15, 10], 0)
        self.assertGreater(shadow_price_metrics.loc[20, 10], 0)
        self.assertGreater(shadow_price_metrics.loc[20, 15], 0)
        self.assertEqual(
            shadow_price_metrics.loc["final_results", 10], shadow_price_metrics.loc[20, 10]
        )
        self.assertEqual(
            shadow_price_metrics.loc["final_results", 15], shadow_price_metrics.loc[20, 15]
        )

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
        expected_shadow_csv = (
            Path(test_data["output_folder"]) / "performance_metrics" / "shadow_price_metrics.csv"
        )

        self.assertTrue(expected_html.exists())
        self.assertTrue(expected_shadow_csv.exists())
        self.assertEqual(metrics.keys(), problem.get_performance_metrics().keys())

        html = expected_html.read_text(encoding="utf-8")
        self.assertIn("Bar Charts", html)
        self.assertIn("Tables", html)
        self.assertIn("Constraint Activity", html)
        self.assertIn("Shadow Prices", html)
        self.assertIn("Performance Metrics Bar Chart", html)
        self.assertIn("Performance Metrics Tables", html)
        self.assertIn("Select all goals", html)
        self.assertIn("Unselect all goals", html)
        self.assertIn("All goals", html)
        self.assertIn("Timeseries Sum", html)
        self.assertIn("Timeseries Average", html)
        self.assertIn("Percentage Below Target", html)
        self.assertIn("Percentage Above Target", html)
        self.assertIn("Active hard constraints", html)
        self.assertIn("Fraction of active hard constraints", html)
        self.assertIn("Active constraints from earlier priorities", html)
        self.assertIn("Constraint activity by priority", html)
        self.assertIn("Shadow prices from earlier priorities", html)
        self.assertIn("From priority 10", html)
        self.assertIn("From priority 15", html)
        self.assertNotIn("Heatmap view", html)
        self.assertNotIn("Metric-focused view", html)

    def test_is_mixed_integer_problem_detects_discrete_variables(self):
        test_data = get_test_data("basic", optimization=True)
        problem = DiscreteFlagOptimizationProblem(
            model_folder=test_data["model_folder"],
            model_name=test_data["model_name"],
            input_folder=test_data["model_input_folder"],
            output_folder=test_data["output_folder"],
        )

        self.assertTrue(problem.is_mixed_integer_problem())

    def test_finite_difference_rhs_sensitivity_returns_empty_for_continuous_problem(self):
        test_data = get_test_data("basic", optimization=True)
        problem = CustomGoalOptimizationProblem(
            model_folder=test_data["model_folder"],
            model_name=test_data["model_name"],
            input_folder=test_data["model_input_folder"],
            output_folder=test_data["output_folder"],
        )

        problem.optimize()

        self.assertEqual(problem.get_finite_difference_rhs_sensitivity_analysis(), {})

    def test_finite_difference_rhs_sensitivity_writes_lower_triangular_csvs(self):
        test_data = get_test_data("basic", optimization=True)
        problem = CustomGoalOptimizationProblem(
            model_folder=test_data["model_folder"],
            model_name=test_data["model_name"],
            input_folder=test_data["model_input_folder"],
            output_folder=test_data["output_folder"],
        )

        problem.optimize()
        baseline_objectives = {
            priority: value
            for priority, value in problem._priority_objective_values.items()
            if priority != "final_results"
        }

        def fake_relaxed_run(*, source_priority, relaxation_percentage):
            objective_offsets = {
                10: {
                    1.0: {15: 1.5, 20: 2.5},
                    2.0: {15: 2.0, 20: 3.0},
                },
                15: {
                    1.0: {20: 4.0},
                    2.0: {20: 5.0},
                },
            }
            offsets = objective_offsets[source_priority][relaxation_percentage]
            return {
                current_priority: baseline_objectives[current_priority] - improvement
                for current_priority, improvement in offsets.items()
            }

        with (
            patch.object(problem, "is_mixed_integer_problem", return_value=True),
            patch.object(
                problem,
                "_run_relaxed_rhs_sensitivity_analysis",
                side_effect=fake_relaxed_run,
            ),
        ):
            matrices = problem.get_finite_difference_rhs_sensitivity_analysis(
                relaxation_percentages=(1.0, 2.0)
            )

        self.assertEqual(set(matrices), {1.0, 2.0})
        self.assertEqual(list(matrices[1.0].columns), [10, 15])
        self.assertIn(10, matrices[1.0].index)
        self.assertIn(20, matrices[1.0].index)
        self.assertIn("final_results", matrices[1.0].index)
        self.assertTrue(matrices[1.0].loc[10].isna().all())
        self.assertEqual(matrices[1.0].loc[15, 10], 1.5)
        self.assertEqual(matrices[1.0].loc[20, 10], 2.5)
        self.assertEqual(matrices[1.0].loc[20, 15], 4.0)
        self.assertEqual(matrices[1.0].loc["final_results", 10], 2.5)
        self.assertEqual(matrices[2.0].loc[20, 15], 5.0)

        sensitivity_output = Path(test_data["output_folder"]) / "sensitivity_analysis"
        self.assertTrue(
            (sensitivity_output / "finite_difference_rhs_sensitivity_1pct.csv").exists()
        )
        self.assertTrue(
            (sensitivity_output / "finite_difference_rhs_sensitivity_2pct.csv").exists()
        )

    def test_finite_difference_rhs_sensitivity_accepts_custom_iterables(self):
        test_data = get_test_data("basic", optimization=True)
        problem = CustomGoalOptimizationProblem(
            model_folder=test_data["model_folder"],
            model_name=test_data["model_name"],
            input_folder=test_data["model_input_folder"],
            output_folder=test_data["output_folder"],
        )

        problem.optimize()
        baseline_objectives = {
            priority: value
            for priority, value in problem._priority_objective_values.items()
            if priority != "final_results"
        }

        def fake_relaxed_run(*, source_priority, relaxation_percentage):
            return {
                current_priority: baseline_objectives[current_priority] - relaxation_percentage
                for current_priority in baseline_objectives
                if current_priority > source_priority
            }

        with (
            patch.object(problem, "is_mixed_integer_problem", return_value=True),
            patch.object(
                problem,
                "_run_relaxed_rhs_sensitivity_analysis",
                side_effect=fake_relaxed_run,
            ),
        ):
            matrices = problem.get_finite_difference_rhs_sensitivity_analysis(
                relaxation_percentages=(percentage for percentage in [0.5, 2, 2, 7.5]),
            )

        self.assertEqual(list(matrices), [0.5, 2.0, 7.5])
        sensitivity_output = Path(test_data["output_folder"]) / "sensitivity_analysis"
        self.assertTrue(
            (sensitivity_output / "finite_difference_rhs_sensitivity_0p5pct.csv").exists()
        )
        self.assertTrue(
            (sensitivity_output / "finite_difference_rhs_sensitivity_7p5pct.csv").exists()
        )

    def test_finite_difference_rhs_sensitivity_rejects_invalid_percentages(self):
        test_data = get_test_data("basic", optimization=True)
        problem = CustomGoalOptimizationProblem(
            model_folder=test_data["model_folder"],
            model_name=test_data["model_name"],
            input_folder=test_data["model_input_folder"],
            output_folder=test_data["output_folder"],
        )

        problem.optimize()

        with self.assertRaisesRegex(ValueError, "at least one"):
            problem.get_finite_difference_rhs_sensitivity_analysis([])

        with self.assertRaisesRegex(ValueError, "negative"):
            problem.get_finite_difference_rhs_sensitivity_analysis([1.0, -2.0])

        with self.assertRaisesRegex(TypeError, "real number"):
            problem.get_finite_difference_rhs_sensitivity_analysis(["1", 2.0])
