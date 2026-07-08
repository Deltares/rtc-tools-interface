"""Tests for performance metrics in GoalGeneratorMixin."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.goal_programming_mixin import Goal
from rtctools.optimization.linearized_order_goal_programming_mixin import (
    LinearizedOrderGoalProgrammingMixin,
)
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.single_pass_goal_programming_mixin import (
    SinglePassGoalProgrammingMixin,
)

from rtctools_interface.optimization.base_optimization_problem import BaseOptimizationProblem
from rtctools_interface.optimization.goal_generator_mixin import (
    GoalGeneratorMixin,
    write_shadow_price_metrics,
)
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


class SinglePassCustomGoalOptimizationProblem(
    GoalGeneratorMixin,
    LinearizedOrderGoalProgrammingMixin,
    SinglePassGoalProgrammingMixin,
    CSVMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    """Single-pass goal-programming variant used for regression testing."""

    def goals(self):
        return super().goals() + [MinimizeIntegralUGoal()]

    def path_goals(self):
        return super().path_goals() + [WaterLevelRangeGoal(), MinimizeUGoal()]


class TestGoalGeneratorMixin(unittest.TestCase):
    """Validate metric collection for goals not defined via a goal table."""

    def test_write_shadow_price_metrics_writes_empty_csv(self):
        with TemporaryDirectory() as output_folder:
            write_shadow_price_metrics(pd.DataFrame(), output_folder)

            shadow_price_metrics_file = (
                Path(output_folder) / "performance_metrics" / "shadow_price_metrics.csv"
            )

            self.assertTrue(shadow_price_metrics_file.exists())

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

    def test_single_pass_goal_programming_supports_performance_metrics(self):
        test_data = get_test_data("basic", optimization=True)
        problem = SinglePassCustomGoalOptimizationProblem(
            model_folder=test_data["model_folder"],
            model_name=test_data["model_name"],
            input_folder=test_data["model_input_folder"],
            output_folder=test_data["output_folder"],
        )

        problem.optimize()
        metrics = problem.get_performance_metrics_with_plot()

        range_goal_id = "WaterLevelRangeGoal__x__path__priority_10__idx_0"
        smooth_goal_id = "MinimizeUGoal__path__priority_15__idx_1"
        integral_goal_id = "MinimizeIntegralUGoal__goal__priority_20__idx_0"

        self.assertEqual(set(metrics), {range_goal_id, smooth_goal_id, integral_goal_id})
        self.assertTrue(Path(problem.performance_metrics_plot_file).exists())
        self.assertFalse(problem.get_active_constraint_metrics().empty)
        self.assertIn("final_results", problem.get_active_constraint_metrics().index)

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

        def fake_relaxed_run(*, source_priority, relaxation_value, mode, relative_floor):
            self.assertEqual(mode, "relative")
            self.assertEqual(relative_floor, 1.0)
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
            offsets = objective_offsets[source_priority][relaxation_value]
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
                relaxation_values=(1.0, 2.0)
            )

        self.assertEqual(set(matrices), {1.0, 2.0})
        self.assertEqual(list(matrices[1.0].columns), [10, 15])
        self.assertIn(10, matrices[1.0].index)
        self.assertIn(20, matrices[1.0].index)
        self.assertIn("final_results", matrices[1.0].index)
        self.assertTrue(matrices[1.0].loc[10].isna().all())
        self.assertGreater(matrices[1.0].loc[15, 10], 0)
        self.assertGreaterEqual(matrices[1.0].loc[20, 10], matrices[1.0].loc[15, 10])
        self.assertGreater(matrices[1.0].loc[20, 15], 0)
        self.assertEqual(matrices[1.0].loc["final_results", 10], matrices[1.0].loc[20, 10])
        self.assertGreater(matrices[2.0].loc[20, 15], matrices[1.0].loc[20, 15])

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

        def fake_relaxed_run(*, source_priority, relaxation_value, mode, relative_floor):
            self.assertEqual(mode, "relative")
            self.assertEqual(relative_floor, 1.0)
            return {
                current_priority: baseline_objectives[current_priority] - relaxation_value
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
                relaxation_values=(percentage for percentage in [0.5, 2, 2, 7.5]),
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

        with self.assertRaisesRegex(ValueError, "mode"):
            problem.get_finite_difference_rhs_sensitivity_analysis(mode="unsupported")

        with self.assertRaisesRegex(ValueError, "negative"):
            problem.get_finite_difference_rhs_sensitivity_analysis(relative_floor=-1.0)

        with self.assertRaisesRegex(
            ValueError, "either relaxation_values or relaxation_percentages"
        ):
            problem.get_finite_difference_rhs_sensitivity_analysis(
                relaxation_values=[1.0],
                relaxation_percentages=[2.0],
            )

    def test_finite_difference_rhs_sensitivity_supports_absolute_mode(self):
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

        def fake_relaxed_run(*, source_priority, relaxation_value, mode, relative_floor):
            self.assertEqual(mode, "absolute")
            self.assertEqual(relative_floor, 0.25)
            return {
                current_priority: baseline_objectives[current_priority] - relaxation_value
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
                relaxation_values=[0.25, 1.5],
                mode="absolute",
                relative_floor=0.25,
            )

        self.assertEqual(list(matrices), [0.25, 1.5])
        sensitivity_output = Path(test_data["output_folder"]) / "sensitivity_analysis"
        self.assertTrue(
            (sensitivity_output / "finite_difference_rhs_sensitivity_absolute_0p25.csv").exists()
        )
        self.assertTrue(
            (sensitivity_output / "finite_difference_rhs_sensitivity_absolute_1p5.csv").exists()
        )

    def test_relax_bound_supports_all_modes(self):
        self.assertEqual(
            CustomGoalOptimizationProblem._relax_bound(
                0.0,
                relaxation_value=5.0,
                mode="relative",
                relative_floor=2.0,
                is_lower_bound=False,
            ),
            0.1,
        )
        self.assertEqual(
            CustomGoalOptimizationProblem._relax_bound(
                0.0,
                relaxation_value=0.5,
                mode="absolute",
                relative_floor=0.0,
                is_lower_bound=True,
            ),
            -0.5,
        )

    def test_finite_difference_rhs_sensitivity_uses_percentage_and_columnwise_maximum(self):
        test_data = get_test_data("basic", optimization=True)
        problem = CustomGoalOptimizationProblem(
            model_folder=test_data["model_folder"],
            model_name=test_data["model_name"],
            input_folder=test_data["model_input_folder"],
            output_folder=test_data["output_folder"],
        )

        problem.optimize()
        problem._priority_objective_values = {10: 4.0, 15: 10.0, 20: 0.0, "final_results": 0.0}

        def fake_relaxed_run(*, source_priority, relaxation_value, mode, relative_floor):
            del relaxation_value, mode, relative_floor
            if source_priority == 10:
                return {15: 9.0, 20: -1.0}
            return {20: -0.2}

        with (
            patch.object(problem, "is_mixed_integer_problem", return_value=True),
            patch.object(
                problem,
                "_run_relaxed_rhs_sensitivity_analysis",
                side_effect=fake_relaxed_run,
            ),
        ):
            matrices = problem.get_finite_difference_rhs_sensitivity_analysis(
                relaxation_values=[1.0]
            )

        matrix = matrices[1.0]
        self.assertEqual(matrix.loc[15, 10], 10.0)
        self.assertEqual(matrix.loc[20, 10], 10.0)
        self.assertEqual(matrix.loc[20, 15], 0.0)
        self.assertEqual(matrix.loc["final_results", 10], 10.0)
        self.assertEqual(matrix.loc["final_results", 15], 0.0)

    def test_store_performance_metrics_handles_missing_current_priority_for_shadow_prices(self):
        test_data = get_test_data("basic", optimization=True)
        problem = CustomGoalOptimizationProblem(
            model_folder=test_data["model_folder"],
            model_name=test_data["model_name"],
            input_folder=test_data["model_input_folder"],
            output_folder=test_data["output_folder"],
        )

        problem.optimize()
        problem._shadow_price_metrics = problem._shadow_price_metrics.iloc[0:0]
        problem.store_performance_metrics("final_results", current_priority=None)

        self.assertIn("final_results", problem.get_active_constraint_metrics().index)
        self.assertIn("final_results", problem.get_shadow_price_metrics().index)
        self.assertTrue(problem.get_shadow_price_metrics().loc["final_results"].isna().all())
