"""Tests for the active constraint mixin."""

import shutil
import unittest
from pathlib import Path

import pandas as pd

from rtctools_interface.optimization.active_constraint_mixin import ActiveConstraintMixin
from rtctools_interface.optimization.base_optimization_problem import BaseOptimizationProblem
from tests.utils.get_test import get_test_data


class ActiveConstraintProblem(ActiveConstraintMixin, BaseOptimizationProblem):
    """Optimization problem with active constraint diagnostics enabled."""


class TestActiveConstraintMixin(unittest.TestCase):
    """Test active constraint CSV generation."""

    def test_active_constraint_csv_files_are_written(self):
        """Solve a goal-programming problem and inspect generated CSV files."""
        test_data = get_test_data("basic", optimization=True)
        output_folder = Path(test_data["output_folder"])
        active_constraint_folder = output_folder / "active_constraints"
        shutil.rmtree(active_constraint_folder, ignore_errors=True)

        problem = ActiveConstraintProblem(
            goal_table_file=test_data["goals_file"],
            model_folder=test_data["model_folder"],
            model_name=test_data["model_name"],
            input_folder=test_data["model_input_folder"],
            output_folder=output_folder,
        )

        self.assertTrue(problem.optimize())

        summary_file = active_constraint_folder / "active_constraints_by_priority.csv"
        previous_goal_file = active_constraint_folder / "previous_goal_constraints.csv"
        self.assertTrue(summary_file.exists())
        self.assertTrue(previous_goal_file.exists())

        summary = pd.read_csv(summary_file)
        self.assertEqual(list(summary["priority"]), [10, 15, 20])
        self.assertTrue((summary["total_constraints"] > 0).all())
        self.assertTrue((summary["active_constraints"] >= 0).all())
        self.assertTrue((summary["active_constraints"] <= summary["total_constraints"]).all())

        previous_goal_constraints = pd.read_csv(previous_goal_file)
        self.assertIn("active_bound_value", previous_goal_constraints.columns)
        self.assertEqual(
            int(previous_goal_constraints.loc[0, "total_previous_goal_constraints"]), 0
        )
        self.assertTrue(
            (previous_goal_constraints["total_previous_goal_constraints"].fillna(0) > 0).any()
        )
        self.assertTrue(
            previous_goal_constraints["active_previous_goal_constraints"].fillna(0).max() > 0
        )
