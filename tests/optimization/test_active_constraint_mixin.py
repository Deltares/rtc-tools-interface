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

        previous_goal_file = active_constraint_folder / "active_constraints_of_previous_goals.csv"
        self.assertFalse((active_constraint_folder / "active_constraints_by_priority.csv").exists())
        self.assertFalse((active_constraint_folder / "previous_goal_constraints.csv").exists())
        self.assertTrue(previous_goal_file.exists())

        previous_goal_constraints = pd.read_csv(previous_goal_file)
        self.assertNotIn("goal_id", previous_goal_constraints.columns)
        self.assertNotIn("component_index", previous_goal_constraints.columns)
        self.assertNotIn("time", previous_goal_constraints.columns)
        self.assertNotIn("is_active", previous_goal_constraints.columns)
        self.assertIn("active_times", previous_goal_constraints.columns)
        self.assertIn("active_bound_value", previous_goal_constraints.columns)
        self.assertNotIn("active_previous_goal_constraints", previous_goal_constraints.columns)
        self.assertIn("active_previous_goals_constraints", previous_goal_constraints.columns)
        self.assertEqual(set(previous_goal_constraints["priority"].to_list()), {10, 15, 20})
        self.assertTrue((previous_goal_constraints["total_previous_goal_constraints"] >= 0).all())
        self.assertTrue(
            (
                previous_goal_constraints["total_previous_goal_constraints"]
                >= previous_goal_constraints["active_previous_goals_constraints"]
            ).all()
        )
        self.assertEqual(
            previous_goal_constraints[previous_goal_constraints["priority"] == 10][
                "active_previous_goals_constraints"
            ].iloc[0],
            0,
        )
        self.assertEqual(
            previous_goal_constraints[previous_goal_constraints["priority"] == 15][
                "active_previous_goals_constraints"
            ].iloc[0],
            0,
        )
        self.assertTrue(
            (
                previous_goal_constraints[previous_goal_constraints["priority"] == 20][
                    "active_previous_goals_constraints"
                ]
                > 0
            ).all()
        )
        path_goal_constraints = previous_goal_constraints[
            previous_goal_constraints["constraint_source"] == "path_goal"
        ]
        self.assertEqual(len(path_goal_constraints[path_goal_constraints["priority"] == 15]), 0)
        self.assertEqual(len(path_goal_constraints[path_goal_constraints["priority"] == 20]), 2)
        self.assertTrue(
            path_goal_constraints[path_goal_constraints["priority"] == 20]["active_times"]
            .notna()
            .all()
        )
