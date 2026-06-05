"""Mixin for reporting active constraints after goal-programming priorities."""

import csv
import logging
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np
from rtctools.optimization.timeseries import Timeseries

logger = logging.getLogger("rtctools")


class ActiveConstraintMixin:
    """Write active-constraint diagnostics for RTC-Tools optimizations.

    Add this mixin before ``GoalProgrammingMixin``/``BaseOptimizationProblem`` in
    the inheritance list. After optimization, two CSV files are written to
    ``<output_folder>/active_constraints``:

    * ``active_constraints_by_priority.csv`` with counts for all transcribed
      constraints after every priority solve.
    * ``previous_goal_constraints.csv`` with details for constraints that RTC-Tools
      created from goals optimized in previous priorities.
    """

    active_constraint_output_folder = "active_constraints"
    active_constraint_tolerance = 1e-7

    _ACTIVE_CONSTRAINT_SUMMARY_FIELDS = [
        "priority",
        "total_constraints",
        "active_constraints",
        "active_lower_bounds",
        "active_upper_bounds",
        "active_equalities",
    ]

    _PREVIOUS_GOAL_CONSTRAINT_FIELDS = [
        "priority",
        "total_previous_goal_constraints",
        "active_previous_goal_constraints",
        "ensemble_member",
        "constraint_source",
        "function_key",
        "goal_id",
        "goal_priority",
        "goal_class",
        "component_index",
        "time",
        "value",
        "lower_bound",
        "upper_bound",
        "is_active",
        "active_bound",
        "active_bound_value",
    ]

    def __init__(self, **kwargs):
        self._active_constraint_summary_rows = []
        self._previous_goal_constraint_rows = []
        super().__init__(**kwargs)

    def priority_completed(self, priority: int) -> None:
        """Collect active-constraint diagnostics after a priority is solved."""
        self._collect_active_constraint_summary(priority)
        self._collect_previous_goal_constraint_details(priority)
        super().priority_completed(priority)

    def post(self) -> None:
        """Write active-constraint diagnostics after optimization."""
        super().post()
        self._write_active_constraint_csv_files()

    @staticmethod
    def _as_flat_float_array(value: Any) -> np.ndarray:
        """Convert CasADi/numeric values to a one-dimensional float array."""
        if isinstance(value, Timeseries):
            value = value.values
        if isinstance(value, (list, tuple)):
            value = ca.veccat(*value) if value else ca.DM.zeros(0)
        array = np.array(value, dtype=float)
        return array.reshape(-1)

    @staticmethod
    def _bound_to_array(bound: Any, size: int) -> np.ndarray:
        """Return a flat bound array matching an evaluated constraint size."""
        if isinstance(bound, Timeseries):
            bound = bound.values
        array = np.array(bound, dtype=float)
        if array.size == 1 and size != 1:
            return np.full(size, float(array.reshape(-1)[0]))
        return array.reshape(-1)

    def _active_bound_masks(
        self, values: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return masks for lower-bound and upper-bound active components."""
        tolerance = self.active_constraint_tolerance
        lower_active = np.isfinite(lower_bounds) & np.isclose(
            values, lower_bounds, rtol=tolerance, atol=tolerance
        )
        upper_active = np.isfinite(upper_bounds) & np.isclose(
            values, upper_bounds, rtol=tolerance, atol=tolerance
        )
        return lower_active, upper_active

    def _collect_active_constraint_summary(self, priority: int) -> None:
        """Collect active-constraint counts for the full transcribed NLP."""
        transcribed_problem = self.transcribed_problem
        constraint_expression = transcribed_problem["nlp"]["g"]
        constraint_function = ca.Function(
            f"active_constraints_g_priority_{priority}",
            [transcribed_problem["nlp"]["x"]],
            [constraint_expression],
        )

        values = self._as_flat_float_array(constraint_function(self.solver_output))
        lower_bounds = self._as_flat_float_array(transcribed_problem["lbg"])
        upper_bounds = self._as_flat_float_array(transcribed_problem["ubg"])

        lower_active, upper_active = self._active_bound_masks(values, lower_bounds, upper_bounds)
        active = lower_active | upper_active
        equalities = (
            np.isfinite(lower_bounds)
            & np.isfinite(upper_bounds)
            & np.isclose(
                lower_bounds,
                upper_bounds,
                rtol=self.active_constraint_tolerance,
                atol=self.active_constraint_tolerance,
            )
        )

        self._active_constraint_summary_rows.append(
            {
                "priority": priority,
                "total_constraints": int(values.size),
                "active_constraints": int(np.count_nonzero(active)),
                "active_lower_bounds": int(np.count_nonzero(lower_active)),
                "active_upper_bounds": int(np.count_nonzero(upper_active)),
                "active_equalities": int(np.count_nonzero(active & equalities)),
            }
        )

    def _collect_previous_goal_constraint_details(self, priority: int) -> None:
        """Collect rows for constraints created from goals of previous priorities."""
        rows = []
        rows.extend(self._goal_constraint_rows(priority, "goal", is_path_goal=False))
        rows.extend(self._goal_constraint_rows(priority, "path_goal", is_path_goal=True))
        rows.extend(
            self._goal_constraint_rows(
                priority, "kept_soft_goal", is_path_goal=False, include_problem_constraints=True
            )
        )
        rows.extend(
            self._goal_constraint_rows(
                priority,
                "kept_soft_path_goal",
                is_path_goal=True,
                include_problem_constraints=True,
            )
        )

        total_constraints = len(rows)
        active_constraints = sum(row["is_active"] for row in rows)
        if rows:
            for row in rows:
                row["total_previous_goal_constraints"] = total_constraints
                row["active_previous_goal_constraints"] = active_constraints
            self._previous_goal_constraint_rows.extend(rows)
        else:
            self._previous_goal_constraint_rows.append(
                {
                    "priority": priority,
                    "total_previous_goal_constraints": 0,
                    "active_previous_goal_constraints": 0,
                    "ensemble_member": "",
                    "constraint_source": "",
                    "function_key": "",
                    "goal_id": "",
                    "goal_priority": "",
                    "goal_class": "",
                    "component_index": "",
                    "time": "",
                    "value": "",
                    "lower_bound": "",
                    "upper_bound": "",
                    "is_active": False,
                    "active_bound": "",
                    "active_bound_value": "",
                }
            )

    def _goal_constraint_rows(
        self,
        priority: int,
        constraint_source: str,
        *,
        is_path_goal: bool,
        include_problem_constraints: bool = False,
    ) -> list[dict[str, Any]]:
        """Evaluate private RTC-Tools goal constraints and build detail rows."""
        rows = []
        constraint_collections = self._get_goal_constraint_collections(
            is_path_goal=is_path_goal, include_problem_constraints=include_problem_constraints
        )
        times = self.times() if is_path_goal else None

        for ensemble_member, constraints in enumerate(constraint_collections):
            for function_key, constraint in constraints:
                goal = getattr(constraint, "goal", None)
                if goal is not None and int(goal.priority) >= int(priority):
                    continue

                values = self._evaluate_goal_constraint(
                    constraint, ensemble_member, is_path_goal=is_path_goal
                )
                lower_bounds = self._bound_to_array(constraint.min, values.size)
                upper_bounds = self._bound_to_array(constraint.max, values.size)
                lower_active, upper_active = self._active_bound_masks(
                    values, lower_bounds, upper_bounds
                )

                for component_index, value in enumerate(values):
                    active_bound, active_bound_value = self._active_bound_description(
                        lower_active[component_index],
                        upper_active[component_index],
                        lower_bounds[component_index],
                        upper_bounds[component_index],
                    )
                    rows.append(
                        {
                            "priority": priority,
                            "total_previous_goal_constraints": "",
                            "active_previous_goal_constraints": "",
                            "ensemble_member": ensemble_member,
                            "constraint_source": constraint_source,
                            "function_key": function_key,
                            "goal_id": getattr(goal, "goal_id", "") if goal is not None else "",
                            "goal_priority": (
                                getattr(goal, "priority", "") if goal is not None else ""
                            ),
                            "goal_class": goal.__class__.__name__ if goal is not None else "",
                            "component_index": component_index,
                            "time": self._component_time(times, component_index),
                            "value": value,
                            "lower_bound": lower_bounds[component_index],
                            "upper_bound": upper_bounds[component_index],
                            "is_active": bool(
                                lower_active[component_index] or upper_active[component_index]
                            ),
                            "active_bound": active_bound,
                            "active_bound_value": active_bound_value,
                        }
                    )
        return rows

    def _get_goal_constraint_collections(
        self, *, is_path_goal: bool, include_problem_constraints: bool
    ) -> list[list[tuple[str, Any]]]:
        """Read goal-constraint stores from RTC-Tools goal programming internals."""
        if include_problem_constraints:
            attribute = (
                "_GoalProgrammingMixin__problem_path_constraints"
                if is_path_goal
                else "_GoalProgrammingMixin__problem_constraints"
            )
        else:
            attribute = (
                "_GoalProgrammingMixin__path_constraint_store"
                if is_path_goal
                else "_GoalProgrammingMixin__constraint_store"
            )

        constraint_store = getattr(self, attribute, [])
        collections = []
        for ensemble_member in range(self.ensemble_size):
            try:
                constraints = constraint_store[ensemble_member]
            except (IndexError, KeyError, TypeError):
                constraints = []
            if hasattr(constraints, "items"):
                collections.append(list(constraints.items()))
            else:
                collections.append(
                    [
                        (f"{attribute}_{ensemble_member}_{i}", constraint)
                        for i, constraint in enumerate(constraints)
                    ]
                )
        return collections

    def _evaluate_goal_constraint(
        self, constraint: Any, ensemble_member: int, *, is_path_goal: bool
    ) -> np.ndarray:
        """Evaluate a goal constraint at the current solver solution."""
        expression = constraint.function(self)
        if is_path_goal:
            expression = self.map_path_expression(expression, ensemble_member)
        function = ca.Function(
            f"active_goal_constraint_{ensemble_member}_{id(constraint)}",
            [self.solver_input],
            [expression],
        )
        return self._as_flat_float_array(function(self.solver_output))

    @staticmethod
    def _component_time(times: np.ndarray | None, component_index: int) -> float | str:
        """Return the time associated with a flattened path-constraint component."""
        if times is None or len(times) == 0:
            return ""
        return times[component_index % len(times)]

    @staticmethod
    def _active_bound_description(
        lower_active: bool, upper_active: bool, lower_bound: float, upper_bound: float
    ) -> tuple[str, str | float]:
        """Return a text label and value for the bound hit by a constraint."""
        if lower_active and upper_active:
            if np.isclose(lower_bound, upper_bound, rtol=0.0, atol=0.0):
                return "both", lower_bound
            return "both", f"{lower_bound};{upper_bound}"
        if lower_active:
            return "lower", lower_bound
        if upper_active:
            return "upper", upper_bound
        return "", ""

    def _write_active_constraint_csv_files(self) -> None:
        """Write collected active-constraint diagnostics to CSV files."""
        output_folder = Path(self._output_folder) / self.active_constraint_output_folder
        output_folder.mkdir(parents=True, exist_ok=True)

        self._write_csv(
            output_folder / "active_constraints_by_priority.csv",
            self._ACTIVE_CONSTRAINT_SUMMARY_FIELDS,
            self._active_constraint_summary_rows,
        )
        self._write_csv(
            output_folder / "previous_goal_constraints.csv",
            self._PREVIOUS_GOAL_CONSTRAINT_FIELDS,
            self._previous_goal_constraint_rows,
        )
        logger.info("Active constraint diagnostics written to %s", output_folder)

    @staticmethod
    def _write_csv(file_path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
        """Write rows to a CSV file with a stable header."""
        with file_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
