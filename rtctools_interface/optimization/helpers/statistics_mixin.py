"""Base mixin for retrieving particular stats for goals and plotting."""

import logging
import re

import casadi as ca
import numpy as np
from rtctools.optimization.goal_programming_mixin import GoalProgrammingMixin
from rtctools.optimization.timeseries import Timeseries

from rtctools_interface.optimization.base_goal import BaseGoal
from rtctools_interface.optimization.goal_performance_metrics import ABS_TOL
from rtctools_interface.utils.type_definitions import TargetDict

logger = logging.getLogger("rtctools")


class StatisticsMixin:
    # TODO: remove pylint disable below once we have more public functions.
    # pylint: disable=too-few-public-methods
    """A mixin class providing methods for collecting data and statistics from optimization results,
    useful for solution performance analysis."""

    def collect_range_target_values(
        self,
        base_goals: list[BaseGoal],
    ) -> dict[str, TargetDict]:
        """For the goals with targets, collect the actual timeseries with these targets."""
        target_series: dict[str, TargetDict] = {}
        for goal in base_goals:
            if goal.goal_type in ["range", "range_rate_of_change"]:
                target_dict = self.collect_range_target_values_from_basegoal(goal)
                target_series[str(goal.goal_id)] = target_dict
        return target_series

    def collect_range_target_values_from_basegoal(self, goal: BaseGoal) -> TargetDict:
        """Collect the target timeseries for a single basegoal."""
        t = self.times()

        def get_parameter_ranges(goal) -> tuple[np.ndarray, np.ndarray]:
            target_min = np.full_like(t, 1) * float(goal.target_min)
            target_max = np.full_like(t, 1) * float(goal.target_max)
            return target_min, target_max

        def get_value_ranges(goal) -> tuple[np.ndarray, np.ndarray]:
            target_min = np.full_like(t, 1) * float(goal.target_min)
            target_max = np.full_like(t, 1) * float(goal.target_max)
            return target_min, target_max

        def get_timeseries_ranges(goal) -> tuple[np.ndarray, np.ndarray]:
            try:
                target_min = goal.target_min.values
            except AttributeError:
                target_min = goal.target_min
            try:
                target_max = goal.target_max.values
            except AttributeError:
                target_max = goal.target_max
            return target_min, target_max

        supported_goal_types = ["range", "range_rate_of_change"]
        if goal.goal_type in supported_goal_types:
            if goal.target_data_type == "parameter":
                target_min, target_max = get_parameter_ranges(goal)
            elif goal.target_data_type == "value":
                target_min, target_max = get_value_ranges(goal)
            elif goal.target_data_type == "timeseries":
                target_min, target_max = get_timeseries_ranges(goal)
            else:
                message = f"Target type {goal.target_data_type} not known for goal {goal.goal_id}."
                logger.error(message)
                raise ValueError(message)
        else:
            message = f"Goal type {goal.goal_type} not supported for target collection."
            logger.error(message)
            raise ValueError(message)
        target_dict: TargetDict = {"target_min": target_min, "target_max": target_max}
        return target_dict

    @staticmethod
    def get_performance_metric_id(goal, *, is_path_goal: bool, goal_index: int) -> str:
        """Build a deterministic identifier for performance metrics of a goal."""
        for attribute in ("goal_id", "function_value_timeseries_id", "violation_timeseries_id"):
            value = getattr(goal, attribute, None)
            if value not in [None, ""]:
                base_id = str(value)
                break
        else:
            goal_parts = [goal.__class__.__name__]
            if getattr(goal, "state", None):
                goal_parts.append(str(goal.state))
            goal_parts.append("path" if is_path_goal else "goal")
            goal_parts.append(f"priority_{getattr(goal, 'priority', 'unknown')}")
            goal_parts.append(f"idx_{goal_index}")
            base_id = "__".join(goal_parts)

        return re.sub(r"[^0-9A-Za-z._-]+", "_", base_id).strip("_")

    def collect_target_values_for_goal(self, goal, *, is_path_goal: bool) -> TargetDict | None:
        """Collect target values for any RTC-Tools goal instance."""
        if not getattr(goal, "has_target_bounds", False):
            return None

        if isinstance(goal, BaseGoal):
            return self.collect_range_target_values_from_basegoal(goal)

        target_shape = len(self.times()) if is_path_goal else None
        target_min, target_max = self._gp_min_max_arrays(goal, target_shape)
        return {
            "target_min": np.asarray(target_min, dtype=float),
            "target_max": np.asarray(target_max, dtype=float),
        }

    def evaluate_goal_function(
        self, goal, *, ensemble_member: int, is_path_goal: bool
    ) -> np.ndarray:
        """Evaluate the goal function on the current solver output."""
        expression = goal.function(self, ensemble_member)
        if is_path_goal:
            expression = self.map_path_expression(expression, ensemble_member)
        else:
            expression = ca.transpose(ca.vertcat(expression))

        evaluator = ca.Function("performance_metrics_goal_eval", [self.solver_input], [expression])
        return np.array(evaluator(self.solver_output))

    @staticmethod
    def _flatten_constraint_values(values) -> np.ndarray:
        """Convert constraint values or bounds to a one-dimensional array."""
        if isinstance(values, Timeseries):
            values = values.values

        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            return np.array([float(array)])
        return array.reshape(-1, order="F")

    def _normalize_constraint_bound_shape(self, values: np.ndarray, bound) -> np.ndarray:
        """Broadcast a constraint bound to the shape of the evaluated values."""
        flattened_values = self._flatten_constraint_values(values)
        flattened_bound = self._flatten_constraint_values(bound)
        if flattened_bound.shape != flattened_values.shape:
            flattened_bound = np.broadcast_to(flattened_bound, flattened_values.shape)
        return flattened_bound

    def _evaluate_constraint_expression(
        self, expression, *, ensemble_member: int, is_path_constraint: bool
    ) -> np.ndarray:
        """Evaluate a scalar or path constraint expression on the current solver output."""
        if is_path_constraint:
            expression = self.map_path_expression(expression, ensemble_member)
        else:
            expression = ca.transpose(ca.vertcat(expression))

        evaluator = ca.Function(
            "performance_metrics_constraint_eval", [self.solver_input], [expression]
        )
        return np.array(evaluator(self.solver_output))

    def _count_active_constraint_entries(
        self, values: np.ndarray, minimum, maximum
    ) -> tuple[int, int]:
        """Count total and active constraint entries for one expanded constraint."""
        flattened_values = self._flatten_constraint_values(values)
        flattened_min = self._normalize_constraint_bound_shape(values, minimum)
        flattened_max = self._normalize_constraint_bound_shape(values, maximum)

        total_entries = 0
        active_entries = 0
        for value, minimum_value, maximum_value in zip(
            flattened_values, flattened_min, flattened_max, strict=False
        ):
            has_lower_bound = np.isfinite(minimum_value)
            has_upper_bound = np.isfinite(maximum_value)
            if not has_lower_bound and not has_upper_bound:
                continue

            total_entries += 1
            if (
                has_lower_bound
                and has_upper_bound
                and abs(minimum_value - maximum_value) <= ABS_TOL
            ):
                if abs(value - minimum_value) <= ABS_TOL:
                    active_entries += 1
            elif (has_lower_bound and abs(value - minimum_value) <= ABS_TOL) or (
                has_upper_bound and abs(value - maximum_value) <= ABS_TOL
            ):
                active_entries += 1

        return total_entries, active_entries

    def _count_tuple_constraint_activity(
        self, constraint: tuple, *, ensemble_member: int, is_path_constraint: bool
    ) -> tuple[int, int]:
        """Count total and active entries for a plain RTC-Tools constraint tuple."""
        expression, minimum, maximum = constraint
        values = self._evaluate_constraint_expression(
            expression,
            ensemble_member=ensemble_member,
            is_path_constraint=is_path_constraint,
        )
        return self._count_active_constraint_entries(values, minimum, maximum)

    def _count_goal_constraint_activity(
        self, constraint, *, ensemble_member: int
    ) -> tuple[int, int]:
        """Count total and active entries for an internal goal-programming hard constraint."""
        is_path_constraint = isinstance(constraint.min, Timeseries)
        values = self._evaluate_constraint_expression(
            constraint.function(self),
            ensemble_member=ensemble_member,
            is_path_constraint=is_path_constraint,
        )
        return self._count_active_constraint_entries(values, constraint.min, constraint.max)

    def _get_goal_constraint_entry_count(self, constraint, *, ensemble_member: int) -> int:
        """Return the number of scalar entries contributed by a hard goal constraint."""
        total_entries, _ = self._count_goal_constraint_activity(
            constraint, ensemble_member=ensemble_member
        )
        return total_entries

    def _warn_shadow_prices_unavailable_once(self, message: str):
        """Log a shadow-price availability warning at most once per problem instance."""
        if getattr(self, "_shadow_price_warning_issued", False):
            return
        logger.warning(message)
        self._shadow_price_warning_issued = True

    def get_previous_priority_shadow_prices(
        self, *, ensemble_member: int = 0, current_priority: int | None = None
    ) -> dict[int, float]:
        """Aggregate absolute shadow prices for hard constraints from earlier priorities."""
        if current_priority is None:
            current_priority = getattr(self, "_gp_current_priority", None)
        if current_priority is None:
            return {}

        previous_priority_constraints = [
            constraint
            for constraint in self._GoalProgrammingMixin__constraint_store[ensemble_member].values()
            if getattr(getattr(constraint, "goal", None), "priority", None) is not None
            and int(constraint.goal.priority) < int(current_priority)
        ]
        previous_priority_constraints.extend(
            constraint
            for constraint in self._GoalProgrammingMixin__path_constraint_store[
                ensemble_member
            ].values()
            if getattr(getattr(constraint, "goal", None), "priority", None) is not None
            and int(constraint.goal.priority) < int(current_priority)
        )
        if not previous_priority_constraints:
            return {}

        lam_g = getattr(self, "_OptimizationProblem__lam_g", None)
        if lam_g is None:
            return {}

        multiplier_values = np.asarray(lam_g, dtype=float).reshape(-1, order="F")
        if multiplier_values.size == 0:
            return {}
        if not np.isfinite(multiplier_values).any():
            self._warn_shadow_prices_unavailable_once(
                "Shadow prices are unavailable because the solver did not return finite "
                "constraint multipliers. This commonly happens for mixed-integer solves, "
                "where shadow prices are generally not well-defined."
            )
            return {}

        base_constraints = super(GoalProgrammingMixin, self).constraints(ensemble_member)
        full_constraints = self.constraints(ensemble_member)
        base_path_constraints = super(GoalProgrammingMixin, self).path_constraints(ensemble_member)
        full_path_constraints = self.path_constraints(ensemble_member)

        total_base_constraints = sum(
            self._count_tuple_constraint_activity(
                constraint, ensemble_member=ensemble_member, is_path_constraint=False
            )[0]
            for constraint in base_constraints
        )
        total_full_constraints = sum(
            self._count_tuple_constraint_activity(
                constraint, ensemble_member=ensemble_member, is_path_constraint=False
            )[0]
            for constraint in full_constraints
        )
        total_base_path_constraints = sum(
            self._count_tuple_constraint_activity(
                constraint, ensemble_member=ensemble_member, is_path_constraint=True
            )[0]
            for constraint in base_path_constraints
        )
        total_full_path_constraints = sum(
            self._count_tuple_constraint_activity(
                constraint, ensemble_member=ensemble_member, is_path_constraint=True
            )[0]
            for constraint in full_path_constraints
        )

        total_additional_constraints = total_full_constraints - total_base_constraints
        total_additional_path_constraints = (
            total_full_path_constraints - total_base_path_constraints
        )
        if total_additional_constraints < 0 or total_additional_path_constraints < 0:
            self._warn_shadow_prices_unavailable_once(
                "Cannot extract shadow prices because the computed number of additional goal "
                "constraints is inconsistent with the full constraint set."
            )
            return {}

        total_goal_constraint_entries = (
            total_additional_constraints + total_additional_path_constraints
        )
        if total_goal_constraint_entries == 0:
            return {}
        if total_goal_constraint_entries > multiplier_values.size:
            self._warn_shadow_prices_unavailable_once(
                "Cannot extract shadow prices for previous priorities because the number of "
                "goal-programming constraint entries exceeds the available Lagrange multipliers."
            )
            return {}

        path_multiplier_tail = multiplier_values[-total_full_path_constraints:]
        additional_path_multiplier_tail = path_multiplier_tail[-total_additional_path_constraints:]

        constraint_multiplier_end = multiplier_values.size - total_full_path_constraints
        constraint_multiplier_tail = multiplier_values[
            constraint_multiplier_end - total_full_constraints : constraint_multiplier_end
        ]
        additional_constraint_multiplier_tail = constraint_multiplier_tail[
            -total_additional_constraints:
        ]

        shadow_prices: dict[int, float] = {}

        offset = 0
        for constraint in self._GoalProgrammingMixin__constraint_store[ensemble_member].values():
            entry_count = self._get_goal_constraint_entry_count(
                constraint, ensemble_member=ensemble_member
            )
            multiplier_slice = additional_constraint_multiplier_tail[offset : offset + entry_count]
            offset += entry_count

            priority = getattr(getattr(constraint, "goal", None), "priority", None)
            if priority is None or int(priority) >= int(current_priority):
                continue

            finite_multiplier_slice = multiplier_slice[np.isfinite(multiplier_slice)]
            if finite_multiplier_slice.size == 0:
                continue

            priority_int = int(priority)
            shadow_prices[priority_int] = shadow_prices.get(priority_int, 0.0) + float(
                np.sum(np.abs(finite_multiplier_slice))
            )

        offset = 0
        for constraint in self._GoalProgrammingMixin__path_constraint_store[
            ensemble_member
        ].values():
            entry_count = self._get_goal_constraint_entry_count(
                constraint, ensemble_member=ensemble_member
            )
            multiplier_slice = additional_path_multiplier_tail[offset : offset + entry_count]
            offset += entry_count

            priority = getattr(getattr(constraint, "goal", None), "priority", None)
            if priority is None or int(priority) >= int(current_priority):
                continue

            finite_multiplier_slice = multiplier_slice[np.isfinite(multiplier_slice)]
            if finite_multiplier_slice.size == 0:
                continue

            priority_int = int(priority)
            shadow_prices[priority_int] = shadow_prices.get(priority_int, 0.0) + float(
                np.sum(np.abs(finite_multiplier_slice))
            )

        return shadow_prices

    def get_constraint_activity_metrics(
        self, *, ensemble_member: int = 0, current_priority: int | None = None
    ) -> dict[str, float | int]:
        """Return summary metrics for active hard constraints in the current subproblem."""
        base_constraints = super(GoalProgrammingMixin, self).constraints(ensemble_member)
        base_path_constraints = super(GoalProgrammingMixin, self).path_constraints(ensemble_member)

        total_hard_constraints = 0
        active_hard_constraints = 0

        for constraint in base_constraints:
            total_entries, active_entries = self._count_tuple_constraint_activity(
                constraint,
                ensemble_member=ensemble_member,
                is_path_constraint=False,
            )
            total_hard_constraints += total_entries
            active_hard_constraints += active_entries

        for constraint in base_path_constraints:
            total_entries, active_entries = self._count_tuple_constraint_activity(
                constraint,
                ensemble_member=ensemble_member,
                is_path_constraint=True,
            )
            total_hard_constraints += total_entries
            active_hard_constraints += active_entries

        goal_constraints = list(
            self._GoalProgrammingMixin__constraint_store[ensemble_member].values()
        )
        goal_constraints.extend(
            self._GoalProgrammingMixin__path_constraint_store[ensemble_member].values()
        )

        if current_priority is None:
            current_priority = getattr(self, "_gp_current_priority", None)
        active_previous_priority_constraints = 0

        for constraint in goal_constraints:
            total_entries, active_entries = self._count_goal_constraint_activity(
                constraint, ensemble_member=ensemble_member
            )
            total_hard_constraints += total_entries
            active_hard_constraints += active_entries

            goal_priority = getattr(constraint.goal, "priority", None)
            if current_priority is not None and goal_priority is not None:
                if int(goal_priority) < int(current_priority):
                    active_previous_priority_constraints += active_entries

        active_fraction = 0.0
        if total_hard_constraints > 0:
            active_fraction = active_hard_constraints / total_hard_constraints

        return {
            "active_hard_constraints": active_hard_constraints,
            "active_hard_constraints_fraction": active_fraction,
            "active_previous_priority_constraints": active_previous_priority_constraints,
        }
