"""Base mixin for retrieving particular stats for goals and plotting."""

import logging
import re

import casadi as ca
import numpy as np

from rtctools_interface.optimization.base_goal import BaseGoal
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
