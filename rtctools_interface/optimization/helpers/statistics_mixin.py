"""Base mixin for retrieving particular stats for goals and plotting."""

import logging

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
