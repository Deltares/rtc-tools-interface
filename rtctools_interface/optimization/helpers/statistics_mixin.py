"""Base class/mixin for with some methods for retrieving particular data/stats for goals and plotting. """

import logging
from typing import Dict, List, Tuple, Union

import numpy as np

from rtctools_interface.optimization.plot_and_goal_schema import (
    MinimizationGoalCombinedModel,
    MaximizationGoalCombinedModel,
    RangeGoalCombinedModel,
    RangeRateOfChangeGoalCombinedModel,
    PlotTableRow,
)
from rtctools_interface.optimization.type_definitions import TargetDict

logger = logging.getLogger("rtctools")


class StatisticsMixin:
    """A mixin class providing methods for collecting data and statistics from optimization results,
    useful for solution performance analysis."""

    def collect_range_target_values(
        self,
        all_goals: List[
            Union[
                MinimizationGoalCombinedModel,
                MaximizationGoalCombinedModel,
                RangeGoalCombinedModel,
                RangeRateOfChangeGoalCombinedModel,
                PlotTableRow,
            ]
        ],
    ) -> Dict[str, TargetDict]:
        """For the goals with targets, collect the actual timeseries with these targets."""
        t = self.times()

        def get_parameter_ranges(goal) -> Tuple[np.ndarray, np.ndarray]:
            try:
                target_min = np.full_like(t, 1) * self.parameters(0)[goal.target_min]
                target_max = np.full_like(t, 1) * self.parameters(0)[goal.target_max]
            except TypeError:
                target_min = np.full_like(t, 1) * self.io.get_parameter(goal.target_min)
                target_max = np.full_like(t, 1) * self.io.get_parameter(goal.target_max)
            return target_min, target_max

        def get_value_ranges(goal) -> Tuple[np.ndarray, np.ndarray]:
            target_min = np.full_like(t, 1) * float(goal.target_min)
            target_max = np.full_like(t, 1) * float(goal.target_max)
            return target_min, target_max

        def get_timeseries_ranges(goal) -> Tuple[np.ndarray, np.ndarray]:
            if isinstance(goal.target_min, str):
                target_min = self.get_timeseries(goal.target_min).values
            else:
                target_min = np.full_like(t, 1) * goal.target_min
            if isinstance(goal.target_max, str):
                target_max = self.get_timeseries(goal.target_max).values
            else:
                target_max = np.full_like(t, 1) * goal.target_max
            return target_min, target_max

        target_series: Dict[str, TargetDict] = {}
        for goal in all_goals:
            if goal.get("goal_type") in ["range", "range_rate_of_change"]:
                if goal.target_data_type == "parameter":
                    target_min, target_max = get_parameter_ranges(goal)
                elif goal.target_data_type == "value":
                    target_min, target_max = get_value_ranges(goal)
                elif goal.target_data_type == "timeseries":
                    target_min, target_max = get_timeseries_ranges(goal)
                else:
                    message = "Target type {} not known for goal {}.".format(goal.target_data_type, goal.goal_id)
                    logger.error(message)
                    raise ValueError(message)
                target_series[goal.goal_id] = {"target_min": target_min, "target_max": target_max}
        return target_series
