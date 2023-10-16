"""This file contains functions to get performance metrics for the BaseGoal."""
import logging
from typing import Dict

import numpy as np
from rtctools_interface.optimization.base_goal import BaseGoal
from rtctools_interface.optimization.goal_table_schema import (
    GOAL_TYPES,
    BaseGoalModel,
    MaximizationGoalModel,
    MinimizationGoalModel,
    RangeGoalModel,
)


logger = logging.getLogger("rtctools")


def performance_metrics_minmaximization(results: Dict[str, np.array], goal: MinimizationGoalModel):
    """Returns total sum over time horizon"""
    state_timeseries = results[goal.state]
    metrics = {
        "timeseries_sum": sum(state_timeseries),
        "timeseries_min": min(state_timeseries),
        "timeseries_max": max(state_timeseries),
        "timeseries_avg": np.mean(state_timeseries),
    }
    return metrics


def get_performance_metrics(results, goal: BaseGoalModel):
    """Returns a dict with performance metrics for each goal."""
    if (
        type(goal) == MinimizationGoalModel  # pylint: disable=unidiomatic-typecheck
        or type(goal) == MaximizationGoalModel  # pylint: disable=unidiomatic-typecheck
    ):
        return performance_metrics_minmaximization(results, goal)
