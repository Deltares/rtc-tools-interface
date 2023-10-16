"""This file contains functions to get performance metrics for the BaseGoal."""
import logging
from typing import Dict

import pandas as pd
import numpy as np

from rtctools_interface.optimization.goal_table_schema import (
    BaseGoalModel,
    MaximizationGoalModel,
    MinimizationGoalModel,
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
    return pd.Series(metrics)


def get_performance_metrics(results, goal: BaseGoalModel):
    """Returns a dict with performance metrics for each goal."""
    if (
        type(goal) == MinimizationGoalModel  # pylint: disable=unidiomatic-typecheck
        or type(goal) == MaximizationGoalModel  # pylint: disable=unidiomatic-typecheck
    ):
        return performance_metrics_minmaximization(results, goal)
    logger.info("No performance metrics are implemented for goal of type: %s", str(type(goal)))
    return None
