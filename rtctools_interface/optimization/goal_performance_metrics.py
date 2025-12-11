"""This file contains functions to get performance metrics for the BaseGoal."""

import logging

import numpy as np
import pandas as pd

from rtctools_interface.optimization.goal_table_schema import (
    BaseGoalModel,
    MaximizationGoalModel,
    MinimizationGoalModel,
    RangeGoalModel,
    RangeRateOfChangeGoalModel,
)
from rtctools_interface.utils.type_definitions import TargetDict

logger = logging.getLogger("rtctools")

ABS_TOL = 0.001


def get_mean_absolute_percentual_difference(timeseries: np.ndarray) -> float:
    """Calculate the mean absolute percentual difference, ignoring entries where timeseries = 0."""
    nonzero_indices = np.nonzero(timeseries)
    timeseries = timeseries[nonzero_indices]
    differences = np.diff(timeseries)
    if len(timeseries) <= 1:
        return 0
    mapd = np.mean(np.abs(differences / timeseries[:-1]))
    return mapd


def get_absolute_sum_difference(timeseries: np.ndarray) -> float:
    """Calculate the mean of absolute first-order difference."""
    if len(timeseries) <= 1:
        return 0
    mad = np.mean(np.abs(np.diff(timeseries)))
    return mad


def get_max_difference(timeseries: np.ndarray) -> float:
    """Get maximum one step difference"""
    return max(np.diff(timeseries))


def get_basic_metrics(timeseries: np.ndarray) -> dict[str, float]:
    """Get general metrics applicable for each goal type."""
    metrics = {
        "timeseries_sum": sum(timeseries),
        "timeseries_min": min(timeseries),
        "timeseries_max": max(timeseries),
        "timeseries_avg": np.mean(timeseries),
        "mean_absolute_percentual_difference": get_mean_absolute_percentual_difference(timeseries),
        "mean_absolute_difference": get_absolute_sum_difference(timeseries),
        "max_difference": get_max_difference(timeseries),
    }
    return metrics


def performance_metrics_minmaximization(
    results: dict[str, np.ndarray], goal: MinimizationGoalModel
) -> pd.Series:
    """Get all relevant statistics for a min/maximization goal."""
    state_timeseries = results[goal.state]
    metrics = get_basic_metrics(state_timeseries)
    return pd.Series(metrics)


def get_range_percentual_exceedance(
    timeseries: np.ndarray, goal: RangeGoalModel, targets: TargetDict
) -> dict[str, float | None] | None:
    """Calculate percentage of timesteps in which target is exceeded"""
    if goal.goal_type not in ["range", "range_rate_of_change"]:
        below_target = None
        above_target = None
    else:
        below_target = float(
            sum(np.where(timeseries + ABS_TOL < targets["target_min"], 1, 0)) / len(timeseries)
        )
        above_target = float(
            sum(np.where(timeseries - ABS_TOL > targets["target_max"], 1, 0)) / len(timeseries)
        )
    return {"perc_below_target": below_target, "perc_above_target": above_target}


def get_range_total_exceedance(
    timeseries: np.ndarray, goal: RangeGoalModel, targets: TargetDict
) -> dict[str, float | None] | None:
    """Calculate sum of absolute exceedances of the target"""
    if goal.goal_type not in ["range", "range_rate_of_change"]:
        below_target = None
        above_target = None
    else:
        below_target = float(
            sum(
                np.abs(
                    np.where(
                        timeseries < targets["target_min"], timeseries - targets["target_min"], 0
                    )
                )
            )
        )
        above_target = float(
            sum(
                np.abs(
                    np.where(
                        timeseries > targets["target_max"], timeseries - targets["target_max"], 0
                    )
                )
            )
        )
    return {"sum_below_target": below_target, "sum_above_target": above_target}


def performance_metrics_range(
    results: dict[str, np.ndarray], goal: RangeGoalModel, targets: TargetDict
) -> pd.Series:
    """Get all relevant statistics for a range goal."""
    metrics: dict = {}
    state_timeseries = results[goal.state]
    metrics = metrics | get_basic_metrics(state_timeseries)
    metrics = metrics | get_range_percentual_exceedance(state_timeseries, goal, targets)
    metrics = metrics | get_range_total_exceedance(state_timeseries, goal, targets)
    return pd.Series(metrics)


def performance_metrics_rangerateofchange(
    results: dict[str, np.ndarray], goal: RangeGoalModel, _targets: TargetDict
) -> pd.Series:
    """Get all relevant statistics for a range-rate-of-change goal."""
    metrics: dict[str, float | None] = {}
    state_timeseries = results[goal.state]
    metrics = metrics | get_basic_metrics(state_timeseries)
    return pd.Series(metrics)


def get_performance_metrics(results, goal: BaseGoalModel, targets: TargetDict) -> pd.Series | None:
    """Returns a series with performance metrics for each goal."""
    if type(goal) in [MinimizationGoalModel, MaximizationGoalModel]:  # pylint: disable=unidiomatic-typecheck
        return performance_metrics_minmaximization(results, goal)
    if type(goal) in [RangeGoalModel]:
        return performance_metrics_range(results, goal, targets)
    if type(goal) in [RangeRateOfChangeGoalModel]:
        return performance_metrics_rangerateofchange(results, goal, targets)
    logger.info("No performance metrics are implemented for goal of type: %s", str(type(goal)))
    return None
