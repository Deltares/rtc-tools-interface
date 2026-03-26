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


def _flatten_metric_values(values: np.ndarray) -> np.ndarray:
    """Convert goal values to a one-dimensional array for metric calculation."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.array([float(array)])
    return array.reshape(-1, order="F")


def _normalize_target_shape(
    values: np.ndarray, target: np.ndarray | float | None
) -> np.ndarray | None:
    """Broadcast a goal target to the shape of the evaluated goal values."""
    if target is None:
        return None

    value_array = np.asarray(values, dtype=float)
    target_array = np.asarray(target, dtype=float)

    if value_array.ndim == 0:
        value_array = np.array([float(value_array)])
    if target_array.ndim == 0:
        target_array = np.array([float(target_array)])

    if (
        target_array.ndim == 2
        and value_array.ndim == 2
        and target_array.shape == value_array.shape[::-1]
    ):
        target_array = target_array.transpose()

    if target_array.shape != value_array.shape:
        target_array = np.broadcast_to(target_array, value_array.shape)

    return _flatten_metric_values(target_array)


def get_mean_absolute_percentual_difference(timeseries: np.ndarray) -> float:
    """Calculate the mean absolute percentual difference, ignoring entries where timeseries = 0."""
    timeseries = _flatten_metric_values(timeseries)
    nonzero_indices = np.nonzero(timeseries)
    timeseries = timeseries[nonzero_indices]
    differences = np.diff(timeseries)
    if len(timeseries) <= 1:
        return 0
    mapd = np.mean(np.abs(differences / timeseries[:-1]))
    return mapd


def get_absolute_sum_difference(timeseries: np.ndarray) -> float:
    """Calculate the mean of absolute first-order difference."""
    timeseries = _flatten_metric_values(timeseries)
    if len(timeseries) <= 1:
        return 0
    mad = np.mean(np.abs(np.diff(timeseries)))
    return mad


def get_max_difference(timeseries: np.ndarray) -> float:
    """Get maximum one step difference"""
    timeseries = _flatten_metric_values(timeseries)
    if len(timeseries) <= 1:
        return 0
    return max(np.diff(timeseries))


def get_basic_metrics(timeseries: np.ndarray) -> dict[str, float]:
    """Get general metrics applicable for each goal type."""
    timeseries = _flatten_metric_values(timeseries)
    metrics = {
        "timeseries_sum": float(np.sum(timeseries)),
        "timeseries_min": float(np.min(timeseries)),
        "timeseries_max": float(np.max(timeseries)),
        "timeseries_avg": np.mean(timeseries),
        "mean_absolute_percentual_difference": get_mean_absolute_percentual_difference(timeseries),
        "mean_absolute_difference": get_absolute_sum_difference(timeseries),
        "max_difference": get_max_difference(timeseries),
    }
    return metrics


def get_range_percentual_exceedance_from_targets(
    timeseries: np.ndarray,
    target_min: np.ndarray | float | None,
    target_max: np.ndarray | float | None,
) -> dict[str, float | None]:
    """Calculate percentage of entries for which the target is exceeded."""
    timeseries = _flatten_metric_values(timeseries)
    target_min = _normalize_target_shape(timeseries, target_min)
    target_max = _normalize_target_shape(timeseries, target_max)

    below_target = None
    above_target = None
    if target_min is not None and np.any(np.isfinite(target_min)):
        below_target = float(
            sum(np.where(timeseries + ABS_TOL < target_min, 1, 0)) / len(timeseries)
        )
    if target_max is not None and np.any(np.isfinite(target_max)):
        above_target = float(
            sum(np.where(timeseries - ABS_TOL > target_max, 1, 0)) / len(timeseries)
        )

    return {"perc_below_target": below_target, "perc_above_target": above_target}


def get_range_total_exceedance_from_targets(
    timeseries: np.ndarray,
    target_min: np.ndarray | float | None,
    target_max: np.ndarray | float | None,
) -> dict[str, float | None]:
    """Calculate the total absolute exceedance of a target."""
    timeseries = _flatten_metric_values(timeseries)
    target_min = _normalize_target_shape(timeseries, target_min)
    target_max = _normalize_target_shape(timeseries, target_max)

    below_target = None
    above_target = None
    if target_min is not None and np.any(np.isfinite(target_min)):
        below_target = float(
            sum(np.abs(np.where(timeseries < target_min, timeseries - target_min, 0)))
        )
    if target_max is not None and np.any(np.isfinite(target_max)):
        above_target = float(
            sum(np.abs(np.where(timeseries > target_max, timeseries - target_max, 0)))
        )

    return {"sum_below_target": below_target, "sum_above_target": above_target}


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
        return {"perc_below_target": None, "perc_above_target": None}
    return get_range_percentual_exceedance_from_targets(
        timeseries, targets["target_min"], targets["target_max"]
    )


def get_range_total_exceedance(
    timeseries: np.ndarray, goal: RangeGoalModel, targets: TargetDict
) -> dict[str, float | None] | None:
    """Calculate sum of absolute exceedances of the target"""
    if goal.goal_type not in ["range", "range_rate_of_change"]:
        return {"sum_below_target": None, "sum_above_target": None}
    return get_range_total_exceedance_from_targets(
        timeseries, targets["target_min"], targets["target_max"]
    )


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


def get_custom_performance_metrics(
    values: np.ndarray,
    target_min: np.ndarray | float | None = None,
    target_max: np.ndarray | float | None = None,
) -> pd.Series:
    """Return performance metrics for an evaluated custom RTC-Tools goal."""
    metrics = get_basic_metrics(values)
    if target_min is not None or target_max is not None:
        metrics = metrics | get_range_percentual_exceedance_from_targets(
            values, target_min, target_max
        )
        metrics = metrics | get_range_total_exceedance_from_targets(values, target_min, target_max)
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
