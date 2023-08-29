"""Module for a basic Goal."""
import logging

import numpy as np

from rtctools.optimization.goal_programming_mixin import Goal
from rtctools.optimization.optimization_problem import OptimizationProblem

logger = logging.getLogger("rtctools")

GOAL_TYPES = [
    "range",
    "minimization",
]

TARGET_DATA_TYPES = [
    "value",
    "parameter",
    "timeseries",
]


class BaseGoal(Goal):
    """
    Basic optimization goal for a given state.

    :cvar goal_type:
        Type of goal ('range' or 'minimization').
    :cvar target_data_type:
        Type of target data ('value', 'parameter', 'timeseries').
        If 'value', set the target bounds by value.
        If 'parameter', set the bounds by a parameter. The target_min
        and/or target_max are expected to be the name of the parameter.
        If 'timeseries', set the bounds by a timeseries. The target_min
        and/or target_max are expected to be the name of the timeseries.
    """

    def __init__(
        self,
        optimization_problem: OptimizationProblem,
        state,
        goal_type="minimization",
        function_min=np.nan,
        function_max=np.nan,
        function_nominal=np.nan,
        target_data_type="value",
        target_min=np.nan,
        target_max=np.nan,
        priority=1,
        weight=1.0,
        order=2,
    ):
        self.state = state
        self.goal_type = None
        self._set_goal_type(goal_type)
        if goal_type == "range":
            self._set_function_bounds(
                optimization_problem=optimization_problem,
                function_min=function_min,
                function_max=function_max)
        self._set_function_nominal(function_nominal)
        if goal_type == "range":
            self._set_target_bounds(
                optimization_problem=optimization_problem,
                target_data_type=target_data_type,
                target_min=target_min,
                target_max=target_max)
        self.priority = priority if np.isfinite(priority) else 1
        self.weight = weight if np.isfinite(weight) else 1.0
        self.order = order if np.isfinite(order) else 2

    def function(self, optimization_problem, ensemble_member):
        del ensemble_member
        return optimization_problem.state(self.state)

    def _set_goal_type(
        self,
        goal_type,
    ):
        """Set the goal type."""
        if goal_type in GOAL_TYPES:
            self.goal_type = goal_type
        else:
            raise ValueError(f"goal_type should be one of {GOAL_TYPES}.")

    def _set_function_bounds(
        self,
        optimization_problem: OptimizationProblem,
        function_min=np.nan,
        function_max=np.nan,
    ):
        """Set function bounds and nominal."""
        self.function_range = [function_min, function_max]
        if not np.isfinite(function_min):
            self.function_range[0] = optimization_problem.bounds()[self.state][0]
        if not np.isfinite(function_max):
            self.function_range[1] = optimization_problem.bounds()[self.state][1]

    def _set_function_nominal(self, function_nominal):
        """Set function nominal"""
        self.function_nominal = function_nominal
        if not np.isfinite(self.function_nominal):
            if np.all(np.isfinite(self.function_range)):
                self.function_nominal = np.sum(self.function_range) / 2
            else:
                self.function_nominal = 1.0
                logger.warning("Function nominal not specified, nominal is set to 1.0")

    def _set_target_bounds(
        self,
        optimization_problem: OptimizationProblem,
        target_data_type="value",
        target_min=np.nan,
        target_max=np.nan,
    ):
        """Set the target bounds."""
        if target_data_type not in TARGET_DATA_TYPES:
            raise ValueError(f"target_data_type should be one of {TARGET_DATA_TYPES}.")
        if target_data_type == "value":
            self.target_min = float(target_min)
            self.target_max = float(target_max)
        elif target_data_type == "parameter":
            self.target_min = optimization_problem.parameters(0)[target_min]
            self.target_max = optimization_problem.parameters(0)[target_max]
        elif target_data_type == "timeseries":
            self.target_min = optimization_problem.get_timeseries(target_min)
            self.target_max = optimization_problem.get_timeseries(target_max)
