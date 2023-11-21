"""Module for a basic Goal."""
import logging

import numpy as np

from rtctools.optimization.goal_programming_mixin import Goal
from rtctools.optimization.optimization_problem import OptimizationProblem
from rtctools.optimization.timeseries import Timeseries

from rtctools_interface.optimization.goal_table_schema import GOAL_TYPES, TARGET_DATA_TYPES
from rtctools_interface.optimization.type_definitions import GoalConfig


logger = logging.getLogger("rtctools")


class BaseGoal(Goal):
    """
    Basic optimization goal for a given state.

    :cvar goal_type:
        Type of goal ('range' or 'minimization_path' or 'maximization_path')
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
        goal_type="minimization_path",
        function_min=np.nan,
        function_max=np.nan,
        function_nominal=np.nan,
        target_data_type="value",
        target_min=np.nan,
        target_max=np.nan,
        priority=1,
        weight=1.0,
        order=2,
        goal_id=None,
        **_kwargs,
    ):
        self.goal_id = goal_id
        self.state = state
        self.target_data_type = target_data_type
        self._set_goal_type(goal_type)
        if goal_type in ["range", "range_rate_of_change"]:
            self._set_function_bounds(
                optimization_problem=optimization_problem,
                function_min=function_min,
                function_max=function_max,
            )
        self._set_function_nominal(function_nominal)
        if goal_type in ["range", "range_rate_of_change"]:
            self._set_target_bounds(
                optimization_problem=optimization_problem,
                target_min=target_min,
                target_max=target_max,
            )
        self.priority = priority if np.isfinite(priority) else 1
        self.weight = weight if np.isfinite(weight) else 1.0
        self._set_order(order)

    def function(self, optimization_problem, ensemble_member):
        del ensemble_member
        if self.goal_type == "maximization_path":
            return -optimization_problem.state(self.state)
        if self.goal_type in ["minimization_path", "range"]:
            return optimization_problem.state(self.state)
        if self.goal_type in ["range_rate_of_change"]:
            return optimization_problem.der(self.state)
        raise ValueError("Unsupported goal type '{}', supported are {}".format(self.goal_type, GOAL_TYPES.keys()))

    def _set_order(self, order):
        """Set the order of the goal."""
        if np.isfinite(order):
            self.order = order
        elif self.goal_type in ["maximization_path", "minimization_path"]:
            self.order = 1
        else:
            self.order = 2
        if self.goal_type == "maximization_path" and self.order % 2 == 0:
            logger.warning(
                "Using even order '%i' for a maximization_path goal" + " results in a minimization_path goal.",
                self.order,
            )

    def _set_goal_type(
        self,
        goal_type,
    ):
        """Set the goal type."""
        if goal_type in GOAL_TYPES:
            self.goal_type = goal_type
        else:
            raise ValueError(f"goal_type should be one of {GOAL_TYPES.keys()}.")

    def _get_state_range(self, optimization_problem, state_name):
        if isinstance(optimization_problem.bounds()[state_name][0], float):
            state_range_0 = optimization_problem.bounds()[state_name][0]
        elif isinstance(optimization_problem.bounds()[state_name][0], Timeseries):
            state_range_0 = optimization_problem.bounds()[state_name][0].values
        else:
            state_range_0 = np.nan
        if isinstance(optimization_problem.bounds()[state_name][1], float):
            state_range_1 = optimization_problem.bounds()[state_name][1]
        elif isinstance(optimization_problem.bounds()[state_name][1], Timeseries):
            state_range_1 = optimization_problem.bounds()[state_name][1].values
        else:
            state_range_1 = np.nan
        return (state_range_0, state_range_1)

    def _set_function_bounds(
        self,
        optimization_problem: OptimizationProblem,
        function_min=np.nan,
        function_max=np.nan,
    ):
        """Set function bounds either by user specified value or calculated"""
        state_range = self._get_state_range(optimization_problem, self.state)
        if (not np.isfinite(function_min) and not np.isfinite(state_range[0])) or (
            not np.isfinite(function_max) and not np.isfinite(state_range[1])
        ):
            raise ValueError(
                f"The upper/lower bound for state {self.state} for goal with id={self.goal_id} is not specified"
                + " so the function range should be specified!"
            )
        if self.goal_type in ["range_rate_of_change"]:
            maximum_scaled_difference = (state_range[1] - state_range[0]) / np.diff(optimization_problem.times()).min()
            calculated_range = (-maximum_scaled_difference, maximum_scaled_difference)
        else:
            calculated_range = state_range

        function_range_0 = function_min if np.isfinite(function_min) else calculated_range[0]
        function_range_1 = function_max if np.isfinite(function_max) else calculated_range[1]
        self.function_range = (function_range_0, function_range_1)

    def _set_function_nominal(self, function_nominal):
        """Set function nominal"""
        self.function_nominal = function_nominal
        if not np.isfinite(self.function_nominal):
            if isinstance(self.function_range, (list, tuple)):
                if np.all(np.isfinite(self.function_range)):
                    self.function_nominal = abs(self.function_range[0]) + abs(self.function_range[1]) / 2
                    return
            self.function_nominal = 1.0
            logger.warning("Function nominal not specified, nominal is set to 1.0")

    def _set_target_bounds(
        self,
        optimization_problem: OptimizationProblem,
        target_min=np.nan,
        target_max=np.nan,
    ):
        """Set the target bounds."""

        def set_value_target():
            if self.goal_type == "range_rate_of_change":
                self.target_min = float(target_min) / 100 * self.function_nominal
                self.target_max = float(target_max) / 100 * self.function_nominal
            else:
                self.target_min = float(target_min)
                self.target_max = float(target_max)

        def set_parameter_target():
            if isinstance(target_max, str):
                self.target_max = optimization_problem.parameters(0)[target_max]
                if self.target_max is None:
                    self.target_max = optimization_problem.io.get_parameter(target_max)
            elif np.isnan(target_max):
                self.target_max = np.nan
            if isinstance(target_min, str):
                self.target_min = optimization_problem.parameters(0)[target_min]
                if self.target_min is None:
                    self.target_min = optimization_problem.io.get_parameter(target_min)
            elif np.isnan(target_min):
                self.target_min = np.nan

        def set_timeseries_target():
            if isinstance(target_max, str):
                self.target_max = optimization_problem.get_timeseries(target_max)
            elif np.isnan(target_max):
                self.target_max = np.nan
            if isinstance(target_min, str):
                self.target_min = optimization_problem.get_timeseries(target_min)
            elif np.isnan(target_min):
                self.target_min = np.nan

        if self.target_data_type not in TARGET_DATA_TYPES:
            raise ValueError(f"target_data_type should be one of {TARGET_DATA_TYPES}.")

        if self.goal_type == "range_rate_of_change" and self.target_data_type != "value":
            raise ValueError("For range_rate_of_change goal only the `value` target type is supported.")

        if self.target_data_type == "value":
            set_value_target()
        elif self.target_data_type == "parameter":
            set_parameter_target()
        elif self.target_data_type == "timeseries":
            set_timeseries_target()

        self._target_dict = optimization_problem.collect_range_target_values_from_basegoal(self)

    def get_goal_config(self) -> GoalConfig:
        """
        Serialize the goal configuration into a dictionary.
        """
        goal_config: GoalConfig = {
            "goal_id": self.goal_id,
            "state": self.state,
            "goal_type": self.goal_type,
            "function_min": self.function_range[0] if np.isfinite(self.function_range[0]) else None,
            "function_max": self.function_range[1] if np.isfinite(self.function_range[1]) else None,
            "function_nominal": self.function_nominal if np.isfinite(self.function_nominal) else None,
            "target_min_series": None,
            "target_max_series": None,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "priority": self.priority,
            "weight": self.weight,
            "order": self.order,
        }
        if isinstance(self.target_min, Timeseries):
            goal_config["target_min"] = self.target_min.values
        if isinstance(self.target_max, Timeseries):
            goal_config["target_max"] = self.target_max.values

        if hasattr(self, "_target_dict"):
            goal_config["target_min_series"] = self._target_dict["target_min"]
            goal_config["target_max_series"] = self._target_dict["target_max"]
        return goal_config
