"""Module for a basic Goal."""
import logging
import numpy as np

from rtctools.optimization.goal_programming_mixin import Goal
from rtctools.optimization.optimization_problem import OptimizationProblem

logger = logging.getLogger("rtctools")

GOAL_TYPES = [
    "range",
    "minimization",
    "minimization_sum",
    "maximization_sum",
    "maximization",
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
        Type of goal ('range' or 'minimization' or 'maximization or 'minimization_sum' or "maximization_sum").
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
        elif goal_type in ["minimization_sum", "maximization_sum"]:
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
        elif goal_type in ["minimization_sum", "maximization_sum"]:
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
        if self.goal_type == "maximization":
            return -optimization_problem.state(self.state)
        elif self.goal_type in ["minimization_sum", "maximization_sum"]:
            times = optimization_problem.times(self.state)
            tot = 0.0
            for t in times:
                tot += optimization_problem.state_at(self.state, t)
            if self.goal_type == "minimization_sum":
                return tot
            else:
                return tot
        else:
            return optimization_problem.state(self.state)
        # except KeyError:
        #     # state_vars = self.state.split('.')[0]
        #     # state_vars = state_vars.split(',')
        #     # state_operators = self.state.split('.')[1]
        #     # state_operators = state_operators.split(',')
        #     # op = {'+': lambda x, y: x + y,
        #     #       '-': lambda x, y: x - y}
        #     # self.state = ''
        #     # for i in range(0, len(state_vars)):
        #     return optimization_problem.state(self.state)
        #     # return optimization_problem.state([op[state_operators[i]] statevars[i] for i in range(0,len(state_vars))])
        #     # return optimization_problem.state([2*a for a in x if a % 2 == 1])

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
            try:
                self.function_range[0] = optimization_problem.bounds()[self.state][0].values
            except AttributeError:
                self.function_range[0] = optimization_problem.bounds()[self.state][0]
        if not np.isfinite(function_max):
            try:
                self.function_range[1] = optimization_problem.bounds()[self.state][1].values
            except AttributeError:
                self.function_range[1] = optimization_problem.bounds()[self.state][1]

    def _set_function_nominal(self, function_nominal):
        """Set function nominal"""
        self.function_nominal = function_nominal
        if not np.isfinite(self.function_nominal):
            try:
                if np.all(np.isfinite(self.function_range)):
                    self.function_nominal = np.sum(self.function_range) / 2
                else:
                    self.function_nominal = 1.0
                    logger.warning("Function nominal not specified, nominal is set to 1.0")
            except TypeError:
                if np.all(np.isfinite(self.function_range[1].values)):
                    self.function_nominal = np.sum(self.function_range[1].values) / 2
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
            try:
                if np.isnan(target_max):
                    self.target_max = np.nan
                else:
                    self.target_max = optimization_problem.parameters(0)[target_max]
            except TypeError:
                self.target_max = optimization_problem.parameters(0)[target_max]
            try:
                if np.isnan(target_min):
                    self.target_min = np.nan
                else:
                    self.target_min = optimization_problem.parameters(0)[target_min]
            except TypeError:
                self.target_min = optimization_problem.parameters(0)[target_min]
            # self.target_min = optimization_problem.parameters(0)[target_min]
        elif target_data_type == "timeseries":
            try:
                if np.isnan(target_max):
                    self.target_max = np.nan
                else:
                    self.target_max = optimization_problem.get_timeseries(target_max)
            except TypeError:
                self.target_max = optimization_problem.get_timeseries(target_max)
            try:
                if np.isnan(target_min):
                    self.target_min = np.nan
                else:
                    self.target_min = optimization_problem.get_timeseries(target_min)
            except TypeError:
                self.target_min = optimization_problem.get_timeseries(target_min)

