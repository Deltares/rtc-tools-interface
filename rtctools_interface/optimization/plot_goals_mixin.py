"""Module for plotting."""
import logging
import math
import os
import copy

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import numpy as np

from rtctools_interface.optimization.read_plot_table import read_plot_table

logger = logging.getLogger("rtctools")


def get_subplot(i_plot, n_rows, axs):
    """Determine the row and column index and returns the corresponding subplot object."""
    i_c = math.ceil((i_plot + 1) / n_rows) - 1
    i_r = i_plot - i_c * n_rows
    subplot = axs[i_r, i_c]
    return subplot


def get_timedeltas(optimization_problem):
    """Get delta_t for each timestep."""
    return [np.nan] + [
        optimization_problem.times()[i] - optimization_problem.times()[i - 1]
        for i in range(1, len(optimization_problem.times()))
    ]


class Subplot:
    """Wrapper class for a subplot in the figure".

    Contains the axis object and all configuration settings and data
    that belongs to the subplot."""

    def __init__(self, optimization_problem, axis, goal, subplot_config, results, results_prev):
        self.axis = axis
        self.goal = goal
        self.config = subplot_config
        self.function_nominal = self.goal.function_nominal if self.goal else 1
        self.results = results
        self.results_prev = results_prev
        self.datetimes = optimization_problem.io.datetimes
        self.time_deltas = get_timedeltas(optimization_problem)
        self.rate_of_change = self.config["goal_type"] in ["range_rate_of_change"]

    def get_differences(self, timeseries):
        """Get rate of change timeseries for input timeseries, relative to the function nominal."""
        timeseries = list(timeseries)
        return [
            (st - st_prev) / dt / self.function_nominal * 100
            for st, st_prev, dt in zip(timeseries, [np.nan] + timeseries[:-1], self.time_deltas)
        ]

    def plot_timeseries(self, label, timeseries_data, **plot_kwargs):
        """Actually plot a timeseries.

        If subplot is of rate_of_change type, the difference series will be plotted."""
        if self.rate_of_change:
            label = "Rate of Change of " + label
            series_to_plot = self.get_differences(timeseries_data)
        else:
            series_to_plot = timeseries_data
        self.axis.plot(self.datetimes, series_to_plot, label=label, **plot_kwargs)

    def plot_with_previous(self, state_name):
        """Add line with the results for a particular state. If previous results
        are available, a line with the timeseries for those results is also plotted."""
        label = state_name

        timeseries_data = self.results[state_name]
        self.plot_timeseries(label, timeseries_data)

        if self.results_prev:
            timeseries_data = self.results_prev["extract_result"][state_name]
            label += " (at previous priority optimization)"
            self.plot_timeseries(
                label,
                timeseries_data,
                color="gray",
                linestyle="dotted",
            )

    def plot_additional_variables(self):
        """Plot the additional variables defined in the plot_table"""
        for var in self.config.get("variables_style_1", []):
            self.plot_timeseries(var, self.results[var])
        for var in self.config.get("variables_style_2", []):
            self.plot_timeseries(var, self.results[var], linestyle="solid", linewidth="0.5")
        for var in self.config.get("variables_with_previous_result", []):
            self.plot_with_previous(var)

    def format_subplot(self):
        """Format the current axis and set legend and title."""
        self.axis.set_ylabel(self.config["y_axis_title"])
        self.axis.legend()
        if "custom_title" in self.config and isinstance(self.config["custom_title"], str):
            self.axis.set_title(self.config["custom_title"])
        elif self.config["specified_in"] == "goal_generator":
            self.axis.set_title(
                "Goal for {} (active from priority {})".format(self.config["state"], self.config["priority"])
            )

        date_format = mdates.DateFormatter("%d%b%H")
        self.axis.xaxis.set_major_formatter(date_format)
        if self.rate_of_change:
            self.axis.yaxis.set_major_formatter(mtick.PercentFormatter())
        self.axis.grid(which="both", axis="x")

    def add_ranges(self, optimization_problem):
        """Add lines for the lower and upper target."""
        t = optimization_problem.times()
        if self.config["target_data_type"] == "parameter":
            try:
                target_min = np.full_like(t, 1) * optimization_problem.parameters(0)[self.config["target_min"]]
                target_max = np.full_like(t, 1) * optimization_problem.parameters(0)[self.config["target_max"]]
            except TypeError:
                target_min = np.full_like(t, 1) * optimization_problem.io.get_parameter(self.config["target_min"])
                target_max = np.full_like(t, 1) * optimization_problem.io.get_parameter(self.config["target_max"])
        elif self.config["target_data_type"] == "value":
            target_min = np.full_like(t, 1) * float(self.config["target_min"])
            target_max = np.full_like(t, 1) * float(self.config["target_max"])
        elif self.config["target_data_type"] == "timeseries":
            if isinstance(self.config["target_min"], str):
                target_min = optimization_problem.get_timeseries(self.config["target_min"]).values
            else:
                target_min = np.full_like(t, 1) * self.config["target_min"]
            if isinstance(self.config["target_max"], str):
                target_max = optimization_problem.get_timeseries(self.config["target_max"]).values
            else:
                target_max = np.full_like(t, 1) * self.config["target_max"]
        else:
            message = "Target type {} not known.".format(self.config["target_data_type"])
            logger.error(message)
            raise ValueError(message)

        if np.array_equal(target_min, target_max, equal_nan=True):
            self.axis.plot(self.datetimes, target_min, "r--", label="Target")
        else:
            self.axis.plot(self.datetimes, target_min, "r--", label="Target min")
            self.axis.plot(self.datetimes, target_max, "r--", label="Target max")


class PlotGoalsMixin:
    """
    Class for plotting results.
    """

    plot_max_rows = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            plot_table_file = self.plot_table_file
        except AttributeError:
            plot_table_file = os.path.join(self._input_folder, "plot_table.csv")
        self.plot_table = read_plot_table(plot_table_file, self.goal_table_file)

        # Store list of variable-names that may not be present in the results.
        variables_style_1 = [var for var_list in self.plot_table.get("variables_style_1", []) for var in var_list]
        variables_style_2 = [var for var_list in self.plot_table.get("variables_style_2", []) for var in var_list]
        variables_with_previous_result = [
            var for var_list in self.plot_table.get("variables_with_previous_result", []) for var in var_list
        ]
        self.custom_variables = variables_style_1 + variables_style_2 + variables_with_previous_result

    def get_goal(self, subplot_config):
        """Find the goal belonging to a subplot"""
        all_goals = self.goals() + self.path_goals()
        for goal in all_goals:
            if goal.goal_id == subplot_config["id"]:
                return goal
        return None

    def pre(self):
        """Tasks before optimizing."""
        super().pre()
        self.intermediate_results = []

    def plot_goal_results_from_dict(self, result_dict, results_dict_prev=None):
        """Plot results, given a dict."""
        self.plot_goals_results(result_dict, results_dict_prev)

    def plot_goal_results_from_self(self, priority=None):
        """Plot results."""
        result_dict = {
            "extract_result": self.extract_results(),
            "priority": priority,
        }
        self.plot_goals_results(result_dict)

    def plot_goals_results(self, result_dict, results_prev=None):
        # pylint: disable=too-many-locals
        """Creates a figure with a subplot for each row in the plot_table."""
        results = result_dict["extract_result"]
        plot_config = self.plot_table.to_dict("records")

        if len(plot_config) == 0:
            logger.info(
                "PlotGoalsMixin did not find anything to plot."
                + " Are there any goals that are active and described in the plot_table?"
            )
            return

        # Initalize figure
        n_cols = math.ceil(len(plot_config) / self.plot_max_rows)
        n_rows = math.ceil(len(plot_config) / n_cols)
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 9, n_rows * 3), dpi=80, squeeze=False)
        fig.suptitle("Results after optimizing until priority {}".format(result_dict["priority"]), fontsize=14)
        i_plot = -1

        # Add subplot for each row in the plot_table
        for subplot_config in plot_config:
            i_plot += 1
            axis = get_subplot(i_plot, n_rows, axs)
            goal = self.get_goal(subplot_config)
            subplot = Subplot(self, axis, goal, subplot_config, results, results_prev)
            if subplot.config["specified_in"] == "goal_generator":
                subplot.plot_with_previous(subplot.config["state"])
            subplot.plot_additional_variables()
            subplot.format_subplot()
            if subplot.config["goal_type"] in ["range", "range_rate_of_change"]:
                subplot.add_ranges(self)

        # Save figure
        for i in range(0, n_cols):
            axs[n_rows - 1, i].set_xlabel("Time")
        os.makedirs("goal_figures", exist_ok=True)
        fig.tight_layout()
        new_output_folder = os.path.join(self._output_folder, "goal_figures")
        os.makedirs(os.path.join(self._output_folder, "goal_figures"), exist_ok=True)
        fig.savefig(os.path.join(new_output_folder, "after_priority_{}.png".format(result_dict["priority"])))

    def priority_completed(self, priority: int) -> None:
        """Store results required for plotting"""
        extracted_results = copy.deepcopy(self.extract_results())
        results_custom_variables = {
            custom_variable: self.get_timeseries(custom_variable)
            for custom_variable in self.custom_variables
            if custom_variable not in extracted_results
        }
        extracted_results.update(results_custom_variables)
        to_store = {"extract_result": extracted_results, "priority": priority}
        self.intermediate_results.append(to_store)
        super().priority_completed(priority)

    def post(self):
        """Tasks after optimizing. Creates a plot for for each priority."""
        super().post()
        for intermediate_result_prev, intermediate_result in zip(
            [None] + self.intermediate_results[:-1], self.intermediate_results
        ):
            self.plot_goal_results_from_dict(intermediate_result, intermediate_result_prev)
