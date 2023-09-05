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

def get_timedeltas(optimization_problem):
    """Get delta_t for each timestep."""
    return [np.nan] + [
        optimization_problem.times()[i] - optimization_problem.times()[i - 1]
        for i in range(1, len(optimization_problem.times()))
    ]

def get_goal_nominal(subplot_config, all_goals):
    """Find the goal belonging to a subplot and return its function nominal."""
    for goal in all_goals:
        if goal.goal_id == subplot_config["id"]:
            return goal.function_nominal
    return 1


def get_subplot(i_plot, n_rows, axs):
    """Determine the row and column index and returns the corresponding subplot object."""
    i_c = math.ceil((i_plot + 1) / n_rows) - 1
    i_r = i_plot - i_c * n_rows
    subplot = axs[i_r, i_c]
    return subplot


def get_differences(timeseries, time_deltas, nominal=1):
    """Get rate of change timeseries for input timeseries."""
    timeseries = list(timeseries)
    return [
        (st - st_prev) / dt / nominal * 100
        for st, st_prev, dt in zip(timeseries + [np.nan], [np.nan] + timeseries[:-1], time_deltas)
    ]


def plot_data(subplot, label, data, t_datetime, time_deltas, rate_of_change=False, nominal=1, **plot_kwargs):
    """Actually plot a timeseries. If rate_of_change is true, the difference series will be plotted."""
    if rate_of_change:
        to_plot = get_differences(data, time_deltas, nominal)
    else:
        to_plot = data

    subplot.plot(t_datetime, to_plot, label=label, **plot_kwargs)


def plot_with_previous(
    subplot, state_name, t_datetime, results, results_dict_prev, time_deltas, rate_of_change=False, nominal=1
):
    """Add line with the results for a particular state. If previous results
    are available, a line with the timeseries for those results is also plotted."""
    label = state_name
    if rate_of_change:
        label = "Rate of Change of " + label

    plot_data(subplot, label, results[state_name], t_datetime, time_deltas, rate_of_change, nominal)

    if results_dict_prev:
        prev_data = results_dict_prev["extract_result"][state_name]
        label += " (at previous priority optimization)"
        plot_data(
            subplot,
            label,
            prev_data,
            t_datetime,
            time_deltas,
            rate_of_change,
            nominal,
            color="gray",
            linestyle="dotted",
        )


def plot_additional_variables(subplot, t_datetime, results, results_dict_prev, subplot_config, time_deltas):
    """Plot the additional variables defined in the plot_table"""
    for var in subplot_config.get("variables_style_1", []):
        subplot.plot(t_datetime, results[var], label=var)
    for var in subplot_config.get("variables_style_2", []):
        subplot.plot(t_datetime, results[var], linestyle="solid", linewidth="0.5", label=var)
    for var in subplot_config.get("variables_with_previous_result", []):
        plot_with_previous(subplot, var, t_datetime, results, results_dict_prev, time_deltas, rate_of_change=False)


def format_subplot(subplot, subplot_config):
    """Format the current axis and set legend and title."""
    subplot.set_ylabel(subplot_config["y_axis_title"])
    subplot.legend()
    if "custom_title" in subplot_config and isinstance(subplot_config["custom_title"], str):
        subplot.set_title(subplot_config["custom_title"])
    elif subplot_config["specified_in"] == "goal_generator":
        subplot.set_title(
            "Goal for {} (active from priority {})".format(subplot_config["state"], subplot_config["priority"])
        )

    date_format = mdates.DateFormatter("%d%b%H")
    subplot.xaxis.set_major_formatter(date_format)
    if subplot_config["goal_type"] == "ramping_range":
        subplot.yaxis.set_major_formatter(mtick.PercentFormatter())
    subplot.grid(which="both", axis="x")


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
        time_deltas = get_timedeltas(self)

        # Add subplot for each row in the plot_table
        for subplot_config in plot_config:
            if subplot_config["specified_in"] == "goal_generator":
                nominal = get_goal_nominal(subplot_config, self.goals() + self.path_goals())
            else:
                nominal = 1
            i_plot += 1
            subplot = get_subplot(i_plot, n_rows, axs)
            rate_of_change = subplot_config["goal_type"] in ["ramping_range", "minimization_ramping"]
            if subplot_config["specified_in"] == "goal_generator":
                plot_with_previous(
                    subplot,
                    subplot_config["state"],
                    np.array(self.io.datetimes),
                    results,
                    results_prev,
                    time_deltas,
                    rate_of_change,
                    nominal=nominal,
                )
            plot_additional_variables(
                subplot, np.array(self.io.datetimes), results, results_prev, subplot_config, time_deltas
            )
            format_subplot(subplot, subplot_config)
            if subplot_config["goal_type"] in ["range", "ramping_range"]:
                self.add_ranges(subplot, np.array(self.io.datetimes), subplot_config)

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

    def add_ranges(self, subplot, t_datetime, subplot_config):
        """Add lines for the lower and upper target."""
        t = self.times()
        if subplot_config["target_data_type"] == "parameter":
            try:
                target_min = np.full_like(t, 1) * self.parameters(0)[subplot_config["target_min"]]
                target_max = np.full_like(t, 1) * self.parameters(0)[subplot_config["target_max"]]
            except TypeError:
                target_min = np.full_like(t, 1) * self.io.get_parameter(subplot_config["target_min"])
                target_max = np.full_like(t, 1) * self.io.get_parameter(subplot_config["target_max"])
        elif subplot_config["target_data_type"] == "value":
            target_min = np.full_like(t, 1) * float(subplot_config["target_min"])
            target_max = np.full_like(t, 1) * float(subplot_config["target_max"])
        elif subplot_config["target_data_type"] == "timeseries":
            if isinstance(subplot_config["target_min"], str):
                target_min = self.get_timeseries(subplot_config["target_min"]).values
            else:
                target_min = np.full_like(t, 1) * subplot_config["target_min"]
            if isinstance(subplot_config["target_max"], str):
                target_max = self.get_timeseries(subplot_config["target_max"]).values
            else:
                target_max = np.full_like(t, 1) * subplot_config["target_max"]
        else:
            message = "Target type {} not known.".format(subplot_config["target_data_type"])
            logger.error(message)
            raise ValueError(message)

        if np.array_equal(target_min, target_max, equal_nan=True):
            subplot.plot(t_datetime, target_min, "r--", label="Target")
        else:
            subplot.plot(t_datetime, target_min, "r--", label="Target min")
            subplot.plot(t_datetime, target_max, "r--", label="Target max")
