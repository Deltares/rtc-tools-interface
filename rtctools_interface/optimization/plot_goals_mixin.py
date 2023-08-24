import logging
import math
import os
import copy


import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import numpy as np

from rtctools_interface.optimization.read_plot_table import read_plot_table

logger = logging.getLogger("rtctools")


def next_subplot(i_plot, n_rows):
    """Determine the the next row and column index"""
    i_c = math.ceil((i_plot + 1) / n_rows) - 1
    i_r = i_plot - i_c * n_rows
    i_plot += 1
    return i_c, i_r, i_plot


def plot_with_history(axs, state_name, i_r, i_c, t_datetime, results, results_dict_prev):
    """Add line with the results for a particular state. If previous results
    are available, a line with the timeseries for those results is also plotted.
    """
    axs[i_r, i_c].plot(t_datetime, results[state_name], label=state_name)

    if results_dict_prev:
        results_prev = results_dict_prev["extract_result"]
        axs[i_r, i_c].plot(
            t_datetime,
            results_prev[state_name],
            label=state_name + " at previous priority optimization",
            color="gray",
            linestyle="dotted",
        )


def plot_additional_variables(axs, i_r, i_c, t_datetime, results, results_dict_prev, goal):
    """Plot the additional variables defined in the plot_table"""
    for var in goal["variables_plot_1"]:
        axs[i_r, i_c].plot(t_datetime, results[var], label=var)
    for var in goal["variables_plot_2"]:
        axs[i_r, i_c].plot(t_datetime, results[var], linestyle="solid", linewidth="0.5", label=var)
    for var in goal["variables_plot_history"]:
        plot_with_history(axs, var, i_r, i_c, t_datetime, results, results_dict_prev)


def format_axs(axs, i_r, i_c, goal):
    """Format the current axis and set legend and title."""
    axs[i_r, i_c].set_ylabel(goal["y_axis_title"])
    axs[i_r, i_c].legend()
    if isinstance(goal["custom_title"], str):
        axs[i_r, i_c].set_title(goal["custom_title"])
    else:
        axs[i_r, i_c].set_title("Goal for {} (active from priority {})".format(goal["state"], goal["priority"]))

    dateFormat = mdates.DateFormatter("%d%b%H")
    axs[i_r, i_c].xaxis.set_major_formatter(dateFormat)
    axs[i_r, i_c].grid(which="both", axis="x")


class PlotGoalsMixin:
    plot_max_rows = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            plot_table_file = self.plot_table_file
        except AttributeError:
            plot_table_file = os.path.join(self._input_folder, "plot_table.csv")
        self.plot_table = read_plot_table(plot_table_file, self.goal_table_file)

        # Store list of variable-names that may not be present in the results.
        variables_plot_1 = [var for var_list in self.plot_table["variables_plot_1"] for var in var_list]
        variables_plot_2 = [var for var_list in self.plot_table["variables_plot_2"] for var in var_list]
        variables_plot_history = [var for var_list in self.plot_table["variables_plot_history"] for var in var_list]
        self.custom_variables = variables_plot_1 + variables_plot_2 + variables_plot_history

    def pre(self):
        super().pre()
        self.intermediate_results = []

    def plot_goal_results_from_dict(self, result_dict, results_dict_prev=None):
        self.plot_goals_results(result_dict, results_dict_prev)

    def plot_goal_results_from_self(self, priority=None):
        result_dict = {
            "extract_result": self.extract_results(),
            "priority": priority,
        }
        self.plot_goals_results(result_dict)

    def plot_goals_results(self, result_dict, results_dict_prev=None):
        """Creates a figure with a subplot for each row in the plot_table."""
        t_datetime = np.array(self.io.datetimes)
        results = result_dict["extract_result"]
        priority = result_dict["priority"]
        all_goals = self.plot_table.to_dict("records")

        n_plots = len(all_goals)
        if n_plots == 0:
            logger.info(
                "PlotGoalsMixin did not find anything to plot."
                + " Are there any goals that are active and described in the plot_table?"
            )
            return

        # Initalize figure
        n_cols = math.ceil(n_plots / self.plot_max_rows)
        n_rows = math.ceil(n_plots / n_cols)
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 9, n_rows * 3), dpi=80, squeeze=False)
        fig.suptitle("Results after optimizing until priority {}".format(priority), fontsize=14)
        i_plot = 0

        # Add subplot for each goal
        for goal in all_goals:
            i_c, i_r, i_plot = next_subplot(i_plot, n_rows)
            if goal["specified_in"] == "goal_generator":
                plot_with_history(axs, goal["state"], i_r, i_c, t_datetime, results, results_dict_prev)
            plot_additional_variables(axs, i_r, i_c, t_datetime, results, results_dict_prev, goal)
            format_axs(axs, i_r, i_c, goal)
            if goal["goal_type"] in ["range"]:
                self.add_ranges(axs, i_r, i_c, t_datetime, goal)

        # Save figure
        for i in range(0, n_cols):
            axs[n_rows - 1, i].set_xlabel("Time")
        os.makedirs("goal_figures", exist_ok=True)
        fig.tight_layout()
        new_output_folder = os.path.join(self._output_folder, "goal_figures")
        os.makedirs(new_output_folder, exist_ok=True)
        fig.savefig(os.path.join(new_output_folder, "after_priority_{}.png".format(priority)))

    def priority_completed(self, priority: int) -> None:
        """Store results required for plotting"""
        extracted_results = copy.deepcopy(self.extract_results())
        # TODO Make more robust on non-existant variables/timeseries (get_timeseries part)
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
        super().post()
        for intermediate_result_prev, intermediate_result in zip(
            [None] + self.intermediate_results[:-1], self.intermediate_results
        ):
            self.plot_goal_results_from_dict(intermediate_result, intermediate_result_prev)

    def add_ranges(self, axs, i_r, i_c, t_datetime, goal):
        t = self.times()
        if goal["target_data_type"] == "parameter":
            try:
                target_min = np.full_like(t, 1) * self.parameters(0)[goal["target_min"]]
                target_max = np.full_like(t, 1) * self.parameters(0)[goal["target_max"]]
            except TypeError:
                target_min = np.full_like(t, 1) * self.io.get_parameter(goal["target_min"])
                target_max = np.full_like(t, 1) * self.io.get_parameter(goal["target_max"])
        elif goal["target_data_type"] == "value":
            target_min = np.full_like(t, 1) * float(goal["target_min"])
            target_max = np.full_like(t, 1) * float(goal["target_max"])
        elif goal["target_data_type"] == "timeseries":
            if isinstance(goal["target_min"], str):
                target_min = self.get_timeseries(goal["target_min"]).values
            else:
                target_min = np.full_like(t, 1) * goal["target_min"]
            if isinstance(goal["target_max"], str):
                target_max = self.get_timeseries(goal["target_max"]).values
            else:
                target_max = np.full_like(t, 1) * goal["target_max"]
        else:
            message = "Target type {} not known.".format(goal["target_data_type"])
            logger.error(message)
            raise ValueError(message)

        if np.array_equal(target_min, target_max, equal_nan=True):
            axs[i_r, i_c].plot(t_datetime, target_min, "r--", label="Target")
        else:
            axs[i_r, i_c].plot(t_datetime, target_min, "r--", label="Target min")
            axs[i_r, i_c].plot(t_datetime, target_max, "r--", label="Target max")
