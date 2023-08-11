import logging
import math
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import numpy as np

from rtctools_interface.optimization.read_plot_table import read_plot_table

logger = logging.getLogger("rtctools")


class PlotGoalsMixin:
    plot_max_rows = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            plot_table_file = self.plot_table_file
        except AttributeError:
            plot_table_file = os.path.join(self._input_folder, "plot_table.csv")
        self.plot_table = read_plot_table(plot_table_file, self.goal_table_file)

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
        timeseries_import_times = self.io.datetimes
        extract_result = result_dict["extract_result"]
        all_goals = self.plot_table.to_dict("records")
        range_goals = [goal for goal in all_goals if goal["goal_type"] == "range"]
        min_q_goals = [goal for goal in all_goals if goal["goal_type"] == "minimization"]
        max_q_goals = [goal for goal in all_goals if goal["goal_type"] == "maximization"]
        min_sum_goals = [goal for goal in all_goals if goal["goal_type"] == "minimization_sum"]
        max_sum_goals = [goal for goal in all_goals if goal["goal_type"] == "maximization_sum"]
        priority = result_dict["priority"]

        t = self.times()
        t_datetime = np.array(timeseries_import_times)
        results = extract_result

        # Prepare the plot
        n_plots = len(range_goals + min_q_goals + max_q_goals + min_sum_goals + max_sum_goals)
        n_cols = math.ceil(n_plots / self.plot_max_rows)
        n_rows = math.ceil(n_plots / n_cols)
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 9, n_rows * 3), dpi=80, squeeze=False)
        fig.suptitle("Results after optimizing until priority {}".format(priority), fontsize=14)
        i_plot = -1

        # Function to apply the general settings used by all goal types
        def apply_general_settings():
            """Add line with the results for a particular goal. If previous results
            are available, a line with the timeseries for those results is also plotted.

            Note that this function does also determine the current row and column index
            """
            i_c = math.ceil((i_plot + 1) / n_rows) - 1
            i_r = i_plot - i_c * n_rows

            goal_variable = g["state"]
            axs[i_r, i_c].plot(t_datetime, results[goal_variable], label=goal_variable)

            if results_dict_prev:
                results_prev = results_dict_prev["extract_result"]
                axs[i_r, i_c].plot(
                    t_datetime,
                    results_prev[goal_variable],
                    label=goal_variable + " at previous priority optimization",
                    color="gray",
                    linestyle="dotted",
                )
            return i_c, i_r

        def apply_additional_settings(goal_settings):
            """Sets some additional settings, like additional variables to plot.
            The second list of variables has a specific style, the first not.
            """

            for var in goal_settings["variables_plot_1"]:
                if var in results:
                    axs[i_row, i_col].plot(t_datetime, results[var], label=var)
                else:
                    axs[i_row, i_col].plot(t_datetime, self.get_timeseries(var), label=var)
            for var in goal_settings["variables_plot_2"]:
                if var in results:
                    axs[i_row, i_col].plot(t_datetime,
                                           results[var],
                                           linestyle="solid",
                                           linewidth="0.5",
                                           label=var)
                else:
                    axs[i_row, i_col].plot(t_datetime,
                                           self.get_timeseries(var),
                                           linestyle="solid",
                                           linewidth="0.5",
                                           label=var)
            axs[i_row, i_col].set_ylabel(goal_settings["y_axis_title"])
            axs[i_row, i_col].legend()
            axs[i_row, i_col].set_title(
                "Goal for {} (active from priority {})".format(goal_settings["state"], goal_settings["priority"])
            )
            dateFormat = mdates.DateFormatter("%d%b%H")
            axs[i_row, i_col].xaxis.set_major_formatter(dateFormat)
            axs[i_row, i_col].grid(which="both", axis="x")

        # Add plots needed for range goals
        for g in sorted(range_goals, key=lambda goal: goal["priority"]):
            i_plot += 1

            i_col, i_row = apply_general_settings()

            if g["target_data_type"] == "parameter":
                try:
                    target_min = np.full_like(t, 1) * self.parameters(0)[g["target_min"]]
                    target_max = np.full_like(t, 1) * self.parameters(0)[g["target_max"]]
                except TypeError:
                    target_min = np.full_like(t, 1) * self.io.get_parameter(g["target_min"])
                    target_max = np.full_like(t, 1) * self.io.get_parameter(g["target_max"])
            elif g["target_data_type"] == "value":
                target_min = np.full_like(t, 1) * g["target_min"]
                target_max = np.full_like(t, 1) * g["target_max"]
            elif g["target_data_type"] == "timeseries":
                if isinstance(g["target_min"], str):
                    target_min = self.get_timeseries(g["target_min"]).values
                else:
                    target_min = np.full_like(t, 1) * g["target_min"]
                if isinstance(g["target_max"], str):
                    target_max = self.get_timeseries(g["target_max"]).values
                else:
                    target_max = np.full_like(t, 1) * g["target_max"]
            else:
                message = "Target type {} not known.".format(g["target_data_type"])
                logger.error(message)
                raise ValueError(message)

            if np.array_equal(target_min, target_max, equal_nan=True):
                axs[i_row, i_col].plot(t_datetime, target_min, "r--", label="Target")
            else:
                axs[i_row, i_col].plot(t_datetime, target_min, "r--", label="Target min")
                axs[i_row, i_col].plot(t_datetime, target_max, "r--", label="Target max")

            apply_additional_settings(g)

        # Add plots needed for minimization of discharge
        for g in min_q_goals:
            i_plot += 1
            i_col, i_row = apply_general_settings()
            apply_additional_settings(g)

        for g in max_q_goals:
            i_plot += 1
            i_col, i_row = apply_general_settings()
            apply_additional_settings(g)
        for g in min_sum_goals:
            i_plot += 1
            i_col, i_row = apply_general_settings()
            apply_additional_settings(g)

        for g in max_sum_goals:
            i_plot += 1
            i_col, i_row = apply_general_settings()
            apply_additional_settings(g)

        # TODO: this should be expanded when there are more columns
        for i in range(0, n_cols):
            axs[n_rows - 1, i].set_xlabel("Time")
        os.makedirs("goal_figures", exist_ok=True)
        fig.tight_layout()
        new_output_folder = os.path.join(self._output_folder, "goal_figures")
        os.makedirs(new_output_folder, exist_ok=True)
        fig.savefig(os.path.join(new_output_folder, "after_priority_{}.png".format(priority)))

    def priority_completed(self, priority: int) -> None:
        # Store results required for plotting
        to_store = {"extract_result": self.extract_results(), "priority": priority}
        self.intermediate_results.append(to_store)
        super().priority_completed(priority)

    def post(self):
        super().post()
        for intermediate_result_prev, intermediate_result in zip(
            [None] + self.intermediate_results[:-1], self.intermediate_results
        ):
            self.plot_goal_results_from_dict(intermediate_result, intermediate_result_prev)
