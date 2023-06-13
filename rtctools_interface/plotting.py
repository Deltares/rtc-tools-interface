import logging
import math
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import numpy as np

logger = logging.getLogger("rtctools")


class PlotGoalsMixin:

    def pre(self):
        super().pre()
        self.intermediate_results = []

    def plot_goal_results_from_dict(self, result_dict, results_dict_prev=None):
        self.plot_goals_results(result_dict, results_dict_prev)

    def plot_goal_results_from_self(self, priority=None):
        result_dict = {
            "timeseries_import_times": self.timeseries_import.times,
            "extract_result": self.extract_results(),
            "min_q_goals": self.min_q_goals,
            "range_goals": self.range_goals,
            "priority": priority,
        }
        self.plot_goals_results(result_dict)

    def plot_goals_results(self, result_dict, results_dict_prev=None):
        timeseries_import_times = result_dict["timeseries_import_times"]
        extract_result = result_dict["extract_result"]
        range_goals = result_dict["range_goals"]
        min_q_goals = result_dict["min_q_goals"]
        priority = result_dict["priority"]

        t = self.times()
        t_datetime = np.array(timeseries_import_times)
        results = extract_result

        # Prepare the plot
        n_plots = len(range_goals + min_q_goals)
        n_cols = math.ceil(n_plots / self.plot_max_rows)
        n_rows = math.ceil(n_plots / n_cols)
        fig, axs = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(n_cols * 9, n_rows * 3), dpi=80, squeeze=False
        )
        fig.suptitle("Results after optimizing until priority {}".format(
            priority), fontsize=14)
        i_plot = -1

        # Function to apply the general settings used by all goal types
        def apply_general_settings():
            """Add line with the results for a particular goal. If previous results
            are available, a line with the timeseries for those results is also plotted.

            Note that this function does also determine the current row and column index
            """
            i_c = math.ceil((i_plot + 1) / n_rows) - 1
            i_r = i_plot - i_c * n_rows

            goal_variable = g[0]
            axs[i_r, i_c].plot(
                t_datetime, results[goal_variable], label=goal_variable)

            if results_dict_prev:
                results_prev = results_dict_prev["extract_result"]
                axs[i_r, i_c].plot(
                    t_datetime,
                    results_prev[goal_variable],
                    label=goal_variable + " at previous priority optimization",
                    color="gray",
                    linestyle="dotted",
                )

            # prio = result_dict["priority"]

            # def add_variable_effects(constraints):
            #     if goal_variable in constraints:
            #         for xr in constraints[goal_variable]["timesteps"]:
            #             if constraints[goal_variable]["effect_direction"] == "+":
            #                 modification = "Increase"
            #                 marker_type = matplotlib.markers.CARETUPBASE
            #                 marker_color = "g"

            #             else:
            #                 modification = "Decrease"
            #                 marker_type = matplotlib.markers.CARETDOWNBASE
            #                 marker_color = "r"

            #             label = "{} {} to improve {}".format(modification, goal_variable, prio)
            #             if label in axs[i_r, i_c].get_legend_handles_labels()[1]:
            #                 label = "_nolegend_"
            #             axs[i_r, i_c].plot(
            #                 t_datetime[int(xr)],
            #                 results[goal_variable][int(xr)],
            #                 marker=marker_type,
            #                 color=marker_color,
            #                 label=label,
            #                 markersize=5,
            #                 alpha=0.6,
            #             )

            # upper_constraints = {
            #     name.replace(".", "_"): value
            #     for name, value in result_dict["upper_constraint_dict"].items()
            # }
            # lower_constraints = {
            #     name.replace(".", "_"): value
            #     for name, value in result_dict["lower_constraint_dict"].items()
            # }
            # add_variable_effects(upper_constraints)
            # add_variable_effects(lower_constraints)

            return i_c, i_r

        def apply_additional_settings(goal_settings):
            """ Sets some additional settings, like additional variables to plot.
            The second list of variables has a specific style, the first not.
            """
            add_settings = goal_settings[-1]

            for var in add_settings[1]:
                axs[i_row, i_col].plot(t_datetime, results[var], label=var)
            for var in add_settings[2]:
                axs[i_row, i_col].plot(
                    t_datetime, results[var], linestyle="solid", linewidth="0.5", label=var
                )
            axs[i_row, i_col].set_ylabel(add_settings[0])
            axs[i_row, i_col].legend()
            axs[i_row, i_col].set_title(
                "Goal for {} (active from priority {})".format(
                    goal_settings[0], goal_settings[4])
            )
            dateFormat = mdates.DateFormatter("%d%b%H")
            axs[i_row, i_col].xaxis.set_major_formatter(dateFormat)
            axs[i_row, i_col].grid(which="both", axis="x")

        # Add plots needed for range goals
        for g in sorted(self.range_goals, key=lambda goal: goal[4]):
            i_plot += 1

            i_col, i_row = apply_general_settings()

            if g[1] == "parameter":
                target_min = np.full_like(t, 1) * self.parameters(0)[g[2]]
                target_max = np.full_like(t, 1) * self.parameters(0)[g[3]]
            elif g[1] == "timeseries":
                target_min = self.get_timeseries(g[2]).values
                target_max = self.get_timeseries(g[3]).values
            else:
                logger.error("Target type {} not known.".format(g[1]))
                raise

            if np.array_equal(target_min, target_max, equal_nan=True):
                axs[i_row, i_col].plot(
                    t_datetime, target_min, "r--", label="Target")
            else:
                axs[i_row, i_col].plot(
                    t_datetime, target_min, "r--", label="Target min")
                axs[i_row, i_col].plot(
                    t_datetime, target_max, "r--", label="Target max")

            apply_additional_settings(g)

        # Add plots needed for minimization of discharge
        for g in min_q_goals:
            i_plot += 1

            i_col, i_row = apply_general_settings()

            apply_additional_settings(g)

        # TODO: this should be expanded when there are more columns
        for i in range(0, n_cols):
            axs[n_rows - 1, i].set_xlabel("Time")
        os.makedirs("goal_figures", exist_ok=True)
        fig.tight_layout()
        fig.savefig("goal_figures/after_priority_{}.png".format(priority))
        # plt.show()

    def priority_completed(self, priority: int) -> None:
        # Store results required for plotting
        to_store = {
            "extract_result": self.extract_results(),
            "range_goals": self.range_goals,
            "min_q_goals": self.min_q_goals,
            "timeseries_import_times": self.timeseries_import.times,
            "priority": priority
        }
        self.intermediate_results.append(to_store)
        super().priority_completed(priority)

    def post(self):
        super().post()
        for intermediate_result_prev, intermediate_result in zip(
            [None] + self.intermediate_results[:-1], self.intermediate_results
        ):
            self.plot_goal_results_from_dict(
                intermediate_result, intermediate_result_prev)
