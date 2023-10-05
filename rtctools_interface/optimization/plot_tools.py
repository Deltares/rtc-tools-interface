"""Functions to create plots."""
from io import StringIO
import logging
import math
import os
from typing import Dict, Union
import matplotlib

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import numpy as np
from rtctools_interface.optimization.base_goal import BaseGoal
from rtctools_interface.optimization.plot_and_goal_schema import (
    MinimizationGoalCombinedModel,
    MaximizationGoalCombinedModel,
    RangeGoalCombinedModel,
    RangeRateOfChangeGoalCombinedModel,
)
from rtctools_interface.optimization.plot_table_schema import PlotTableRow
from rtctools_interface.optimization.type_definitions import PlotDataAndConfig, PrioIndependentData


logger = logging.getLogger("rtctools")


def get_row_col_number(i_plot, n_rows):
    i_c = math.ceil((i_plot + 1) / n_rows) - 1
    i_r = i_plot - i_c * n_rows
    return i_c, i_r


def get_subplot_axis(i_plot, n_rows, axs):
    """Determine the row and column index and returns the corresponding subplot object."""
    i_c, i_r = get_row_col_number(i_plot, n_rows)
    subplot = axs[i_r, i_c]
    return subplot


def get_timedeltas(times):
    """Get delta_t for each timestep."""
    return [np.nan] + [times[i] - times[i - 1] for i in range(1, len(times))]


class Subplot:
    """Wrapper class for a subplot in the figure".

    Contains the axis object and all configuration settings and data
    that belongs to the subplot."""

    def __init__(
        self,
        axis,
        subplot_config,
        goal,
        results,
        results_prev,
        prio_independent_data: PrioIndependentData,
    ):
        self.axis = axis
        self.config: Union[
            MinimizationGoalCombinedModel,
            MaximizationGoalCombinedModel,
            RangeGoalCombinedModel,
            RangeRateOfChangeGoalCombinedModel,
            PlotTableRow,
        ] = subplot_config
        self.goal: BaseGoal = goal
        self.function_nominal = self.goal.function_nominal if self.goal else 1
        self.results = results
        self.results_prev = results_prev
        self.datetimes = prio_independent_data["io_datetimes"]
        self.time_deltas = get_timedeltas(prio_independent_data["times"])
        self.rate_of_change = (
            self.config.goal_type in ["range_rate_of_change"] if self.config.specified_in == "goal_generator" else 0
        )

        if self.config.specified_in == "goal_generator" and self.config.goal_type in ["range", "range_rate_of_change"]:
            targets = prio_independent_data["target_series"][self.config.goal_id]
            self.target_min, self.target_max = targets["target_min"], targets["target_max"]
        else:
            self.target_min, self.target_max = None, None

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
        for var in self.config.variables_style_1:
            self.plot_timeseries(var, self.results[var])
        for var in self.config.variables_style_2:
            self.plot_timeseries(var, self.results[var], linestyle="solid", linewidth="0.5")
        for var in self.config.variables_with_previous_result:
            self.plot_with_previous(var)

    def format_subplot(self):
        """Format the current axis and set legend and title."""
        self.axis.set_ylabel(self.config.y_axis_title)
        self.axis.legend()
        if "custom_title" in self.config.__dict__ and isinstance(self.config.custom_title, str):
            self.axis.set_title(self.config.custom_title)
        elif self.config.specified_in == "goal_generator":
            self.axis.set_title("Goal for {} (active from priority {})".format(self.config.state, self.config.priority))

        date_format = mdates.DateFormatter("%d%b%H")
        self.axis.xaxis.set_major_formatter(date_format)
        if self.rate_of_change:
            self.axis.yaxis.set_major_formatter(mtick.PercentFormatter())
        self.axis.grid(which="both", axis="x")

    def add_ranges(self):
        """Add lines for the lower and upper target."""
        if np.array_equal(self.target_min, self.target_max, equal_nan=True):
            self.axis.plot(self.datetimes, self.target_min, "r--", label="Target")
        else:
            self.axis.plot(self.datetimes, self.target_min, "r--", label="Target min")
            self.axis.plot(self.datetimes, self.target_max, "r--", label="Target max")

    def plot(self):
        """Plot the data in the subplot and format."""
        if self.config.specified_in == "goal_generator":
            self.plot_with_previous(self.config.state)
        self.plot_additional_variables()
        self.format_subplot()
        if self.config.specified_in == "goal_generator" and self.config.goal_type in [
            "range",
            "range_rate_of_change",
        ]:
            self.add_ranges()


def save_fig_as_png(fig, output_folder, priority) -> matplotlib.figure.Figure:
    """Save matplotlib figure to output folder."""
    os.makedirs("goal_figures", exist_ok=True)
    new_output_folder = os.path.join(output_folder, "goal_figures")
    os.makedirs(os.path.join(output_folder, "goal_figures"), exist_ok=True)
    fig.savefig(os.path.join(new_output_folder, "after_priority_{}.png".format(priority)))
    return fig


def get_goal(subplot_config, all_goals) -> Union[BaseGoal, None]:
    """Find the goal belonging to a subplot"""
    for goal in all_goals:
        if goal.goal_id == subplot_config.id:
            return goal
    return None


def save_fig_as_stringio(fig):
    """Save figure as stringio in self."""
    svg_data = StringIO()
    fig.savefig(svg_data, format="svg")
    return svg_data


def save_figure(fig, save_plot_to, output_folder, priority) -> Union[StringIO, matplotlib.figure.Figure]:
    """Save figure."""
    if save_plot_to == "image":
        return save_fig_as_png(fig, output_folder, priority)
    if save_plot_to == "stringio":
        return save_fig_as_stringio(fig)
    raise ValueError("Unsupported method of saving the plot results.")


def create_priority_plot(
    result_dict, results_prev, plot_data_and_config: PlotDataAndConfig
) -> Union[StringIO, matplotlib.figure.Figure]:
    # pylint: disable=too-many-locals
    """Creates a figure with a subplot for each row in the plot_table."""
    results = result_dict["extract_result"]
    plot_config = plot_data_and_config["plot_options"]["plot_config"]
    plot_max_rows = plot_data_and_config["plot_options"]["plot_max_rows"]
    if len(plot_config) == 0:
        logger.info(
            "PlotGoalsMixin did not find anything to plot."
            + " Are there any goals that are active and described in the plot_table?"
        )
        return None

    # Initalize figure
    n_cols = math.ceil(len(plot_config) / plot_max_rows)
    n_rows = math.ceil(len(plot_config) / n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 9, n_rows * 3), dpi=80, squeeze=False)
    fig.suptitle("Results after optimizing until priority {}".format(result_dict["priority"]), fontsize=14)
    i_plot = -1

    all_goals = plot_data_and_config["prio_independent_data"]["all_goals"]
    # Add subplot for each row in the plot_table
    for subplot_config in plot_config:
        i_plot += 1
        axis = get_subplot_axis(i_plot, n_rows, axs)
        goal = get_goal(subplot_config, all_goals)
        subplot = Subplot(
            axis, subplot_config, goal, results, results_prev, plot_data_and_config["prio_independent_data"]
        )
        subplot.plot()

    for i in range(0, n_cols):
        axs[n_rows - 1, i].set_xlabel("Time")
    fig.tight_layout()
    return save_figure(
        fig,
        plot_data_and_config["plot_options"]["save_plot_to"],
        plot_data_and_config["plot_options"]["output_folder"],
        result_dict["priority"],
    )


def create_plot_each_priority(
    plot_data_and_config: PlotDataAndConfig,
) -> Dict[str, Union[StringIO, matplotlib.figure.Figure]]:
    """Create all plots for one optimization run, for each priority one seperate plot."""
    intermediate_results = plot_data_and_config["intermediate_results"]
    plot_results = {}
    for intermediate_result_prev, intermediate_result in zip([None] + intermediate_results[:-1], intermediate_results):
        priority = intermediate_result["priority"]
        plot_results[priority] = create_priority_plot(
            intermediate_result, intermediate_result_prev, plot_data_and_config
        )
    return plot_results
