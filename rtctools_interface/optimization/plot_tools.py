"""Functions to create plots."""
from abc import abstractmethod, ABC
from io import StringIO
import logging
import math
import os
from typing import Dict, Union
import matplotlib

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    """Get row and col number given a plot number."""
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


class SubplotBase(ABC):
    """Base class for creating subplots."""

    def __init__(
        self,
        subplot_config,
        goal,
        results,
        results_prev,
        prio_independent_data: PrioIndependentData,
    ):
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

        if "custom_title" in self.config.__dict__ and isinstance(self.config.custom_title, str):
            self.subplot_title = self.config.custom_title
        elif self.config.specified_in == "goal_generator":
            self.subplot_title = "Goal for {} (active from priority {})".format(self.config.state, self.config.priority)

    def get_differences(self, timeseries):
        """Get rate of change timeseries for input timeseries, relative to the function nominal."""
        timeseries = list(timeseries)
        return [
            (st - st_prev) / dt / self.function_nominal * 100
            for st, st_prev, dt in zip(timeseries, [np.nan] + timeseries[:-1], self.time_deltas)
        ]

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

    def add_ranges(self):
        """Add lines for the lower and upper target."""
        if np.array_equal(self.target_min, self.target_max, equal_nan=True):
            self.plot_dashed_line(self.datetimes, self.target_min, "Target", "r")
        else:
            self.plot_dashed_line(self.datetimes, self.target_min, "Target min", "r")
            self.plot_dashed_line(self.datetimes, self.target_max, "Target max", "r")

    def plot_timeseries(self, label, timeseries_data, color=None, linewidth=None, linestyle=None):
        """Plot a timeseries with the given style.
        If subplot is of rate_of_change type, the difference series will be plotted."""
        if self.rate_of_change:
            label = "Rate of Change of " + label
            series_to_plot = self.get_differences(timeseries_data)
        else:
            series_to_plot = timeseries_data

        self.plot_line(self.datetimes, series_to_plot, label, color, linewidth, linestyle)

    @abstractmethod
    def plot_line(self, xarray, yarray, label, color=None, linewidth=None, linestyle=None):
        """Given the input and output array, add a line plot to the subplot."""

    @abstractmethod
    def plot_dashed_line(self, xarray, yarray, label, color):
        """Given the input and output array, add dashed line plot to the subplot."""

    @abstractmethod
    def format_subplot(self):
        """Format the current subplot."""


class SubplotMatplotlib(SubplotBase):
    """Class for creating subplots using matplotlib. Expects an axis object
    which refers to that subplot."""

    def __init__(
        self,
        axis,
        subplot_config,
        goal,
        results,
        results_prev,
        prio_independent_data: PrioIndependentData,
    ):
        super().__init__(subplot_config, goal, results, results_prev, prio_independent_data)
        self.axis = axis

    def plot_dashed_line(self, xarray, yarray, label, color="red"):
        """Given the input and output array, add dashed line plot to the subplot."""
        self.axis.plot(xarray, yarray, "--", label=label, color=color)

    def plot_line(self, xarray, yarray, label, color=None, linewidth=None, linestyle=None):
        self.axis.plot(xarray, yarray, label=label, color=color, linewidth=linewidth, linestyle=linestyle)

    def format_subplot(self):
        """Format the current axis and set legend and title."""
        # Format y-axis
        self.axis.set_ylabel(self.config.y_axis_title)
        self.axis.legend()
        # Set title
        self.axis.set_title(self.subplot_title)
        # Format x-axis
        data_format_str = "%d%b%H"
        date_format = mdates.DateFormatter(data_format_str)
        self.axis.xaxis.set_major_formatter(date_format)
        self.axis.set_xlabel("Time")
        # Format y-axis for rate-of-change-goals
        if self.rate_of_change:
            self.axis.yaxis.set_major_formatter(mtick.PercentFormatter())
        # Add grid lines
        self.axis.grid(which="both", axis="x")


class SubplotPlotly(SubplotBase):
    # As this class is still work in progress...
    """Class for creating subplots using plotly. Expects to be part of
    a figure object with subplots."""

    def __init__(
        self,
        subplot_config,
        goal,
        results,
        results_prev,
        prio_independent_data: PrioIndependentData,
        figure=None,
        row_num=0,
        col_num=0,
        i_plot=None,
    ):
        super().__init__(subplot_config, goal, results, results_prev, prio_independent_data)
        self.row_num = row_num
        self.col_num = col_num
        self.use_plotly = True
        self.figure = figure
        self.i_plot = i_plot

    def map_color_code(self, color):
        """Map a color code to a plotly supported color code."""
        color_mapping = {"r": "red"}
        return color_mapping.get(color, color)

    def plot_dashed_line(self, xarray, yarray, label, color="red"):
        """Given the input and output array, add dashed line plot to the subplot."""
        self.figure.add_trace(
            go.Scatter(
                legendgroup=self.i_plot,
                x=xarray,
                y=yarray,
                name=label,
                mode="lines",
                line={"color": self.map_color_code(color), "dash": "dot"},
            ),
            row=self.row_num,
            col=self.col_num,
        )

    def plot_line(self, xarray, yarray, label, color=None, linewidth=None, linestyle=None):
        linewidth = float(linewidth) * 1.3 if linewidth else None
        linestyle = "dot" if linestyle == "dotted" else None
        self.figure.add_trace(
            go.Scatter(
                mode="lines",
                legendgroup=self.i_plot,
                legendgrouptitle_text=self.subplot_title,
                x=xarray,
                y=yarray,
                name=label,
                line={"width": linewidth, "dash": linestyle, "color": color},
            ),
            row=self.row_num,
            col=self.col_num,
        )

    def format_subplot(self):
        """Format the current axis and set legend and title."""
        # Format y-axis
        self.figure.update_yaxes(title_text=self.config.y_axis_title, row=self.row_num, col=self.col_num)
        # Set title
        self.figure.layout.annotations[self.i_plot]["text"] = self.subplot_title
        # Format x-axis
        data_format_str = "%d%b%H"
        self.figure.update_xaxes(tickformat=data_format_str, row=self.row_num, col=self.col_num)
        # Format y-axis for rate-of-change-goals
        if self.rate_of_change:
            self.figure.update_yaxes(tickformat=".1", row=self.row_num, col=self.col_num)
        # Add grid lines
        self.figure.update_xaxes(showgrid=True, row=self.row_num, col=self.col_num, gridwidth=1, gridcolor="gray")
        self.figure.update_xaxes(showticklabels=True, row=self.row_num, col=self.col_num)


def get_file_write_path(output_folder, file_name="figure"):
    """Get path to to file."""
    new_output_folder = os.path.join(output_folder, "goal_figures")
    os.makedirs(new_output_folder, exist_ok=True)
    return os.path.join(new_output_folder, file_name)


def get_file_name(priority: int, final_result: bool):
    """Get the file name for the figure to be written."""
    if final_result:
        file_name = "final_results"
    else:
        file_name = "after_priority_{}".format(priority)
    return file_name


def save_fig_as_png(fig, output_folder, priority, final_result) -> matplotlib.figure.Figure:
    """Save matplotlib figure to output folder."""
    file_name = get_file_name(priority, final_result)
    figure_path = get_file_write_path(output_folder, file_name)
    fig.savefig(figure_path + ".png")
    return fig


def save_fig_as_html(fig, output_folder, priority, final_result) -> dict:
    """Save plotly figure as html"""
    file_name = get_file_name(priority, final_result)
    figure_path = get_file_write_path(output_folder, file_name)
    fig.write_html(figure_path + ".html")
    return fig.to_json()


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


def save_figure(fig, save_plot_to, output_folder, priority, final_result) -> Union[StringIO, matplotlib.figure.Figure]:
    """Save figure."""
    if save_plot_to == "image":
        return save_fig_as_png(fig, output_folder, priority, final_result)
    if save_plot_to == "stringio":
        return save_fig_as_stringio(fig)
    raise ValueError("Unsupported method of saving the plot results.")


def create_matplotlib_figure(
    result_dict, results_prev, plot_data_and_config: PlotDataAndConfig, final_result=False
) -> Union[StringIO, matplotlib.figure.Figure]:
    # pylint: disable=too-many-locals
    """Creates a figure with a subplot for each row in the plot_table."""
    results = result_dict["extract_result"]
    plot_config = plot_data_and_config["plot_options"]["plot_config"]
    plot_max_rows = plot_data_and_config["plot_options"]["plot_max_rows"]
    if len(plot_config) == 0:
        logger.info("Nothing to plot." + " Are there any goals that are active and described in the plot_table?")
        return None

    # Initalize figure
    n_cols = math.ceil(len(plot_config) / plot_max_rows)
    n_rows = math.ceil(len(plot_config) / n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 9, n_rows * 3), dpi=80, squeeze=False)
    if final_result:
        main_title = "Results after optimizing for all priorities"
    else:
        main_title = "Results after optimizing until priority {}".format(result_dict["priority"])
    fig.suptitle(main_title, fontsize=14)
    i_plot = -1

    all_goals = plot_data_and_config["prio_independent_data"]["all_goals"]
    # Add subplot for each row in the plot_table
    for subplot_config in plot_config:
        i_plot += 1
        axis = get_subplot_axis(i_plot, n_rows, axs)
        goal = get_goal(subplot_config, all_goals)
        subplot = SubplotMatplotlib(
            axis,
            subplot_config,
            goal,
            results,
            results_prev,
            plot_data_and_config["prio_independent_data"],
        )
        subplot.plot()

    fig.tight_layout()
    return save_figure(
        fig,
        plot_data_and_config["plot_options"]["save_plot_to"],
        plot_data_and_config["plot_options"]["output_folder"],
        result_dict["priority"],
        final_result,
    )


def create_plotly_figure(
    result_dict, results_prev, plot_data_and_config: PlotDataAndConfig, final_result=False
) -> dict:
    # pylint: disable=too-many-locals
    """Creates a figure with a subplot for each row in the plot_table."""
    results = result_dict["extract_result"]
    plot_config = plot_data_and_config["plot_options"]["plot_config"]
    plot_max_rows = plot_data_and_config["plot_options"]["plot_max_rows"]
    if len(plot_config) == 0:
        logger.info("Nothing to plot." + " Are there any goals that are active and described in the plot_table?")
        return None

    # Initalize figure
    n_cols = math.ceil(len(plot_config) / plot_max_rows)
    n_rows = math.ceil(len(plot_config) / n_cols)
    if final_result:
        main_title = "Result after optimizing for all priorities"
    else:
        main_title = "Results after optimizing until priority {}".format(result_dict["priority"])
    i_plot = -1

    plotly_figure = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=len(plot_config) * [" "], shared_xaxes=True)

    all_goals = plot_data_and_config["prio_independent_data"]["all_goals"]
    # Add subplot for each row in the plot_table
    for subplot_config in plot_config:
        i_plot += 1
        i_c, i_r = get_row_col_number(i_plot, n_rows)
        goal = get_goal(subplot_config, all_goals)
        subplot = SubplotPlotly(
            subplot_config,
            goal,
            results,
            results_prev,
            plot_data_and_config["prio_independent_data"],
            plotly_figure,
            i_r + 1,
            i_c + 1,
            i_plot,
        )
        subplot.plot()

    plotly_figure.update_layout(title_text=main_title)

    # Scale text
    scale_factor = 0.8
    plotly_figure.update_layout(
        font={"size": scale_factor * 12},
        title_font={"size": scale_factor * 16},
    )
    plotly_figure.update_annotations(font_size=scale_factor * 14)

    return save_fig_as_html(
        plotly_figure, plot_data_and_config["plot_options"]["output_folder"], result_dict["priority"], final_result
    )


def create_plot_each_priority(
    plot_data_and_config: PlotDataAndConfig, plotting_library: str = "plotly"
) -> Dict[str, Union[StringIO, matplotlib.figure.Figure]]:
    """Create all plots for one optimization run, for each priority one seperate plot."""
    intermediate_results = plot_data_and_config["intermediate_results"]
    plot_results = {}
    for intermediate_result_prev, intermediate_result in zip([None] + intermediate_results[:-1], intermediate_results):
        priority = intermediate_result["priority"]
        if plotting_library == "plotly":
            plot_results[priority] = create_plotly_figure(
                intermediate_result, intermediate_result_prev, plot_data_and_config
            )
        elif plotting_library == "matplotlib":
            plot_results[priority] = create_matplotlib_figure(
                intermediate_result, intermediate_result_prev, plot_data_and_config
            )
        else:
            raise ValueError("Invalid plotting library.")
    return plot_results


def create_plot_final_results(
    plot_data_and_config: PlotDataAndConfig, plotting_library: str = "plotly"
) -> Dict[str, Union[StringIO, matplotlib.figure.Figure]]:
    """Create a plot for the final results."""
    plot_results = {}
    intermediate_result = sorted(plot_data_and_config["intermediate_results"], key=lambda x: x["priority"])[-1]
    result_name = "final_results"
    if plotting_library == "plotly":
        plot_results[result_name] = create_plotly_figure(
            intermediate_result, None, plot_data_and_config, final_result=True
        )
    elif plotting_library == "matplotlib":
        plot_results[result_name] = create_matplotlib_figure(
            intermediate_result, None, plot_data_and_config, final_result=True
        )
    else:
        raise ValueError("Invalid plotting library.")
    return plot_results
