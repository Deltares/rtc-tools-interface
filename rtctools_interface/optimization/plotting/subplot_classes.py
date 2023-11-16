"""Classes for plotting, either using plotly or matplotlib. The classes generate one subplot in the full figure.."""
from abc import abstractmethod, ABC
import logging
import random
from typing import Any, Dict, Optional

import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import plotly.graph_objects as go

import numpy as np
from rtctools_interface.optimization.plot_table_schema import PlotTableRow
from rtctools_interface.optimization.type_definitions import GoalConfig, IntermediateResult, PrioIndependentData

logger = logging.getLogger("rtctools")

COMPARISON_RUN_SUFFIX = " (previous run)"


def get_timedeltas(times):
    """Get delta_t for each timestep."""
    return [np.nan] + [times[i] - times[i - 1] for i in range(1, len(times))]


def generate_unique_color(used_colors):
    """Get a color. Adds the new color to used_colors."""
    color_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    available_colors = [color for color in color_palette if color not in used_colors]

    if available_colors:
        new_color = available_colors[0]
    else:  # Generate a new color, may be similar to the existing colors.
        new_color = "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    used_colors.append(new_color)
    return new_color


class SubplotBase(ABC):
    """Base class for creating subplots."""

    def __init__(
        self,
        subplot_config: PlotTableRow,
        goal: GoalConfig,
        results: Dict[str, Any],
        results_prev: IntermediateResult,
        prio_independent_data: PrioIndependentData,
        used_colors,
        results_compare: Optional[IntermediateResult] = None,
    ):
        self.config: PlotTableRow = subplot_config
        self.goal: GoalConfig = goal
        self.function_nominal = self.goal["function_nominal"] if self.goal else 1
        self.results = results
        self.results_prev = results_prev
        self.results_compare = results_compare
        self.datetimes = prio_independent_data["io_datetimes"]
        self.time_deltas = get_timedeltas(prio_independent_data["times"])
        self.rate_of_change = self.goal.get("goal_type") in ["range_rate_of_change"]
        self.used_colors = used_colors

        if self.goal.get("goal_type") in ["range", "range_rate_of_change"]:
            self.target_min, self.target_max = self.goal["target_min_series"], self.goal["target_max_series"]
        else:
            self.target_min, self.target_max = None, None

        if "custom_title" in self.config.__dict__ and isinstance(self.config.custom_title, str):
            self.subplot_title = self.config.custom_title
        elif self.config.specified_in == "goal_generator":
            self.subplot_title = "Goal for {} (active from priority {})".format(
                self.goal["state"], self.goal["priority"]
            )
        else:
            self.subplot_title = ""

    def get_differences(self, timeseries):
        """Get rate of change timeseries for input timeseries, relative to the function nominal."""
        timeseries = list(timeseries)
        return [
            (st - st_prev) / dt / self.function_nominal * 100
            for st, st_prev, dt in zip(timeseries, [np.nan] + timeseries[:-1], self.time_deltas)
        ]

    def plot_with_comparison(self, label, state_name, linestyle=None, linewidth=None):
        """Plot the state both for the recent run and the comparison run."""
        timeseries_data = self.results[state_name]
        color = generate_unique_color(self.used_colors)
        self.plot_timeseries(label, timeseries_data, color=color, linestyle=linestyle, linewidth=linewidth)
        if self.results_compare:
            timeseries_data = self.results_compare["timeseries_data"][state_name]
            label += COMPARISON_RUN_SUFFIX
            self.plot_timeseries(label, timeseries_data, linestyle="dotted", color=color, linewidth=linewidth)

    def plot_with_previous(self, label, state_name, linestyle=None, linewidth=None):
        """Add line with the results for a particular state. If the results for the previous
        priority are availab, also add a (gray) line with those."""
        self.plot_with_comparison(label, state_name, linestyle=linestyle, linewidth=linewidth)

        if self.results_prev:
            timeseries_data = self.results_prev["timeseries_data"][state_name]
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
            self.plot_with_comparison(var, var)
        for var in self.config.variables_style_2:
            self.plot_with_comparison(var, var, linestyle="solid", linewidth="0.5")
        for var in self.config.variables_with_previous_result:
            self.plot_with_previous(var, var)

    def plot(self):
        """Plot the data in the subplot and format."""
        if self.config.specified_in == "goal_generator":
            self.plot_with_previous(self.goal["state"], self.goal["state"])
        self.plot_additional_variables()
        self.format_subplot()
        if self.config.specified_in == "goal_generator" and self.goal["goal_type"] in [
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
        subplot_config: PlotTableRow,
        goal: GoalConfig,
        results: Dict[str, Any],
        results_prev: IntermediateResult,
        prio_independent_data: PrioIndependentData,
        used_colors,
    ):
        super().__init__(subplot_config, goal, results, results_prev, prio_independent_data, used_colors)
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
        subplot_config: PlotTableRow,
        goal: GoalConfig,
        results: Dict[str, Any],
        results_prev: IntermediateResult,
        prio_independent_data: PrioIndependentData,
        used_colors,
        results_compare: Optional[IntermediateResult] = None,
        figure=None,
        row_num=0,
        col_num=0,
        i_plot=None,
    ):
        super().__init__(
            subplot_config, goal, results, results_prev, prio_independent_data, used_colors, results_compare
        )
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
                line={"color": self.map_color_code(color), "dash": "dot"},
            ),
            row=self.row_num,
            col=self.col_num,
        )

    def plot_line(self, xarray, yarray, label, color=None, linewidth=None, linestyle=None):
        linewidth = float(linewidth) * 1.3 if linewidth else linewidth
        linestyle = "dot" if linestyle == "dotted" else linestyle
        self.figure.add_trace(
            go.Scatter(
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
