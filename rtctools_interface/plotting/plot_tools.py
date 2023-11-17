"""Functions to create plots."""
from io import StringIO
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import matplotlib

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

from rtctools_interface.plotting.subplot_classes import (
    COMPARISON_RUN_SUFFIX,
    SubplotMatplotlib,
    SubplotPlotly,
)
from rtctools_interface.utils.type_definitions import GoalConfig, IntermediateResult, PlotDataAndConfig

logger = logging.getLogger("rtctools")


def get_row_col_number(i_plot, n_rows, n_cols, row_first=False):
    """Get row and col number given a plot number."""
    if row_first:
        i_r = math.ceil((i_plot + 1) / n_cols) - 1
        i_c = i_plot - i_r * n_cols
    else:  # Count along column direction first.
        i_c = math.ceil((i_plot + 1) / n_rows) - 1
        i_r = i_plot - i_c * n_rows
    return i_c, i_r


def get_subplot_axis(i_plot, n_rows, n_cols, axs):
    """Determine the row and column index and returns the corresponding subplot object."""
    i_c, i_r = get_row_col_number(i_plot, n_rows, n_cols)
    subplot = axs[i_r, i_c]
    return subplot


def get_file_write_path(output_folder: Union[str, Path], file_name="figure"):
    """Get path to to file."""
    new_output_folder = Path(output_folder) / "goal_figures"
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
    return fig


def get_goal(subplot_config, base_goals) -> Union[GoalConfig, None]:
    """Find the goal belonging to a subplot. Only checks for goals as specified in the goal table."""
    for goal in base_goals:
        if goal.get("goal_id") == subplot_config.id:
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


def add_buttons_to_plotly(plotly_figure):
    """Add buttons to the plotly figure.

    Currently only a button to select whether previous results should also be visible"""

    def comparison_run(name):
        """Returns bool indicating whether the trace corresponds to a comparison run."""
        return COMPARISON_RUN_SUFFIX in name

    all_names = [True for _ in plotly_figure.data]
    hide_comparison = [not comparison_run(trace.name) for trace in plotly_figure.data]
    buttons = [
        {
            "label": "Show results from previous run",
            "method": "update",
            "args": [{"visible": all_names}],
        },
        {
            "label": "Hide results from previous run",
            "method": "update",
            "args": [{"visible": hide_comparison}],
        },
    ]

    # Add the buttons to the layout
    plotly_figure.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "x": 1.0,
                "y": 1.0,
                "xanchor": "left",
                "yanchor": "bottom",
                "pad": {"r": 2, "t": 2, "l": 20, "b": 10},
            }
        ]
    )


def check_empty_plot_table(plot_config):
    """Chech whether there are any elements in the plot table."""
    if len(plot_config) == 0:
        logger.info("Nothing to plot." + " Are there any goals that are active and described in the plot_table?")
        return True
    return False


def get_main_title(final_result: bool, result_dict):
    """Generate main title."""
    if final_result:
        main_title = "Result after optimizing for all priorities"
    else:
        main_title = "Results after optimizing until priority {}".format(result_dict["priority"])
    return main_title


def create_matplotlib_figure(
    result_dict, results_prev, current_run: PlotDataAndConfig, final_result=False
) -> Union[StringIO, matplotlib.figure.Figure]:
    # pylint: disable=too-many-locals
    """Creates a figure with a subplot for each row in the plot_table."""
    used_colors: list = []
    results = result_dict["timeseries_data"]
    plot_config = current_run["plot_options"]["plot_config"]
    plot_max_rows = current_run["plot_options"]["plot_max_rows"]
    if check_empty_plot_table(plot_config):
        return None

    # Initalize figure
    n_cols = math.ceil(len(plot_config) / plot_max_rows)
    n_rows = math.ceil(len(plot_config) / n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 9, n_rows * 3), dpi=80, squeeze=False)
    main_title = get_main_title(final_result, result_dict)
    fig.suptitle(main_title, fontsize=14)
    i_plot = -1

    base_goals = current_run["prio_independent_data"]["base_goals"]
    # Add subplot for each row in the plot_table
    for subplot_config in plot_config:
        i_plot += 1
        axis = get_subplot_axis(i_plot, n_rows, n_cols, axs)
        goal = get_goal(subplot_config, base_goals)
        subplot = SubplotMatplotlib(
            axis,
            subplot_config,
            goal,
            results,
            results_prev,
            current_run["prio_independent_data"],
            used_colors,
        )
        subplot.plot()

    fig.tight_layout()
    return save_figure(
        fig,
        current_run["plot_options"]["save_plot_to"],
        current_run["plot_options"]["output_folder"],
        result_dict["priority"],
        final_result,
    )


def set_plotly_layout(plotly_figure, final_result, result_dict, results_compare):
    """Set the layout for the plotly figure."""
    main_title = get_main_title(final_result, result_dict)
    plotly_figure.update_layout(title_text=main_title)
    scale_factor = 0.8
    plotly_figure.update_layout(
        font={"size": scale_factor * 12},
        title_font={"size": scale_factor * 16},
    )
    plotly_figure.update_traces(hovertemplate="%{y}")
    plotly_figure.update_layout(hovermode="x")
    plotly_figure.update_annotations(font_size=scale_factor * 14)
    if results_compare:
        add_buttons_to_plotly(plotly_figure)
    plotly_figure.update_traces(marker={"size": 5})


def create_plotly_figure(
    result_dict: IntermediateResult,
    results_prev: Optional[IntermediateResult],
    current_run: PlotDataAndConfig,
    final_result=False,
    results_compare: Optional[IntermediateResult] = None,
) -> Any:
    # pylint: disable=too-many-locals
    """Creates a figure with a subplot for each row in the plot_table."""
    plot_config = current_run["plot_options"]["plot_config"]
    if check_empty_plot_table(plot_config):
        return None

    n_cols = math.ceil(len(plot_config) / current_run["plot_options"]["plot_max_rows"])
    n_rows = math.ceil(len(plot_config) / n_cols)
    i_plot = -1
    plotly_figure = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=len(plot_config) * [" "], shared_xaxes=True)
    base_goals = current_run["prio_independent_data"]["base_goals"]

    # Add subplot for each row in the plot_table
    used_colors: list = []
    for subplot_config in plot_config:
        i_plot += 1
        i_c, i_r = get_row_col_number(i_plot, n_rows, n_cols, row_first=True)
        goal = get_goal(subplot_config, base_goals)
        subplot = SubplotPlotly(
            subplot_config,
            goal,
            result_dict["timeseries_data"],
            results_prev,
            current_run["prio_independent_data"],
            used_colors,
            results_compare,
            plotly_figure,
            i_r + 1,
            i_c + 1,
            i_plot,
        )
        subplot.plot()

    set_plotly_layout(plotly_figure, final_result, result_dict, results_compare)

    return save_fig_as_html(
        plotly_figure, current_run["plot_options"]["output_folder"], result_dict["priority"], final_result
    )


def create_plot_each_priority(current_run: PlotDataAndConfig, plotting_library: str = "plotly") -> Dict[int, Any]:
    """Create all plots for one optimization run, for each priority one seperate plot."""
    intermediate_results = current_run["intermediate_results"]
    plot_results = {}
    for intermediate_result_prev, intermediate_result in zip([None] + intermediate_results[:-1], intermediate_results):
        priority = intermediate_result["priority"]
        if plotting_library == "plotly":
            plot_results[priority] = create_plotly_figure(intermediate_result, intermediate_result_prev, current_run)
        elif plotting_library == "matplotlib":
            plot_results[priority] = create_matplotlib_figure(
                intermediate_result, intermediate_result_prev, current_run
            )
        else:
            raise ValueError("Invalid plotting library.")
    return plot_results


def create_plot_final_results(
    current_run: PlotDataAndConfig,
    previous_run: Optional[PlotDataAndConfig] = None,
    output_folder=None,
    plotting_library: str = "plotly",
) -> Dict[str, Union[StringIO, matplotlib.figure.Figure]]:
    """Create a plot for the final results."""
    current_final_result = sorted(current_run["intermediate_results"], key=lambda x: x["priority"])[-1]
    if previous_run:
        previous_final_result = sorted(previous_run["intermediate_results"], key=lambda x: x["priority"])[-1]
    else:
        previous_final_result = None

    if output_folder:
        current_run["plot_options"]["output_folder"] = output_folder

    if plotting_library == "plotly":
        final_results_plot = create_plotly_figure(
            current_final_result, None, current_run, final_result=True, results_compare=previous_final_result
        )
    elif plotting_library == "matplotlib":
        final_results_plot = create_matplotlib_figure(current_final_result, None, current_run, final_result=True)
    else:
        raise ValueError("Invalid plotting library.")
    result_name = "final_results"
    return {result_name: final_results_plot}
