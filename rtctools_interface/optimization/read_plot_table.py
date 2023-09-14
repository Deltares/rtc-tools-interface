"""Module for reading goals from a csv file."""
import pandas as pd

from rtctools_interface.optimization.plot_table_schema import GOAL_TYPE_COMBINED_MODEL, PlotTable
from rtctools_interface.optimization.read_goals import get_goals_from_csv
from rtctools_interface.utils.parse_and_validate_table import parse_and_validate_table

PLOT_PARAMETERS = [
    "id",
    "y_axis_title",
    "variables_style_1",
    "variables_style_2",
    "variables_with_previous_result",
    "custom_title",
    "specified_in",
]


def read_and_check_plot_table(plot_table_file):
    """Read plot information from csv file and check values"""
    raw_plot_table = pd.read_csv(plot_table_file, sep=",")
    parsed_plot_table = parse_and_validate_table(raw_plot_table, PlotTable, "plot_table")
    return parsed_plot_table


def read_plot_table(plot_table_file, goal_table_file):
    """Read plot table for PlotGoals and merge with goals table"""
    plot_table = read_and_check_plot_table(plot_table_file)

    goals = get_goals_from_csv(goal_table_file)
    goals_by_id = {goal.goal_id: goal for _goal_type, goals in goals.items() for goal in goals}
    joined_config = []
    for subplot_config in plot_table:
        if subplot_config.id in goals_by_id.keys():
            goal_config = goals_by_id[subplot_config.id]
            if subplot_config.specified_in == "python":
                joined_config.append(subplot_config)
            else:
                joined_config.append(
                    GOAL_TYPE_COMBINED_MODEL[goal_config.goal_type](**(subplot_config.__dict__ | goal_config.__dict__))
                )
    return joined_config
