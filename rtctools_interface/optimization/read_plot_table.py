"""Module for reading goals from a csv file."""
import pandas as pd

from rtctools_interface.optimization.plot_table_schema import plot_table_column_spec
from rtctools_interface.optimization.read_goals import GOAL_PARAMETERS, read_and_check_goal_table
from rtctools_interface.utils.check_pandas_table import check_pandas_table

PLOT_PARAMETERS = [
    "id",
    "y_axis_title",
    "variables_plot_1",
    "variables_plot_2",
    "custom_state",
    "custom_title",
    "specified_in",
]


def string_to_list(string):
    """
    Convert a string to a list of strings
    """
    if string == "" or not isinstance(string, str):
        return []
    string_without_whitespace = string.replace(" ", "")
    list_of_strings = string_without_whitespace.split(",")
    return list_of_strings


def read_and_check_plot_table(plot_table_file):
    """Read plot information from csv file and check values"""
    plot_table = pd.read_csv(plot_table_file, sep=",")
    check_pandas_table(plot_table, plot_table_column_spec, "plot_table")
    return plot_table


def read_plot_table(plot_table_file, goal_table_file):
    """Read plot table for PlotGoals and merge with goals table"""
    plot_table = read_and_check_plot_table(plot_table_file)
    plot_table[["variables_plot_1", "variables_plot_2"]] = plot_table[
        ["variables_plot_1", "variables_plot_2"]
    ].applymap(string_to_list)
    goals = read_and_check_goal_table(goal_table_file)
    joined_table = plot_table.merge(goals, on="id", how="left")
    joined_table["active"].replace(pd.NA, 1, inplace=True)
    is_active = (joined_table["active"] == 1) | (joined_table["specified_in"] == "python")
    return joined_table.loc[is_active, PLOT_PARAMETERS + GOAL_PARAMETERS]
