"""Module for reading goals from a csv file."""
import pandas as pd

from rtctools_interface.optimization.read_goals import GOAL_PARAMETERS

PLOT_PARAMETERS = [
    "id",
    "y_axis_title",
    "variables_plot_1",
    "variables_plot_2"
]

def string_to_list(string):
    """
    Convert a string to a list of strings
    """
    if string == "" or not isinstance(string, str):
        return []
    string_without_whitespace = string.replace(' ', '')
    list_of_strings = string_without_whitespace.split(',')
    return list_of_strings


def read_plot_table(plot_table_file, goal_table_file):
    """Read plot table for PlotGoals and merge with goals table
    """
    plot_table = pd.read_csv(plot_table_file, sep=",")
    plot_table[["variables_plot_1", "variables_plot_2"]] = plot_table[["variables_plot_1", "variables_plot_2"]].applymap(string_to_list)
    goals = pd.read_csv(goal_table_file, sep=",")
    joined_table = plot_table.merge(goals, on="id")
    is_active = (goals['active'] == 1)
    return joined_table.loc[is_active, PLOT_PARAMETERS + GOAL_PARAMETERS]

