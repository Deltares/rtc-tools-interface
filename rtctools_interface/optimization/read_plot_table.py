"""Module for reading goals from a csv file."""
from typing import List, Union
import pandas as pd
from rtctools_interface.optimization.plot_and_goal_schema import (
    GOAL_TYPE_COMBINED_MODEL,
    MinimizationGoalCombinedModel,
    MaximizationGoalCombinedModel,
    RangeGoalCombinedModel,
    RangeRateOfChangeGoalCombinedModel,
)

from rtctools_interface.optimization.plot_table_schema import PlotTableRow
from rtctools_interface.optimization.read_goals import read_goals


def read_plot_config_from_csv(plot_table_file) -> List[PlotTableRow]:
    """Read plot information from csv file and check values"""
    raw_plot_table = pd.read_csv(plot_table_file, sep=",")
    parsed_rows: List[PlotTableRow] = []
    for _, row in raw_plot_table.iterrows():
        parsed_rows.append(PlotTableRow(**row))
    return parsed_rows


def read_plot_config_from_list(plot_config_list: List[PlotTableRow]) -> List[PlotTableRow]:
    """Read plot config from a list. Validates whether the elements are of correct type."""
    if not isinstance(plot_config_list, list):
        raise TypeError(f"Pass a list of PlotTableRow elements, not a {type(plot_config_list)}")
    for plot_config in plot_config_list:
        if not isinstance(plot_config, PlotTableRow):
            raise TypeError("Each element in the passed plot table should be of type 'PlotTableRow'")
    return plot_config_list


def get_plot_config(plot_table_file=None, plot_config_list=None, read_from="csv_table"):
    """Get plot config rows."""
    if read_from == "csv_table":
        return read_plot_config_from_csv(plot_table_file)
    if read_from == "passed_list":
        return read_plot_config_from_list(plot_config_list)
    raise ValueError("PlotGoalsMixin should either read from 'csv_table' or 'passed_list'")


def get_joined_plot_config(
    plot_table_file, goal_table_file, plot_config_list, read_from, goals_to_generate
) -> list[
    Union[
        MinimizationGoalCombinedModel,
        MaximizationGoalCombinedModel,
        RangeGoalCombinedModel,
        RangeRateOfChangeGoalCombinedModel,
        PlotTableRow,
    ]
]:
    """Read plot table for PlotGoals and merge with goals table"""
    plot_table = get_plot_config(
        plot_table_file=plot_table_file, plot_config_list=plot_config_list, read_from=read_from
    )

    # goals = read_goals_from_csv(goal_table_file)
    path_goals = read_goals(
        file=goal_table_file, path_goal=True, read_from=read_from, goals_to_generate=goals_to_generate
    )
    non_path_goals = read_goals(
        file=goal_table_file, path_goal=False, read_from=read_from, goals_to_generate=goals_to_generate
    )
    goals = path_goals + non_path_goals
    goals_by_id = {goal.goal_id: goal for goal in goals}
    joined_plot_config = []
    for subplot_config in plot_table:
        if subplot_config.id in goals_by_id.keys():
            goal_config = goals_by_id[subplot_config.id]
            if subplot_config.specified_in == "python":
                joined_plot_config.append(subplot_config)
            else:
                joined_plot_config.append(
                    GOAL_TYPE_COMBINED_MODEL[goal_config.goal_type](**(subplot_config.__dict__ | goal_config.__dict__))
                )
    return joined_plot_config
