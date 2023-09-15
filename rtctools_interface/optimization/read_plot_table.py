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
from rtctools_interface.optimization.read_goals import read_goals_from_csv


def read_plot_config_from_csv(plot_table_file) -> List[PlotTableRow]:
    """Read plot information from csv file and check values"""
    raw_plot_table = pd.read_csv(plot_table_file, sep=",")
    parsed_rows: List[PlotTableRow] = []
    for _, row in raw_plot_table.iterrows():
        parsed_rows.append(PlotTableRow(**row))
    return parsed_rows


def get_joined_plot_config(
    plot_table_file, goal_table_file
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
    plot_table = read_plot_config_from_csv(plot_table_file)

    goals = read_goals_from_csv(goal_table_file)
    goals_by_id = {goal.goal_id: goal for _goal_type, goals in goals.items() for goal in goals}
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
