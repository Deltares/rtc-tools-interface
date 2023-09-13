"""Module for reading goals from a csv file."""
import pandas as pd

from rtctools_interface.optimization.base_goal import PATH_GOALS, NON_PATH_GOALS
from rtctools_interface.optimization.goal_table_schema import GoalTableRow

from rtctools_interface.utils.parse_and_validate_table import parse_and_validate_table

GOAL_PARAMETERS = [
    "id",
    "state",
    "goal_type",
    "function_min",
    "function_max",
    "function_nominal",
    "target_data_type",
    "target_min",
    "target_max",
    "priority",
    "weight",
    "order",
]


def read_and_check_goal_table(file):
    """Read goals from csv file and check values"""
    raw_goal_table = pd.read_csv(file, sep=",")
    parsed_goal_table = parse_and_validate_table(raw_goal_table, GoalTableRow, "goal_table")
    return parsed_goal_table


def read_goals(file, path_goal: bool):
    """Read goals from a csv file
    Returns either only the path_goals or only the non_path goals
    """
    goals = read_and_check_goal_table(file)
    is_active = goals["active"] == 1
    if path_goal:
        requested_goal_type = goals["goal_type"].isin(PATH_GOALS)
    else:
        requested_goal_type = goals["goal_type"].isin(NON_PATH_GOALS)
    filter_goals = is_active * requested_goal_type
    return goals.loc[filter_goals, GOAL_PARAMETERS]
