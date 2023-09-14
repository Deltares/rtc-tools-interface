"""Module for reading goals from a csv file."""
from typing import Any, List
import pandas as pd

from rtctools_interface.optimization.goal_table_schema import GOAL_TYPES, NON_PATH_GOALS, PATH_GOALS

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


def get_goals_from_csv(file) -> dict[str, List[Any]]:
    """Read goals from csv file and check values"""
    raw_goal_table = pd.read_csv(file, sep=",")
    if "goal_type" not in raw_goal_table:
        raise ValueError("Goal type column not in goal table.")
    if "active" not in raw_goal_table:
        raise ValueError("Active column not in goal table.")
    parsed_goals = {goal_type: [] for goal_type in GOAL_TYPES.keys()}
    for _, row in raw_goal_table.iterrows():
        if row["goal_type"] not in GOAL_TYPES.keys():
            raise ValueError(f"Goal of type {row['goal_type']} is not allowed. Allowed are {GOAL_TYPES.keys()}")
        if int(row["active"]) == 1:
            parsed_goals[row["goal_type"]].append(GOAL_TYPES[row["goal_type"]](**row))
        elif int(row["active"]) != 0:
            raise ValueError("Value in active column should be either 0 or 1.")
    return parsed_goals


def read_goals(file, path_goal: bool):
    """Read goals from a csv file
    Returns either only the path_goals or only the non_path goals. In either case only the active goals.
    """
    parsed_goals = get_goals_from_csv(file)
    requested_goal_types = PATH_GOALS.keys() if path_goal else NON_PATH_GOALS.keys()
    return [goal for goal_type, goals in parsed_goals.items() if goal_type in requested_goal_types for goal in goals]
