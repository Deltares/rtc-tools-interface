"""Module for reading goals from a csv file."""
from typing import List, Literal, Union
import pandas as pd

from rtctools_interface.optimization.goal_table_schema import (
    GOAL_TYPES,
    NON_PATH_GOALS,
    PATH_GOALS,
    MinMaximizationGoalModel,
    RangeGoalModel,
    RangeRateOfChangeGoalModel,
)


def goal_table_checks(goal_table):
    """Validate input goal table."""
    if "goal_type" not in goal_table:
        raise ValueError("Goal type column not in goal table.")
    if "active" not in goal_table:
        raise ValueError("Active column not in goal table.")
    for _, row in goal_table.iterrows():
        if row["goal_type"] not in GOAL_TYPES.keys():
            raise ValueError(f"Goal of type {row['goal_type']} is not allowed. Allowed are {GOAL_TYPES.keys()}")
        if int(row["active"]) not in [0, 1]:
            raise ValueError("Value in active column should be either 0 or 1.")


def read_goals_from_csv(
    file,
) -> dict[
    Literal["minimization_path", "maximization_path", "range", "range_rate_of_change"],
    List[Union[RangeGoalModel, RangeRateOfChangeGoalModel, MinMaximizationGoalModel]],
]:
    """Read goals from csv file and validate values."""
    raw_goal_table = pd.read_csv(file, sep=",")
    goal_table_checks(raw_goal_table)

    parsed_goals = {goal_type: [] for goal_type in GOAL_TYPES.keys()}
    for _, row in raw_goal_table.iterrows():
        if int(row["active"]) == 1:
            parsed_goals[row["goal_type"]].append(GOAL_TYPES[row["goal_type"]](**row))

    # TODO: Make a pydantic model for the set of all goals and do the following validation there.
    ids = [parsed_goal.goal_id for goal_by_type in parsed_goals.values() for parsed_goal in goal_by_type]
    if len(ids) != len(set(ids)):
        raise ValueError("ID's in goal generator table should be unique!")
    return parsed_goals


def read_goals(
    file, path_goal: bool
) -> List[Union[RangeGoalModel, RangeRateOfChangeGoalModel, MinMaximizationGoalModel]]:
    """Read goals from a csv file
    Returns either only the path_goals or only the non_path goals. In either case only the active goals.
    """
    parsed_goals = read_goals_from_csv(file)
    requested_goal_types = PATH_GOALS.keys() if path_goal else NON_PATH_GOALS.keys()
    return [goal for goal_type, goals in parsed_goals.items() if goal_type in requested_goal_types for goal in goals]
