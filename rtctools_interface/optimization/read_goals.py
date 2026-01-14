"""Module for reading goals from a csv file."""

from pathlib import Path

import pandas as pd

from rtctools_interface.optimization.goal_table_schema import (
    GOAL_TYPES,
    NON_PATH_GOALS,
    PATH_GOALS,
    BaseGoalModel,
    MaximizationGoalModel,
    MinimizationGoalModel,
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
            raise ValueError(
                f"Goal of type {row['goal_type']} is not allowed. Allowed are {GOAL_TYPES.keys()}"
            )
        if int(row["active"]) not in [0, 1]:
            raise ValueError("Value in active column should be either 0 or 1.")


def validate_goal_list(goal_list):
    """Validate list of goals on correct type and uniqueness of id's"""
    ids = [goal.goal_id for goal in goal_list]
    if len(ids) != len(set(ids)):
        raise ValueError("ID's in goal generator table should be unique!")


def read_goals_from_csv(
    file: str | Path, csv_list_separator: str = ","
) -> list[
    RangeGoalModel | RangeRateOfChangeGoalModel | MinimizationGoalModel | MaximizationGoalModel
]:
    """Read goals from csv file and validate values."""
    raw_goal_table = pd.read_csv(file, sep=csv_list_separator)
    goal_table_checks(raw_goal_table)

    parsed_goals = []
    for _, row in raw_goal_table.iterrows():
        if int(row["active"]) == 1:
            parsed_goals.append(GOAL_TYPES[row["goal_type"]](**row))
    return parsed_goals


def read_goals_from_list(
    goals_to_generate: list,
) -> list[
    RangeGoalModel | RangeRateOfChangeGoalModel | MinimizationGoalModel | MaximizationGoalModel
]:
    """Read goals from a list. Validates whether the goals are of correct type."""
    if not isinstance(goals_to_generate, list):
        raise TypeError(f"Pass a list of goal elements, not a {type(goals_to_generate)}")
    for base_goal in goals_to_generate:
        if not isinstance(base_goal, BaseGoalModel):
            raise TypeError(
                "Each element in the list of goals to generate should be a child of BaseGoalModel"
            )
    active_goals = []
    for goal in goals_to_generate:
        if int(goal.active) == 1:
            active_goals.append(goal)
    return active_goals


def read_goals(
    file: str | None = None,
    path_goal: bool = True,
    read_from: str = "csv_table",
    goals_to_generate: list | None = None,
    csv_list_separator: str = ",",
) -> list[
    RangeGoalModel | RangeRateOfChangeGoalModel | MinimizationGoalModel | MaximizationGoalModel
]:
    """Read goals from a csv file

    Returns either only the path_goals or only the non_path goals.
    In either case only the active goals.
    """
    if read_from == "csv_table":
        parsed_goals = read_goals_from_csv(file, csv_list_separator)
    elif read_from == "passed_list":
        parsed_goals = read_goals_from_list(goals_to_generate)
    else:
        raise ValueError("GoalGeneratorMixin should either read from 'csv_table' or 'passed_list'")
    validate_goal_list(parsed_goals)
    requested_goal_types = PATH_GOALS.keys() if path_goal else NON_PATH_GOALS.keys()
    return [goal for goal in parsed_goals if goal.goal_type in requested_goal_types]
