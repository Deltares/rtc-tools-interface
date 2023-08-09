"""Module for reading goals from a csv file."""
import pandas as pd


GOAL_PARAMETERS = [
    'state',
    'goal_type',
    'function_min',
    'function_max',
    'function_nominal',
    'target_data_type',
    'target_min',
    'target_max',
    'priority',
    'weight',
    'order',
]


def read_goals(file, goal_type):
    """Read goals from a cvs file.
    """
    goals = pd.read_csv(file, sep=",")
    is_active = (goals['active'] == 1)
    goal_type = (goals['pathgoal'] == goal_type)
    filter_goals = is_active*goal_type
    return goals.loc[filter_goals, GOAL_PARAMETERS]
