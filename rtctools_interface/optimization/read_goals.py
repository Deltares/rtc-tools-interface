"""Module for reading goals from a csv file."""
import pandas as pd

from rtctools_interface.optimization.base_goal import PATH_GOALS, NON_PATH_GOALS

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


def read_goals(file, path_goal):
    """Read goals from a cvs file.
    Returns either only the path_goals or only the non_path goals
    """
    goals = pd.read_csv(file, sep=",")
    is_active = (goals['active'] == 1)
    if path_goal:
        requested_goal_type = goals['goal_type'].isin(PATH_GOALS)
    else:
        requested_goal_type = goals['goal_type'].isin(NON_PATH_GOALS)
    filter_goals = is_active*requested_goal_type
    return goals.loc[filter_goals, GOAL_PARAMETERS]
