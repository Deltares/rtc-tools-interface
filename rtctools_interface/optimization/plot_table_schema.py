"""Schema for the plot_table."""

from typing import List, Union
from pydantic import BaseModel, field_validator
import numpy as np

from rtctools_interface.optimization.goal_table_schema import (
    MinMaximizationGoalModel,
    RangeGoalModel,
    RangeRateOfChangeGoalModel,
)


def string_to_list(string):
    """
    Convert a string to a list of strings
    """
    if string == "" or not isinstance(string, str):
        return []
    string_without_whitespace = string.replace(" ", "")
    list_of_strings = string_without_whitespace.split(",")
    return list_of_strings


class PlotTableRow(BaseModel):
    """Model for one row in the plot table."""

    specified_in: str
    y_axis_title: str
    id: Union[int, str, float] = np.nan
    variables_style_1: list[str] = []
    variables_style_2: list[str] = []
    variables_with_previous_result: list[str] = []
    custom_title: Union[str, float] = np.nan

    @field_validator("specified_in")
    @classmethod
    def validate_goal_type(cls, value):
        """Check whether the specified_in value is allowed"""
        allowed = ["python", "goal_generator"]
        if value not in allowed:
            raise ValueError(f"Specified_in should be one of {allowed}")
        return value

    @field_validator("variables_style_1", "variables_style_2", "variables_with_previous_result", mode="before")
    @classmethod
    def convert_to_list(cls, value):
        """Convert the inputs to a list."""
        return string_to_list(value)


class PlotTable(BaseModel):
    """Model for the goal table"""

    rows: List[PlotTableRow]


class RangeGoalCombinedModel(PlotTableRow, RangeGoalModel):
    """Model for information in plot table and goal table."""


class MinMaximizationGoalCombinedModel(PlotTableRow, MinMaximizationGoalModel):
    """Model for information in plot table and goal table."""


class RangeRateOfChangeGoalCombinedModel(PlotTableRow, RangeRateOfChangeGoalModel):
    """Model for information in plot table and goal table."""


GOAL_TYPE_COMBINED_MODEL = {
    "minimization_path": MinMaximizationGoalCombinedModel,
    "maximization_path": MinMaximizationGoalCombinedModel,
    "range": RangeGoalCombinedModel,
    "range_rate_of_change": RangeRateOfChangeGoalCombinedModel,
}
