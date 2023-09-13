"""Schema for the plot_table."""

from typing import List, Union
from pydantic import BaseModel, field_validator
import numpy as np


class PlotTableRow(BaseModel):
    """Model for one row in the plot table."""

    specified_in: str
    y_axis_title: str
    id: Union[int, str, float] = np.nan
    variables_style_1: Union[str, float] = np.nan
    variables_style_2: Union[str, float] = np.nan
    variables_with_previous_result: Union[str, float] = np.nan
    custom_title: Union[str, float] = np.nan

    @field_validator("specified_in")
    @classmethod
    def validate_goal_type(cls, value):
        """Check whether the specified_in value is allowed"""
        allowed = ["python", "goal_generator"]
        if value not in allowed:
            raise ValueError(f"Specified_in should be one of {allowed}")
        return value


class PlotTable(BaseModel):
    """Model for the goal table"""

    rows: List[PlotTableRow]
