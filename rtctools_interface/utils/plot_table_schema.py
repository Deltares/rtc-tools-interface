"""Schema for the plot_table."""

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator, model_validator


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

    specified_in: Literal["python", "goal_generator"]
    y_axis_title: str
    id: int | str | float = np.nan
    variables_style_1: list[str] = []
    variables_style_2: list[str] = []
    variables_with_previous_result: list[str] = []
    custom_title: str | float = np.nan

    @field_validator(
        "variables_style_1", "variables_style_2", "variables_with_previous_result", mode="before"
    )
    @classmethod
    def convert_to_list(cls, value):
        """Convert the inputs to a list."""
        if isinstance(value, list):
            return value
        return string_to_list(value)

    @field_validator("id")
    @classmethod
    def convert_to_int(cls, value):
        """Convert value to integer if possible."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return value

    @model_validator(mode="after")
    def check_required_id(self):
        """Check if ID is present if specified in goal_generator."""
        if self.specified_in == "goal_generator" and pd.isna(self.id):
            raise ValueError("ID is required when goal is specified in the goal generator.")
        return self

    def get(self, attribute_name, default=None):
        """Similar functionality as dict-get method."""
        try:
            return getattr(self, attribute_name)
        except AttributeError:
            return default
