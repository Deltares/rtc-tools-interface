"""Schema for the goal_table."""
from typing import Union
from pydantic import BaseModel, field_validator, model_validator
import numpy as np
import pandas as pd

from rtctools_interface.optimization.base_goal import GOAL_TYPES, TARGET_DATA_TYPES


class GoalTableRow(BaseModel):
    """Model for one row in the goal table."""

    id: Union[int, str]
    active: int
    state: str
    goal_type: str
    function_min: float = np.nan
    function_max: float = np.nan
    function_nominal: float = np.nan
    target_data_type: Union[str, float] = np.nan
    target_min: Union[float, str] = np.nan
    target_max: Union[float, str] = np.nan
    priority: int
    weight: float = np.nan
    order: float = np.nan

    @field_validator("goal_type")
    @classmethod
    def validate_goal_type(cls, value):
        """Check whether the supplied goal type is supported"""
        if value not in GOAL_TYPES:
            raise ValueError(f"Invalid goal_type '{value}'. Allowed values are {GOAL_TYPES}.")
        return value

    @field_validator("target_data_type")
    @classmethod
    def validate_target_type(cls, value):
        """Check whether the supplied target_data_type is supported"""
        if not pd.isna(value) and value not in TARGET_DATA_TYPES:
            raise ValueError(f"Invalid target type '{value}'. Allowed values are {TARGET_DATA_TYPES}.")
        return value

    @field_validator("active")
    @classmethod
    def validate_active(cls, value):
        """Check whether active is either 0 or 1"""
        allowed_values = [0, 1]
        if value not in allowed_values:
            raise ValueError(f"Invalid value '{value}' in column 'active'. Allowed values are {allowed_values}.")
        return value

    @model_validator(mode="after")
    def validate_target_type_and_value(self):
        """Check whether the target_min and target_max datatype correspond to the target_data_type"""
        try:
            if isinstance(self.target_data_type, str):
                if self.target_data_type == "value":
                    assert isinstance(self.target_min, float)
                    assert isinstance(self.target_max, float)
                elif self.target_data_type in ["parameter", "timeseries"]:
                    assert isinstance(self.target_min, str) or pd.isna(self.target_min)
                    assert isinstance(self.target_max, str) or pd.isna(self.target_max)
        except AssertionError as exc:
            raise ValueError(
                "The type in the target_min/target_max column does not correspond to the target_data_type."
            ) from exc
        return self

    @model_validator(mode="after")
    def validate_range_goal(self):
        """Check whether required columns for the range_goal are available."""
        try:
            if self.goal_type in ["range", "range_rate_of_change"]:
                assert not (pd.isna(self.target_min) and pd.isna(self.target_max))
        except AssertionError as exc:
            raise ValueError("For a range goal, at least one of target_min and target_max should be set.") from exc
        return self
