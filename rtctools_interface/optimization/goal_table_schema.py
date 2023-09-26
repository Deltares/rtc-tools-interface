"""Schema for the goal_table."""
from typing import Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np
import pandas as pd


class BaseGoalModel(BaseModel):
    """BaseModel for a goal."""

    goal_id: Union[int, str] = Field(..., alias="id")
    active: Literal[0, 1]
    state: str
    goal_type: str
    priority: int
    function_nominal: float = np.nan
    weight: float = np.nan
    order: float = np.nan

    @field_validator("goal_type")
    @classmethod
    def validate_goal_type(cls, value):
        """Check whether the supplied goal type is supported"""
        if value not in GOAL_TYPES.keys():
            raise ValueError(f"Invalid goal_type '{value}'. Allowed values are {GOAL_TYPES.keys()}.")
        return value

    @field_validator("goal_id", "active")
    @classmethod
    def convert_to_int(cls, value):
        """Convert value to integer if possible."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return value


class MaximizationGoalModel(BaseGoalModel):
    """Model for a minimization and maximization goal."""


MinimizationGoalModel = MaximizationGoalModel


class RangeGoalModel(BaseGoalModel):
    """Model for a range goal."""

    target_data_type: str
    function_min: float = np.nan
    function_max: float = np.nan
    target_min: Union[float, str] = np.nan
    target_max: Union[float, str] = np.nan

    @field_validator("target_min", "target_max")
    @classmethod
    def convert_to_float(cls, value):
        """Convert value to float if possible."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return value

    @model_validator(mode="after")
    def validate_targets(self):
        """Check whether required columns for the range_goal are available."""
        try:
            assert not (pd.isna(self.target_min) and pd.isna(self.target_max))
        except AssertionError as exc:
            raise ValueError("For a range goal, at least one of target_min and target_max should be set.") from exc
        return self

    @model_validator(mode="after")
    def validate_target_type_and_value(self):
        """Check whether the target_min and target_max datatype correspond to the target_data_type"""
        try:
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


class RangeRateOfChangeGoalModel(RangeGoalModel):
    """Model for a rate of change range goal."""


PATH_GOALS = {
    "minimization_path": MinimizationGoalModel,
    "maximization_path": MaximizationGoalModel,
    "range": RangeGoalModel,
    "range_rate_of_change": RangeRateOfChangeGoalModel,
}
NON_PATH_GOALS = {}
GOAL_TYPES = PATH_GOALS | NON_PATH_GOALS
TARGET_DATA_TYPES = [
    "value",
    "parameter",
    "timeseries",
]
