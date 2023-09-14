"""Schema for the goal_table."""
from typing import Union
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np
import pandas as pd


class BaseGoalModel(BaseModel):
    """BaseModel for a goal."""

    goal_id: Union[int, str] = Field(..., alias="id")
    active: int
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

    @field_validator("goal_id")
    @classmethod
    def convert_to_int(cls, value):
        """Convert value to integer if possible."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return value

    @field_validator("active")
    @classmethod
    def validate_active(cls, value):
        """Check whether active is either 0 or 1"""
        allowed_values = [0, 1]
        if value not in allowed_values:
            raise ValueError(f"Invalid value '{value}' in column 'active'. Allowed values are {allowed_values}.")
        return value


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


class MinMaximizationGoalModel(BaseGoalModel):
    """Model for a minimization and maximization goal."""


# class AllGoals(BaseModel):
#     """Model for the goal table"""

#     range_goals: List[RangeGoalModel]
#     min_max_goals: List[MinMaximizationGoalModel]
#     range_rate_of_change_goals: List[RangeRateOfChangeGoalModel]

#     @model_validator(mode="after")
#     def validate_unique_ids(self):
#         """Validate whether all id's are unique."""
#         all_ids = [row.id for row in self.rows]  # For now, we also consider inactive goals.
#         if len(all_ids) != len(set(all_ids)):
#             raise ValueError("Non-unique goal-id('s) in goal table! Please give each goal a unique id.")


# class AllGoals(BaseModel):
#     goals = List[Union[RangeGoalModel, MinMaximizationGoalModel, RangeRateOfChangeGoalModel]]


PATH_GOALS = {
    "minimization_path": MinMaximizationGoalModel,
    "maximization_path": MinMaximizationGoalModel,
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
