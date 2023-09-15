"""The models in this module define the combined model for a goal and its plot information."""
from rtctools_interface.optimization.goal_table_schema import (
    MinMaximizationGoalModel,
    RangeGoalModel,
    RangeRateOfChangeGoalModel,
)
from rtctools_interface.optimization.plot_table_schema import PlotTableRow


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
