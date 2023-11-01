"""Type definitions for rtc_tools interface."""
import datetime
import pathlib
from typing import Optional, Tuple, TypedDict, List, Dict, Literal, Union

import numpy as np

from rtctools_interface.optimization.plot_and_goal_schema import (
    MinimizationGoalCombinedModel,
    MaximizationGoalCombinedModel,
    RangeGoalCombinedModel,
    RangeRateOfChangeGoalCombinedModel,
)
from rtctools_interface.optimization.plot_table_schema import PlotTableRow


class TargetDict(TypedDict):
    """Target min and max timeseries for a goal."""

    target_min: np.ndarray
    target_max: np.ndarray


class GoalConfig(TypedDict):
    """Configuration for a goal."""

    goal_id: str
    state: str
    goal_type: str
    function_min: Optional[float]
    function_max: Optional[float]
    function_nominal: Optional[float]
    target_min: Tuple[float, np.ndarray]
    target_max: Tuple[float, np.ndarray]
    priority: int
    weight: float
    order: int


class PrioIndependentData(TypedDict):
    """Data for one optimization run, which is independent of the priority."""

    io_datetimes: List[datetime.datetime]
    times: np.ndarray
    target_series: Dict[str, TargetDict]
    all_goals: List[GoalConfig]


class PlotOptions(TypedDict):
    """Plot configuration for on optimization run."""

    plot_config: List[
        Union[
            MinimizationGoalCombinedModel,
            MaximizationGoalCombinedModel,
            RangeGoalCombinedModel,
            RangeRateOfChangeGoalCombinedModel,
            PlotTableRow,
        ]
    ]
    plot_max_rows: int
    output_folder: pathlib.Path
    save_plot_to: Literal["image", "stringio"]


class IntermediateResult(TypedDict):
    """Dict containing the results (timeseries) for one priority optimization."""

    priority: int
    extract_result: Dict[str, np.ndarray]


class PlotDataAndConfig(TypedDict):
    """All data and options required to create all plots for one optimization run."""

    intermediate_results: List[IntermediateResult]
    plot_options: PlotOptions
    prio_independent_data: PrioIndependentData
