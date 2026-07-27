"""Type definitions for rtc_tools interface."""

import datetime
import pathlib
from typing import Literal, TypedDict

import numpy as np

from rtctools_interface.utils.plot_table_schema import PlotTableRow


class TargetDict(TypedDict):
    """Target min and max timeseries for a goal."""

    target_min: np.ndarray
    target_max: np.ndarray


class GoalConfig(TypedDict):
    """Configuration for a goal."""

    goal_id: str
    state: str
    goal_type: str
    function_min: float | None
    function_max: float | None
    function_nominal: float | None
    target_min: tuple[float, np.ndarray]
    target_max: tuple[float, np.ndarray]
    target_min_series: np.ndarray | None
    target_max_series: np.ndarray | None
    priority: int
    weight: float
    order: int


class PrioIndependentData(TypedDict):
    """Data for one optimization run, which is independent of the priority."""

    io_datetimes: list[datetime.datetime]
    times: np.ndarray
    base_goals: list[GoalConfig]


class PlotOptions(TypedDict):
    """Plot configuration for on optimization run."""

    plot_config: list[PlotTableRow]
    plot_max_rows: int
    output_folder: pathlib.Path
    save_plot_to: Literal["image", "stringio"]


class IntermediateResult(TypedDict):
    """Dict containing the results (timeseries) for one priority optimization."""

    priority: int
    timeseries_data: dict[str, np.ndarray]


class PreviousGoalConstraintRow(TypedDict):
    """CSV row for active constraints created from previous goals."""

    priority: int
    total_previous_goal_constraints: int | str
    active_previous_goals_constraints: int | str
    ensemble_member: int | str
    constraint_source: str
    function_key: str
    goal_priority: int | str
    goal_class: str
    active_times: str
    value: float | str
    lower_bound: float | str
    upper_bound: float | str
    active_bound: str
    active_bound_value: float | str


class PlotDataAndConfig(TypedDict):
    """All data and options required to create all plots for one optimization run."""

    intermediate_results: list[IntermediateResult]
    plot_options: PlotOptions
    prio_independent_data: PrioIndependentData
    config_version: float
