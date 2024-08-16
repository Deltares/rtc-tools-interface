"""Module for calculating optimization periods."""
import bisect
import datetime
from pathlib import Path

import pandas as pd


def get_optimization_ranges_from_file(
    file_path: Path, model_time_range: tuple[datetime.datetime, datetime.datetime]
):
    """Read horizon config from a csv file"""
    if not file_path.exists():
        raise FileNotFoundError(
            f"The closed_loop_dates csv does not exist. Please create a horizon config file in {file_path}."
        )
    try:
        closed_loop_dates = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError(
            "The closed_loop_dates csv is empty. Please provide a valid file with start_date and end_date column."
        )
    closed_loop_dates.columns = closed_loop_dates.columns.str.replace(" ", "")
    if not all([col in closed_loop_dates.columns for col in ["start_date", "end_date"]]):
        raise ValueError("The closed_loop_dates csv should have both 'start_date' and 'end_date' columns.")
    closed_loop_dates["start_date"] = pd.to_datetime(closed_loop_dates["start_date"])
    closed_loop_dates["end_date"] = pd.to_datetime(closed_loop_dates["end_date"])
    for i in range(1, len(closed_loop_dates)):
        if closed_loop_dates["start_date"].iloc[i] > closed_loop_dates["end_date"].iloc[i - 1]:
            raise ValueError(f"Closed loop date table: Start date at row {i} is later than the previous end date. ")
    if any(closed_loop_dates["start_date"] < closed_loop_dates["start_date"].shift(1)):
        raise ValueError("Closed loop date table: The start dates are not in ascending order.")
    if any(closed_loop_dates["end_date"] < closed_loop_dates["end_date"].shift(1)):
        raise ValueError("Closed loop date table: The end dates are not in ascending order.")
    if any(closed_loop_dates["end_date"] < closed_loop_dates["start_date"]):
        raise ValueError("Closed loop date table: For one or more rows the end date is before the start date.")
    if any(closed_loop_dates["start_date"] > closed_loop_dates["end_date"]):
        raise ValueError("Closed loop date table: For one or more rows the start date is after the end date.")
    if (
        any(closed_loop_dates["start_date"].dt.hour != 0)
        or any(closed_loop_dates["start_date"].dt.minute != 0)
        or any(closed_loop_dates["end_date"].dt.hour != 0)
        or any(closed_loop_dates["end_date"].dt.minute != 0)
    ):
        raise ValueError(
            "Closed loop date table: Currently, the date ranges can only be specific up to the level of days."
        )
    assert (
        min(closed_loop_dates["start_date"]).date() == model_time_range[0].date()
    ), (
        "The start day of the first optimization run is not equal"
        " to the start day of the forecast date (or first timestep)."
    )
    assert (
        max(closed_loop_dates["end_date"]).date() <= model_time_range[1].date()
    ), (
        "The end date of one or more optimization runs is later"
        " than the end date of the timeseries import."
    )
    closed_loop_dates = [
        (optimization_range["start_date"], optimization_range["end_date"])
        for _, optimization_range in closed_loop_dates.iterrows()
    ]
    return closed_loop_dates


def _get_next_time_index(
    times: list[datetime.date],
    i_current: int,
    timestep_size: datetime.timedelta
) -> int:
    """
    Get the next timestep index.

    The next timestep index i_next is the highest index such that:
    * times[i_next] > times[i_current]
    * times[i_next] <= times[i_current] + timestep_size
    In case i_current >= i_max, i_next will be set to i_max, where i_max = len(times) - 1.
    """
    i_max = len(times) - 1
    if i_current >= i_max:
        return i_max
    current_time = times[i_current]
    next_time = current_time + timestep_size
    i_next = bisect.bisect_right(times, next_time) - 1
    i_next = max(i_current + 1, i_next)
    return i_next


def get_optimization_ranges(
    model_times: list[datetime.date],
    start_time: datetime.datetime,
    forecast_timestep: datetime.timedelta,
    optimization_period: datetime.timedelta
) -> list[tuple[datetime.datetime, datetime.datetime]]:
    """Calculate a list of optimization periods."""
    if forecast_timestep > optimization_period:
        raise ValueError(
            f"Forecast timestep {forecast_timestep} cannot be larger than"
            f" the optimization period {optimization_period}."
        )
    if start_time not in model_times:
        raise ValueError(f"Start time {start_time} is not in the given model times.")
    i_max = len(model_times) - 1
    i_start = model_times.index(start_time)
    i_end = _get_next_time_index(model_times, i_start, optimization_period)
    optimization_periods = []
    optimization_periods.append((model_times[i_start], model_times[i_end]))
    while i_end < i_max:
        i_start = _get_next_time_index(model_times, i_start, forecast_timestep)
        i_end = _get_next_time_index(model_times, i_start, optimization_period)
        optimization_periods.append((model_times[i_start], model_times[i_end]))
    return optimization_periods


def round_datetime_ranges_to_days(
    datetime_ranges: list[tuple[datetime.datetime, datetime.datetime]]
) -> list[tuple[datetime.datetime, datetime.datetime]]:
    """Round datetimes to dats in datetime ranges.

    The start of the range is rounded to the start of the day
    and the end of the range is rounded to the end of the day.
    """
    rounded_ranges = []
    for start, end in datetime_ranges:
        if start.date() == end.date():
            continue
        start = datetime.datetime.combine(start.date(), datetime.time.min)
        end = datetime.datetime.combine(end.date(), datetime.time.max)
        rounded_ranges.append((start, end))
    return rounded_ranges
