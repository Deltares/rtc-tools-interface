from pathlib import Path

import pandas as pd


def read_closed_loop_dates(file_path: Path):
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
    return closed_loop_dates
