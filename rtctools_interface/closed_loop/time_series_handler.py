import bisect
import datetime
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from rtctools.data import csv, pi, rtc

ns = {"fews": "http://www.wldelft.nl/fews", "pi": "http://www.wldelft.nl/fews/PI"}

logger = logging.getLogger("rtctools")


class TimeSeriesHandler(ABC):
    """ABC for handling timeseries data."""

    # The forecast date determines at which date the optimization starts.
    forecast_date: datetime.datetime | None = None

    @abstractmethod
    def read(self, file_name: str) -> None:
        """Read the timeseries."""

    @abstractmethod
    def select_time_range(self, start_date: datetime.datetime, end_date: datetime.datetime) -> None:
        """Select a time range from the timeseries data. Removes data outside the interval.
        The specified range is inclusive on both sides."""

    @abstractmethod
    def write(self, file_path: Path) -> None:
        """Write the timeseries data to a file."""

    @abstractmethod
    def get_timestep(self) -> datetime.timedelta:
        """Get the timestep of the timeseries data."""

    @abstractmethod
    def get_datetimes(self) -> list[datetime.datetime]:
        """Get the dates of the timeseries."""

    @abstractmethod
    def get_datetime_range(self) -> tuple[datetime.datetime, datetime.datetime]:
        """Get the date range of the timeseries data (min, max)."""

    @abstractmethod
    def get_all_internal_ids(self) -> list[str]:
        """Get all internal id's of the timeseries data."""

    @abstractmethod
    def set_initial_value(self, internal_id: str, value: float) -> None:
        """Set the initial value of a variable in the timeseries data."""

    @abstractmethod
    def is_set(self, internal_id: str) -> bool:
        """Check whether the variable exists in the timeseries data
        and whether it has a least one non-nan value"""


class CSVTimeSeriesFile(TimeSeriesHandler):
    """Timeseries handler for csv files."""

    def __init__(
        self,
        input_folder: Path,
        timeseries_import_basename: str = "timeseries_import",
        csv_delimiter=",",
        initial_state_base_name: str = "initial_state",
    ):
        self.data = pd.DataFrame()
        self.input_folder = input_folder
        self.csv_delimiter = csv_delimiter
        self.data_col = None
        self.initial_state = None
        self.read(timeseries_import_basename, initial_state_base_name)

    def read(self, file_name: str, initial_state_base_name=None):
        timeseries = csv.load(
            (self.input_folder / file_name).with_suffix(".csv"),
            delimiter=self.csv_delimiter,
            with_time=True,
        )
        self.data = pd.DataFrame(timeseries)
        if self.data is not None:
            self.date_col = self.data.columns[0]
            self.forecast_date = self.data[self.date_col].iloc[0]
        else:
            raise ValueError("No data to read.")
        if initial_state_base_name is not None:
            initial_state_file = self.input_folder / initial_state_base_name
            if initial_state_file.with_suffix(".csv").exists():
                initial_state = csv.load(
                    initial_state_file.with_suffix(".csv"),
                    delimiter=self.csv_delimiter,
                    with_time=False,
                )
                self.initial_state: dict | None = {
                    field: float(initial_state[field]) for field in initial_state.dtype.names
                }
        else:
            self.initial_state = None

    def select_time_range(self, start_date: datetime.datetime, end_date: datetime.datetime):
        mask = (self.data[self.date_col] >= start_date) & (self.data[self.date_col] <= end_date)
        self.data = self.data.loc[mask]
        self.forecast_date = start_date

    def write(self, file_path: Path):
        self.write_timeseries(file_path)
        self.write_initial_state(file_path)

    def write_timeseries(self, file_path: Path, file_name: str = "timeseries_import"):
        self.data.to_csv(
            (file_path / file_name).with_suffix(".csv"),
            index=False,
            date_format="%Y-%m-%d %H:%M:%S",
        )

    def write_initial_state(self, file_path: Path, file_name: str = "initial_state"):
        if self.initial_state is not None:
            initial_state = pd.DataFrame(self.initial_state, index=[0])
            initial_state.to_csv(
                (file_path / file_name).with_suffix(".csv"), header=True, index=False
            )

    def get_timestep(self):
        return self.data[self.date_col].diff().min()

    def get_datetimes(self):
        return self.data[self.date_col].to_list()

    def get_datetime_range(self):
        return self.data[self.date_col].min(), self.data[self.date_col].max()

    def get_all_internal_ids(self):
        ids = list(self.data.columns[1:])
        if self.initial_state is not None:
            ids.extend(list(self.initial_state.keys()))
        return ids

    def set_initial_value(self, internal_id, value):
        if self.initial_state is None or internal_id not in self.initial_state:
            self.data[internal_id].iloc[0] = value
        else:
            self.initial_state[internal_id] = value

    def is_set(self, internal_id):
        val_is_set = False
        if internal_id in self.data.columns:
            val_is_set = not self.data[internal_id].isna().all()
        if self.initial_state is not None and internal_id in self.initial_state:
            val_is_set = False
        return val_is_set


class XMLTimeSeriesFile(TimeSeriesHandler):
    """ "Timeseries handler for xml files"""

    def __init__(
        self,
        input_folder: Path,
        timeseries_import_basename: str = "timeseries_import",
    ):
        self.input_folder = input_folder
        self.pi_binary_timeseries = False
        self.pi_validate_timeseries = True
        self.data_config = None
        self.pi_timeseries = None
        self.read(timeseries_import_basename)

        if self.get_datetime_range()[0] < self.forecast_date:
            logger.warning(
                "Currently, the closed loop runner does support data before the forecast date."
            )
            logger.warning("Removing data before forecast date.")
            self.select_time_range(self.forecast_date, self.pi_timeseries.times[-1])

    def read(self, file_name: str):
        """Read the timeseries data from a file."""
        timeseries_import_basename = file_name
        self.data_config = rtc.DataConfig(self.input_folder)
        self.pi_timeseries = pi.Timeseries(
            self.data_config,
            self.input_folder,
            timeseries_import_basename,
            binary=self.pi_binary_timeseries,
            pi_validate_times=self.pi_validate_timeseries,
        )
        self.forecast_date = self.pi_timeseries.forecast_datetime

    def is_set(self, internal_id):
        """Check whether the variable exists in the timeseries data.

        and whether it has a value at at least one of time steps."""
        try:
            var: np.ndarray = self.pi_timeseries.get(variable=internal_id)
            return not np.isnan(var).all()
        except KeyError:
            return False

    def _is_in_dataconfig(self, internal_id):
        """Check if an internal id is in the data configuration."""
        try:
            self.data_config.pi_variable_ids(internal_id)
            return True
        except KeyError:
            return False

    def get_all_internal_ids(self):
        """Get all internal id's of the timeseries data.

        Only returns the id's that are also in the dataconfig."""
        all_ids = [var for var, _ in self.pi_timeseries.items() if self._is_in_dataconfig(var)]
        return all_ids

    def select_time_range(self, start_date: datetime.datetime, end_date: datetime.datetime):
        times = self.pi_timeseries.times
        i_start = bisect.bisect_left(times, start_date)
        i_end = bisect.bisect_right(times, end_date) - 1
        self.pi_timeseries.resize(start_datetime=times[i_start], end_datetime=times[i_end])
        self.pi_timeseries.times = times[i_start : i_end + 1]
        self.pi_timeseries.forecast_datetime = times[i_start]
        self.forecast_date = times[i_start]

    def write(self, file_path: Path, file_name: str = "timeseries_import"):
        # By setting make_new_file headers will be recreated,
        # neceesary for writing new forecast date
        self.pi_timeseries.make_new_file = True
        self.pi_timeseries.write(output_folder=file_path, output_filename=file_name)

    def get_datetimes(self):
        """Get the dates of all timeseries data."""
        return self.pi_timeseries.times

    def get_datetime_range(self):
        """Get the date range of the timeseries data, minimum and maximum over all series"""
        times = self.pi_timeseries.times
        return times[0], times[-1]

    def get_timestep(self):
        """Get the timestep of the timeseries data, raise error if different stepsizes"""
        return self.pi_timeseries.dt

    def set_initial_value(self, internal_id: str, value: float):
        self.pi_timeseries.set(internal_id, [value])
