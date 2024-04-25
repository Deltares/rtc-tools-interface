import copy
import datetime
import logging
from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
from rtctools.data import pi
from rtctools.data import rtc
from rtctools.data import csv

ns = {"fews": "http://www.wldelft.nl/fews", "pi": "http://www.wldelft.nl/fews/PI"}

logger = logging.getLogger("rtctools")


class TimeSeriesHandler(ABC):
    """ABC for handling timeseries data."""

    forecast_date: Optional[datetime.datetime] = None
    has_forecast_date = False

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

    def set_reference_data(self, reference_data: "TimeSeriesHandler"):
        """Set the reference TimeSeriesHandler. Only relevant for XMLTimeSeriesFile.
        Required when setting initial values for variables that are not in the modelling period data range."""

    @abstractmethod
    def get_timestep(self) -> datetime.timedelta:
        """Get the timestep of the timeseries data."""

    @abstractmethod
    def get_datetime_range(self) -> Tuple[datetime.datetime, datetime.datetime]:
        """Get the date range of the timeseries data (min, max)."""

    @abstractmethod
    def get_all_internal_ids(self) -> List[str]:
        """Get all internal id's of the timeseries data."""

    @abstractmethod
    def set_initial_value(self, internal_id: str, value: float) -> None:
        """Set the initial value of a variable in the timeseries data."""

    @abstractmethod
    def is_set(self, internal_id: str) -> bool:
        """Check whether the variable exists in the timeseries data and whether it has a least one non-nan value"""


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
            self.has_forecast_date = True
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
                self.initial_state: Optional[dict] = {
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
            initial_state.to_csv((file_path / file_name).with_suffix(".csv"), header=True, index=False)

    def get_timestep(self):
        return self.data[self.date_col].diff().min()

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

    def __parse_date_time(self, el):
        return datetime.datetime.strptime(el.get("date") + " " + el.get("time"), "%Y-%m-%d %H:%M:%S")

    def __init__(
        self,
        input_folder: Path,
        timeseries_import_basename: str = "timeseries_import",
    ):
        self.input_folder = input_folder
        self.pi_binary_timeseries = False
        self.pi_validate_timeseries = True
        self.read(timeseries_import_basename)

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
        tree = self.pi_timeseries._Timeseries__tree
        self.root = tree.getroot()
        if self.root is None:
            raise ValueError("No data to read.")
        self.set_forecast_date()

    def set_forecast_date(self):
        first_series = self.root.find("pi:series", ns)
        first_header = first_series.find("pi:header", ns)
        forecast_date_element = first_header.find("pi:forecastDate", ns)
        if forecast_date_element is not None:
            self.forecast_date = self.__parse_date_time(forecast_date_element)
            self.has_forecast_date = True
        else:
            first_event = first_series.find("pi:event", ns)
            self.forecast_date = self.__parse_date_time(first_event)
            self.has_forecast_date = False

    def is_set(self, internal_id):
        """Check whether the variable exists in the timeseries data and whether it has a value at at least
        one of time steps."""
        location_id, parameter_id, qualifier_id = self.get_external_id(internal_id)
        for series in self.root.findall("pi:series", ns):
            if (
                location_id,
                parameter_id,
                qualifier_id,
            ) == self.get_location_parameter_from_series(series):
                events = series.findall("pi:event", ns)
                for event in events:
                    if event.get("value") is not None:
                        return True
        return False

    def get_internal_id(self, series):
        """Get the internal id of a series element."""
        pi_header = series.find("pi:header", ns)
        return self.data_config.variable(pi_header)

    def get_external_id(self, internal_id):
        """Get the external id of a series element. Returns a tuple with three elements: location, parameter
        and qualifier ID's."""
        return self.data_config.pi_variable_ids(internal_id)

    def get_all_internal_ids(self):
        """Get all internal id's of the timeseries data. Only returns the id's that are also in the dataconfig"""
        all_ids = [self.get_internal_id(series) for series in self.root.findall("pi:series", ns)]
        return [
            id for id in all_ids if ":" not in id
        ]  # slight workaround, this would mean variables cannot contain ":"

    def set_new_forecast_date(self, forecast_date: datetime.datetime):
        for series in self.root.findall("pi:series", ns):
            header = series.find("pi:header", ns)
            start_date, end_date = self.get_single_date_range_from_series(series)
            if start_date <= forecast_date <= end_date:
                header.find("pi:forecastDate", ns).attrib = {
                    "date": forecast_date.strftime("%Y-%m-%d"),
                    "time": forecast_date.strftime("%H:%M:%S"),
                }
            else:
                raise ValueError("Forecast date is not within the date range of the timeseries data.")

    def select_time_range(self, start_date: datetime.datetime, end_date: datetime.datetime):
        assert isinstance(start_date, datetime.datetime) and isinstance(
            end_date, datetime.datetime
        ), "Dates must be datetime objects"
        for series in self.root.findall("pi:series", ns):
            new_start_date, new_end_date = datetime.datetime.max, datetime.datetime.min
            events = series.findall("pi:event", ns)
            for event in events:
                event_datetime = self.__parse_date_time(event)
                if event_datetime < start_date or event_datetime > end_date:
                    series.remove(event)
                    continue
                if event_datetime < new_start_date:
                    new_start_date = event_datetime
                if event_datetime > new_end_date:
                    new_end_date = event_datetime
            start_date_attrib = {
                "date": new_start_date.strftime("%Y-%m-%d"),
                "time": new_start_date.strftime("%H:%M:%S"),
            }
            end_date_attrib = {
                "date": new_end_date.strftime("%Y-%m-%d"),
                "time": new_end_date.strftime("%H:%M:%S"),
            }
            series.find("pi:header", ns).find("pi:startDate", ns).attrib = start_date_attrib
            series.find("pi:header", ns).find("pi:endDate", ns).attrib = end_date_attrib
            if self.has_forecast_date:
                series.find("pi:header", ns).find("pi:forecastDate", ns).attrib = start_date_attrib
            if not series.findall("pi:event", ns):
                self.root.remove(series)
                logger.warning("Removed series with no events.")
            self.set_forecast_date()

    def write(self, file_path: Path, file_name: str = "timeseries_import"):
        tree = ET.ElementTree(self.root)
        tree.write((file_path / file_name).with_suffix(".xml"))

    def get_location_parameter_from_series(self, series) -> Tuple[str, str, List[str]]:
        header = series.find("pi:header", ns)
        locationId = header.find("pi:locationId", ns).text
        parameterId = header.find("pi:parameterId", ns).text
        qualifier_ids = []
        qualifiers_els = header.findall("pi:qualifierId", ns)
        for qualifier in qualifiers_els:
            qualifier_ids.append(qualifier.text)
        return locationId, parameterId, qualifier_ids

    def get_single_date_range(self, locationId, paramterId, qualifier_ids):
        """Get the date range of the timeseries data for a single location and parameter"""
        min_date = datetime.datetime.max
        max_date = datetime.datetime.min
        for series in self.root.findall("pi:series", ns):
            if (
                locationId,
                paramterId,
                qualifier_ids,
            ) == self.get_location_parameter_from_series(series):
                events = series.findall("pi:event", ns)
                for event in events:
                    event_datetime = self.__parse_date_time(event)
                    if event_datetime < min_date:
                        min_date = event_datetime
                    if event_datetime > max_date:
                        max_date = event_datetime
        if min_date == datetime.datetime.max:
            raise ValueError(
                "No data for locationId {},  parameterId {} and qualifierIds {}".format(
                    locationId, paramterId, qualifier_ids
                )
            )
        return min_date, max_date

    def get_single_date_range_from_series(self, series):
        (
            location_id,
            parameter_id,
            qualifier_ids,
        ) = self.get_location_parameter_from_series(series)
        return self.get_single_date_range(location_id, parameter_id, qualifier_ids)

    def get_datetime_range(self):
        """Get the date range of the timeseries data, minimum and maximum over all series"""
        min_date = datetime.datetime.max
        max_date = datetime.datetime.min
        for series in self.root.findall("pi:series", ns):
            events = series.findall("pi:event", ns)
            for event in events:
                event_datetime = self.__parse_date_time(event)
                if event_datetime < min_date:
                    min_date = event_datetime
                if event_datetime > max_date:
                    max_date = event_datetime
        return min_date, max_date

    def get_timestep(self):
        """Get the timestep of the timeseries data, raise error if different stepsizes"""
        timestep = None
        for series in self.root.findall("pi:series", ns):
            events = series.findall("pi:event", ns)
            for i, event in enumerate(events):
                if i == 0:
                    continue
                event_datetime = self.__parse_date_time(event)
                previous_event_datetime = self.__parse_date_time(events[i - 1])
                if timestep is None:
                    timestep = event_datetime - previous_event_datetime
                elif timestep != event_datetime - previous_event_datetime:
                    raise ValueError("Different timesteps in timeseries data.")
        return timestep

    def set_reference_data(self, reference_data):
        """Set the the reference XMLTimeSeriesFile object to use for creating new series."""
        self._reference_data = reference_data

    def get_series(
        self,
        location_id: str,
        parameter_id: str,
        qualifier_ids: List[str],
        from_reference=False,
    ) -> ET.Element:
        """Get the XML series element for a location and parameter.
        If from_reference is True, the series is taken from the reference data."""
        if from_reference:
            if hasattr(self, "_reference_data"):
                root = self._reference_data.root
            else:
                raise ValueError("No reference data available.")
        elif hasattr(self, "root") and self.root is not None:
            root = self.root
        else:
            raise ValueError("No data to select from, use read first.")
        for series in root.findall("pi:series", ns):
            if (
                location_id,
                parameter_id,
                qualifier_ids,
            ) == self.get_location_parameter_from_series(series):
                return series
        raise ValueError("No series found for locationId {} and parameterId {}".format(location_id, parameter_id))

    def set_initial_value(self, internal_id: str, value: float):
        location_id, parameter_id, qualifier_ids = self.get_external_id(internal_id)
        for series in self.root.findall("pi:series", ns):
            if (
                location_id,
                parameter_id,
                qualifier_ids,
            ) == self.get_location_parameter_from_series(series):
                logger.info(
                    "Overwriting initial value for locationId {} and parameterId {} and qualifier ids {}".format(
                        location_id, parameter_id, qualifier_ids
                    )
                )
                event = series.find("pi:event", ns)
                event.attrib["value"] = str(value)
                return
        else:
            # if no series found, create a new series with just one event (based on the reference data)
            reference_series = self.get_series(location_id, parameter_id, qualifier_ids, from_reference=True)
            new_series = copy.deepcopy(reference_series)
            start_date, end_date = self.get_datetime_range()
            first_event = new_series.find("pi:event", ns)
            if first_event is None:
                raise ValueError("No event found in reference data.")
            first_event.attrib["value"] = str(value)
            first_event.attrib["date"] = start_date.strftime("%Y-%m-%d")
            first_event.attrib["time"] = start_date.strftime("%H:%M:%S")
            header = new_series.find("pi:header", ns)
            if header is None:
                raise ValueError("No header found in reference data.")
            header.find("pi:startDate", ns).attrib = {
                "date": start_date.strftime("%Y-%m-%d"),
                "time": start_date.strftime("%H:%M:%S"),
            }
            header.find("pi:endDate", ns).attrib = {
                "date": end_date.strftime("%Y-%m-%d"),
                "time": end_date.strftime("%H:%M:%S"),
            }
            if self.has_forecast_date:
                header.find("pi:forecastDate", ns).attrib = {
                    "date": start_date.strftime("%Y-%m-%d"),
                    "time": start_date.strftime("%H:%M:%S"),
                }
            self.root.append(new_series)
