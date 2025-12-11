"""Module for configuring a closed-loop optimization problem."""

from datetime import timedelta
from pathlib import Path


class ClosedLoopConfig:
    """Configuration of a closed-loop optimization problem."""

    def __init__(
        self,
        file: Path = None,
        round_to_dates: bool = False,
    ):
        """
        Create a configuration for closed-loop optimization.

        :param file: CSV file with two columns 'start_date' and 'end_date'.
            Each row indicates a time range for which to optimize a given problem.
            Note:
            * The start time of the first time range should coincide with the
              start time of the input timeseries.
            * The start time of the next time range should be less or equal
              to the end time of the current time range.
        :param round_to_dates: If true, the start time and end time of each time range
            is rounded to just the date.
            In particular, the start time is rounded to the start of the day
            and the end time is rounded to the end of the day.
        """
        if file is not None:
            file = Path(file).resolve()
        self._file = file
        self._forecast_timestep: timedelta | None = None
        self._optimization_period: timedelta | None = None
        self.round_to_dates = round_to_dates

    @classmethod
    def from_fixed_periods(
        cls,
        forecast_timestep: timedelta,
        optimization_period: timedelta,
        round_to_dates: bool = False,
    ):
        """
        Create a closed loop configuration based on fixed periods.

        :param forecast_timestep: The time between the start of each optimization time range.
        :param optimization_period: the duration of each optimization time range
        """
        if forecast_timestep > optimization_period:
            raise ValueError(
                f"The forecast timestep ({forecast_timestep}) should be less than or equal to"
                f" the optimization period ({optimization_period})."
            )
        config = cls()
        config._forecast_timestep = forecast_timestep
        config._optimization_period = optimization_period
        config.round_to_dates = round_to_dates
        return config

    @property
    def file(self):
        """Get the file that defines the closed-loop periods."""
        return self._file

    @property
    def forecast_timestep(self):
        """Get the forecast timestep of the closed-loop periods."""
        return self._forecast_timestep

    @property
    def optimization_period(self):
        """Get the optimization period of the closed-loop periods."""
        return self._optimization_period
