"""Module for configuring a closed-loop optimization problem."""
from pathlib import Path
from datetime import timedelta


class ClosedLoopConfig():
    """Configuration of a closed-loop optimization problem."""

    def __init__(
        self,
        file: Path = None,
        round_to_dates: bool = False,
    ):
        self._file = file
        self._forecast_timestep = None
        self._optimization_period = None
        self.round_to_dates = round_to_dates

    @classmethod
    def from_fixed_periods(
        cls,
        forecast_timestep: timedelta,
        optimization_period: timedelta,
        round_to_dates: bool = False,
    ):
        """Create a closed loop configuration based on fixed periods."""
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
