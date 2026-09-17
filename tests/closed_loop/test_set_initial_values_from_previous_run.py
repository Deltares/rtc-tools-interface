"""Tests for set_initial_values_from_previous_run."""

import datetime
import unittest
from unittest.mock import MagicMock

import pytest

from rtctools_interface.closed_loop.runner import set_initial_values_from_previous_run


class TestSetInitialValuesFromPreviousRun(unittest.TestCase):
    """set_initial_values_from_previous_run should fail loudly when no initial value is
    available for a variable, rather than silently skipping it."""

    def _make_timeseries(self, forecast_date):
        timeseries = MagicMock()
        timeseries.forecast_date = forecast_date
        timeseries.is_set.return_value = False
        return timeseries

    def test_sets_initial_value_when_present(self):
        forecast_date = datetime.datetime(2020, 1, 1)
        timeseries = self._make_timeseries(forecast_date)

        set_initial_values_from_previous_run(
            results_previous_run={"var_a": [1.0, 2.0, 3.0]},
            timeseries=timeseries,
            previous_run_datetimes=[forecast_date],
        )

        timeseries.set_initial_value.assert_called_once_with("var_a", 1.0)

    def test_raises_when_value_missing_for_variable(self):
        """A variable with no value from either extract_results() or get_timeseries() must
        be represented as None, not omitted, so the missing-value check still fires."""
        forecast_date = datetime.datetime(2020, 1, 1)
        timeseries = self._make_timeseries(forecast_date)

        with pytest.raises(ValueError, match="Could not find initial value for var_a"):
            set_initial_values_from_previous_run(
                results_previous_run={"var_a": None},
                timeseries=timeseries,
                previous_run_datetimes=[forecast_date],
            )

    def test_absent_key_is_invisible_to_consumer(self):
        """A key left out of results_previous_run entirely (instead of being set to None)
        cannot be detected here: no error is raised and no initial value is set for it.
        Producers must therefore always set missing values to None rather than omitting the
        key, so this consumer's missing-value check can still fire."""
        forecast_date = datetime.datetime(2020, 1, 1)
        timeseries = self._make_timeseries(forecast_date)

        set_initial_values_from_previous_run(
            results_previous_run={},
            timeseries=timeseries,
            previous_run_datetimes=[forecast_date],
        )

        timeseries.set_initial_value.assert_not_called()
