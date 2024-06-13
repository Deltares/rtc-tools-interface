"""Tests for getting optimization periods."""
import unittest
from datetime import datetime, timedelta

import rtctools_interface.closed_loop.optimization_ranges as opt_ranges


class TestGetOptimizationPeriods(unittest.TestCase):
    """Test calculating optimization periods."""
    def check_ranges(
        self,
        ranges: list[tuple[datetime, datetime]],
        expected_ranges: list[tuple[datetime, datetime]]
    ):
        """Check if two lists of ranges are the same."""
        n_exp_ranges = len(expected_ranges)
        n_ranges = len(ranges)
        self.assertEqual(n_ranges, n_exp_ranges)
        for i_period in range(n_ranges):
            exp_start, exp_end = expected_ranges[i_period]
            start, end = expected_ranges[i_period]
            self.assertEqual(start, exp_start)
            self.assertEqual(end, exp_end)

    def test_get_optimization_ranges(self):
        """Test get_optimization_ranges."""
        model_times = [
            datetime(2024,1,1),
            datetime(2024,1,2),
            datetime(2024,1,4),
            datetime(2024,1,5),
            datetime(2024,1,7),
            datetime(2024,1,8),
            datetime(2024,1,10),
        ]
        start_time = datetime(2024,1,2)
        forecast_timestep = timedelta(days=2)
        optimization_period = timedelta(days=5)
        ranges = opt_ranges.get_optimization_ranges(
            model_times=model_times,
            start_time=start_time,
            forecast_timestep=forecast_timestep,
            optimization_period=optimization_period
        )
        expected_ranges = [
            (datetime(2024,1,2), datetime(2024,1,7)),
            (datetime(2024,1,4), datetime(2024,1,8)),
            (datetime(2024,1,5), datetime(2024,1,10)),
        ]
        self.check_ranges(ranges, expected_ranges)

    def test_round_datetime_ranges_to_days(self):
        """Test round_datetime_ranges_to_days."""
        ranges = [
            (datetime(2020,1,1), datetime(2020,1,2)),
            (datetime(2020,1,2), datetime(2020,1,2)),
            (datetime(2020,1,2,12), datetime(2020,1,5,23)),
        ]
        rounded_ranges = opt_ranges.round_datetime_ranges_to_days(ranges)
        expected_ranges = [
            (datetime(2020,1,1), datetime(2020,1,2,23,59,59,999999)),
            (datetime(2020,1,2,12), datetime(2020,1,5,23,59,59,999999)),
        ]
        self.check_ranges(rounded_ranges, expected_ranges)
