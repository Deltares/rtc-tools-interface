"""Tests for reading goals from a csv file."""
import pathlib
import unittest

from rtctools_interface.optimization.read_goals import read_goals


CSV_FILE = pathlib.Path(__file__).parent.parent / "data" / "goals" / "basic.csv"


class TestGoalReader(unittest.TestCase):
    """Test for reading goals."""

    def test_read_csv(self):
        """Test for reading goals from csv."""
        goals = read_goals(CSV_FILE, path_goal=True)
        self.assertEqual(len(goals), 2)
