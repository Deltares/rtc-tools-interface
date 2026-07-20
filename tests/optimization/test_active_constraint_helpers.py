"""Tests for active constraint helper functions."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


def _load_active_constraint_helpers():
    """Load active constraint helpers with runtime dependencies stubbed."""
    module_names = [
        "casadi",
        "rtctools",
        "rtctools.optimization",
        "rtctools.optimization.timeseries",
        "rtctools_interface",
        "rtctools_interface.utils",
        "rtctools_interface.utils.type_definitions",
    ]
    previous_modules = {name: sys.modules.get(name) for name in module_names}

    casadi = types.ModuleType("casadi")
    casadi.veccat = lambda *values: values
    casadi.DM = types.SimpleNamespace(zeros=lambda size: np.zeros(size))

    timeseries = types.ModuleType("rtctools.optimization.timeseries")
    timeseries.Timeseries = type("Timeseries", (), {})

    type_definitions = types.ModuleType("rtctools_interface.utils.type_definitions")
    type_definitions.PreviousGoalConstraintRow = dict

    sys.modules["casadi"] = casadi
    sys.modules["rtctools"] = types.ModuleType("rtctools")
    sys.modules["rtctools.optimization"] = types.ModuleType("rtctools.optimization")
    sys.modules["rtctools.optimization.timeseries"] = timeseries
    sys.modules["rtctools_interface"] = types.ModuleType("rtctools_interface")
    sys.modules["rtctools_interface.utils"] = types.ModuleType("rtctools_interface.utils")
    sys.modules["rtctools_interface.utils.type_definitions"] = type_definitions

    try:
        helper_path = (
            Path(__file__).parents[2]
            / "rtctools_interface"
            / "optimization"
            / "active_constraint_helpers.py"
        )
        spec = importlib.util.spec_from_file_location(
            "active_constraint_helpers_under_test", helper_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module

    return module


active_constraint_helpers = _load_active_constraint_helpers()


class TestActiveConstraintHelpers(unittest.TestCase):
    """Test active constraint helper functions."""

    def test_format_active_times_maps_vector_components_to_timesteps(self):
        """Map flattened vector-valued path-goal components to their timestep."""
        times = np.array([10.0, 20.0, 30.0])
        active_indices = np.array([1, 2, 4])

        self.assertEqual(
            active_constraint_helpers.format_active_times(times, active_indices, values_size=6),
            "10.0;20.0;30.0",
        )

    def test_format_active_times_rejects_uneven_component_count(self):
        """Reject path-goal values that cannot be evenly mapped to timesteps."""
        times = np.array([10.0, 20.0, 30.0])
        active_indices = np.array([0])

        with self.assertRaisesRegex(ValueError, "not divisible"):
            active_constraint_helpers.format_active_times(times, active_indices, values_size=5)
