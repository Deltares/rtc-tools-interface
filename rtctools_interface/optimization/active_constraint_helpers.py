"""Helper functions for active-constraint diagnostics."""

import csv
from collections.abc import Iterable
from pathlib import Path

import casadi as ca
import numpy as np
from rtctools.optimization.timeseries import Timeseries

from rtctools_interface.utils.type_definitions import PreviousGoalConstraintRow


def as_flat_float_array(value: object) -> np.ndarray:
    """Convert CasADi/numeric values to a one-dimensional float array."""
    if isinstance(value, Timeseries):
        value = value.values
    if isinstance(value, (list, tuple)):
        if not value:
            return np.array([], dtype=float)
        value = ca.veccat(*value)
    array = np.array(value, dtype=float)
    return array.reshape(-1)


def bound_to_array(bound: object, size: int) -> np.ndarray:
    """Return a flat bound array matching an evaluated constraint size."""
    if isinstance(bound, Timeseries):
        bound = bound.values
    array = np.array(bound, dtype=float)
    if array.size == 1 and size != 1:
        return np.full(size, float(array.reshape(-1)[0]))
    return array.reshape(-1)


def component_time(times: np.ndarray | None, component_index: int) -> float | str:
    """Return the time associated with a flattened path-constraint component."""
    if times is None or len(times) == 0:
        return ""
    return times[component_index % len(times)]


def format_active_times(times: np.ndarray | None, active_indices: np.ndarray) -> str:
    """Format unique active timesteps for a flattened path-constraint vector."""
    active_times = [component_time(times, int(index)) for index in active_indices]
    return format_values(active_times)


def format_indexed_values(values: np.ndarray, indices: np.ndarray) -> str:
    """Format values at selected indices for aggregated CSV cells."""
    return format_values([values[index] for index in indices])


def format_values(values: Iterable[object]) -> str:
    """Format unique, non-empty values as a semicolon-separated string."""
    formatted_values = []
    for value in values:
        if value == "" or (isinstance(value, float) and np.isnan(value)):
            continue
        formatted_value = str(value)
        if formatted_value not in formatted_values:
            formatted_values.append(formatted_value)
    return ";".join(formatted_values)


def active_bound_description(
    lower_active: bool, upper_active: bool, lower_bound: float, upper_bound: float
) -> tuple[str, str | float]:
    """Return a text label and value for the bound hit by a constraint."""
    if lower_active and upper_active:
        if np.isclose(lower_bound, upper_bound, rtol=0.0, atol=0.0):
            return "both", lower_bound
        return "both", f"{lower_bound};{upper_bound}"
    if lower_active:
        return "lower", lower_bound
    if upper_active:
        return "upper", upper_bound
    return "", ""


def write_csv(
    file_path: Path, fieldnames: list[str], rows: list[PreviousGoalConstraintRow]
) -> None:
    """Write rows to a CSV file with a stable header."""
    with file_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
