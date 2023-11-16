"""Module for reading goals from a csv file."""
import logging
from pathlib import Path
from typing import List, Union
import pandas as pd

from rtctools_interface.optimization.plot_table_schema import PlotTableRow

logger = logging.getLogger("rtctools")


def read_plot_config_from_csv(plot_table_file: Union[Path, str]) -> List[PlotTableRow]:
    """Read plot information from csv file and check values"""
    plot_table_file = Path(plot_table_file)
    if plot_table_file.is_file():
        try:
            raw_plot_table = pd.read_csv(plot_table_file, sep=",")
        except pd.errors.EmptyDataError:  # Empty plot table
            raw_plot_table = pd.DataFrame()
        parsed_rows: List[PlotTableRow] = []
        for _, row in raw_plot_table.iterrows():
            parsed_rows.append(PlotTableRow(**row))
        return parsed_rows
    message = (
        f"No plot table was found at the default location ({plot_table_file.resolve()})."
        + " Please create one before using the PlotGoalsMixin."
        + f" It should have the following columns: '{list(PlotTableRow.model_fields.keys())}'"
    )
    raise FileNotFoundError(message)


def read_plot_config_from_list(plot_config: List[PlotTableRow]) -> List[PlotTableRow]:
    """Read plot config from a list. Validates whether the elements are of correct type."""
    if not isinstance(plot_config, list):
        raise TypeError(f"Pass a list of PlotTableRow elements, not a {type(plot_config)}")
    for plot_table_row in plot_config:
        if not isinstance(plot_table_row, PlotTableRow):
            raise TypeError("Each element in the passed plot table should be of type 'PlotTableRow'")
    return plot_config


def get_plot_config(plot_table_file=None, plot_config_list=None, read_from="csv_table"):
    """Get plot config rows."""
    if read_from == "csv_table":
        return read_plot_config_from_csv(plot_table_file)
    if read_from == "passed_list":
        return read_plot_config_from_list(plot_config_list)
    raise ValueError("PlotGoalsMixin should either read from 'csv_table' or 'passed_list'")


def get_joined_plot_config(plot_table_file, plot_config_list, read_from) -> list[PlotTableRow]:
    """Read plot table for PlotGoals and merge with goals table"""
    plot_table = get_plot_config(
        plot_table_file=plot_table_file, plot_config_list=plot_config_list, read_from=read_from
    )

    return plot_table
