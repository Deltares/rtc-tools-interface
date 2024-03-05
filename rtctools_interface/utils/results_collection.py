"""Mixin to store all required data for plotting."""
import logging
import os
import copy
from pathlib import Path
import time
from typing import Dict, List, Optional

import numpy as np
from rtctools_interface.utils.plot_table_schema import PlotTableRow
from rtctools_interface.utils.read_goals_mixin import ReadGoalsMixin

from rtctools_interface.utils.serialization import deserialize, serialize
from rtctools_interface.utils.read_plot_table import get_plot_config
from rtctools_interface.utils.type_definitions import (
    PlotDataAndConfig,
    PlotOptions,
    PrioIndependentData,
)

MAX_NUM_CACHED_FILES = 5
CONFIG_VERSION: float = 1.0

logger = logging.getLogger("rtctools")


def get_most_recent_cache(cache_folder):
    """Get the most recent pickle file, based on its name."""
    cache_folder = Path(cache_folder)
    json_files = list(cache_folder.glob("*.json"))

    if json_files:
        return max(json_files, key=lambda file: int(file.stem), default=None)
    return None


def clean_cache_folder(cache_folder, max_files=10):
    """Clean the cache folder with pickles, remove the oldest ones when there are more than `max_files`."""
    cache_path = Path(cache_folder)
    files = [f for f in cache_path.iterdir() if f.suffix == ".json"]

    if len(files) > max_files:
        files.sort(key=lambda x: int(x.stem))
        files_to_delete = len(files) - max_files
        for i in range(files_to_delete):
            file_to_delete = cache_path / files[i]
            file_to_delete.unlink()


def write_cache_file(cache_folder: Path, results_to_store: PlotDataAndConfig):
    """Write a file to the cache folder as a pickle file with the linux time as name."""
    os.makedirs(cache_folder, exist_ok=True)
    file_name = int(time.time())
    with open(cache_folder / f"{file_name}.json", "w", encoding="utf-8") as json_file:
        json_file.write(serialize(results_to_store))

    clean_cache_folder(cache_folder, MAX_NUM_CACHED_FILES)


def read_cache_file_from_folder(cache_folder: Path) -> Optional[PlotDataAndConfig]:
    """Read the most recent file from the cache folder."""
    cache_file_path = get_most_recent_cache(cache_folder)
    loaded_data: Optional[PlotDataAndConfig]
    if cache_file_path:
        with open(cache_file_path, "r", encoding="utf-8") as handle:
            loaded_data: dict = deserialize(handle.read())
        if loaded_data.get("config_version", 0) < CONFIG_VERSION:
            logger.warning(
                "The cache file that was found is not supported by the current version of rtc-tools-interface!"
            )
            loaded_data = None
    else:
        loaded_data = None
    return loaded_data


def get_plot_variables(plot_config: list[PlotTableRow]) -> List[str]:
    """Get list of variable-names that are in the plot table."""
    variables_style_1 = [var for subplot_config in plot_config for var in subplot_config.variables_style_1]
    variables_style_2 = [var for subplot_config in plot_config for var in subplot_config.variables_style_2]
    variables_with_previous_result = [
        var for subplot_config in plot_config for var in subplot_config.variables_with_previous_result
    ]
    return variables_style_1 + variables_style_2 + variables_with_previous_result


def filter_plot_config(plot_config: list[PlotTableRow], all_goal_generator_goals) -> list[PlotTableRow]:
    """ "Remove PlotTableRows corresponding to non-existing goals in the goal generator."""
    goal_generator_goal_ids = [goal.goal_id for goal in all_goal_generator_goals]
    new_plot_config = [
        plot_table_row
        for plot_table_row in plot_config
        if plot_table_row.id in goal_generator_goal_ids or plot_table_row.specified_in == "python"
    ]
    return new_plot_config


class PlottingBaseMixin(ReadGoalsMixin):
    """Base class for creating plots.

    Reads the plot table, if available the goal table, and contains functions to store all required data for plots."""

    plot_max_rows = 4
    plot_results_each_priority = True
    plot_final_results = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            plot_table_file = self.plot_table_file
        except AttributeError:
            plot_table_file = os.path.join(self._input_folder, "plot_table.csv")
        plot_config_list = kwargs.get("plot_config_list", [])
        read_from = kwargs.get("read_goals_from", "csv_table")
        self.save_plot_to = kwargs.get("save_plot_to", "image")
        self.plotting_library = kwargs.get("plotting_library", "plotly")
        self.plot_config = get_plot_config(plot_table_file, plot_config_list, read_from)

        self.custom_variables = get_plot_variables(self.plot_config)

        if not hasattr(self, "_all_goal_generator_goals") and self.optimization_problem:
            goals_to_generate = kwargs.get("goals_to_generate", [])
            read_from = kwargs.get("read_goals_from", "csv_table")
            self.load_goals(read_from, goals_to_generate)

        if self.optimization_problem:
            all_goal_generator_goals = self._all_goal_generator_goals
            self.state_variables = list({base_goal.state for base_goal in all_goal_generator_goals})
        else:
            self.state_variables = []
            all_goal_generator_goals = []

        self.plot_config = filter_plot_config(self.plot_config, all_goal_generator_goals)

        self._cache_folder = Path(self._output_folder) / "cached_results"
        if "previous_run_plot_config" in kwargs:
            self._previous_run = kwargs["previous_run_plot_config"]
        else:
            self._previous_run = read_cache_file_from_folder(self._cache_folder)

    def pre(self):
        """Tasks before optimizing."""
        super().pre()
        self._intermediate_results = []

    def collect_timeseries_data(self, all_variables_to_store: List[str]) -> Dict[str, np.ndarray]:
        """Collect the timeseries data for a list of variables."""
        extracted_results = copy.deepcopy(self.extract_results())
        timeseries_data = {}
        for timeseries_name in all_variables_to_store:
            try:
                timeseries_data[timeseries_name] = extracted_results[timeseries_name]
            except KeyError:
                try:
                    timeseries_data[timeseries_name] = self.get_timeseries(timeseries_name)
                except KeyError as exc:
                    raise KeyError("Cannot find timeseries for %s" % timeseries_name) from exc
        return timeseries_data

    def create_plot_data_and_config(self, base_goals: list) -> PlotDataAndConfig:
        """Create the PlotDataAndConfig dict."""
        prio_independent_data: PrioIndependentData = {
            "io_datetimes": self.io.datetimes,
            "times": self.times(),
            "base_goals": base_goals,
        }
        plot_options: PlotOptions = {
            "plot_config": self.plot_config,
            "plot_max_rows": self.plot_max_rows,
            "output_folder": self._output_folder,
            "save_plot_to": self.save_plot_to,
        }
        plot_data_and_config: PlotDataAndConfig = {
            "intermediate_results": self._intermediate_results,
            "plot_options": plot_options,
            "prio_independent_data": prio_independent_data,
            "config_version": CONFIG_VERSION,
        }
        return plot_data_and_config

    def _store_current_results(self, cache_folder, results_to_store):
        write_cache_file(cache_folder, results_to_store)
        self._plot_data_and_config = results_to_store

    @property
    def get_plot_data_and_config(self):
        """Get the plot data and config from the current run."""
        return self._plot_data_and_config
