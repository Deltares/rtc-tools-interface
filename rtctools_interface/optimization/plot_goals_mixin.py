"""Mixin to store all required data for plotting. Can also call the plot function."""
import logging
import os
import copy
from pathlib import Path
import pickle
import time
from typing import Union, List, Dict

import numpy as np

from rtctools_interface.optimization.plot_and_goal_schema import (
    MinimizationGoalCombinedModel,
    MaximizationGoalCombinedModel,
    RangeGoalCombinedModel,
    RangeRateOfChangeGoalCombinedModel,
)
from rtctools_interface.optimization.plot_table_schema import PlotTableRow
from rtctools_interface.optimization.plotting.plot_tools import create_plot_each_priority, create_plot_final_results

from rtctools_interface.optimization.read_plot_table import get_joined_plot_config
from rtctools_interface.optimization.type_definitions import (
    PlotDataAndConfig,
    PlotOptions,
    PrioIndependentData,
    TargetDict,
)

logger = logging.getLogger("rtctools")


def get_most_recent_cache(cache_folder):
    """Get the most recent pickle file, based on its name."""
    cache_folder = Path(cache_folder)
    pickle_files = list(cache_folder.glob("*.pickle"))

    if pickle_files:
        return max(pickle_files, key=lambda file: int(file.stem), default=None)
    return None


def clean_cache_folder(cache_folder, max_files=10):
    """Clean the cache folder with pickles, remove the oldest ones when there are more than `max_files`."""
    cache_path = Path(cache_folder)
    files = [f for f in cache_path.iterdir() if f.suffix == ".pickle"]

    if len(files) > max_files:
        files.sort(key=lambda x: int(x.stem))
        files_to_delete = len(files) - max_files
        for i in range(files_to_delete):
            file_to_delete = cache_path / files[i]
            file_to_delete.unlink()


class PlotGoalsMixin:
    """
    Class for plotting results.
    """

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
        goals_to_generate = kwargs.get("goals_to_generate", [])
        self.save_plot_to = kwargs.get("save_plot_to", "image")
        self.plotting_library = kwargs.get("plotting_library", "plotly")
        self.plot_config = get_joined_plot_config(
            plot_table_file, getattr(self, "goal_table_file", None), plot_config_list, read_from, goals_to_generate
        )

        # Store list of variable-names that may not be present in the results.
        variables_style_1 = [var for subplot_config in self.plot_config for var in subplot_config.variables_style_1]
        variables_style_2 = [var for subplot_config in self.plot_config for var in subplot_config.variables_style_2]
        variables_with_previous_result = [
            var for subplot_config in self.plot_config for var in subplot_config.variables_with_previous_result
        ]
        self.custom_variables = variables_style_1 + variables_style_2 + variables_with_previous_result

    def pre(self):
        """Tasks before optimizing."""
        super().pre()
        self.intermediate_results = []

    def priority_completed(self, priority: int) -> None:
        """Store priority-dependent results required for plotting."""
        extracted_results = copy.deepcopy(self.extract_results())
        results_custom_variables = {
            custom_variable: self.get_timeseries(custom_variable)
            for custom_variable in self.custom_variables
            if custom_variable not in extracted_results
        }
        extracted_results.update(results_custom_variables)
        to_store = {"extract_result": extracted_results, "priority": priority}
        self.intermediate_results.append(to_store)
        super().priority_completed(priority)

    def collect_range_target_values(
        self,
        all_goals: List[
            Union[
                MinimizationGoalCombinedModel,
                MaximizationGoalCombinedModel,
                RangeGoalCombinedModel,
                RangeRateOfChangeGoalCombinedModel,
                PlotTableRow,
            ]
        ],
    ) -> Dict[str, TargetDict]:
        """For the goals with targets, collect the actual timeseries with these targets."""
        t = self.times()

        def get_parameter_ranges(goal):
            try:
                target_min = np.full_like(t, 1) * self.parameters(0)[goal.target_min]
                target_max = np.full_like(t, 1) * self.parameters(0)[goal.target_max]
            except TypeError:
                target_min = np.full_like(t, 1) * self.io.get_parameter(goal.target_min)
                target_max = np.full_like(t, 1) * self.io.get_parameter(goal.target_max)
            return target_min, target_max

        def get_value_ranges(goal):
            target_min = np.full_like(t, 1) * float(goal.target_min)
            target_max = np.full_like(t, 1) * float(goal.target_max)
            return target_min, target_max

        def get_timeseries_ranges(goal):
            if isinstance(goal.target_min, str):
                target_min = self.get_timeseries(goal.target_min).values
            else:
                target_min = np.full_like(t, 1) * goal.target_min
            if isinstance(goal.target_max, str):
                target_max = self.get_timeseries(goal.target_max).values
            else:
                target_max = np.full_like(t, 1) * goal.target_max
            return target_min, target_max

        target_series: Dict[str, TargetDict] = {}
        for goal in all_goals:
            if not goal.specified_in == "python" and goal.goal_type in ["range", "range_rate_of_change"]:
                if goal.target_data_type == "parameter":
                    target_min, target_max = get_parameter_ranges(goal)
                elif goal.target_data_type == "value":
                    target_min, target_max = get_value_ranges(goal)
                elif goal.target_data_type == "timeseries":
                    target_min, target_max = get_timeseries_ranges(goal)
                else:
                    message = "Target type {} not known.".format(goal.target_data_type)
                    logger.error(message)
                    raise ValueError(message)
                target_series[goal.goal_id] = {"target_min": target_min, "target_max": target_max}
        return target_series

    def post(self):
        """Tasks after optimizing. Creates a plot for for each priority."""
        super().post()
        prio_independent_data: PrioIndependentData = {
            "io_datetimes": self.io.datetimes,
            "times": self.times(),
            "target_series": self.collect_range_target_values(self.plot_config),
            "all_goals": self.goals() + self.path_goals(),
        }

        plot_options: PlotOptions = {
            "plot_config": self.plot_config,
            "plot_max_rows": self.plot_max_rows,
            "output_folder": self._output_folder,
            "save_plot_to": self.save_plot_to,
        }

        current_run: PlotDataAndConfig = {
            "intermediate_results": self.intermediate_results,
            "plot_options": plot_options,
            "prio_independent_data": prio_independent_data,
        }

        # load previous results
        cache_folder = Path(self._output_folder) / "cached_results"
        cache_file_path = get_most_recent_cache(cache_folder)
        if cache_file_path:
            with open(cache_file_path, "rb") as handle:
                previous_run: PlotDataAndConfig = pickle.load(handle)
        else:
            previous_run = None

        self.plot_data = {}
        if self.plot_results_each_priority:
            self.plot_data = self.plot_data | create_plot_each_priority(
                current_run, plotting_library=self.plotting_library
            )

        if self.plot_final_results:
            self.plot_data = self.plot_data | create_plot_final_results(
                current_run, previous_run, plotting_library=self.plotting_library
            )

        # Cache results, such that in a next run they can be used for comparison
        os.makedirs(cache_folder, exist_ok=True)
        file_name = int(time.time())
        with open(cache_folder / f"{file_name}.pickle", "wb") as handle:
            pickle.dump(current_run, handle, protocol=pickle.HIGHEST_PROTOCOL)
        clean_cache_folder(cache_folder, 5)
