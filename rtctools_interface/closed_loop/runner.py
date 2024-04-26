import datetime
import os
import copy
from pathlib import Path
import shutil
import sys
from typing import List, Optional
from rtctools.optimization.pi_mixin import PIMixin
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.util import run_optimization_problem, _resolve_folder
from rtctools_interface.closed_loop.closed_loop_dates import read_closed_loop_dates
from rtctools_interface.closed_loop.results_construction import combine_xml_exports
from rtctools_interface.closed_loop.time_series_handler import XMLTimeSeriesFile, CSVTimeSeriesFile, TimeSeriesHandler
import logging

logger = logging.getLogger("rtctools")


def set_initial_values_from_previous_run(
    results_previous_run: Optional[dict],
    timeseries: TimeSeriesHandler,
    previous_run_datetimes: List[datetime.datetime],
    start_date: datetime.datetime,
) -> None:
    """Set initial values from previous run (if any)"""
    if results_previous_run is not None:
        variables_to_set = {key: value for key, value in results_previous_run.items() if not timeseries.is_set(key)}
        if timeseries.forecast_date:
            index_of_initial_value = previous_run_datetimes.index(timeseries.forecast_date)
        else:
            raise ValueError("Could not find forecast date in timeseries import.")
        for key, values in variables_to_set.items():
            if values is not None:
                timeseries.set_initial_value(key, values[index_of_initial_value])
            else:
                raise ValueError(f"Could not find initial value for {key}.")


def write_input_folder(
    modelling_period_input_folder_i: Path,
    original_input_folder: Path,
    timeseries_import: TimeSeriesHandler,
) -> None:
    modelling_period_input_folder_i.mkdir(exist_ok=True)
    for file in original_input_folder.iterdir():
        if file.is_file():
            shutil.copy(file, modelling_period_input_folder_i / file.name)
        elif file.is_dir():
            shutil.copytree(file, modelling_period_input_folder_i / file.name)
    timeseries_import.write(modelling_period_input_folder_i)


def run_optimization_problem_closed_loop(
    optimization_problem_class,
    base_folder="..",
    log_level=logging.INFO,
    profile=False,
    **kwargs,
):
    """Runs an optimization problem in closed loop mode.
    This function is a drop-in replacement for the run_optimization_problem of rtc-tools. The user
    needs to specify a closed_loop_dates.csv in the input folder of the optimization problem. This
    CSV file should contain two columns: start_date and end_date, each row corresponding to one
    modelling period. The function will run optimization problems for each modelling period specified
    in the CSV file, setting the final results from the previous run as initial conditions for the next.
    See the readme.md for more details.

    Notes:
        - The first start_date in your closed_loop_dates.csv should be equal to the start_date of
          your timeseries_import.
        - The different horizons should overlap with at least one day to allow retrieving and setting
          initial values. More days of overlap are allowed.
    """
    base_folder = Path(base_folder)
    if not os.path.isabs(base_folder):
        base_folder = Path(sys.path[0]) / base_folder
    original_input_folder = Path(_resolve_folder(kwargs, base_folder, "input_folder", "input"))
    original_output_folder = Path(_resolve_folder(kwargs, base_folder, "output_folder", "output"))
    original_output_folder.mkdir(exist_ok=True)

    if issubclass(optimization_problem_class, PIMixin):
        original_import = XMLTimeSeriesFile(original_input_folder)
        original_import.set_reference_data(copy.deepcopy(original_import))
    elif issubclass(optimization_problem_class, CSVMixin):
        original_import = CSVTimeSeriesFile(original_input_folder)
    else:
        raise ValueError("Optimization problem class must be derived from PIMixin or CSVMixin.")

    variables_in_import = original_import.get_all_internal_ids()

    modelling_period_input_folder = base_folder / "input_modelling_periods"
    if modelling_period_input_folder.exists():
        shutil.rmtree(modelling_period_input_folder)
    modelling_period_input_folder.mkdir(exist_ok=True)

    modelling_periods_output_folder = original_output_folder / "output_modelling_periods"
    if modelling_periods_output_folder.exists():
        shutil.rmtree(modelling_periods_output_folder)
    modelling_periods_output_folder.mkdir(exist_ok=True)

    original_date_range = original_import.get_datetime_range()
    closed_loop_dates = read_closed_loop_dates(original_input_folder / "closed_loop_dates.csv")
    assert (
        min(closed_loop_dates["start_date"]).date() == original_date_range[0].date()
    ), "The start day of the first optimization run is not equal to the start day of the timeseries import."
    assert (
        max(closed_loop_dates["end_date"]).date() <= original_date_range[1].date()
    ), "The end date of one or more optimization runs is later than the end date of the timeseries import."

    results_previous_run = None
    previous_run_datetimes = None
    for i, date_range in closed_loop_dates.iterrows():
        timeseries_import = copy.deepcopy(original_import)

        timeseries_import.select_time_range(
            start_date=date_range["start_date"],
            end_date=date_range["end_date"] + datetime.timedelta(days=1) - datetime.timedelta(seconds=1),
        )

        set_initial_values_from_previous_run(
            results_previous_run, timeseries_import, previous_run_datetimes, date_range["start_date"]
        )

        modelling_period_name = f"period_{i}"
        modelling_period_output_folder_i = modelling_periods_output_folder / modelling_period_name
        modelling_period_output_folder_i.mkdir(exist_ok=True)
        modelling_period_input_folder_i = modelling_period_input_folder / modelling_period_name

        write_input_folder(modelling_period_input_folder_i, original_input_folder, timeseries_import)

        result = run_optimization_problem(
            optimization_problem_class,
            base_folder,
            log_level,
            profile,
            input_folder=modelling_period_input_folder_i,
            output_folder=modelling_period_output_folder_i,
            **kwargs,
        )
        logger.info(f"Finished optimization run {i} with result {result}")

        results_previous_run = {key: result.extract_results().get(key) for key in variables_in_import}
        previous_run_datetimes = result.io.datetimes
        if len(results_previous_run) != len(variables_in_import):
            logger.warning("Could not find the results for all input variables.")
            logger.warning("Missing variables: " + str(set(variables_in_import) - set(results_previous_run.keys())))
            raise ValueError("Could not find the results for all input variables.")

    logger.info("Finished all optimization runs.")
    if issubclass(optimization_problem_class, PIMixin):
        combine_xml_exports(modelling_periods_output_folder, original_input_folder)
    else:
        logger.info("Combining CSV exports is not yet implemented.")
