import copy
import datetime
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

from rtctools.data.pi import DiagHandler
from rtctools.optimization.pi_mixin import PIMixin
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.util import run_optimization_problem, _resolve_folder
from rtctools_interface.closed_loop.config import ClosedLoopConfig
import rtctools_interface.closed_loop.optimization_ranges as opt_ranges
from rtctools_interface.closed_loop.results_construction import combine_csv_exports, combine_xml_exports
from rtctools_interface.closed_loop.time_series_handler import XMLTimeSeriesFile, CSVTimeSeriesFile, TimeSeriesHandler

logger = logging.getLogger("rtctools")


def set_initial_values_from_previous_run(
    results_previous_run: Optional[dict],
    timeseries: TimeSeriesHandler,
    previous_run_datetimes: List[datetime.datetime],
) -> None:
    """Modifies the initial values of `timeseries` based on the results of the previous run (if any)"""
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
    """Write the input folder for the current modelling period.
    Copies the original input folder to the modelling period input folder and writes the new
    timeseries_import."""
    modelling_period_input_folder_i.mkdir(exist_ok=True)
    for file in original_input_folder.iterdir():
        if file.is_file():
            shutil.copy(file, modelling_period_input_folder_i / file.name)
        elif file.is_dir():
            shutil.copytree(file, modelling_period_input_folder_i / file.name)
    timeseries_import.write(modelling_period_input_folder_i)


def _get_optimization_ranges(
    config: ClosedLoopConfig,
    input_timeseries: TimeSeriesHandler,
) -> list[tuple[datetime.datetime, datetime.datetime]]:
    """Return a list of optimization periods."""
    if config.file is not None:
        datetime_range = input_timeseries.get_datetime_range()
        optimization_ranges = opt_ranges.get_optimization_ranges_from_file(config.file, datetime_range)
    elif config.optimization_period is not None:
        datetimes = input_timeseries.get_datetimes()
        optimization_ranges = opt_ranges.get_optimization_ranges(
            model_times=datetimes,
            start_time=datetimes[0],
            forecast_timestep=config.forecast_timestep,
            optimization_period=config.optimization_period,
        )
    else:
        raise ValueError("The closed-loop configuration should have either a file or optimization_period set.")
    if config.round_to_dates:
        optimization_ranges = opt_ranges.round_datetime_ranges_to_days(optimization_ranges)
    return optimization_ranges


def run_optimization_problem_closed_loop(
    optimization_problem_class,
    base_folder="..",
    log_level=logging.INFO,
    profile=False,
    config: Optional[ClosedLoopConfig] = None,
    modelling_period_input_folder: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Runs an optimization problem in closed loop mode.

    This function is a drop-in replacement for the run_optimization_problem of rtc-tools.
    The user needs to specify a closed-loop configuration that describes the time ranges
    for which to subsequentially solve the optimization problem.
    The results from the previous run will be used to set the initial values of the next run.
    See README.md for more details.
    """
    base_folder = Path(base_folder)
    if not os.path.isabs(base_folder):
        base_folder = Path(sys.path[0]) / base_folder
    original_input_folder = Path(_resolve_folder(kwargs, base_folder, "input_folder", "input"))
    original_output_folder = Path(_resolve_folder(kwargs, base_folder, "output_folder", "output"))
    original_output_folder.mkdir(exist_ok=True)

    # Set logging handlers.
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if issubclass(optimization_problem_class, PIMixin):
        if not any((isinstance(h, DiagHandler) for h in logger.handlers)):
            diag_handler = DiagHandler(original_output_folder)
            logger.addHandler(diag_handler)
    logger.setLevel(log_level)

    if issubclass(optimization_problem_class, PIMixin):
        original_import = XMLTimeSeriesFile(original_input_folder)
    elif issubclass(optimization_problem_class, CSVMixin):
        original_import = CSVTimeSeriesFile(original_input_folder)
    else:
        raise ValueError("Optimization problem class must be derived from PIMixin or CSVMixin.")

    variables_in_import = original_import.get_all_internal_ids()

    fixed_input_config_file = (original_input_folder / "fixed_inputs.json").resolve()
    if not fixed_input_config_file.exists():
        raise FileNotFoundError(
            f"Could not find fixed inputs configuration file: {fixed_input_config_file}"
            "Create a file with a list of strings that represent the fixed inputs (can be an empty list)."
        )
    with open(fixed_input_config_file, "r") as file:
        fixed_input_series = json.load(file)
    if not isinstance(fixed_input_series, list) and all(isinstance(item, str) for item in fixed_input_series):
        raise ValueError("Fixed input config file should be a list of strings (or an empty list).")

    if modelling_period_input_folder is None:
        modelling_period_input_folder = base_folder / "input_modelling_periods"
    if modelling_period_input_folder.exists():
        shutil.rmtree(modelling_period_input_folder)
    modelling_period_input_folder.mkdir(exist_ok=True)

    modelling_periods_output_folder = original_output_folder / "output_modelling_periods"
    if modelling_periods_output_folder.exists():
        shutil.rmtree(modelling_periods_output_folder)
    modelling_periods_output_folder.mkdir(exist_ok=True)

    if config is None:
        config = ClosedLoopConfig(original_input_folder / "closed_loop_dates.csv")
    optimization_ranges = _get_optimization_ranges(config, original_import)
    results_previous_run = None
    previous_run_datetimes = None
    for i, (start_time, end_time) in enumerate(optimization_ranges):
        timeseries_import = copy.deepcopy(original_import)

        timeseries_import.select_time_range(start_date=start_time, end_date=end_time)

        set_initial_values_from_previous_run(results_previous_run, timeseries_import, previous_run_datetimes)

        modelling_period_name = f"period_{i}"
        modelling_period_output_folder_i = modelling_periods_output_folder / modelling_period_name
        modelling_period_output_folder_i.mkdir(exist_ok=True)
        modelling_period_input_folder_i = modelling_period_input_folder / modelling_period_name
        write_input_folder(modelling_period_input_folder_i, original_input_folder, timeseries_import)

        logger.info(f"Running optimization for period {i}: {(str(start_time), str(end_time))}.")
        run_number_in_fallback_list = 1
        # Create a second fallback run with parameter
        for k in range(2):
            if run_number_in_fallback_list < 3:
                result = run_optimization_problem(
                optimization_problem_class,
                base_folder,
                log_level,
                profile,
                input_folder=modelling_period_input_folder_i,
                output_folder=modelling_period_output_folder_i,
                run_number_in_fallback_list = run_number_in_fallback_list,
                **kwargs,
                )
                if hasattr(result, 'run_number_in_fallback_list'):
                   run_number_in_fallback_list = result.run_number_in_fallback_list
                else:
                    run_number_in_fallback_list = 3
                if run_number_in_fallback_list == 2:
                    logger.info("The model failed, it will run it again a fallback option")
        period = f"period {i} {(str(start_time), str(end_time))}"
        if result.solver_stats["success"]:
            logger.info(f"Successful optimization for {period}.")
        else:
            message = f"Failed optimization for {period} with status '{result.solver_stats['return_status']}'."
            logger.error(message)
            raise Exception(message)

        results_previous_run = {}
        for key in variables_in_import:
            if key not in fixed_input_series:
                extracted_value = result.extract_results().get(key)
                if extracted_value is not None:
                    results_previous_run[key] = extracted_value
                else:
                    try:
                        fallback_value = result.get_timeseries(key)
                    except KeyError:
                        fallback_value = None
                    if fallback_value is not None:
                        results_previous_run[key] = fallback_value.values
        previous_run_datetimes = result.io.datetimes

    logger.info("Finished all optimization runs.")
    if issubclass(optimization_problem_class, PIMixin):
        combine_xml_exports(modelling_periods_output_folder, original_input_folder, write_csv_out=True)
    elif issubclass(optimization_problem_class, CSVMixin):
        combine_csv_exports(modelling_periods_output_folder)
    else:
        logger.warning(
            "Could not combine exports because the optimization problem class is not derived from PIMixin or CSVMixin."
        )
    return result.solver_stats
