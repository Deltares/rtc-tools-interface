import copy
import logging
import os
from pathlib import Path
from typing import List
import pandas as pd
from rtctools.data import rtc
from rtctools.data import pi

logger = logging.getLogger("rtctools")


def _get_variables_from_pi(data_config: rtc.DataConfig, timeseries: pi.Timeseries):
    """Get all variables of a PI timeseries that are in the data configuration."""
    variables = []
    for var, _ in timeseries.items():
        try:
            data_config.pi_variable_ids(var)
            variables.append(var)
        except KeyError:
            pass
    return variables


def combine_xml_exports(output_base_path: Path, original_input_timeseries_path: Path, write_csv_out: bool = False):
    """Combine the xml exports of multiple periods into a single xml file."""
    logger.info("Combining XML exports.")
    dataconfig = rtc.DataConfig(folder=original_input_timeseries_path)

    ts_import_orig = pi.Timeseries(
        data_config=dataconfig,
        folder=original_input_timeseries_path,
        basename="timeseries_import",
        binary=False,
    )
    if ts_import_orig.forecast_datetime > ts_import_orig.start_datetime:
        logger.info("Timeseries export will start at original forecast date, disregarding data before forecast date.")
        ts_import_orig.resize(ts_import_orig.forecast_datetime, ts_import_orig.end_datetime)
        ts_import_orig.times = ts_import_orig.times[ts_import_orig.times.index(ts_import_orig.forecast_datetime):]
    orig_start_datetime = ts_import_orig.start_datetime
    orig_end_datetime = ts_import_orig.end_datetime

    ts_export = pi.Timeseries(
        data_config=dataconfig, folder=output_base_path / "period_0", basename="timeseries_export", binary=False
    )  # Use the first timeseries export as a starting point for the combined timeseries export.
    ts_export.resize(orig_start_datetime, orig_end_datetime)

    variables = _get_variables_from_pi(data_config=dataconfig, timeseries=ts_export)

    i = 0
    while os.path.isfile(os.path.join(output_base_path, f"period_{i}", "timeseries_export.xml")):
        ts_export_step = pi.Timeseries(
            data_config=dataconfig,
            folder=os.path.join(output_base_path, f"period_{i}"),
            basename="timeseries_export",
            binary=False,
        )
        all_times = ts_import_orig.times  # Workaround to map indices to times, as ts_export does
        # not contain all times. TODO Check whether the assumption that these times map to
        # the correct indices for ts_export always holds.
        for loc_par in variables:
            try:
                current_values = ts_export.get(loc_par)
                new_values = ts_export_step.get(loc_par)
            except KeyError:
                logger.debug("Variable {} not found in output of model horizon: {}".format(loc_par, i))
                continue
            new_times = ts_export_step.times
            try:
                start_new_data_index = all_times.index(new_times[0])
            except ValueError:
                if all_times[-1] + ts_export.dt == new_times[0]:
                    start_new_data_index = len(all_times)
                else:
                    raise ValueError(
                        "Could not match the start data of the timeseries export file "
                        + "with the end of the previous."
                    )
            combined_values = copy.deepcopy(current_values)
            combined_values[start_new_data_index : start_new_data_index + len(new_values)] = new_values  # noqa
            ts_export.set(loc_par, combined_values)
        i += 1
    ts_export.write(output_folder=output_base_path.parent, output_filename="timeseries_export")

    if write_csv_out:
        data = pd.DataFrame()
        data["date"] = all_times
        for timeseries_id in variables:
            try:
                values = ts_export.get(timeseries_id)
            except KeyError:
                logger.debug("Variable {} not found in output of model horizon: {}".format(timeseries_id, i))
                continue
            data[timeseries_id] = values
        data.round(6).to_csv(output_base_path.parent / "timeseries_export.csv", index=False)


def combine_dataframes(dfs: List[pd.DataFrame], index_col: str = "time"):
    """Combine multiple dataframes with the same index column.
    The dataframes are combined in the order they are passed, with the last dataframe taking precedence
    in case of overlapping indices."""
    combined_df = pd.DataFrame()
    for df in dfs:
        df.set_index(index_col, inplace=True)
        combined_df = df.combine_first(combined_df)
    combined_df.reset_index(inplace=True)
    return combined_df


def combine_csv_exports(output_base_path: Path):
    """Combine the csv exports of multiple periods into a single csv file."""
    i = 0
    dfs = []
    while os.path.isfile(os.path.join(output_base_path, f"period_{i}", "timeseries_export.csv")):
        df = pd.read_csv(os.path.join(output_base_path, f"period_{i}", "timeseries_export.csv"))
        dfs.append(df)
        i += 1
    combined_df = combine_dataframes(dfs)
    combined_df.round(6).to_csv(output_base_path.parent / "timeseries_export.csv", index=False)


if __name__ == "__main__":
    closed_loop_test_folder = Path(__file__).parents[2] / "tests" / "closed_loop"
    output_base_path = closed_loop_test_folder / Path(
        r"test_models\goal_programming_xml\output\output_modelling_periods_reference"
    )
    original_input_timeseries_path = closed_loop_test_folder / Path(r"test_models\goal_programming_xml\input")
    combine_xml_exports(output_base_path, original_input_timeseries_path, True)
