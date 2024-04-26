import copy
import logging
import os
from pathlib import Path
import numpy as np
from rtctools.data import rtc
from rtctools.data import pi
from rtctools.data.csv import save as csv_save

logger = logging.getLogger("rtctools")


def write_csv(output_variables, times, results, output_folder):
    names = ["time"] + output_variables
    formats = ["O"] + (len(names) - 1) * ["f8"]
    dtype = dict(names=names, formats=formats)
    data = np.zeros(len(times), dtype=dtype)
    data["time"] = times
    for output_variable in output_variables:
        try:
            values = results[output_variable]
        except Exception as e:
            logger.error("\nException {} thrown when trying to write csv file".format(e))
            values = np.full_like(times, -999)
        data[output_variable] = values
    fname = os.path.join(output_folder, "timeseries_export.csv")
    csv_save(fname, data, delimiter=",", with_time=True)


def combine_xml_exports(output_base_path, original_input_timeseries_path, write_csv_out=False):
    logger.info("Combining XML exports.")
    dataconfig = rtc.DataConfig(folder=original_input_timeseries_path)

    ts_import_orig = pi.Timeseries(
        data_config=dataconfig, folder=original_input_timeseries_path, basename="timeseries_import", binary=False
    )
    orig_start_datetime = ts_import_orig.start_datetime
    orig_end_datetime = ts_import_orig.end_datetime

    ts_export = pi.Timeseries(
        data_config=dataconfig, folder=output_base_path / "period_0", basename="timeseries_export", binary=False
    )
    ts_export._Timeseries__path_xml = os.path.join(output_base_path.parent, "timeseries_export.xml")
    ts_export.resize(orig_start_datetime, orig_end_datetime)

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
        for loc_par in dataconfig._DataConfig__location_parameter_ids:
            try:
                current_values = ts_export.get(loc_par)
                new_values = ts_export_step.get(loc_par)
            except KeyError:
                logger.debug("Variable {} not found in output of model horizon: {}".format(loc_par, i))
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
    ts_export.write()


if __name__ == "__main__":
    closed_loop_test_folder = Path(__file__).parents[2] / "tests" / "closed_loop"
    output_base_path = closed_loop_test_folder / Path(
        r"test_models\goal_programming_xml\output\output_modelling_periods_reference"
    )
    original_input_timeseries_path = closed_loop_test_folder / Path(r"test_models\goal_programming_xml\input")
    combine_xml_exports(output_base_path, original_input_timeseries_path)
