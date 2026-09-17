"""Tests for the base optimization problem class."""

import os
import unittest
import xml.etree.ElementTree as ET
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from rtctools_interface.closed_loop.results_construction import combine_xml_exports


def read_timeseries_data_from_xml(path, location_id_to_extract):
    tree = ET.parse(path)
    root = tree.getroot()

    ns = {"pi": "http://www.wldelft.nl/fews/PI"}

    for series in root.findall("pi:series", ns):
        header = series.find("pi:header", ns)
        location_id = header.find("pi:locationId", ns)
        if location_id_to_extract in location_id.text:
            values = []
            for event in series.findall("pi:event", ns):
                values.append(float(event.get("value")))
            break
    return values


class TestCombineXmlExports(unittest.TestCase):
    """Combining XML output files."""

    def test_combine_xml_exports(self):
        TEST_DIR = Path(__file__).parent

        original_input_timeseries_path = TEST_DIR / "test_models" / "goal_programming_xml" / "input"

        output_base_path = (
            TEST_DIR
            / "test_models"
            / "goal_programming_xml"
            / "output"
            / "output_modelling_periods_reference"
        )

        combine_xml_exports(output_base_path, original_input_timeseries_path)
        forecast_timestep = timedelta(days=2)
        timestep = timedelta(hours=8)

        values_per_period = {}

        for period_number in range(3):
            xml_path = os.path.join(
                output_base_path, Path("period_" + str(period_number) + "/timeseries_export.xml")
            )
            values = read_timeseries_data_from_xml(xml_path, "Q_orifice")
            values_per_period[period_number] = values

        # Logic for glueing the series together without taking the first timestep
        aggregates_values = values_per_period[0][0 : int(forecast_timestep / timestep) + 1]
        aggregates_values = (
            aggregates_values + values_per_period[1][1 : int(forecast_timestep / timestep) + 1]
        )
        aggregates_values = aggregates_values + values_per_period[2][1:]

        reference_values = aggregates_values

        xml_path = os.path.join(output_base_path, Path("../timeseries_export.xml"))
        values_to_be_compared = read_timeseries_data_from_xml(xml_path, "Q_orifice")

        # Ensure we checked multiple periods
        assert reference_values == values_to_be_compared

    @patch("rtctools_interface.closed_loop.results_construction.pi.Timeseries")
    @patch("rtctools_interface.closed_loop.results_construction.rtc.DataConfig")
    @patch("rtctools_interface.closed_loop.results_construction.os.path.isfile")
    def test_combine_xml_exports_skips_single_timestep_period(
        self, isfile_mock, dataconfig_mock, timeseries_mock
    ):
        dataconfig = MagicMock()
        dataconfig.pi_variable_ids.return_value = ("location", "parameter")
        dataconfig_mock.return_value = dataconfig

        ts_import_orig = MagicMock()
        ts_import_orig.forecast_datetime = 0
        ts_import_orig.start_datetime = 0
        ts_import_orig.end_datetime = 2
        ts_import_orig.times = [0, 1, 2]

        ts_export = MagicMock()
        ts_export.items.return_value = [("Q_orifice", None)]
        ts_export.get.return_value = [1.0, 2.0, 3.0]

        ts_export_step = MagicMock()
        ts_export_step.times = [1]
        ts_export_step.get.return_value = [9.0]

        timeseries_mock.side_effect = [ts_import_orig, ts_export, ts_export_step]
        isfile_mock.side_effect = [True, False]

        combine_xml_exports(Path("/unused/output"), Path("/unused/input"))

        ts_export.set.assert_not_called()
        ts_export.write.assert_called_once_with(
            output_folder=Path("/unused/output").parent, output_filename="timeseries_export"
        )
