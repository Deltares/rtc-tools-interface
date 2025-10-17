"""Tests for the base optimization problem class."""
import os
import unittest
from rtctools.data import rtc
from rtctools.data import pi
from rtctools_interface.closed_loop.results_construction import combine_xml_exports
from pathlib import Path
from datetime import timedelta
import xml.etree.ElementTree as ET

def read_timeseries_data_from_xml(path, location_id_to_extract):
    tree = ET.parse(path)
    root = tree.getroot()

    ns = {'pi': 'http://www.wldelft.nl/fews/PI'}

    for series in root.findall('pi:series', ns):
        header = series.find('pi:header', ns)
        location_id = header.find('pi:locationId', ns)
        if location_id_to_extract in location_id.text:
            values = []
            for event in series.findall('pi:event', ns):
                values.append(float(event.get('value')))
            break
    return values

class TestCombineXmlExports(unittest.TestCase):
    """Combining XML output files."""

    def test_combine_xml_exports(self):
        original_input_timeseries_path =Path(os.path.join(os.getcwd())) / Path(
        r"test_models\goal_programming_xml\input"
    )
        output_base_path = Path(os.path.join(Path(os.getcwd()), 'test_models/goal_programming_xml/output/output_modelling_periods_reference'))
        dataconfig = rtc.DataConfig(folder=original_input_timeseries_path)

        ts_import_orig = pi.Timeseries(data_config=dataconfig,
            folder=original_input_timeseries_path, basename="timeseries_import", binary=False, )


        combine_xml_exports(output_base_path, original_input_timeseries_path)
        forecast_timestep = timedelta(days=2)
        timestep = timedelta(hours=8)

        values_per_period = {}

        for period_number in range(3):
            xml_path = os.path.join(output_base_path, Path('Period_' + str(period_number) + '/timeseries_export.xml'))
            values = read_timeseries_data_from_xml(xml_path, 'Q_orifice')
            values_per_period[period_number] = values

        # Logic for glueing the series together without taking the first timestep
        aggregates_values = values_per_period[0][0:int(forecast_timestep / timestep) + 1]
        aggregates_values = aggregates_values + values_per_period[1][1:int(forecast_timestep / timestep) + 1]
        aggregates_values = aggregates_values +values_per_period[2][1:]

        reference_values = aggregates_values

        xml_path = os.path.join(output_base_path, Path(
            '../timeseries_export.xml'))
        values_to_be_compared = read_timeseries_data_from_xml(xml_path, 'Q_orifice')

        # Ensure we checked multiple periods
        assert sorted(reference_values) == sorted(values_to_be_compared)