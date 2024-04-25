"""Test the closed loop runner"""
import math
from operator import is_
import xml.etree.ElementTree as ET
from unittest import TestCase
from pathlib import Path
import pandas as pd
from rtctools_interface.closed_loop.runner import run_optimization_problem_closed_loop
from .test_models.goal_programming_xml.src.example import Example as ExampleXml
from .test_models.goal_programming_csv.src.example import Example as ExampleCsv

ns = {"fews": "http://www.wldelft.nl/fews", "pi": "http://www.wldelft.nl/fews/PI"}

# Elementwise comparisons are practially disabled.
A_TOL = 10
R_TOL = 1


def compare_xml_files(output_modelling_period_folder: Path, reference_folder: Path):
    """Compare the timeseries_export.xml files in the output and reference folders."""
    for folder in output_modelling_period_folder.iterdir():
        if not folder.is_dir():
            continue
        file_name = "timeseries_export.xml"
        tree_result = ET.parse(folder / file_name)
        tree_ref = ET.parse(reference_folder / folder.name / file_name)
        series_result = tree_result.findall("pi:series", ns)
        series_ref = tree_ref.findall("pi:series", ns)
        assert len(series_result) == len(series_ref), "Different number of series found in exports."
        for serie_result, serie_ref in zip(series_result, series_ref):
            for event_result, event_ref in zip(serie_result.findall("pi:event", ns), serie_ref.findall("pi:event", ns)):
                value_result = float(event_result.attrib["value"])
                value_ref = float(event_ref.attrib["value"])
                assert math.isclose(
                    value_result, value_ref, rel_tol=R_TOL, abs_tol=A_TOL
                ), f"Difference found in event: {value_result} != {value_ref}"


class TestClosedLoop(TestCase):
    """
    Class for testing closed loop runner.
    """

    def test_running_closed_loop_csv(self):
        """
        Check if test model runs without problems and generates same results.
        """
        base_folder = Path(__file__).parent / "test_models" / "goal_programming_csv"
        run_optimization_problem_closed_loop(ExampleCsv, base_folder=base_folder)

        output_modelling_period_folder = base_folder / "output" / "output_modelling_periods"
        self.assertTrue(output_modelling_period_folder.exists(), "Output modelling period folder should be created.")
        self.assertEqual(
            len(list(output_modelling_period_folder.iterdir())), 3, "Three modelling periods should be created."
        )
        for folder in output_modelling_period_folder.iterdir():
            self.assertTrue((folder / "timeseries_export.csv").exists())
        reference_folder = base_folder / "output" / "output_modelling_periods_reference"
        for folder in output_modelling_period_folder.iterdir():
            reference_folder_i = reference_folder / folder.name
            for file in folder.iterdir():
                df_result = pd.read_csv(file)
                df_ref = pd.read_csv(reference_folder_i / file.name)
                pd.testing.assert_frame_equal(df_result, df_ref, atol=A_TOL, rtol=R_TOL)

    def test_running_closed_loop_xml(self):
        """
        Check if test model runs without problems and generates same results.
        """
        base_folder = Path(__file__).parent / "test_models" / "goal_programming_xml"
        run_optimization_problem_closed_loop(ExampleXml, base_folder=base_folder)

        output_modelling_period_folder = base_folder / "output" / "output_modelling_periods"
        self.assertTrue(output_modelling_period_folder.exists(), "Output modelling period folder should be created.")
        self.assertEqual(
            len([f for f in output_modelling_period_folder.iterdir() if f.is_dir()]),
            3,
            "Three modelling periods should be created.",
        )
        for folder in output_modelling_period_folder.iterdir():
            if folder.is_dir():
                self.assertTrue((folder / "timeseries_export.xml").exists())
        reference_folder = base_folder / "output" / "output_modelling_periods_reference"
        compare_xml_files(output_modelling_period_folder, reference_folder)
