"""Test the closed loop runner"""
import math
import xml.etree.ElementTree as ET
from datetime import timedelta
from pathlib import Path
from unittest import TestCase

import pandas as pd

from rtctools_interface.closed_loop.config import ClosedLoopConfig
from rtctools_interface.closed_loop.runner import run_optimization_problem_closed_loop
from .test_models.goal_programming_xml.src.example import Example as ExampleXml
from .test_models.goal_programming_csv.src.example import Example as ExampleCsv

ns = {"fews": "http://www.wldelft.nl/fews", "pi": "http://www.wldelft.nl/fews/PI"}

# Elementwise comparisons are practially disabled.
A_TOL = 0.1
R_TOL = 0.1


def compare_xml_file(file_result: Path, file_ref: Path):
    """Compare two timeseries_export files elementwise."""
    tree_result = ET.parse(file_result)
    tree_ref = ET.parse(file_ref)
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


def compare_xml_files(output_modelling_period_folder: Path, reference_folder: Path):
    """Compare the timeseries_export.xml files in the output and reference folders."""
    for folder in output_modelling_period_folder.iterdir():
        if not folder.is_dir():
            continue
        file_name = "timeseries_export.xml"
        file_result = folder / file_name
        file_ref = reference_folder / folder.name / file_name
        compare_xml_file(file_result, file_ref)


class TestClosedLoop(TestCase):
    """
    Class for testing closed loop runner.
    """

    def compare_csv_files(self, output_folder: Path, reference_folder: Path, n_periods: int):
        """Compare the csv files in output and reference subfolders."""
        self.assertTrue(
            output_folder.exists(),
            "Output modelling period folder should be created."
        )
        self.assertEqual(
            len(list(output_folder.iterdir())),
            n_periods,
            f"Error: {n_periods} modelling periods should be created."
        )
        for folder in output_folder.iterdir():
            self.assertTrue((folder / "timeseries_export.csv").exists())
        for folder in output_folder.iterdir():
            reference_folder_i = reference_folder / folder.name
            for file in folder.iterdir():
                df_result = pd.read_csv(file)
                df_ref = pd.read_csv(reference_folder_i / file.name)
                pd.testing.assert_frame_equal(df_result, df_ref, atol=A_TOL, rtol=R_TOL)

    def compare_csv_file(self, output_file: Path, reference_file: Path):
        """Compare the main csv output file in output and reference folder."""
        df_result = pd.read_csv(output_file)
        df_ref = pd.read_csv(reference_file)
        pd.testing.assert_frame_equal(df_result, df_ref, atol=A_TOL, rtol=R_TOL)

    def compare_xml_files(self, output_folder: Path, reference_folder: Path, n_periods: int):
        """Compare the xml files in output and reference subfolders."""
        self.assertTrue(
            output_folder.exists(),
            "Output modelling period folder should be created."
        )
        self.assertEqual(
            len([f for f in output_folder.iterdir() if f.is_dir()]),
            n_periods,
            f"Error: {n_periods} modelling periods should be created.",
        )
        for folder in output_folder.iterdir():
            if folder.is_dir():
                self.assertTrue((folder / "timeseries_export.xml").exists())
        compare_xml_files(output_folder, reference_folder)

    def compare_xml_file(self, output_file: Path, reference_file: Path):
        """Compare the main xml output file in output and reference folder."""
        compare_xml_file(output_file, reference_file)

    def test_running_closed_loop_csv(self):
        """
        Check if test model runs without problems and generates same results.
        """
        base_folder = Path(__file__).parent / "test_models" / "goal_programming_csv"
        config = ClosedLoopConfig(
            file=base_folder / "input" / "closed_loop_dates.csv",
            round_to_dates=True
        )
        run_optimization_problem_closed_loop(ExampleCsv, base_folder=base_folder, config=config)
        self.compare_csv_files(
            output_folder=base_folder / "output" / "output_modelling_periods",
            reference_folder=base_folder / "output" / "output_modelling_periods_reference",
            n_periods=3
        )
        self.compare_csv_file(
            output_file=base_folder / "output" / "timeseries_export.csv",
            reference_file=base_folder / "output" / "timeseries_export_reference.csv"
        )

    def test_running_closed_loop_csv_fixed_periods(self):
        """
        Check if test model runs for fixed optimization periods.
        """
        base_folder = Path(__file__).parent / "test_models" / "goal_programming_csv"
        output_folder = "output_fixed_periods"
        config = ClosedLoopConfig.from_fixed_periods(
            optimization_period=timedelta(days=3),
            forecast_timestep=timedelta(days=2)
        )
        run_optimization_problem_closed_loop(
            ExampleCsv,
            base_folder=base_folder,
            config=config,
            output_folder=output_folder
        )
        self.compare_csv_file(
            output_file=base_folder / output_folder / "timeseries_export.csv",
            reference_file=base_folder / "output" / "timeseries_export_reference.csv"
        )

    def test_running_closed_loop_xml(self):
        """
        Check if test model runs without problems and generates same results.
        """
        test_cases = [
            {
                "description": "without forecast date",
                "input_folder": "input",
            },
            {
                "description": "with forecast date unequal to first date",
                "input_folder": "input_with_forecast_date",
            },
            {
                "description": "with forecast date equal to first date",
                "input_folder": "input_with_forecast_date_equal_first_date",
            }
        ]

        base_folder = Path(__file__).parent / "test_models" / "goal_programming_xml"

        for case in test_cases:
            with self.subTest(case["description"]):
                config = ClosedLoopConfig(
                    file=base_folder / case["input_folder"] / "closed_loop_dates.csv",
                    round_to_dates=True
                )
                run_optimization_problem_closed_loop(
                    ExampleXml,
                    base_folder=base_folder,
                    config=config,
                    input_folder=case["input_folder"]
                )

                self.compare_xml_files(
                    output_folder=base_folder / "output" / "output_modelling_periods",
                    reference_folder=base_folder / "output" / "output_modelling_periods_reference",
                    n_periods=3
                )
                self.compare_xml_file(
                    output_file=base_folder / "output" / "timeseries_export.xml",
                    reference_file=base_folder / "output" / "timeseries_export_reference.xml"
                )

    def test_running_closed_loop_xml_fixed_periods(self):
        """
        Check if test model runs for fixed optimization periods.
        """
        test_cases = [
            {
                "description": "without forecast date",
                "output_folder": "output_fixed_periods",
                "input_folder": "input",
                "optimization_period": timedelta(days=3),
                "forecast_timestep": timedelta(days=2)
            },
            {
                "description": "with forecast date unequal to first date",
                "output_folder": "output_fixed_periods",
                "input_folder": "input_with_forecast_date",
                "optimization_period": timedelta(days=3),
                "forecast_timestep": timedelta(days=2)
            },
            {
                "description": "with forecast date equal to first date",
                "output_folder": "output_fixed_periods",
                "input_folder": "input_with_forecast_date_equal_first_date",
                "optimization_period": timedelta(days=3),
                "forecast_timestep": timedelta(days=2)
            }
        ]

        base_folder = Path(__file__).parent / "test_models" / "goal_programming_xml"

        for case in test_cases:
            with self.subTest(case["description"]):
                config = ClosedLoopConfig.from_fixed_periods(
                    optimization_period=case["optimization_period"],
                    forecast_timestep=case["forecast_timestep"]
                )
                run_optimization_problem_closed_loop(
                    ExampleXml,
                    base_folder=base_folder,
                    config=config,
                    output_folder=case["output_folder"],
                    input_folder=case["input_folder"]
                )
                self.compare_xml_file(
                    output_file=base_folder / case["output_folder"] / "timeseries_export.xml",
                    reference_file=base_folder / "output" / "timeseries_export_reference.xml"
                )
