"""Tools for getting optimization test data."""
import pathlib

import pandas as pd


DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


def get_test_data(test: str):
    """
    Get the input data and output folder for a given test.
    """
    tests_df = pd.read_csv(DATA_DIR / "optimization" / "tests.csv", sep=",")
    tests_df.set_index("test", inplace=True)
    test_data = tests_df.loc[test]
    return {
        "model_folder": DATA_DIR / "models" / test_data["model_folder"],
        "model_name": test_data["model_name"],
        "model_input_folder": DATA_DIR / "model_input" / test_data["model_input_folder"],
        "goals_file": DATA_DIR / "goals" / test_data["goals_file"],
        "plot_table_file": DATA_DIR / "plot_table" / test_data["plot_table_file"],
        "output_folder": DATA_DIR / "optimization" / "output" / test_data["output_folder"],
    }
