"""Tools for getting optimization or simulation test data."""

import pathlib

import pandas as pd

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


def get_test_data(test: str, optimization: bool = True) -> dict:
    """
    Get the input data and output folder for a given test.
    """
    sub_path = "optimization" if optimization else "simulation"
    tests_df = pd.read_csv(DATA_DIR / sub_path / "tests.csv", sep=",")
    tests_df.set_index("test", inplace=True)
    test_data = tests_df.loc[test]
    test_data_dict = {
        "model_folder": DATA_DIR / "models" / test_data["model_folder"],
        "model_name": test_data["model_name"],
        "model_input_folder": DATA_DIR / "model_input" / test_data["model_input_folder"],
        "plot_table_file": DATA_DIR / "plot_table" / test_data["plot_table_file"],
        "output_folder": DATA_DIR / sub_path / "output" / test_data["output_folder"],
    }
    if optimization:
        test_data_dict["goals_file"] = DATA_DIR / "goals" / test_data["goals_file"]
    return test_data_dict
