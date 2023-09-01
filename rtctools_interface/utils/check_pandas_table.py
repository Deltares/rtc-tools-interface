import pandas as pd
import logging


logger = logging.getLogger("rtctools")


def check_pandas_table(table, column_specs, table_name):
    """Validate values and columns in pandas dataframe"""

    for column, spec in column_specs.items():
        if column not in table.columns and spec["required"]:
            raise ValueError(f"Required column '{column}' is missing in {table_name}.")

    for column in table.columns:
        if column not in column_specs.keys():
            logger.info(f"Found unexpected column '{column}' in {table_name}.")

    for column, spec in column_specs.items():
        if column in table.columns:
            for value in table[column]:
                if not pd.isna(value) and not isinstance(value, tuple(spec["allowed_types"])):
                    raise ValueError(
                        f"Invalid type in '{column}' column of {table_name}."
                        + f" Expected {spec['allowed_types']}, got {type(value)}."
                    )
                if spec["required"] and pd.isna(value):
                    raise ValueError(f"Missing value(s) in required column '{column}' of {table_name}.")
                if spec["allowed_values"] is not None and value not in spec["allowed_values"] and spec["required"]:
                    raise ValueError(
                        f"Invalid value in '{column}' column of {table_name}."
                        + f" Allowed values are {spec['allowed_values']}."
                    )
