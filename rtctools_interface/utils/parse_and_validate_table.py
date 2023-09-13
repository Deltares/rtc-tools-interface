import pandas as pd
import logging
from pydantic import ValidationError


logger = logging.getLogger("rtctools")


def parse_and_validate_table(table, row_model_class, table_name):
    """ "Given a pandas table, verify each row according to the specified column_model."""
    table_items = table.to_dict(orient="records")
    parsed_goals = []
    row_num = 0
    try:
        for row_num, item in enumerate(table_items):
            parsed_goals.append(row_model_class(**item))
    except (ValueError, ValidationError) as exc:
        raise ValueError(f"While validating row {row_num+1} of {table_name}: {exc}") from exc
    return pd.DataFrame([s.__dict__ for s in parsed_goals])
