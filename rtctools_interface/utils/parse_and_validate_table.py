import pandas as pd
import logging
from pydantic import ValidationError


logger = logging.getLogger("rtctools")


def parse_and_validate_table(table, tabel_model_class, table_name):
    """ "Given a pandas table, verify each row according to the specified table_model."""
    try:
        parsed_table = tabel_model_class(rows=table.to_dict(orient="records"))
    except (ValueError, ValidationError) as exc:
        raise ValueError(f"While validating {table_name}: {exc}") from exc
    return pd.DataFrame([row.dict() for row in parsed_table.rows])
