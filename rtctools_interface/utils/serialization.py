"""Methods to serialize and deserialize the PlotDataAndConfig objects."""
import json
import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rtctools_interface.utils.plot_table_schema import PlotTableRow


def custom_encoder(obj):
    """Custom JSON encoder for types not supported by default."""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return {"__type__": "datetime" if isinstance(obj, datetime.datetime) else "date", "value": obj.isoformat()}
    if isinstance(obj, np.ndarray):
        return {"__type__": "ndarray", "data": obj.tolist()}
    if isinstance(obj, Path):
        return {"__type__": "path", "value": str(obj)}
    if isinstance(obj, pd.DataFrame):
        return {"__type__": "pandas_dataframe", "value": obj.to_json()}
    if isinstance(obj, PlotTableRow):
        return {"__type__": "plot_table_row", "value": obj.model_dump()}
    if hasattr(obj, "dict"):
        return obj.dict()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def serialize(object_to_serialize: Any) -> str:
    """Serialize an object to a JSON string."""
    return json.dumps(object_to_serialize, default=custom_encoder)


def custom_decoder(dct: Any) -> Any:
    """Custom JSON decoder for types not supported by default."""
    # pylint: disable=too-many-return-statements
    if isinstance(dct, dict):
        for key, value in dct.items():
            dct[key] = custom_decoder(value)  # Recursively process each value
        if dct.get("__type__") == "datetime":
            return datetime.datetime.fromisoformat(dct["value"])
        if dct.get("__type__") == "date":
            return datetime.date.fromisoformat(dct["value"])
        if dct.get("__type__") == "ndarray":
            return np.array(dct["data"])
        if dct.get("__type__") == "path" and dct["value"]:
            return Path(dct["value"])
        if dct.get("__type__") == "pandas_dataframe":
            return pd.read_json(dct["value"])
        if dct.get("__type__") == "plot_table_row":
            return PlotTableRow(**dct["value"])
        return dct
    if isinstance(dct, list):
        return [custom_decoder(item) for item in dct]  # Recursively process each item in the list
    return dct


def deserialize(serialized_str: str) -> dict:
    """Deserialize the JSON string."""
    return json.loads(serialized_str, object_hook=custom_decoder)
