"""Methods to serialize and deserialize the PlotDataAndConfig objects."""
import json
import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rtctools_interface.optimization.plot_and_goal_schema import GOAL_TYPE_COMBINED_MODEL
from rtctools_interface.optimization.plot_table_schema import PlotTableRow

from rtctools_interface.optimization.type_definitions import PlotDataAndConfig


def custom_encoder(obj):
    """Custom JSON encoder for types not supported by default."""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return {"__type__": "datetime" if isinstance(obj, datetime.datetime) else "date", "value": obj.isoformat()}
    if isinstance(obj, np.ndarray):
        return {"__type__": "ndarray", "data": obj.tolist()}
    if isinstance(obj, Path):
        return {"__type__": "path", "value": str(obj)}
    if hasattr(obj, "dict"):
        return obj.dict()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def serialize(plot_data_and_config: PlotDataAndConfig) -> str:
    """Serialize the PlotDataAndConfig object to a JSON string."""
    return json.dumps(plot_data_and_config, default=custom_encoder)


def custom_decoder(dct: Any) -> Any:
    """Custom JSON decoder for types not supported by default."""
    if isinstance(dct, dict):
        for key, value in dct.items():
            dct[key] = custom_decoder(value)  # Recursively process each value
        if dct.get("__type__") == "datetime":
            return datetime.datetime.fromisoformat(dct["value"])
        elif dct.get("__type__") == "date":
            return datetime.date.fromisoformat(dct["value"])
        elif dct.get("__type__") == "ndarray":
            return np.array(dct["data"])
        elif dct.get("__type__") == "path" and dct["value"]:
            return Path(dct["value"])
        return dct
    elif isinstance(dct, list):
        return [custom_decoder(item) for item in dct]  # Recursively process each item in the list
    else:
        return dct


def deserialize(serialized_str: str) -> PlotDataAndConfig:
    """Deserialize the JSON string back to a PlotDataAndConfig object."""
    data = json.loads(serialized_str, object_hook=custom_decoder)

    # Reconstruct pydantic models
    goal_generator_goals = [
        GOAL_TYPE_COMBINED_MODEL[item["goal_type"]](**item)
        for item in data["plot_options"]["plot_config"]
        if item["specified_in"] == "goal_generator"
    ]
    python_goals = [
        PlotTableRow(**item) for item in data["plot_options"]["plot_config"] if item["specified_in"] == "python"
    ]
    data["plot_options"]["plot_config"] = goal_generator_goals + python_goals

    return data
