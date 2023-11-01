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
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return {"__type__": "ndarray", "data": obj.tolist()}
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "dict"):
        return obj.dict()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def serialize(plot_data_and_config: PlotDataAndConfig) -> str:
    """Serialize the PlotDataAndConfig object to a JSON string."""
    return json.dumps(plot_data_and_config, default=custom_encoder)


def custom_decoder(dct: Any) -> Any:
    """Custom JSON decoder for types not supported by default."""
    for key, value in dct.items():
        if isinstance(value, str):
            try:
                if "T" in value:
                    dct[key] = datetime.datetime.fromisoformat(value)
                elif "-" in value:
                    dct[key] = datetime.date.fromisoformat(value)
                elif Path(value).exists():
                    dct[key] = Path(value)
            except ValueError:
                pass
        elif isinstance(value, dict) and value.get("__type__") == "ndarray":
            dct[key] = np.array(value["data"])
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
