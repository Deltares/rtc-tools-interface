"""Schema for the goal_table."""
from rtctools_interface.optimization.base_goal import GOAL_TYPES

goal_table_column_spec = {
    "id": {"allowed_types": [int, str, float], "allowed_values": None, "required": True},
    "active": {"allowed_types": [int, float], "allowed_values": [0, 1], "required": True},
    "state": {"allowed_types": [str], "allowed_values": None, "required": True},
    "goal_type": {"allowed_types": [str], "allowed_values": GOAL_TYPES, "required": True},
    "function_min": {"allowed_types": [int, float], "allowed_values": None, "required": False},
    "function_max": {"allowed_types": [int, float], "allowed_values": None, "required": False},
    "function_nominal": {"allowed_types": [int, float], "allowed_values": None, "required": False},
    "target_data_type": {"allowed_types": [int, float, str], "allowed_values": None, "required": False},
    "target_min": {"allowed_types": [int, float, str], "allowed_values": None, "required": False},
    "target_max": {"allowed_types": [int, float, str], "allowed_values": None, "required": False},
    "priority": {"allowed_types": [int, float], "allowed_values": None, "required": True},
    "weight": {"allowed_types": [int, float], "allowed_values": None, "required": False},
    "order": {"allowed_types": [int, float], "allowed_values": None, "required": False},
}
