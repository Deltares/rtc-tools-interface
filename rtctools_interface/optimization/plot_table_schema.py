"""Schema for the plot_table."""
plot_table_column_spec = {
    "id": {"allowed_types": [int, float, str], "allowed_values": None, "required": False},
    "y_axis_title": {"allowed_types": [str], "allowed_values": None, "required": True},
    "variables_style_1": {"allowed_types": [str], "allowed_values": None, "required": False},
    "variables_style_2": {"allowed_types": [str], "allowed_values": None, "required": False},
    "variables_with_previous_result": {"allowed_types": [str], "allowed_values": None, "required": False},
    "custom_title": {"allowed_types": [str], "allowed_values": None, "required": False},
    "specified_in": {"allowed_types": [str], "allowed_values": ["python", "goal_generator"], "required": True},
}
