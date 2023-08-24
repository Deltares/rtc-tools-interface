plot_table_column_spec = {
    "id": {"allowed_types": [int, float, str], "allowed_values": None, "required": False},
    "y_axis_title": {"allowed_types": [str], "allowed_values": None, "required": True},
    "variables_plot_1": {"allowed_types": [str], "allowed_values": None, "required": False},
    "variables_plot_2": {"allowed_types": [str], "allowed_values": None, "required": False},
    "variables_plot_history": {"allowed_types": [str], "allowed_values": None, "required": False},
    "custom_title": {"allowed_types": [str], "allowed_values": None, "required": False},
    "specified_in": {"allowed_types": [str], "allowed_values": ["python", "goal_generator"], "required": True},
}
