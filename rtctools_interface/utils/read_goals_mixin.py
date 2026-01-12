"""Mixin to read the goal table and store as class variables."""

import os
from typing import Literal

from rtctools_interface.optimization.read_goals import read_goals


class ReadGoalsMixin:
    """Read the goal table either from the default or specified path."""

    def load_goals(
        self,
        read_from: Literal["csv_table", "passed_list"] = "csv_table",
        goals_to_generate: list | None = None,
        csv_list_separator=",",
    ):
        """Read goal table and store as instance variable."""
        goals_to_generate = goals_to_generate if goals_to_generate else []
        if not hasattr(self, "goal_table_file"):
            self.goal_table_file = os.path.join(self._input_folder, "goal_table.csv")

        if (
            read_from == "csv_table"
            and os.path.isfile(self.goal_table_file)
            or read_from == "passed_list"
        ):
            self._goal_generator_path_goals = read_goals(
                self.goal_table_file,
                path_goal=True,
                read_from=read_from,
                goals_to_generate=goals_to_generate,
                csv_list_separator=csv_list_separator
            )
            self._goal_generator_non_path_goals = read_goals(
                self.goal_table_file,
                path_goal=False,
                read_from=read_from,
                goals_to_generate=goals_to_generate,
                csv_list_separator=csv_list_separator
            )
            self._all_goal_generator_goals = (
                self._goal_generator_path_goals + self._goal_generator_non_path_goals
            )
        else:
            self._all_goal_generator_goals = []
