"""Mixin to store all required data for plotting. Can also call the plot function."""
import logging

from rtctools_interface.optimization.helpers.statistics_mixin import StatisticsMixin
from rtctools_interface.plotting.plot_tools import create_plot_each_priority, create_plot_final_results
from rtctools_interface.optimization.base_goal import BaseGoal
from rtctools_interface.utils.results_collection import PlottingBaseMixin

logger = logging.getLogger("rtctools")


class PlotGoalsMixin(StatisticsMixin, PlottingBaseMixin):
    """
    Class for plotting results.
    """

    def priority_completed(self, priority: int) -> None:
        """Store priority-dependent results required for plotting."""
        timeseries_data = self.collect_timeseries_data(list(set(self.custom_variables + self.state_variables)))
        to_store = {"timeseries_data": timeseries_data, "priority": priority}
        self.intermediate_results.append(to_store)
        super().priority_completed(priority)

    def post(self):
        """Tasks after optimizing. Creates a plot for for each priority."""
        super().post()

        if self.solver_stats["success"]:
            base_goals = [
                goal.get_goal_config() for goal in self.goals() + self.path_goals() if isinstance(goal, BaseGoal)
            ]
            current_run = self.create_plot_data_and_config(base_goals)
            # Cache results, such that in a next run they can be used for comparison
            self._store_current_results(self._cache_folder, current_run)

            # Create the plots
            plot_data = {}
            if self.plot_results_each_priority:
                plot_data = plot_data | create_plot_each_priority(current_run, plotting_library=self.plotting_library)

            if self.plot_final_results:
                plot_data = plot_data | create_plot_final_results(
                    current_run, self._previous_run, plotting_library=self.plotting_library
                )
