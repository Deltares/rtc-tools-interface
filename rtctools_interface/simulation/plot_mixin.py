"""Mixin to store all required data for plotting. Can also call the plot function."""
import logging

from rtctools_interface.plotting.plot_tools import create_plot_final_results
from rtctools_interface.utils.results_collection import PlottingBaseMixin

logger = logging.getLogger("rtctools")


class PlotMixin(PlottingBaseMixin):
    """
    Class for plotting results.
    """

    def post(self):
        """Tasks after optimizing. Creates a plot for for each priority."""
        super().post()

        timeseries_data = self.collect_timeseries_data(self.custom_variables)
        self.intermediate_results.append({"timeseries_data": timeseries_data, "priority": 0})
        current_run = self.create_plot_data_and_config([])
        self._store_current_results(self._cache_folder, current_run)

        if self.plot_final_results:
            create_plot_final_results(current_run, self._previous_run, plotting_library=self.plotting_library)
