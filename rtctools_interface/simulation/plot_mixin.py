"""Mixin to store all required data for plotting. Can also call the plot function."""

import logging
from typing import Dict, List

import numpy as np

from rtctools_interface.plotting.plot_tools import create_plot_final_results
from rtctools_interface.utils.results_collection import PlottingBaseMixin

logger = logging.getLogger("rtctools")


class PlotMixin(PlottingBaseMixin):
    """
    Class for plotting results based on the plot_table.
    """

    optimization_problem = False
    _manual_extracted_states: Dict[str, list] = {}

    def manual_extraction_from_state_vector(self):
        for variable in self.custom_variables:
            try:
                self._manual_extracted_states[variable].append(self.get_var(variable))
            except KeyError:
                logger.debug("Variable {} not found in output of model.".format(variable))

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._manual_extracted_states = {variable: [] for variable in self.custom_variables}
        self.manual_extraction_from_state_vector()

    def update(self, dt):
        super().update(dt)
        self.manual_extraction_from_state_vector()

    def post(self):
        """Tasks after optimizing."""
        super().post()

        # find empty arrays in self._manual_extracted_states
        # for these variables try to use self.get_timeseries(variable)
        for variable in self.custom_variables:
            if not self._manual_extracted_states[variable] or len(self._manual_extracted_states[variable]) == 0:
                logger.debug(f"Variable {variable} has empty data collected.")
                try:
                    self._manual_extracted_states[variable] = self.get_timeseries(variable)
                except KeyError:
                    logger.warning(f"Variable {variable} not found in output of model.")

        timeseries_data = self.collect_timeseries_data(self.custom_variables)
        self._intermediate_results.append({"timeseries_data": timeseries_data, "priority": 0})
        current_run = self.create_plot_data_and_config([])
        self._store_current_results(self._cache_folder, current_run)

        if self.plot_final_results:
            create_plot_final_results(current_run, self._previous_run, plotting_library=self.plotting_library)

    def collect_timeseries_data(self, all_variables_to_store: List[str]) -> Dict[str, np.ndarray]:
        return {variable: np.array(self._manual_extracted_states[variable]) for variable in all_variables_to_store}
