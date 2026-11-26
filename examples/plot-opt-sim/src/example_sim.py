"""
Small simulation example based on rtc-electra, using rtc-tools-interface for plotting.

The pumping stations are controlled by a simple on/off (hysteresis) rule on the
boezem water level. The plots are configured in ``input/plot_table_sim.csv``.

Figures are written to ``output/simulation/figures/final_results.html``.
"""

import os

from rtctools.simulation.csv_mixin import CSVMixin
from rtctools.simulation.simulation_problem import SimulationProblem
from rtctools.util import run_simulation_problem

from rtctools_interface.simulation.plot_mixin import PlotMixin

BASE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "output", "simulation")


class ExampleSim(PlotMixin, CSVMixin, SimulationProblem):
    """Rule-based simulation of the simplified Electra boezem."""

    plot_table_file = os.path.join(BASE_FOLDER, "input", "plot_table_sim.csv")
    plot_max_rows = 4

    # (switch-on level, switch-off level, capacity) per pumping station
    pump_rules = {
        "Q_KGM155": (-0.88, -0.95, 33.5),
        "Q_DenDeel": (-0.86, -0.92, 10.6),
    }

    def initialize(self, config_file=None):
        self._pump_on = dict.fromkeys(self.pump_rules, False)
        for pump in self.pump_rules:
            self.set_var(pump, 0.0)
        super().initialize(config_file)

    def update(self, dt):
        h = self.get_var("H")
        for pump, (h_on, h_off, capacity) in self.pump_rules.items():
            if h > h_on:
                self._pump_on[pump] = True
            elif h < h_off:
                self._pump_on[pump] = False
            self.set_var(pump, capacity if self._pump_on[pump] else 0.0)
        super().update(dt)


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    run_simulation_problem(
        ExampleSim,
        base_folder=BASE_FOLDER,
        output_folder=OUTPUT_FOLDER,
        plotting_library="plotly",  # or "matplotlib" for png figures
    )
