"""
Small optimization example based on rtc-electra, using rtc-tools-interface for plotting.

- The goals are read from ``input/goal_table.csv`` by the GoalGeneratorMixin.
- The plots are configured in ``input/plot_table.csv`` and created by the PlotMixin.

Figures are written to ``output/optimization/figures``:
  * ``after_priority_<p>.html``: result after each priority (incl. result of previous priority)
  * ``final_results.html``: final result (with a toggle to compare with the previous run)

After the optimization, ``post()`` runs the simulation (``example_sim.py``), whose figures
are written to ``output/simulation/figures``.
"""

import os

from example_sim import OUTPUT_FOLDER as SIM_OUTPUT_FOLDER
from example_sim import ExampleSim
from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.goal_programming_mixin import GoalProgrammingMixin
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.util import run_optimization_problem, run_simulation_problem

from rtctools_interface.optimization.goal_generator_mixin import GoalGeneratorMixin
from rtctools_interface.optimization.plot_mixin import PlotMixin

BASE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "output", "optimization")


class Example(
    PlotMixin,
    GoalGeneratorMixin,
    GoalProgrammingMixin,
    CSVMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    """Optimize the pump discharges of the simplified Electra boezem."""

    plot_table_file = os.path.join(BASE_FOLDER, "input", "plot_table.csv")
    goal_table_file = os.path.join(BASE_FOLDER, "input", "goal_table.csv")

    # Plot settings of rtc-tools-interface
    plot_max_rows = 4
    plot_results_each_priority = True
    plot_final_results = True

    def __init__(self, *args, **kwargs):
        self._plotting_library = kwargs.get("plotting_library", "plotly")
        super().__init__(*args, **kwargs)

    def solver_options(self):
        options = super().solver_options()
        options["ipopt"]["print_level"] = 0
        options["print_time"] = False
        return options

    def post(self):
        # Default post-processing (write output, make optimization plots)
        super().post()

        # Run the simulation after the optimization
        os.makedirs(SIM_OUTPUT_FOLDER, exist_ok=True)
        run_simulation_problem(
            ExampleSim,
            base_folder=BASE_FOLDER,
            output_folder=SIM_OUTPUT_FOLDER,
            plotting_library=self._plotting_library,
        )


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    run_optimization_problem(
        Example,
        base_folder=BASE_FOLDER,
        output_folder=OUTPUT_FOLDER,
        plotting_library="plotly",  # or "matplotlib" for png figures
    )
