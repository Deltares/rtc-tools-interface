"""Module for a basic simulation problem."""
from rtctools.simulation.csv_mixin import CSVMixin
from rtctools.simulation.simulation_problem import SimulationProblem


class BaseSimulationProblem(
    CSVMixin,
    SimulationProblem,
):
    # Ignore too many ancestors, since the use of mixin classes is how rtc-tools is set up.
    # pylint: disable=too-many-ancestors
    """
    Basic simulation problem for a given state.

    :cvar goal_table_file:
        path to csv file containing a list of goals.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def update(self, dt):
        self.set_var("u", 1)
        super().update(dt)

    def initialize(self):
        self.set_var("u", 1)
        super().initialize()
