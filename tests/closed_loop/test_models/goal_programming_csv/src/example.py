""" This example model is a modified version of the goal_programming example model of
rtc-tools: https://gitlab.com/deltares/rtc-tools"""
import numpy as np
from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin, StateGoal
from rtctools.optimization.modelica_mixin import ModelicaMixin

from rtctools_interface.closed_loop.runner import run_optimization_problem_closed_loop


class WaterLevelRangeGoal(StateGoal):
    # Applying a state goal to every time step is easily done by defining a goal
    # that inherits StateGoal. StateGoal is a helper class that uses the state
    # to determine the function, function range, and function nominal
    # automatically.
    state = "storage.HQ.H"
    # One goal can introduce a single or two constraints (min and/or max). Our
    # target water level range is 0.43 - 0.44. We might not always be able to
    # realize this, but we want to try.
    target_min = 0.43
    target_max = 0.44

    # Because we want to satisfy our water level target first, this has a
    # higher priority (=lower number).
    priority = 1


class MinimizeQpumpGoal(Goal):
    # This goal does not use a helper class, so we have to define the function
    # method, range and nominal explicitly. We do not specify a target_min or
    # target_max in this class, so the goal programming mixin will try to
    # minimize the expression returned by the function method.
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.integral("Q_pump")

    # The nominal is used to scale the value returned by
    # the function method so that the value is on the order of 1.
    function_nominal = 100.0
    # The lower the number returned by this function, the higher the priority.
    priority = 2
    # The penalty variable is taken to the order'th power.
    order = 1


class MinimizeChangeInQpumpGoal(Goal):
    # To reduce pump power cycles, we add a third goal to minimize changes in
    # Q_pump. This will be passed into the optimization problem as a path goal
    # because it is an an individual goal that should be applied at every time
    # step.
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.der("Q_pump")

    function_nominal = 5.0
    priority = 3
    # Default order is 2, but we want to be explicit
    order = 2


class Example(GoalProgrammingMixin, CSVMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    """
    An introductory example to goal programming in RTC-Tools
    """

    def path_constraints(self, ensemble_member):
        # We want to add a few hard constraints to our problem. The goal
        # programming mixin however also generates constraints (and objectives)
        # from on our goals, so we have to call super() here.
        constraints = super().path_constraints(ensemble_member)

        # Release through orifice downhill only. This constraint enforces the
        # fact that water only flows downhill
        constraints.append((self.state("Q_orifice") + (1 - self.state("is_downhill")) * 10, 0.0, 10.0))

        # Make sure is_downhill is true only when the sea is lower than the
        # water level in the storage.
        M = 2  # The so-called "big-M"
        constraints.append(
            (
                self.state("H_sea") - self.state("storage.HQ.H") - (1 - self.state("is_downhill")) * M,
                -np.inf,
                0.0,
            )
        )
        constraints.append(
            (
                self.state("H_sea") - self.state("storage.HQ.H") + self.state("is_downhill") * M,
                0.0,
                np.inf,
            )
        )

        # Orifice flow constraint. Uses the equation:
        # Q(HUp, HDown, d) = width * C * d * (2 * g * (HUp - HDown)) ^ 0.5
        # Note that this equation is only valid for orifices that are submerged
        #          units:  description:
        w = 3.0  # m       width of orifice
        d = 0.8  # m       hight of orifice
        C = 1.0  # none    orifice constant
        g = 9.8  # m/s^2   gravitational acceleration
        constraints.append(
            (
                ((self.state("Q_orifice") / (w * C * d)) ** 2) / (2 * g)
                + self.state("orifice.HQDown.H")
                - self.state("orifice.HQUp.H")
                - M * (1 - self.state("is_downhill")),
                -np.inf,
                0.0,
            )
        )

        return constraints

    def goals(self):
        return [MinimizeQpumpGoal()]

    def path_goals(self):
        # Sorting goals on priority is done in the goal programming mixin. We
        # do not have to worry about order here.
        return [WaterLevelRangeGoal(self), MinimizeChangeInQpumpGoal()]

    def pre(self):
        # Call super() class to not overwrite default behaviour
        super().pre()
        # We keep track of our intermediate results, so that we can print some
        # information about the progress of goals at the end of our run.
        self.intermediate_results = []

    # Any solver options can be set here
    def solver_options(self):
        options = super().solver_options()
        solver = options["solver"]
        options[solver]["print_level"] = 1
        return options


# Run
if __name__ == "__main__":
    run_optimization_problem_closed_loop(Example)
