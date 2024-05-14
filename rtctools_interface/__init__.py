from . import _version

__version__ = _version.get_versions()["version"]

from rtctools_interface.closed_loop.runner import run_optimization_problem_closed_loop

__all__ = ["run_optimization_problem_closed_loop"]
