"""Deprecated, use PlotMixin."""
import warnings

from rtctools_interface.optimization.plot_mixin import PlotMixin


class PlotGoalsMixin(PlotMixin):
    """
    Deprecated class, use PlotMixin instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn("PlotGoalsMixin is deprecated, use PlotMixin instead", FutureWarning, stacklevel=1)
        super().__init__(*args, **kwargs)
