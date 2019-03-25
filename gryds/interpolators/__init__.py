#! /usr/bin/env python
#
# Interpolation code


from __future__ import division, print_function, absolute_import

from .grid import Grid
from .bspline import BSplineInterpolator
from .bspline_cuda import BSplineInterpolatorCUDA
from .linear import LinearInterpolator
from .color import MultiChannelInterpolator

Interpolator = BSplineInterpolator  # Default interpolator
