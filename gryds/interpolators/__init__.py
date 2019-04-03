#! /usr/bin/env python
#
# Interpolation code


from __future__ import division, print_function, absolute_import

from .grid import Grid
from .bspline import BSplineInterpolator
from .linear import LinearInterpolator
from .color import MultiChannelInterpolator
from .cuda import BSplineInterpolatorCuda

Interpolator = BSplineInterpolator  # Default interpolator
