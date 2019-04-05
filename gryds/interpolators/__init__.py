#! /usr/bin/env python
#
# Interpolation code


from __future__ import division, print_function, absolute_import

from .grid import Grid
from .bspline import BSplineInterpolator
from .linear import LinearInterpolator
from .color import MultiChannelInterpolator

try:
	from .cuda import BSplineInterpolatorCuda
except ImportError:
	pass

Interpolator = BSplineInterpolator  # Default interpolator
