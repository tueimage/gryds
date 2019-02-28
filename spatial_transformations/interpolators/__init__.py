#! /usr/bin/env python
#
# Interpolation code
#
# @author: Koen Eppenhof
# @email: k.a.j.eppenhof@tue.nl
# @date: 2019/01/24


from __future__ import division, print_function, absolute_import

from .grid import Grid
from .bspline import BSplineInterpolator
from .linear import LinearInterpolator

Interpolator = BSplineInterpolator  # Default interpolator
