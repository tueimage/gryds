#! /usr/bin/env python
#
# Composed transformations combine multiple transform objects into one
#
# @author: Koen Eppenhof
# @email: k.a.j.eppenhof@tue.nl
# @date: 2018/08/30


from __future__ import division, print_function, absolute_import

import numpy as np
from ..config import DTYPE
from .base import Transformation


class ComposedTransformation(Transformation):
    """Composed transform that turns multiple Transform objects into a net
    tranform through composition.

    Given two transformations t1 and t2, and a points x0, the following result
    in the same transformed points x2:

    >>> # Manual composition
    >>> x1 = t1.transform(x0)
    >>> x2 = t2.transform(x1)

    >>> # With ComposedTransform
    >>> t12 = ComposedTransform(t1, t2)
    >>> x2 = t12.transform(x0)
    """

    def __init__(self, *transformations):
        ndims = [x.ndim for x in transformations]
        if not np.all(np.array(ndims) == ndims[0]):
            raise ValueError('Number of dimensions for transformations {} do not '
                             ' match: {}.'.format(', '.join(
                                 tuple([x.__class__.__name__ for x in transformations])
                             ), ndims))
        self.ndim = ndims[0]
        self.transformations = transformations

    def _transform_points(self, points):
        points_copy = points.copy()
        for transform in self.transformations:
            points_copy = transform.transform(points_copy)
        assert points_copy.dtype == DTYPE
        return points_copy
