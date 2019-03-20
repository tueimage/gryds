#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Composed transformations combine multiple transform objects into one


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

    Attributes:
        ndim (int): The number of dimensions.
        parameters (np.ndarray): Left empty
        transformations (Iterable): A sequence of Transformation objects
    """

    def __init__(self, *transformations):
        """
        Args:
            transformations (iterable): A sequence (list, tuple) of transformations.
        Raises:
            ValueError: If the number of dimenions the transformations operate
                on are not the same.
            ValueError: If transformations is empty.
        """
        if not transformations or len(transformations) == 0:
            raise ValueError('No transformations supplied.')
        ndims = [x.ndim for x in transformations]
        if not np.all(np.array(ndims) == ndims[0]):
            raise ValueError('Number of dimensions for transformations {} do not '
                             ' match: {}.'.format(', '.join(
                                 tuple([x.__class__.__name__ for x in transformations])
                             ), ndims))
        self.ndim = ndims[0]
        self.transformations = transformations

    def __repr__(self):
        return '{}({}D, {})'.format(self.__class__.__name__, self.ndim,
            '∘'.join([str(x) for x in self.transformations]))

    def _transform_points(self, points):
        points_copy = points.copy()
        for transform in self.transformations:
            points_copy = transform.transform(points_copy)
        assert points_copy.dtype == DTYPE
        return points_copy
