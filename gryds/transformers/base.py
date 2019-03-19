#! /usr/bin/env python
#
# Transformation base class


from __future__ import division, print_function, absolute_import

import numpy as np
from ..config import DTYPE


class Transformation(object):
    """Base class for transformations, i.e. maps from *points* in one spatial
    domain to points in another spatial domain. This base class enforces checks
    of the number of points against the number of dimensions of the transform,
    i.e. an error is thrown if self.ndim does not match the number of dimensions
    in the points.

    All transformations are applied to relative coordinates, i.e. coordinates in
    the [0, 1)^ndim domain. The base class scales the points if necessary to
    this domain using the scale parameter of the transform function supplied by
    the user.

    Attributes:
        ndim (int): The number of dimensions.
        parameters (iterable/array): Some array-like representation of the 
            transformation parameters, dependant on kind of transformation.
    """

    def __init__(self, ndim, parameters):
        """
        Args:
            ndim (int): Number of dimensions of the transformation, used for
                checking the dimensions of points to be transformed.
        """
        self.ndim = ndim
        self.parameters = parameters

    def __repr__(self):
        return '{}({}D)'.format(self.__class__.__name__, self.ndim)

    def _dimension_check(self, points):
        """Checks if the points are compatible with the number of dimensions
        in the transformation.

        Args:
            points (np.array): A (self.ndim x N) array of N points.
            scale (np.array): An array of self.ndim scaling factors.
        Raises:
            ValueError: If the points and self.ndim are not compatible
        """
        if points.ndim != 2:
            raise ValueError(
                'Points should be expressed as an (ndim x N) matrix.')
        if self.ndim != points.shape[0]:
            raise ValueError(
                'Dimensions not compatible: {}D point cannot be transformed'
                ' by {}D transformer.'.format(points.shape[0], self.ndim))

    def _transform_points(self, points):
        """Template for transformer function."""
        raise NotImplementedError

    def transform(self, points, scale=None):
        """Calling the _transform_points function with dimension checks.

        Args:
            points (np.array): A (self.ndim x N) array of N points.
            scale (np.array): An array of self.ndim scaling factors.
        Returns:
            (np.array): The (self.ndim x N) array of N transformed points.
        Raises:
            ValueError: If the points and self.ndim are not compatible.
        """
        points = np.array(points, dtype=DTYPE)

        if not scale:
            scale = [1]

        scale = np.array(scale, dtype=DTYPE)[:, None]
        scaled_points = points / scale

        self._dimension_check(scaled_points)
        result = self._transform_points(scaled_points).astype('float32') * scale

        return result.astype(DTYPE)

    def __call__(self, points, scale=None):
        """Calling the transformation as a function invokes the `transform`
        function.

        Args:
            points (np.array): A (self.ndim x N) array of N points.
            scale (np.array): An array of self.ndim scaling factors.
        Returns:
            (np.array): The (self.ndim x N) array of N transformed points.
        Raises:
            ValueError: If the points and self.ndim are not compatible.
        """
        return self.transform(points, scale)
