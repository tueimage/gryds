#! /usr/bin/env python
#
# Linear transformations


from __future__ import division, print_function, absolute_import

import numpy as np
from ..config import DTYPE
from .base import Transformation


class LinearTransformation(Transformation):
    """Linear transformation for 2D or 3D augmented coordinates.

    Attributes:
        ndim (int): The number of dimensions.
        parameters (np.ndarray): The (ndim) x (ndim + 1) transformation matrix.
    """

    def __init__(self, matrix):
        """
        Args:
            matrix (np.array): An (ndim ) x (ndim + 1) array
                representing the augmented affine matrix.
        Raises:
            ValueError: If the matrix is not shaped correctly.
        """
        matrix = np.array(matrix, dtype=DTYPE)

        if matrix.shape[0] != matrix.shape[1] - 1:
            raise ValueError(
                'Incorrect matrix shape, should be (ndim) x (ndim + 1),'
                'is {}.'.format(matrix.shape))

        super(LinearTransformation, self).__init__(len(matrix), matrix)

    def _transform_points(self, points):
        augmented_points = np.ones((self.ndim + 1, points.shape[1]),
                                   dtype=DTYPE)
        augmented_points[:self.ndim] = points

        transformed_points = np.dot(self.parameters, augmented_points)
        result = transformed_points[:self.ndim, :]

        assert result.dtype == DTYPE
        return result
