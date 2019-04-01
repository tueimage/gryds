#! /usr/bin/env python
#
# Resample images on a new Grid instance using linear interplation.
# This class is mostly here to test pure Numpy implementations of 
# linear interpolation but can be mostly ignored for standard code.


from __future__ import division, print_function, absolute_import

import numpy as np
from ..config import DTYPE
from .grid import Grid
from .base import Interpolator


class LinearInterpolator(Interpolator):
    """A pure Numpy implementation of linear interpolation.

    Attributes:
        self.image (np.ndarray): The wrapped ND image.
        self.grid (Grid): The image's default sampling grid.
    """
    def __init__(self, image, **kwargs):
        """
        Args:
            image (np.array): A 2D or 3D image array.
        """
        super(LinearInterpolator, self).__init__(
            image
        )
        if kwargs:
            print('WARNING: ignored options: {}'.format(kwargs))
        if image.ndim == 2:
            self._sample = self.__sample2
        elif image.ndim == 3:
            self._sample = self.__sample3
        else:
            raise ValueError('Image should be 2D or 3D array.')

    def sample(self, points, **kwargs):
        """
        Samples the image at given points.

        Args:
            points (np.array): An N x ndims array of points.
            **kwargs (dict): ignored
        Returns:
            np.array: N-shaped array of intensities at the points.
        """
        if kwargs:
            print('WARNING: ignored options: {}'.format(kwargs))
        p = np.array(points)
        return self._sample(*p)

    def resample(self, grid, **kwargs):
        """
        Reamples the image at a given grid.

        Args:
            grid (Grid): The new grid.
            **kwargs (dict): ignored
        Returns:
            np.array: The resampled image at the new grid.
        """
        if kwargs:
            print('WARNING: ignored options: {}'.format(kwargs))
        g = grid.scaled_to(self.image.shape).grid
        return self._sample(*g)

    def __sample2(self, X, Y):
        X0 = np.floor(X).astype('int')
        Y0 = np.floor(Y).astype('int')
        X1 = X0 + 1
        Y1 = Y0 + 1

        X0 = np.clip(X0, 0, self.image.shape[0] - 1)
        X1 = np.clip(X1, 0, self.image.shape[0] - 1)
        Y0 = np.clip(Y0, 0, self.image.shape[1] - 1)
        Y1 = np.clip(Y1, 0, self.image.shape[1] - 1)

        b_x0y0 = self.image[X0, Y0]
        b_x1y0 = self.image[X1, Y0]
        b_x0y1 = self.image[X0, Y1]
        b_x1y1 = self.image[X1, Y1]

        b = (X1 - X) * (Y1 - Y) * b_x0y0 + \
            (X - X0) * (Y1 - Y) * b_x1y0 + \
            (X1 - X) * (Y - Y0) * b_x0y1 + \
            (X - X0) * (Y - Y0) * b_x1y1
        return b

    def __sample3(self, X, Y, Z):
        X0 = np.floor(X).astype('int64')
        Y0 = np.floor(Y).astype('int64')
        Z0 = np.floor(Z).astype('int64')
        X1 = X0 + 1
        Y1 = Y0 + 1
        Z1 = Z0 + 1

        X0 = np.clip(X0, 0, self.image.shape[0] - 1)
        X1 = np.clip(X1, 0, self.image.shape[0] - 1)
        Y0 = np.clip(Y0, 0, self.image.shape[1] - 1)
        Y1 = np.clip(Y1, 0, self.image.shape[1] - 1)
        Z0 = np.clip(Z0, 0, self.image.shape[2] - 1)
        Z1 = np.clip(Z1, 0, self.image.shape[2] - 1)

        b_x0y0z0 = self.image[X0, Y0, Z0]
        b_x1y0z0 = self.image[X1, Y0, Z0]
        b_x0y1z0 = self.image[X0, Y1, Z0]
        b_x1y1z0 = self.image[X1, Y1, Z0]
        b_x0y0z1 = self.image[X0, Y0, Z1]
        b_x1y0z1 = self.image[X1, Y0, Z1]
        b_x0y1z1 = self.image[X0, Y1, Z1]
        b_x1y1z1 = self.image[X1, Y1, Z1]

        b = (X1 - X) * (Y1 - Y) * (Z1 - Z) * b_x0y0z0 + \
            (X - X0) * (Y1 - Y) * (Z1 - Z) * b_x1y0z0 + \
            (X1 - X) * (Y - Y0) * (Z1 - Z) * b_x0y1z0 + \
            (X - X0) * (Y - Y0) * (Z1 - Z) * b_x1y1z0 + \
            (X1 - X) * (Y1 - Y) * (Z - Z0) * b_x0y0z1 + \
            (X - X0) * (Y1 - Y) * (Z - Z0) * b_x1y0z1 + \
            (X1 - X) * (Y - Y0) * (Z - Z0) * b_x0y1z1 + \
            (X - X0) * (Y - Y0) * (Z - Z0) * b_x1y1z1
        return b
