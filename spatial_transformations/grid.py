#! /usr/bin/env python
#
# Implementation of sampling grids
#
# @author: Koen Eppenhof
# @email: k.a.j.eppenhof@tue.nl
# @date: 2018/08/30


from __future__ import division, print_function, absolute_import

import numpy as np
from .config import DTYPE


class Grid(object):
    """Sampling grid that can be transformed."""

    def __init__(self, shape=None):
        """
        Args:
            shape (iterable): an interable of length ndim for the shape of the
                grid.
        """
        if shape:
            self.grid = np.array(np.meshgrid(
                *[np.arange(d) / d for d in shape],
                indexing='ij'
            ), dtype=DTYPE)

    def __repr__(self):
        return self.__module__ + '.' + self.__class__.__name__ + \
            '(\n\t' + '\n\t'.join(str(self.grid).split('\n')) + '\n)'

    def scaled_to(self, size):
        """
        Scale the grid to the given size, for example to fit an image size.

        Args:
            size (iterable): An iterable of length ndim for the shape of the
                grid.
        Returns:
            Grid: A scaled version of the grid.
        """
        if len(size) != len(self.grid):
            raise ValueError(
                'Number of dimensions in size ({}) and grid ({}), do not'
                ' match'.format(
                    len(size), len(self.grid))
            )
        size = np.array(size)

        new_grid_instance = Grid()
        new_grid_instance.grid = np.array(
            [x * y for x, y in zip(size, self.grid)], dtype=DTYPE
        )
        return new_grid_instance

    def transform(self, *transforms):
        """
        Transform the grid with a one or multiple transforms.

        Args:
            transforms (*list): A list of Transform objects.
        Returns:
            Grid: a new grid instance with a transformed version of the points.
        """
        org_shape = self.grid.shape
        new_grid = self.grid.copy()

        for transform in transforms:
            rshp_grid = new_grid.reshape(self.grid.shape[0], -1)
            new_grid = transform(rshp_grid)

        new_grid_instance = Grid()
        new_grid_instance.grid = new_grid.reshape(org_shape)

        assert new_grid_instance.grid.dtype == DTYPE
        return new_grid_instance

    def jacobian(self, *transforms):
        """
        Calculate the Jacobian for the points on the grid after the transforms
        have been applied.

        Args:
            transforms (*list): A list of Transform objects.
        Returns:
            np.array: An array of the size of the grid with the Jacobian
                vectors, (i.e. ndim x Na x Nb x ... x ND)
        """
        new_grid = self.transform(*transforms)
        # scaled_grid = new_grid.scaled_to(self.grid.shape[1:])
        jacobian = np.zeros(
            (self.grid.shape[0], self.grid.shape[0]) + self.grid.shape[1:]
        )
        for i in range(jacobian.shape[0]):
            for j in range(jacobian.shape[1]):
                padding = self.grid.shape[0] * [(0, 0)]
                padding[j] = (0, 1)
                jacobian[i, j] = np.pad(
                    np.diff(new_grid.grid[i], axis=j),
                    padding, mode='constant')

        assert jacobian.dtype == DTYPE
        return jacobian

    def jacobian_det(self, *transforms):
        """
        Calculate the Jacobian determinant for the points on the grid after the
         transforms have been applied.

        Args:
            *transforms (list): A list of Transform objects.
        Returns:
            np.array: An array of the size of the grid with the Jacobian
                determinant, (i.e. Na x ... x ND)
        """
        jac = self.jacobian(*transforms)
        jac = np.transpose(jac, list(range(2, jac.ndim)) + [0, 1])

        jacdet = np.linalg.det(jac)

        assert jacdet.dtype == DTYPE
        return jacdet
