#! /usr/bin/env python
#
# Implementation of sampling grids


from __future__ import division, print_function, absolute_import

import numpy as np
from ..config import DTYPE


class Grid(object):
    """Sampling grid that can be transformed.

    Attributes:
        self.grid (nd.array): The grid as an ndim x Ni x Nj x ... x Nndim array
    """

    def __init__(self, shape=None, grid=None):
        """
        Args:
            shape (iterable): an interable of length ndim for the shape of the
                grid.
            grid (np.ndarray): a pre-defined grid as an ndim x Ni x Nj x ... x Nndim array

        Raises:
            ValueError: when neither the shape or the grid are defined.
        """
        if grid is not None and shape is None:
            self.grid = grid.astype(DTYPE)
        elif shape is not None and grid is None:
            self.grid = np.array(np.meshgrid(
                *[np.arange(d) / d for d in shape],
                indexing='ij'
            ), dtype=DTYPE)
        else:
            raise ValueError('Either the shape or the grid parameters should be defined')

    def __repr__(self):
        return '{}({}D, {})'.format(self.__class__.__name__, self.grid.shape[0],
            'x'.join([str(x) for x in self.grid.shape[1:]]))

    def scaled_to(self, size):
        """
        Scale the grid to the given size, for example to fit an image size.

        Args:
            size (iterable): An iterable of length ndim for the shape of the
                grid.
        Raises:
            ValueError: when the number of dimensions and size parameters
                do not match.
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

        new_grid_instance = Grid(grid=np.array(
            [x * y for x, y in zip(size, self.grid)], dtype=DTYPE
        ))
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

        new_grid_instance = Grid(grid=new_grid.reshape(org_shape))

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
        diff_grid = self.transform(*transforms).scaled_to(self.grid.shape[1:]).grid
        # scaled_grid = new_grid.scaled_to(self.grid.shape[1:])
        jacobian = np.zeros(
            (self.grid.shape[0], self.grid.shape[0]) + self.grid.shape[1:]
        )
        for i in range(jacobian.shape[0]):
            for j in range(jacobian.shape[1]):
                padding = self.grid.shape[0] * [(0, 0)]
                padding[j] = (0, 1)
                jacobian[i, j] = np.pad(
                    np.diff(diff_grid[i], axis=j),
                    padding, mode='edge')

        return jacobian.astype(DTYPE)

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

        return jacdet.astype(DTYPE)
