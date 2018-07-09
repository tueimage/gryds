#! /usr/bin/env python
#
# Resample images using transformations on grids.
#
# @author: Koen Eppenhof
# @email: k.a.j.eppenhof@tue.nl
# @date: 2017/08/17


from __future__ import division, print_function

import numpy as np
import scipy.ndimage as nd


DTYPE = 'float32'


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
                *[np.arange(d, dtype=DTYPE) / d for d in shape],
                indexing='ij'
            ))

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
        size = np.array(size).astype(DTYPE)

        new_grid_instance = Grid()
        new_grid_instance.grid = np.array(
            [x * y for x, y in zip(size, self.grid)]
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
        return np.linalg.det(jac)


class Interpolator(object):
    """An interpolator for an image, that can resample an image on a new grid,
    or transform an image."""

    def __init__(self, image, **default_sampling_options):
        """
        Args:
            image (np.array): An image array.
            **default_sampling_options (dict): Sampling kwargs accepted by
                scipy.ndimage.map_coordinates().

            See https://docs.scipy.org/doc/scipy-0.14.0/reference/
            generated/scipy.ndimage.interpolation.map_coordinates.html
        """
        self.image = image
        self.default_sampling_options = default_sampling_options

    @property
    def grid(self):
        """Returns the image's grid."""
        return Grid(shape=self.image.shape)

    def sample(self, points, **sampling_options):
        """
        Samples the image at given points.

        Args:
            points (np.array): An N x ndims array of points.
            **sampling_options (dict): Sampling kwargs accepted by
                scipy.ndimage.map_coordinates().
        Returns:
            np.array: N-shaped array of intensities at the points.
        """
        new_sampling_options = self.default_sampling_options.copy()
        new_sampling_options.update(sampling_options)
        return nd.map_coordinates(self.image, points,
                                  **new_sampling_options)

    def resample(self, grid, **sampling_options):
        """
        Reamples the image at a given grid.

        Args:
            grid (Grid): The new grid.
            **sampling_options (dict): Sampling kwargs accepted by
                scipy.ndimage.map_coordinates().
        Returns:
            np.array: The resampled image at the new grid.
        """
        new_sampling_options = self.default_sampling_options.copy()
        new_sampling_options.update(sampling_options)

        rescaled_grid = grid.scaled_to(self.image.shape)
        new_image = self.sample(rescaled_grid.grid, **new_sampling_options)
        return new_image

    def transform(self, *transforms, **sampling_options):
        """
        Transforms the image by transforming the original image's grid and
        resampling the image at the transformed grid.

        Args:
            *transforms (list): A list of Transform objects.
            **sampling_options (dict): Sampling kwargs accepted by
                scipy.ndimage.map_coordinates().
        Returns:
            np.array: The transformed image.
        """
        new_sampling_options = self.default_sampling_options.copy()
        new_sampling_options.update(sampling_options)

        transformed_grid = self.grid.transform(*transforms)
        return self.resample(transformed_grid, **new_sampling_options)
