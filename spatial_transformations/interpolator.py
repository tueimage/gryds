#! /usr/bin/env python
#
# Resample images on a new Grid instance
#
# @author: Koen Eppenhof
# @email: k.a.j.eppenhof@tue.nl
# @date: 2018/08/30


from __future__ import division, print_function, absolute_import


import scipy.ndimage as nd
from .config import DTYPE
from .grid import Grid


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
        sample = nd.map_coordinates(self.image, points,
                                    **new_sampling_options)
        return sample.astype(DTYPE)

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
        return new_image.astype(DTYPE)

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
        new_grid = self.resample(transformed_grid, **new_sampling_options)
        return new_grid.astype(DTYPE)
