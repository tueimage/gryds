#! /usr/bin/env python
#
# Resample multi-channel images on a new Grid


from __future__ import division, print_function, absolute_import

import numpy as np
from ..config import DTYPE
from .grid import Grid
from .bspline import BSplineInterpolator


class MultiChannelInterpolator:
    """Wrapper for an interpolator that is applied to each channel of a
    multi-channel (e.g. color) image.

    Attributes:
        image (np.ndarray): The wrapped ND image.
        grid (Grid): The image's default sampling grid.
        default_mode (str): Determines how edges are treated.
        default_order (int): B-Spline order.
        default_cval (numeric): Constant value for mode='constant'.
    """

    def __init__(self, image, interpolator=BSplineInterpolator,
                 data_format='channels_last', cval=None, **kwargs):
        """
        Args:
            image (np.array): An image array.
            interpolator (Interpolator): The interpolator that will be applied.
            data_format (str): The format of the multi-channel image. Options
                are 'channels_last' (for [[[R,G,B]]] images)
                and 'channels_first' (for [[[R]], [[G]], [[B]]] images).
            cval (numeric): Constant value for mode='constant' if the wrapped
                Interpolator class supports it.
            **kwargs (dict): Options for the wrapped Interpolator class.
        Raises:
            ValueError: when the data_format is something other than
                'channels_first' or 'channels_last'.
        """        
        self.image = image
        self.data_format = data_format

        self.nchan = image.shape[-1] if data_format == 'channels_last' \
            else image.shape[0]
        if cval is not None:
            assert len(cval) == self.nchan
        else:
            cval = self.nchan * [0]

        if data_format == 'channels_last':
            self.grid = Grid(shape=self.image.shape[:-1])
            self.interpolators = []
            for i, x in enumerate(np.rollaxis(image, -1)):
                self.interpolators.append(interpolator(x, **kwargs))
                try:
                    if self.interpolators[i].default_cval is not None:
                        self.interpolators[i].default_cval = cval[i]
                except AttributeError:
                    pass

        elif data_format == 'channels_first':
            self.grid = Grid(shape=self.image.shape[1:])
            self.interpolators = []
            for i, x in enumerate(image):
                self.interpolators.append(interpolator(x, **kwargs))
                try:
                    if self.interpolators[i].default_cval is not None:
                        self.interpolators[i].default_cval = cval[i]
                except AttributeError:
                    pass

        else:
            raise ValueError('Option data_format of MultiChannelInterpolator '
                             'should be either'
                             ' \'channels_first\' or \'channels_last\'.')

    def __repr__(self):
        return '{}({}D, {})'.format(
            self.__class__.__name__, self.image.ndim - 1, self.data_format)

    @property
    def shape(self):
        return self.image.shape

    def sample(self, points, **kwargs):
        """
        Samples the image at given points.

        Args:
            points (np.array): An N x ndims array of points.
            **kwargs (dict): redirected to wrapped Interpolator's sample() method
        Returns:
            np.array: nchan x N-shaped or N x nchan-shaped array of intensities
                at the points (depending on data_format).
        """
        cvals = kwargs.pop('cval', self.nchan * [0])

        if self.data_format == 'channels_last':
            return np.rollaxis(np.array([
                x.sample(points, cval=cval, **kwargs) for cval, x in zip(cvals, self.interpolators)
            ]), 0, self.image.ndim)
        if self.data_format == 'channels_first':    
            return np.array([
                x.sample(points, cval=cval, **kwargs) for cval, x in zip(cvals, self.interpolators)
            ])

    def resample(self, grid, **kwargs):
        """
        Reamples the image at a given grid.

        Args:
            grid (Grid): The new grid.
            **kwargs (dict): redirected to wrapped Interpolator's resample() method
        Returns:
            np.array: The resampled image at the new grid.
        """
        rescaled_grid = grid.scaled_to(self.interpolators[0].image.shape)
        return self.sample(rescaled_grid.grid,
                           **kwargs)

    def transform(self, *transforms, **kwargs):
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
        transformed_grid = self.interpolators[0].grid.transform(*transforms)
        return self.resample(transformed_grid, **kwargs)
