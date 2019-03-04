#! /usr/bin/env python
#
# Resample images on a new Grid instance using B-spline interplation
#
# @author: Koen Eppenhof
# @email: k.a.j.eppenhof@tue.nl
# @date: 2018/08/30


from __future__ import division, print_function, absolute_import

from ..config import DTYPE
from .grid import Grid
from .bspline import BSplineInterpolator

import numpy as np


class MultiChannelInterpolator:

    def __init__(self, image, interpolator=BSplineInterpolator,
            data_format='channels_last', cval=None, **kwargs):
        self.image = image
        self.data_format = data_format

        n_channels = image.shape[-1] if data_format == 'channels_last' \
            else image.shape[0]
        if cval is not None:
            assert len(cval) == n_channels
        else:
            cval = n_channels * [0]

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

    @property
    def shape(self):
        return self.image.shape

    def sample(self, points, **kwargs):
        if self.data_format == 'channels_last':
            return np.rollaxis(np.array([
                x.sample(points, **kwargs) for x in self.interpolators
            ]), 0, self.image.ndim)
        if self.data_format == 'channels_first':    
            return np.array([
                x.sample(points, **kwargs) for x in self.interpolators
            ])

    def resample(self, grid, **kwargs):
        if self.data_format == 'channels_last':
            return np.rollaxis(np.array([
                x.resample(grid, **kwargs) for x in self.interpolators
            ]), 0, self.image.ndim)
        if self.data_format == 'channels_first':    
            return np.array([
                x.resample(grid, **kwargs) for x in self.interpolators
            ])

    def transform(self, *transforms, **kwargs):
        if self.data_format == 'channels_last':
            return np.rollaxis(np.array([
                x.transform(*transforms, **kwargs) for x in self.interpolators
            ]), 0, self.image.ndim)
        if self.data_format == 'channels_first':    
            return np.array([
                x.transform(*transforms, **kwargs) for x in self.interpolators
            ])
