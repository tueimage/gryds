#! /usr/bin/env python
#
# Transformation base class.
#
# @author: Koen Eppenhof
# @email: k.a.j.eppenhof@tue.nl
# @date: 2018/08/30


from __future__ import division, print_function, absolute_import

from .grid import Grid
from ..config import DTYPE


class Interpolator(object):
    """Base class for interpolators, that implements the minimum requirements
    for sampling on grid in new images.
    """

    def __init__(self, image):
        """
        Args:
            ndim (int): Number of dimensions of the transformation, used for
                checking the dimensions of points to be transformed.
        """
        self.image = image
        self.grid = Grid(shape=self.image.shape)

    @property
    def shape(self):
        return self.image.shape

    def sample(self, points, **kwargs):
        raise NotImplementedError()

    def resample(self, points, **kwargs):
        raise NotImplementedError()

    def transform(self, *transforms):
        transformed_grid = self.grid.transform(*transforms)
        new_grid = self.resample(transformed_grid)
        return new_grid.astype(DTYPE)
