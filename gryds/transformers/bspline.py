#! /usr/bin/env python
#
# BSpline transformation


from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.ndimage as nd
from ..config import DTYPE
from .base import Transformation
from .affine import _center_of


class BSplineTransformation(Transformation):
    """BSpline transformation of points."""

    def __init__(self, grid, order=3, mode='mirror', cval=0):
        """
        Args:
            grid (np.array): An (ndim x N1 x N2 x ... Nndim) sized array of
                displacements for grid points.
            order (int): The order of the B-spline. Default is 3. Use 0 for
                binary images. Use 1 for normal linear interpolation.
            mode (str): How edges of image domain should be treated when
                transformed of 'constant', 'nearest', 'mirror', 'reflect',
                'wrap'. Default is 'constant'. See https://docs.scipy.org/doc/
                scipy-0.14.0/reference/generated/
                scipy.ndimage.interpolation.map_coordinates.html for more
                information about modes.
            cval (numeric): Constant value for mode='constant'
        Raises:
            ValueError: If grid.shape[0] is not equal to grid.ndim -1
        """
        grid = np.array(grid, dtype=DTYPE)
        if grid.shape[0] is not grid.ndim - 1:
            raise ValueError('First axis of grid should be equal to '
                             'transform\'s ndim {}.'.format(grid.ndim - 1))
        self.bspline_order = order
        self.mode = mode
        self.cval = cval
        super(BSplineTransformation, self).__init__(
            ndim=len(grid),
            parameters=grid
        )

    @classmethod
    def from_deformation_field(cls, field, order=3, mode='mirror', cval=0):
        """Start from a given deformation field.

        TODO: check if this makes a difference.
        """
        padding = [(0, 0)]
        for i in range(1, field.ndim):
            padding.append((0, 1))
        return cls(grid=np.pad(field, padding, mode='constant'),
            order=order, mode=mode, cval=cval)

    def _transform_points(self, points):
        assert points.dtype == DTYPE
        # Empty list for the interpolated B-spline grid's components.
        displacement = []

        # Points is in the [0, 1)^ndim domain. Here it is scaled to the
        # B-spline grid's size.
        scaled_points = points * (
            np.array(self.parameters.shape[1:], dtype=DTYPE) - 1)[:, None]

        # Every component (e.g. Tx, Ty, Tz in 3D) of the B-spline grid is
        # interpolated at the scaled point's positions.
        for bspline_component in self.parameters:
            displacement.append(
                nd.map_coordinates(bspline_component, scaled_points,
                                   order=self.bspline_order,
                                   mode=self.mode,
                                   cval=self.cval)
            )
        result = (points + np.array(displacement))
        assert result.dtype == DTYPE
        return result
