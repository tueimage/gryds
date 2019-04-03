#! /usr/bin/env python
#
# BSpline transformation


from __future__ import division, print_function, absolute_import

import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as nd
from ..config import DTYPE
from .base import Transformation
from .bspline import BSplineTransformation


class BSplineTransformationCuda(BSplineTransformation):
    """BSpline transformation of points.

    Attributes:
        ndim (int): The number of dimensions.
        parameters (np.ndarray): The control point grid in 
            ndim x Ni x Nj x ... x Nndim format.
        bspline_order (int): The order of the B-spline.
        mode (str): How edges of image domain should be treated when transformed.
        cval (numeric): Constant value for mode='constant'
    """

    def __init__(self, grid, order=1, mode='mirror', cval=0):
        """
        Args:
            grid (np.array): An (ndim x N1 x N2 x ... Nndim) sized array of
                displacements for grid points.
            order (int): B-Spline order. Currently, only 0 and 1 are
                supported.
            mode (str): How edges of image domain should be treated when
                transformed. One of 'constant', 'nearest', 'mirror', 'reflect',
                'wrap'. Default is 'constant'. See https://docs.scipy.org/doc/
                scipy-0.14.0/reference/generated/
                scipy.ndimage.interpolation.map_coordinates.html for more
                information about modes.
            cval (numeric): Constant value for mode='constant'
        Raises:
            ValueError: If grid.shape[0] is not equal to grid.ndim -1
        """
        super(BSplineTransformationCuda, self).__init__(
            grid=grid,
            order=order,
            mode=mode,
            cval=cval
        )

    def _transform_points(self, points):
        assert points.dtype == DTYPE
        # Empty list for the interpolated B-spline grid's components.
        displacement = []

        # Reshape points for the cupy map_coordinates function to
        # receive coordinates in the expected shape
        points_gpu = points.reshape(self.ndim, -1)

        # Points is in the [0, 1)^ndim domain. Here it is scaled to the
        # B-spline grid's size.
        scaled_points = points_gpu * (
            np.array(self.parameters.shape[1:], dtype=DTYPE) - 1)[:, None]

        # Every component (e.g. Tx, Ty, Tz in 3D) of the B-spline grid is
        # interpolated at the scaled point's positions.
        for bspline_component in self.parameters:
            displacement.append(
                cp.asnumpy(
                    nd.map_coordinates(input=cp.array(bspline_component),
                                   coordinates=cp.array(scaled_points),
                                   order=self.bspline_order,
                                   mode=self.mode,
                                   cval=self.cval)
                )
            )
        result = points + np.array(displacement).reshape(points.shape)

        assert result.dtype == DTYPE
        return result
