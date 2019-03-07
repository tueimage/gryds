from __future__ import absolute_import

import sys
import os

sys.path.append(os.path.abspath('../gryds'))

from unittest import TestCase
import numpy as np
import gryds
DTYPE = gryds.DTYPE


class TestAnisotropic(TestCase):
    """Tests anisotropic scaling, and associated effect on grids and Jacobians"""

    def test_2d_downscaling(self):
        trf = gryds.AffineTransformation(ndim=2, scaling=[1.5, 1]) # scale grid by 150% isotropically

        grid = gryds.Grid((10, 20))
        new_grid = grid.transform(trf)

        # The original grid runs from 0 to 0.9 for the i-coordinates
        # The transformed grid should run from 0 to 1.35
        np.testing.assert_equal(new_grid.grid[0, 0], np.array(0, DTYPE))
        np.testing.assert_almost_equal(
            new_grid.grid[0, 9],
            np.array(1.35, DTYPE),
            decimal=6
        )

        # The jacobian of this transformation should be 1 everywhere, i.e. no
        # scaling should have happened
        np.testing.assert_almost_equal(
            grid.jacobian_det(trf),
            np.array(1.5, DTYPE), # i.e. 1.5*1.5
            decimal=4)

    def test_5d_downscaling(self):
        matrix = np.zeros((5, 6))
        for i in range(5):
            matrix[i, i] = 1
        matrix[2, 2] = 1.5

        trf = gryds.LinearTransformation(matrix) # scale grid by 150$ isotropically

        grid = gryds.Grid((2, 3, 4, 5, 6))
        new_grid = grid.transform(trf)

        # The original grid runs from 0 to 0.9
        # The transformed grid should run from 0 to 1.35
        np.testing.assert_equal(new_grid.grid[0, 0], np.array(0, DTYPE))
        np.testing.assert_almost_equal(
            new_grid.grid[2, :, :, 3],
            np.array(1.125, DTYPE), # 1.5 * 0.75
            decimal=6
        )

        # The jacobian of this transformation should be 1 everywhere, i.e. no
        # scaling should have happened
        np.testing.assert_almost_equal(
            grid.jacobian_det(trf),
            np.array(1.5, DTYPE), # i.e. 1.5^5
            decimal=4)
