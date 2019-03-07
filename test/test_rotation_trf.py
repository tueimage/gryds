from __future__ import absolute_import

from unittest import TestCase
import numpy as np
import gryds
DTYPE = gryds.DTYPE


class TestRotation(TestCase):
    """Tests rotation, and associated effect on grids and Jacobians"""

    def test_2d_90_deg_rotation(self):
        trf = gryds.AffineTransformation(ndim=2, angles=[0.5 * np.pi]) # rotate grid 90° clockwise

        grid = gryds.Grid((10, 20))
        new_grid = grid.transform(trf)

        # The grid runs from 0 to 0.95 on the j-axis
        # 90 deg rot means the i-axis will run from 0 to -0.95
        np.testing.assert_equal(new_grid.grid[0, 0, 0], np.array(0, DTYPE))
        np.testing.assert_equal(new_grid.grid[0, 0, -1], np.array(-0.95, DTYPE))

        # The jacobian of this transformation should be 1 everywhere, i.e. no
        # scaling should have happened
        np.testing.assert_almost_equal(
            grid.jacobian_det(trf),
            np.array(1, DTYPE),
            decimal=4)

    def test_2d_45_deg_rotation(self):
        trf = gryds.AffineTransformation(ndim=2, angles=[0.25 * np.pi]) # rotate grid 90° clockwise

        grid = gryds.Grid((10, 20))
        new_grid = grid.transform(trf)

        # The grid runs from 0 to 0.95 on the j-axis
        # 90 deg rot means the i-axis will run from 0 to -0.95
        np.testing.assert_equal(new_grid.grid[0, 0, 0], np.array(0, DTYPE))
        np.testing.assert_almost_equal(
            new_grid.grid[0, 0, -1],
            np.array(-0.671751442, DTYPE), # 0.95 / sqrt(2)
            decimal=6)

        # The jacobian of this transformation should be 1 everywhere, i.e. no
        # scaling should have happened
        np.testing.assert_almost_equal(
            grid.jacobian_det(trf),
            np.array(1, DTYPE),
            decimal=4)

    def test_3d_90_deg_rotation(self):
        trf = gryds.AffineTransformation(ndim=3, angles=[0, 0.5 * np.pi, 0]) # rotate grid 90° clockwise

        grid = gryds.Grid((10, 20, 20))
        new_grid = grid.transform(trf)

        # The grid runs from 0 to 0.95 on the j-axis
        # 90 deg rot means the i-axis will run from 0 to -0.95
        np.testing.assert_equal(new_grid.grid[0, 0, 0, 0], np.array(0, DTYPE))

        np.testing.assert_almost_equal(
            new_grid.grid[0, 0, 0, -1],
            np.array(0.95, DTYPE),
            decimal=6)

        # The jacobian of this transformation should be 1 everywhere, i.e. no
        # scaling should have happened
        np.testing.assert_almost_equal(
            grid.jacobian_det(trf),
            np.array(1, DTYPE),
            decimal=4)

    def test_3d_45_deg_rotation(self):
        trf = gryds.AffineTransformation(ndim=3, angles=[0, 0.25 * np.pi, 0]) # rotate grid 90° clockwise

        grid = gryds.Grid((10, 20, 20))
        new_grid = grid.transform(trf)

        # The grid runs from 0 to 0.95 on the j-axis
        # 90 deg rot means the i-axis will run from 0 to -0.95
        np.testing.assert_equal(new_grid.grid[0, 0, 0, 0], np.array(0, DTYPE))

        np.testing.assert_almost_equal(
            new_grid.grid[0, 0, 0, -1],
            np.array(0.671751442, DTYPE),
            decimal=6)

        # The jacobian of this transformation should be 1 everywhere, i.e. no
        # scaling should have happened
        np.testing.assert_almost_equal(
            grid.jacobian_det(trf),
            np.array(1, DTYPE),
            decimal=4)

