from __future__ import absolute_import

import sys
import os

sys.path.append(os.path.abspath('../gryds'))

from unittest import TestCase
import numpy as np
import gryds
DTYPE = gryds.DTYPE


class TestBSplineTransformation(TestCase):
    """Tests BSpline transformations, and associated effect on grids and Jacobians"""

    def test_translation_bspline_2d(self):
        bspline_grid = np.ones((2, 2, 2))
        trf = gryds.BSplineTransformation(bspline_grid)

        grid = gryds.Grid((10, 20))
        new_grid = grid.transform(trf)

        # The grid runs from 0 to 0.9 on the i-axis
        # Translation by 100% will mean that the i-axis will now run from 1 to 1.9
        np.testing.assert_equal(new_grid.grid[0, 0, 0], np.array(1, DTYPE))
        np.testing.assert_equal(new_grid.grid[0, -1, 0], np.array(1.9, DTYPE))

        # The jacobian of this transformation should be 1 everywhere, i.e. no
        # scaling should have happened
        np.testing.assert_almost_equal(
            grid.jacobian_det(trf),
            np.array(1, DTYPE),
            decimal=4)

    def test_translation_bspline_5d(self):
        bspline_grid = np.ones((5, 2, 2, 2, 2, 2))
        trf = gryds.BSplineTransformation(bspline_grid)

        grid = gryds.Grid((3, 3, 3, 3, 3))
        new_grid = grid.transform(trf)

        # The grid runs from 0 to 0.9 on the i-axis
        # Translation by 100% will mean that the i-axis will now run from 1 to 1.9
        np.testing.assert_equal(new_grid.grid[0, 0, 0, 0, 0], np.array(1, DTYPE))
        np.testing.assert_almost_equal(new_grid.grid[0, -1, 0, 0, 0], np.array(1.6666667, DTYPE))

        # The jacobian of this transformation should be 1 everywhere, i.e. no
        # scaling should have happened
        np.testing.assert_almost_equal(
            grid.jacobian_det(trf),
            np.array(1, DTYPE),
            decimal=4)

    def test_bspline_2d(self):
        bspline_grid = np.array([
            [[0.1, 0], [0, 0]],
            [[0, 0], [0, 0]]
        ])
        trf = gryds.BSplineTransformation(bspline_grid)

        grid = gryds.Grid((10, 20))
        new_grid = grid.transform(trf)

        # The top left has been displaced by 10% or 0.1 pixels in the i-direction
        np.testing.assert_almost_equal(new_grid.grid[0, 0, 0], np.array(0.1, DTYPE))

        # The jacobian of this transformation should NOT be 1 everywhere, i.e.
        # scaling should have happened, and the new volume should be smaller
        # as the top left has been folded in
        self.assertTrue(np.all(grid.jacobian_det(trf) < np.array(1, DTYPE)))

    def test_bspline_2d_folding(self):
        bspline_grid = np.array([
            [[0.51, 0.51], [-0.5, -0.5]], # Folds the top half of the image slightly over the bottom half.
            [[0, 0], [0, 0]]
        ])
        trf = gryds.BSplineTransformation(bspline_grid, order=1)

        grid = gryds.Grid((100, 20))

        # The jacobian of this transformation should be below 0 everywhere
        self.assertTrue(np.all(grid.jacobian_det(trf) < 0))

    def test_bspline_wrong_grid_size(self):
        bspline_grid = np.random.rand(3, 10, 10)
        self.assertRaises(ValueError, gryds.BSplineTransformation, bspline_grid)
