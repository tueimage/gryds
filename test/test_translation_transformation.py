from __future__ import absolute_import

import sys
import os

sys.path.append(os.path.abspath('../gryds'))

from unittest import TestCase
import numpy as np
import gryds
DTYPE = gryds.DTYPE


class TestTranslation(TestCase):
    """Tests translation, and associated effect on grids and Jacobians"""

    def test_2d_translation(self):
        trf = gryds.TranslationTransformation([0.1, 0]) # move grid 10% downwards (moves image 10% upwards)

        grid = gryds.Grid((10, 20))
        new_grid = grid.transform(trf)

        # The original grid runs from 0 to 0.9 for the i-coordinates
        # The transformed grid should run from 0.1 to 1
        np.testing.assert_equal(new_grid.grid[0, 0], np.array(0.1, DTYPE))
        np.testing.assert_equal(new_grid.grid[0, 9], np.array(1.0, DTYPE))

        # The jacobian of this transformation should be 1 everywhere, i.e. no
        # scaling should have happened
        np.testing.assert_almost_equal(
            grid.jacobian_det(trf),
            np.array(1, DTYPE),
            decimal=4)

    def test_5d_translation(self):
        trf = gryds.TranslationTransformation([0, 0, 0.1, 0, 0])

        grid = gryds.Grid((10, 10, 10, 10, 10))
        new_grid = grid.transform(trf)

        # The original grid runs from 0 to 0.9
        # The transformed grid should run from 0.1 to 1
        self.assertTrue(np.all(new_grid.grid[2, :, :, 0] == np.array(0.1, DTYPE)))
        self.assertTrue(np.all(new_grid.grid[2, :, :, 9] == np.array(1.0, DTYPE)))

        # The jacobian of this transformation should be 1 everywhere, i.e. no
        # scaling should have happened
        np.testing.assert_almost_equal(
            grid.jacobian_det(trf),
            np.array(1, DTYPE),
            decimal=4)

    def test_repr(self):
        self.assertEqual(str(gryds.TranslationTransformation([3, 4])), 'TranslationTransformation(2D, t=[3 4])')
