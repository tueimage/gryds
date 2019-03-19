from __future__ import absolute_import

import sys
import os

sys.path.append(os.path.abspath('../gryds'))

from unittest import TestCase
import numpy as np
import gryds
DTYPE = gryds.DTYPE


class TestGrid(TestCase):
    """Tests grid initialization and scaling."""
    
    def test_2d_grid_ranges(self):
        a_grid = gryds.Grid((10, 20))
        
        self.assertEqual(
            a_grid.grid[0, 9, 0],
            np.array(0.9, dtype=DTYPE)
        )        
        self.assertEqual(
            a_grid.grid[1, 0, 19],
            np.array(0.95, dtype=DTYPE)
        )        

    def test_2d_grid_scaling(self):
        a_grid = gryds.Grid((10, 20))

        new_grid = a_grid.scaled_to((3, 4))
        self.assertAlmostEqual(
            new_grid.grid[0, 9, 0],
            np.array(2.7, dtype=DTYPE),
            places=6
        )
        self.assertAlmostEqual(
            new_grid.grid[1, 0, 19],
            np.array(3.8, dtype=DTYPE),
            places=6
        )

    def test_5d_grid_range(self):
        a_grid = gryds.Grid((10, 10, 10, 10, 10))

        self.assertEqual(
            a_grid.grid[0, 9, 0, 0, 0, 0],
            np.array(0.9, dtype=DTYPE)
        )
        self.assertEqual(
            a_grid.grid[4, 0, 0, 0, 0, 9],
            np.array(0.9, dtype=DTYPE)
        )

    def test_5d_grid_scaling(self):
        a_grid = gryds.Grid((10, 10, 10, 10, 10))

        new_grid = a_grid.scaled_to((3, 4, 5, 6, 7))

        self.assertAlmostEqual(
            new_grid.grid[0, 9, 0, 0, 0, 0],
            np.array(2.7, dtype=DTYPE), # 3 x 9 / 10
            places=6
        )
        self.assertAlmostEqual(
            new_grid.grid[4, 0, 0, 0, 0, 9],
            np.array(6.3, dtype=DTYPE), # 7 x 9 / 10
            places=6
        )

    def test_grid_repr(self):
        a_grid = gryds.Grid((2, 2))
        self.assertEqual(str(a_grid), 'Grid(2D, 2x2)')

    def test_grid_init(self):
        gryds.Grid(grid=np.zeros((2, 10, 10)))

    def test_grid_wrong_scale_shape(self):
        a_grid = gryds.Grid((2, 2))
        self.assertRaises(ValueError, a_grid.scaled_to, [1, 2, 3])

    def test_no_grid_no_shape(self):
        self.assertRaises(ValueError, gryds.Grid)
