from __future__ import absolute_import

from unittest import TestCase
import numpy as np
import gryds
DTYPE = gryds.DTYPE


class TestGrid(TestCase):

    def __init__(self, *args, **kwargs):
        self.places = 6
        super(TestGrid, self).__init__(*args, **kwargs)

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
            places=self.places
        )
        self.assertAlmostEqual(
            new_grid.grid[1, 0, 19],
            np.array(3.8, dtype=DTYPE),
            places=self.places
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
            places=self.places
        )
        self.assertAlmostEqual(
            new_grid.grid[4, 0, 0, 0, 0, 9],
            np.array(6.3, dtype=DTYPE), # 7 x 9 / 10
            places=self.places
        )

