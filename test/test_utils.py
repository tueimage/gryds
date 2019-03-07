from __future__ import absolute_import

import sys
import os

sys.path.append(os.path.abspath('../gryds'))

from unittest import TestCase
import numpy as np
import gryds
DTYPE = gryds.DTYPE


class TestUtils(TestCase):

    def test_dvf_opts(self):
        self.assertEqual(
            gryds.dvf_opts(np.array([1, 2, 3])),
            {
                'cmap': 'bwr',
                'vmin': -3,
                'vmax': 3
            }
        )

    def test_dvf_show(self):
        dvf = np.array([1, 2, 3])
        self.assertDictEqual(
            gryds.dvf_show(dvf),
            {
                'X': dvf,
                'cmap': 'bwr',
                'vmin': -3,
                'vmax': 3
            }
        )        

    def test_max_no_fold(self):
        np.random.seed(0)
        random_grid = gryds.utils.max_no_fold((2, 200, 300))
        print(random_grid.max())
        self.assertTrue(
            np.all(random_grid <= .000628141)
        )
        self.assertTrue(
            np.all(random_grid >= -.000628141)
        )


    def test_phantom(self):
        phantom = gryds.utils.phantom_image((3, 10), spacing=4)

        np.testing.assert_equal(phantom[0], 1)
        np.testing.assert_equal(phantom[1], [1, 0, 0, 0, 1, 0, 0, 0, 1, 0])
        np.testing.assert_equal(phantom[2], [1, 0, 0, 0, 1, 0, 0, 0, 1, 0])

