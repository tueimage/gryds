from __future__ import absolute_import

import sys
import os

sys.path.append(os.path.abspath('../gryds'))

from unittest import TestCase
import numpy as np
import gryds
DTYPE = gryds.DTYPE


class TestBSplineInterpolator(TestCase):

    def test_2d_bspline_interpolator_90_deg_rotation(self):
        image = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ], dtype=DTYPE)
        intp = gryds.Interpolator(image)
        trf = gryds.AffineTransformation(ndim=2, angles=[np.pi/2.], center=[0.4, 0.4])
        new_image = intp.transform(trf, mode='mirror').astype(DTYPE)
        np.testing.assert_almost_equal(image, new_image, decimal=4)

    def test_2d_bspline_interpolator_45_deg_rotation(self):
        image = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ], dtype=DTYPE)
        expected = np.array([
            [1., 0.2929, 0., 0.2929, 1.],
            [0.2929, 1., 0.5, 1., 0.2929],
            [0., 0.5, 1., 0.5, 0.],
            [0.2929, 1., 0.5, 1., 0.2929],
            [1., 0.2929, 0., 0.2929, 1.]
        ], dtype=DTYPE)
        intp = gryds.Interpolator(image, order=1, mode='mirror')
        trf = gryds.AffineTransformation(ndim=2, angles=[np.pi/4.], center=[0.4, 0.4])
        new_image = intp.transform(trf).astype(DTYPE)
        np.testing.assert_almost_equal(expected, new_image, decimal=4)

    def test_repr(self):
        self.assertEqual(str(gryds.BSplineTransformation(np.random.rand(2, 5, 7))), 'BSplineTransformation(2D, 5x7)')
