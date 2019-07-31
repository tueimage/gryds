from __future__ import absolute_import

import sys
import os

sys.path.append(os.path.abspath('../gryds'))

from unittest import TestCase
import numpy as np
import gryds
DTYPE = gryds.DTYPE


try:
    gryds.BSplineInterpolatorCuda
except AttributeError:
    print('Cuda tests not run because Cupy was not installed.')
else:
    class TestBSplineCudaInterpolator(TestCase):
        """Tests grid initialization and scaling."""
        
        def test_2d_cuda_interpolator_90_deg_rotation(self):
            image = np.array([
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0]
            ], dtype=DTYPE)
            expected = np.array([
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0]
            ], dtype=DTYPE) # Borders will be zero due to being outside of image domain
            intp = gryds.BSplineInterpolatorCuda(image)
            trf = gryds.AffineTransformation(ndim=2, angles=[np.pi/2.], center=[0.4, 0.4])
            new_image = intp.transform(trf).astype(DTYPE)
            np.testing.assert_almost_equal(expected, new_image, decimal=4)

        def test_2d_cuda_interpolator_45_deg_rotation(self):
            image = np.array([
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0]
            ], dtype=DTYPE)
            expected = np.array([
                [0, 0, 0, 0, 0],
                [0, 1., 0.5, 1., 0],
                [0., 0.5, 1., 0.5, 0.],
                [0, 1., 0.5, 1., 0],
                [0, 0, 0, 0, 0]
            ], dtype=DTYPE) # Borders will be zero due to being outside of image domain
            intp = gryds.BSplineInterpolatorCuda(image)
            trf = gryds.AffineTransformation(ndim=2, angles=[np.pi/4.], center=[0.4, 0.4])
            new_image = intp.transform(trf).astype(DTYPE)
            np.testing.assert_almost_equal(expected, new_image, decimal=4)

        def test_3d_cuda_interpolator_90_deg_rotation(self):
            image = np.zeros((2, 5, 5))
            image[1] = np.array([[
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0]
            ]], dtype=DTYPE)
            image[0] = image[1]
            expected = np.zeros((2, 5, 5))
            expected[0] = np.array([
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0]
            ], dtype=DTYPE) # Borders will be zero due to being outside of image domain
            expected[1] = expected[0]
            intp = gryds.BSplineInterpolatorCuda(image)
            trf = gryds.AffineTransformation(ndim=3, angles=[np.pi/2., 0, 0], center=[0.4, 0.4, 0.4])
            new_image = intp.transform(trf).astype(DTYPE)
            np.testing.assert_almost_equal(expected, new_image, decimal=4)

        def test_3d_cuda_interpolator_45_deg_rotation(self):
            image = np.array([
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0]
            ], dtype=DTYPE)
            expected = np.array([
                [0, 0, 0., 0, 0],
                [0., 1., 0.5, 1., 0.],
                [0., 0.5, 1., 0.5, 0.],
                [0., 1., 0.5, 1., 0.],
                [0., 0., 0., 0., 0.]
            ], dtype=DTYPE) # Borders will be zero due to being outside of image domain
            intp = gryds.BSplineInterpolatorCuda(image)
            trf = gryds.AffineTransformation(ndim=2, angles=[np.pi/4.], center=[0.4, 0.4])
            new_image = intp.transform(trf).astype(DTYPE)
            np.testing.assert_almost_equal(expected, new_image, decimal=4)

        def test_normal_bspline_equal(self):
            bsp = gryds.BSplineTransformation(0.01 * (np.random.rand(2, 32, 32) - 0.5), order=1)
            image = np.zeros((128, 128))
            image[32:-32] = 0.5
            image[:, 32:-32] += 0.5
            intp_cpu = gryds.BSplineInterpolator(image, order=1).transform(bsp)
            intp_gpu = gryds.BSplineInterpolatorCuda(image).transform(bsp)
            np.testing.assert_equal(intp_cpu, intp_gpu)
