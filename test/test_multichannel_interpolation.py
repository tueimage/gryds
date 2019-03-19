from __future__ import absolute_import

import sys
import os

sys.path.append(os.path.abspath('../gryds'))

from unittest import TestCase
import numpy as np
import gryds
DTYPE = gryds.DTYPE


class TestMultiChannelInterpolator(TestCase):

    def test_bspline_channels_first(self):
        image = np.array(3 * [[
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ]], dtype=DTYPE)

        intp = gryds.MultiChannelInterpolator(image, data_format='channels_first', cval=[0, 0, 0])
        trf = gryds.AffineTransformation(ndim=2, angles=[np.pi/2.], center=[0.4, 0.4])
        new_image = intp.transform(trf, mode='mirror').astype(DTYPE)
        np.testing.assert_almost_equal(image, new_image, decimal=4)

        intp = gryds.MultiChannelInterpolator(image, data_format='channels_first')
        trf = gryds.AffineTransformation(ndim=2, angles=[np.pi/2.], center=[0.4, 0.4])
        new_image = intp.transform(trf, mode='mirror').astype(DTYPE)
        np.testing.assert_almost_equal(image, new_image, decimal=4)
    
    def test_bspline_channels_last(self):
        image = np.array(3 * [[
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ]], dtype=DTYPE).transpose(1, 2, 0)

        intp = gryds.MultiChannelInterpolator(image, data_format='channels_last', cval=[0, 0, 0])
        trf = gryds.AffineTransformation(ndim=2, angles=[np.pi/2.], center=[0.4, 0.4])
        new_image = intp.transform(trf, mode='mirror').astype(DTYPE)
        np.testing.assert_almost_equal(image, new_image, decimal=4)

        intp = gryds.MultiChannelInterpolator(image, data_format='channels_last')
        trf = gryds.AffineTransformation(ndim=2, angles=[np.pi/2.], center=[0.4, 0.4])
        new_image = intp.transform(trf, mode='mirror').astype(DTYPE)
        np.testing.assert_almost_equal(image, new_image, decimal=4)

    def test_linear_channels_first(self):
        image = np.array(3 * [[
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ]], dtype=DTYPE)
     
        expected = np.array(3 * [[
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]], dtype=DTYPE)

        intp = gryds.MultiChannelInterpolator(image, gryds.LinearInterpolator, data_format='channels_first', cval=[0, 0, 0])
        trf = gryds.AffineTransformation(ndim=2, angles=[np.pi/2.], center=[0.4, 0.4])
        new_image = intp.transform(trf).astype(DTYPE)
        np.testing.assert_almost_equal(expected, new_image, decimal=4)

        intp = gryds.MultiChannelInterpolator(image, gryds.LinearInterpolator, data_format='channels_first')
        trf = gryds.AffineTransformation(ndim=2, angles=[np.pi/2.], center=[0.4, 0.4])
        new_image = intp.transform(trf).astype(DTYPE)
        np.testing.assert_almost_equal(expected, new_image, decimal=4)
    
    def test_linear_channels_last(self):
        image = np.array(3 * [[
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ]], dtype=DTYPE).transpose(1, 2, 0)

        expected = np.array(3 * [[
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]], dtype=DTYPE).transpose(1, 2, 0)

        intp = gryds.MultiChannelInterpolator(image, gryds.LinearInterpolator, data_format='channels_last', cval=[0, 0, 0])
        trf = gryds.AffineTransformation(ndim=2, angles=[np.pi/2.], center=[0.4, 0.4])
        new_image = intp.transform(trf).astype(DTYPE)
        np.testing.assert_almost_equal(expected, new_image, decimal=4)

        intp = gryds.MultiChannelInterpolator(image, gryds.LinearInterpolator, data_format='channels_last')
        trf = gryds.AffineTransformation(ndim=2, angles=[np.pi/2.], center=[0.4, 0.4])
        new_image = intp.transform(trf).astype(DTYPE)
        np.testing.assert_almost_equal(expected, new_image, decimal=4)

    def test_data_format_error(self):
        # Test if data_format error is raised
        self.assertRaises(ValueError,
            gryds.MultiChannelInterpolator, np.array([]), data_format='channels_second')

    def test_shape_prop(self):
        image = np.array(3 * [[
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ]], dtype=DTYPE).transpose(1, 2, 0)
        intp = gryds.MultiChannelInterpolator(image, gryds.LinearInterpolator, data_format='channels_last', cval=[0, 0, 0])
        self.assertEqual(intp.shape, (5, 5, 3))

    def test_repr(self):
        self.assertEqual(
            str(gryds.MultiChannelInterpolator(np.random.rand(3, 20, 20))),
            'MultiChannelInterpolator(2D, channels_last)')
