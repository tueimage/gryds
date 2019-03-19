from __future__ import absolute_import

import sys
import os

sys.path.append(os.path.abspath('../gryds'))

from unittest import TestCase
import numpy as np
import gryds
DTYPE = gryds.DTYPE


class TestBaseInterpolator(TestCase):
    """Tests grid initialization and scaling."""
    
    def test_base_interpolator_shape(self):
        im = np.random.rand(10, 10, 10, 10)
        intp = gryds.base.Interpolator(im)
        np.testing.assert_equal(intp.shape, (10, 10, 10, 10))

    def test_base_interpolator_sample(self):
        im = np.random.rand(10, 10, 10, 10)
        intp = gryds.base.Interpolator(im)
        self.assertRaises(NotImplementedError, intp.sample, 0)

    def test_base_interpolator_resample(self):
        im = np.random.rand(10, 10, 10, 10)
        intp = gryds.base.Interpolator(im)
        self.assertRaises(NotImplementedError, intp.resample, 0)

    def test_base_interpolator_transform(self):
        trf = gryds.TranslationTransformation([1, 2, 3, 4, 5])
        im = np.random.rand(10, 10, 10, 10, 10)
        intp = gryds.base.Interpolator(im)
        self.assertRaises(NotImplementedError, intp.transform, trf)

    def test_repr(self):
        self.assertEqual(
            str(gryds.base.Interpolator(np.random.rand(20, 20))),
            'Interpolator(2D)'
        )
