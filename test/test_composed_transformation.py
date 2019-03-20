#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys
import os

sys.path.append(os.path.abspath('../gryds'))

from unittest import TestCase
import numpy as np
import gryds
DTYPE = gryds.DTYPE


class TestComposition(TestCase):
    """Tests composed transformations by applying inverse transformations"""

    def test_translation(self):
        image = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ], dtype=DTYPE)
        intp = gryds.Interpolator(image, mode='mirror')

        trf1 = gryds.TranslationTransformation([0.1, 0])
        trf2 = gryds.TranslationTransformation([-0.1, 0])

        trf = gryds.ComposedTransformation(trf2, trf1)

        new_image = intp.transform(trf)
        np.testing.assert_almost_equal(image, new_image)
   
    def test_rotation(self):
        image = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ], dtype=DTYPE)
        intp = gryds.Interpolator(image, mode='mirror')

        trf1 = gryds.AffineTransformation(ndim=2, angles=[0.1])
        trf2 = gryds.AffineTransformation(ndim=2, angles=[-0.1])

        trf = gryds.ComposedTransformation(trf2, trf1)

        new_image = intp.transform(trf)
        np.testing.assert_almost_equal(image, new_image, decimal=6)
   
    def test_rotation_translation(self):
        image = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ], dtype=DTYPE)
        intp = gryds.Interpolator(image, mode='mirror')

        trf1 = gryds.TranslationTransformation([0.1, 0])
        trf2 = gryds.AffineTransformation(ndim=2, angles=[0.1])
        trf3 = gryds.AffineTransformation(ndim=2, angles=[-0.1])
        trf4 = gryds.TranslationTransformation([-0.1, 0])

        trf = gryds.ComposedTransformation(trf1, trf2, trf3, trf4)

        new_image = intp.transform(trf)
        np.testing.assert_almost_equal(image, new_image, decimal=6)

    def test_incompatible_error(self):
        trf1 = gryds.TranslationTransformation([0.1, 0, 0])
        trf2 = gryds.TranslationTransformation([-0.1, 0])

        self.assertRaises(ValueError, gryds.ComposedTransformation, trf2, trf1)

    def test_repr(self):
        bsp = gryds.BSplineTransformation(np.random.rand(2, 3, 4))
        self.assertEqual(str(gryds.ComposedTransformation(bsp, bsp)),
            'ComposedTransformation(2D, BSplineTransformation(2D, 3x4)∘BSplineTransformation(2D, 3x4))'
        )

    def test_no_transformations_supplied(self):
        self.assertRaises(ValueError, gryds.ComposedTransformation)
