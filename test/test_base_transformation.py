from __future__ import absolute_import

import sys
import os

sys.path.append(os.path.abspath('../gryds'))

from unittest import TestCase
import numpy as np
import gryds
DTYPE = gryds.DTYPE


class TestBaseTransformation(TestCase):

    def test_wrong_format_for_points(self):

        t = gryds.transformers.base.Transformation(ndim=3, parameters=[])
        points = np.random.rand(2, 10) # incompatible with t.ndim=3
        self.assertRaises(ValueError, t._dimension_check, points)

        points = np.random.rand(2, 3, 10) # not a list of points
        self.assertRaises(ValueError, t._dimension_check, points)

    def test_base_transformation_transform(self):

        t = gryds.transformers.base.Transformation(ndim=3, parameters=[])
        points = np.random.rand(2, 10) # incompatible with t.ndim=3
        self.assertRaises(NotImplementedError, t._transform_points, points)
