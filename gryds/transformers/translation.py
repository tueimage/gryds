#! /usr/bin/env python
#
# Translation transformation


from __future__ import division, print_function, absolute_import

import numpy as np
from ..config import DTYPE
from .base import Transformation


class TranslationTransformation(Transformation):
    """Translation of points.

    Attributes:
        ndim (int): The number of dimensions.
        parameters (np.ndarray): Translation vector.
    """

    def __init__(self, translation):
        """
        Args:
            translation (np.array): Translation vector.
        """
        super(TranslationTransformation, self).__init__(
            ndim=len(translation),
            parameters=np.array(translation)
        )

    def __repr__(self):
        return '{}({}D, t={})'.format(self.__class__.__name__, self.ndim, self.parameters)

    def _transform_points(self, points):
        result = (points + self.parameters[:, None])
        return result
