#! /usr/bin/env python
#
# Translation transformation
#
# @author: Koen Eppenhof
# @email: k.a.j.eppenhof@tue.nl
# @date: 2018/08/30


from __future__ import division, print_function, absolute_import

import numpy as np
from ..config import DTYPE
from .base import Transformation


class TranslationTransformation(Transformation):
    """Translation of points."""

    def __init__(self, translation):
        """
        Args:
            translation (np.array): Translation vector.
        """
        super(TranslationTransformation, self).__init__(
            ndim=len(translation),
            parameters=np.array(translation)
        )

    def _transform_points(self, points):
        result = (points + self.parameters[:, None])
        assert result.dtype == DTYPE
        return result
