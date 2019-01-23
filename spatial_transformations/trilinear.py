#! /usr/bin/env python
#
# @author: Koen Eppenhof
# @email: k.a.j.eppenhof@tue.nl
# @date: 2018/08/30


from __future__ import division, print_function, absolute_import

from .config import DTYPE
from .grid import Grid
import numpy as np


class Trilinear(object):

    def __init__(self, image):
        self.image = image
    
    def resample(self, grid):
        X, Y, Z = grid.scaled_to(self.image.shape).grid
        X0 = np.floor(X).astype('int')
        Y0 = np.floor(Y).astype('int')
        Z0 = np.floor(Z).astype('int')
        X1 = X0 + 1
        Y1 = Y0 + 1
        Z1 = Z0 + 1

        X0 = np.clip(X0, 0, self.image.shape[0] - 1)
        X1 = np.clip(X1, 0, self.image.shape[0] - 1)
        Y0 = np.clip(Y0, 0, self.image.shape[1] - 1)
        Y1 = np.clip(Y1, 0, self.image.shape[1] - 1)
        Z0 = np.clip(Z0, 0, self.image.shape[2] - 1)
        Z1 = np.clip(Z1, 0, self.image.shape[2] - 1)

        b_x0y0z0 = self.image[X0, Y0, Z0]
        b_x1y0z0 = self.image[X1, Y0, Z0]
        b_x0y1z0 = self.image[X0, Y1, Z0]
        b_x1y1z0 = self.image[X1, Y1, Z0]
        b_x0y0z1 = self.image[X0, Y0, Z1]
        b_x1y0z1 = self.image[X1, Y0, Z1]
        b_x0y1z1 = self.image[X0, Y1, Z1]
        b_x1y1z1 = self.image[X1, Y1, Z1]
        
        b = (X1 - X) * (Y1 - Y) * (Z1 - Z) * b_x0y0z0 + \
        (X - X0) * (Y1 - Y) * (Z1 - Z) * b_x1y0z0 + \
        (X1 - X) * (Y - Y0) * (Z1 - Z) * b_x0y1z0 + \
        (X - X0) * (Y - Y0) * (Z1 - Z) * b_x1y1z0 + \
        (X1 - X) * (Y1 - Y) * (Z - Z0) * b_x0y0z1 + \
        (X - X0) * (Y1 - Y) * (Z - Z0) * b_x1y0z1 + \
        (X1 - X) * (Y - Y0) * (Z - Z0) * b_x0y1z1 + \
        (X - X0) * (Y - Y0) * (Z - Z0) * b_x1y1z1
        return b
