#! /usr/bin/env python
#
# @author: Koen Eppenhof
# @email: k.a.j.eppenhof@tue.nl
# @date: 2018/08/30


import theano
import theano.tensor as T
import numpy as np

from lasagne.utils import as_tuple, floatX
from lasagne.layers.base import Layer, MergeLayer

__all__ = [
    "TransformationCompositionLayer",
]


class TransformationCompositionLayer(MergeLayer):
    def __init__(self, grid1, grid2, **kwargs):
        super(TransformationCompositionLayer, self).__init__(
            [grid1, grid2], **kwargs)

        grid1_shp, grid2_shp = self.input_shapes
        assert grid1_shp == grid2_shp

        if len(grid1_shp) != 5 or grid1_shp[1] != 3:
            raise ValueError("The input network must have a 5-dimensional "
                             "output shape: (batch_size, 3, "
                             "input_rows, input_columns, input_depths)")
        if len(grid2_shp) != 5 or grid2_shp[1] != 3:
            raise ValueError("The input network must have a 5-dimensional "
                             "output shape: (batch_size, 3, "
                             "input_rows, input_columns, input_depths)")
        base_grid_1 = np.array(
            [np.meshgrid(*[np.arange(x) for x in grid1_shp[2:]], indexing='ij')]
        )
        base_grid_2 = np.array(
            [np.meshgrid(*[np.arange(x) for x in grid2_shp[2:]], indexing='ij')]
        )

        self.grid_base_1 = theano.shared(base_grid_1)
        self.grid_base_2 = theano.shared(base_grid_2)

    def get_output_shape_for(self, input_shapes):
        shape = list(input_shapes[1])
        shape[1] = 3
        return shape

    def get_output_for(self, inputs, **kwargs):
        grid1, grid2 = inputs
        grid1 = grid1 + self.grid_base_1
        X = grid1[0, 0]
        Y = grid1[0, 1]
        Z = grid1[0, 2]
        dX = resample(X, self.grid_base_2 + grid2).dimshuffle('x', 'x', 0, 1, 2)
        dY = resample(Y, self.grid_base_2 + grid2).dimshuffle('x', 'x', 0, 1, 2)
        dZ = resample(Z, self.grid_base_2 + grid2).dimshuffle('x', 'x', 0, 1, 2)
        return T.concatenate([dX, dY, dZ], axis=1) - self.grid_base_1


def resample(image, grid):
    X, Y, Z = grid[0, 0], grid[0, 1], grid[0, 2]
    X0 = T.cast(T.floor(X), 'int64')
    Y0 = T.cast(T.floor(Y), 'int64')
    Z0 = T.cast(T.floor(Z), 'int64')
    X1 = X0 + 1
    Y1 = Y0 + 1
    Z1 = Z0 + 1

    X0 = T.clip(X0, 0, image.shape[0] - 1)
    X1 = T.clip(X1, 0, image.shape[0] - 1)
    Y0 = T.clip(Y0, 0, image.shape[1] - 1)
    Y1 = T.clip(Y1, 0, image.shape[1] - 1)
    Z0 = T.clip(Z0, 0, image.shape[2] - 1)
    Z1 = T.clip(Z1, 0, image.shape[2] - 1)

    b_x0y0z0 = image[X0, Y0, Z0]
    b_x1y0z0 = image[X1, Y0, Z0]
    b_x0y1z0 = image[X0, Y1, Z0]
    b_x1y1z0 = image[X1, Y1, Z0]
    b_x0y0z1 = image[X0, Y0, Z1]
    b_x1y0z1 = image[X1, Y0, Z1]
    b_x0y1z1 = image[X0, Y1, Z1]
    b_x1y1z1 = image[X1, Y1, Z1]

    b = (X1 - X) * (Y1 - Y) * (Z1 - Z) * b_x0y0z0 + \
        (X - X0) * (Y1 - Y) * (Z1 - Z) * b_x1y0z0 + \
        (X1 - X) * (Y - Y0) * (Z1 - Z) * b_x0y1z0 + \
        (X - X0) * (Y - Y0) * (Z1 - Z) * b_x1y1z0 + \
        (X1 - X) * (Y1 - Y) * (Z - Z0) * b_x0y0z1 + \
        (X - X0) * (Y1 - Y) * (Z - Z0) * b_x1y0z1 + \
        (X1 - X) * (Y - Y0) * (Z - Z0) * b_x0y1z1 + \
        (X - X0) * (Y - Y0) * (Z - Z0) * b_x1y1z1
    return b
