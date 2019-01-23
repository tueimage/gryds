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
    "TransformerLayer3D",
]


class ResampleLayer(MergeLayer):
    def __init__(self, image, grid, **kwargs):
        super(ResampleLayer, self).__init__(
            [image, grid], **kwargs)

        input_shp, grid_shp = self.input_shapes

        if len(grid_shp) != 5 or grid_shp[1] != 3:
            raise ValueError("The input network must have a 5-dimensional "
                             "output shape: (batch_size, 3, "
                             "input_rows, input_columns, input_depths)")
        if len(input_shp) != 5:
            raise ValueError("The input network must have a 5-dimensional "
                             "output shape: (batch_size, num_input_channels, "
                             "input_rows, input_columns, input_depths)")
        base_grid = np.array(
            [np.meshgrid(*[np.arange(x) for x in input_shp[2:]], indexing='ij')]
        )
        print(base_grid.shape)
        self.grid_base = theano.shared(base_grid)

    def get_output_shape_for(self, input_shapes):
        shape = list(input_shapes[1])
        shape[1] = 1
        return shape

    def get_output_for(self, inputs, **kwargs):
        image, grid = inputs
        return resample(image, self.grid_base + grid)


def resample(image, grid):

    X, Y, Z = grid[0, 0], grid[0, 1], grid[0, 2]
    X0 = T.cast(T.floor(X), 'int64')
    Y0 = T.cast(T.floor(Y), 'int64')
    Z0 = T.cast(T.floor(Z), 'int64')
    X1 = X0 + 1
    Y1 = Y0 + 1
    Z1 = Z0 + 1

    X0 = T.clip(X0, 0, image.shape[2] - 1)
    X1 = T.clip(X1, 0, image.shape[2] - 1)
    Y0 = T.clip(Y0, 0, image.shape[3] - 1)
    Y1 = T.clip(Y1, 0, image.shape[3] - 1)
    Z0 = T.clip(Z0, 0, image.shape[4] - 1)
    Z1 = T.clip(Z1, 0, image.shape[4] - 1)

    b_x0y0z0 = image[0, 0, X0, Y0, Z0]
    b_x1y0z0 = image[0, 0, X1, Y0, Z0]
    b_x0y1z0 = image[0, 0, X0, Y1, Z0]
    b_x1y1z0 = image[0, 0, X1, Y1, Z0]
    b_x0y0z1 = image[0, 0, X0, Y0, Z1]
    b_x1y0z1 = image[0, 0, X1, Y0, Z1]
    b_x0y1z1 = image[0, 0, X0, Y1, Z1]
    b_x1y1z1 = image[0, 0, X1, Y1, Z1]

    b = (X1 - X) * (Y1 - Y) * (Z1 - Z) * b_x0y0z0 + \
    (X - X0) * (Y1 - Y) * (Z1 - Z) * b_x1y0z0 + \
    (X1 - X) * (Y - Y0) * (Z1 - Z) * b_x0y1z0 + \
    (X - X0) * (Y - Y0) * (Z1 - Z) * b_x1y1z0 + \
    (X1 - X) * (Y1 - Y) * (Z - Z0) * b_x0y0z1 + \
    (X - X0) * (Y1 - Y) * (Z - Z0) * b_x1y0z1 + \
    (X1 - X) * (Y - Y0) * (Z - Z0) * b_x0y1z1 + \
    (X - X0) * (Y - Y0) * (Z - Z0) * b_x1y1z1
    return b
