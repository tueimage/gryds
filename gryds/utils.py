#! /usr/bin/env python
#
# Utils file
#
# @author: Koen Eppenhof
# @email: k.a.j.eppenhof@tue.nl
# @date: 2018/08/30


import numpy as np
DTYPE = numpy.float64


def dvf_opts(dvf):
    """plt.imshow kwargs to show a 2D deformation field as a normalized
    blue-white-red map.

    Example usage: plt.imshow(dvf, **dvf_opts(dvf))
    """
    return {
        'cmap': 'bwr',
        'vmin': -max(dvf.min(), dvf.max()),
        'vmax': max(dvf.min(), dvf.max())
    }


def dvf_show(dvf):
    """Plot a 2D deformation field as a normalized blue-white-red map.

    Example usage: plt.imshow(**dvf_show(dvf))
    """
    return {
        'X': dvf,
        'cmap': 'bwr',
        'vmin': -max(dvf.min(), dvf.max()),
        'vmax': max(dvf.min(), dvf.max())
    }


def max_no_fold(size):
    """Find a B-spline grid with maximal range without folding."""
    scale = 0.5 * 1. / (4 * (size[-1] - 1))
    return scale * 2 * (np.random.rand(*size) - 0.5)


def unif(scale, size):
    """Returns a uniformly distributed grid of given size
    and displacement scale"""

    size = np.array([size]).flatten()
    return scale * 2 * (np.random.rand(*size) - 0.5)


def phantom_image(size, spacing=16, thickness=1, offset=0):
    """Returns a 3D grid image of given size, with certain grid spacing."""
    im = np.zeros(size)
    N = spacing

    for i in range(thickness):
        im[offset + i::N] = 1
        im[:, offset + i::N] = 1
        im[:, :, offset + i::N] = 1
    return im
