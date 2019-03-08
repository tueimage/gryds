#! /usr/bin/env python
#
# Utils file


import numpy as np


def dvf_opts(dvf):
    """plt.imshow kwargs to show a 2D deformation field as a normalized
    blue-white-red map.

    Example usage: plt.imshow(dvf, **dvf_opts(dvf))
    """
    return {
        'cmap': 'bwr',
        'vmin': -np.abs(max(dvf.min(), dvf.max())),
        'vmax': np.abs(max(dvf.min(), dvf.max()))
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
    scale = [0.5 * 1. / (4 * (x - 1)) for x in size[1:]]
    return unif(scale, size)


def unif(scale, size):
    """Returns a uniformly distributed grid of given size
    and displacement scale"""

    size = np.array([size]).flatten()
    # return scale * 2 * (np.random.rand(*size) - 0.5)
    return np.array([
        x * y for x, y in zip(scale, 2 * (np.random.rand(*size) - 0.5))])


def phantom_image(size, spacing=4, thickness=1, offset=0):
    """Returns a 3D grid image of given size, with certain grid spacing."""
    im = np.zeros(size)
    N = spacing

    for i in range(thickness):
        sl = slice(offset + i, None, N)
        for axis in range(im.ndim):
            id = [slice(None)] * im.ndim
            id[axis] = sl
            im[tuple(id)] = 1

    return im
