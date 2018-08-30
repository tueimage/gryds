#! /usr/bin/env python
#
# Transformations of points and grids of points
#
# @author: Koen Eppenhof
# @email: k.a.j.eppenhof@tue.nl
# @date: 2017/08/17


from __future__ import division, print_function

import numpy as np
import scipy.ndimage as nd

# This package assumes 32-bit floats for images and transformed grids everywhere
DTYPE = 'float32'


class Transform(object):
    """Base class for transformations, i.e. maps from *points* in one spatial
    domain to points in another spatial domain. This base class enforces checks
    of the number of points against the number of dimensions of the transform,
    i.e. an error is thrown if self.ndim does not match the number of dimensions
    in the points.

    All transforms are applied to relative coordinates, i.e. coordinates in the
    [0, 1)^ndim domain. The base class scales the points if necessary to this
    domain using the scale parameter of the transform function supplied by the
    user.
    """

    def __init__(self, ndim, parameters):
        """
        Args:
            ndim (int): Number of dimensions of the transform, used for
                checking the dimensions of points to be transformed.
        """
        self.ndim = ndim
        self.parameters = parameters

    def _dimension_check(self, points):
        """Checks if the points are compatible with the number of dimensions
        in the transform.

        Args:
            points (np.array): A (self.ndim x N) array of N points.
            scale (np.array): An array of self.ndim scaling factors.
        Raises:
            ValueError: If the points and self.ndim are not compatible
        """
        if points.ndim != 2:
            raise ValueError(
                'Points should be expressed as an (ndim x N) matrix.')
        if self.ndim != points.shape[0]:
            raise ValueError(
                'Dimensions not compatible: {}D point cannot be transformed'
                ' by {}D transformer.'.format(points.shape[0], self.ndim))

    def _transform_points(self, points):
        """Template for transformer function."""
        raise NotImplementedError

    def transform(self, points, scale=None):
        """Calling the _transform_points function with dimension checks.

        Args:
            points (np.array): A (self.ndim x N) array of N points.
            scale (np.array): An array of self.ndim scaling factors.
        Returns:
            (np.array): The (self.ndim x N) array of N transformed points.
        Raises:
            ValueError: If the points and self.ndim are not compatible.
        """
        points = np.array(points)

        if not scale:
            scale = [1]

        scale = np.array(scale).astype(DTYPE)[:, None]
        scaled_points = points / scale

        self._dimension_check(scaled_points)
        return self._transform_points(scaled_points) * scale

    def __call__(self, points, scale=None):
        """Calling the transform as a function invokes the transform function.

        Args:
            points (np.array): A (self.ndim x N) array of N points.
            scale (np.array): An array of self.ndim scaling factors.
        Returns:
            (np.array): The (self.ndim x N) array of N transformed points.
        Raises:
            ValueError: If the points and self.ndim are not compatible.
        """
        return self.transform(points, scale)


class ComposedTransform(Transform):
    """Composed transform that turns multiple Transform objects into a net
    tranform through composition.

    Given two transforms t1 and t2, and a points x0, the following result in
    the same transformed points x2:

    >>> # Manual composition
    >>> x1 = t1.transform(x0)
    >>> x2 = t2.transform(x1)

    >>> # With ComposedTransform
    >>> t12 = ComposedTransform(t1, t2)
    >>> x2 = t12.transform(x0)
    """

    def __init__(self, *transforms):
        ndims = [x.ndim for x in transforms]
        if not np.all(np.array(ndims) == ndims[0]):
            raise ValueError('Number of dimensions for transforms {} do not '
                             ' match: {}.'.format(', '.join(
                                 tuple([x.__class__.__name__ for x in transforms])), ndims))
        self.ndim = ndims[0]
        self.transforms = transforms

    def _transform_points(self, points):
        points_copy = points.copy()
        for transform in self.transforms:
            points_copy = transform.transform(points_copy)
        return points_copy


class TranslationTransform(Transform):
    """Translation of points."""

    def __init__(self, translation):
        """
        Args:
            translation (np.array): Translation vector.
        """
        super(TranslationTransform, self).__init__(
            ndim=len(translation),
            parameters=np.array(translation).astype(DTYPE)
        )

    def _transform_points(self, points):
        return (points + self.parameters[:, None])


class BSplineTransform(Transform):
    """BSpline transformation of points."""

    def __init__(self, grid, order=3):
        """
        Args:
            grid (np.array): An (ndim x N1 x N2 x ... Nndim) sized array of
                displacements for grid points.
            order (int): The B-spline order.
        Raises:
            ValueError: If grid.shape[0] is not equal to grid.ndim -1
        """
        grid = np.array(grid)
        if grid.shape[0] is not grid.ndim - 1:
            raise ValueError('First axis of grid should be equal to '
                             'transform\'s ndim {}.'.format(grid.ndim - 1))
        super(BSplineTransform, self).__init__(
            ndim=len(grid),
            parameters=grid.astype(DTYPE)
        )
        self.bspline_order = order

    def _transform_points(self, points):
        # Empty list for the interpolated B-spline grid's components.
        displacement = []

        # Points is in the [0, 1)^ndim domain. Here it is scaled to the
        # B-spline grid's size.
        scaled_points = points * (
            np.array(self.parameters.shape[1:]) - 1)[:, None]

        # Every component (e.g. Tx, Ty, Tz in 3D) of the B-spline grid is
        # interpolated at the scaled point's positions.
        for bspline_component in self.parameters:
            displacement.append(
                nd.map_coordinates(bspline_component, scaled_points,
                                   order=self.bspline_order)
            )
        return (points + np.array(displacement))


class LinearTransform(Transform):
    """Linear transformation for 2D or 3D augmented coordinates."""

    def __init__(self, matrix):
        """
        Args:
            matrix (np.array): An (ndim ) x (ndim + 1) array
                representing the augmented affine matrix, where ndim is either
                2 or 3.

        Raises:
            ValueError: If the matrix is not shaped correctly.
        """
        matrix = np.array(matrix).astype(DTYPE)

        if matrix.shape[0] != matrix.shape[1] - 1:
            raise ValueError(
                'Incorrect matrix shape, should be (ndim) x (ndim + 1),'
                'is {}.'.format(matrix.shape))

        super(LinearTransform, self).__init__(len(matrix), matrix)

    def _transform_points(self, points):
        augmented_points = np.ones((self.ndim + 1, points.shape[1]))
        augmented_points[:self.ndim] = points
        transformed_points = np.dot(self.parameters, augmented_points)
        return transformed_points[:self.ndim, :]


def affine_matrix(center=None, shear_matrix=None, scaling=None,
                  angles=None, translation=None):
    """
    Args:
        shear_matrix (np.array): An (ndim x ndim) matrix with shear
            components.
        scaling (np.array): An (ndim) length array of scaling factors.
        angles (np.array): A size 1 or 3 array (for 2D and 3D transforms
            respectively) of angles in radians.
        translation (np.array): The (ndim) array of translation.
        center (np.array): The (ndim) array of the center of rotation in
            relative coordinates (i.e. in the [0, 1)^ndim domain.

    Raises:
        ValueError: If the number of angles is not 1 or 3.
        ValueError: If the number of elements in the shear_matrix, scaling,
            angles, and translation array do not match ndim.
    """
    if center is not None:
        ndim = len(center)
    elif shear_matrix is not None:
        ndim = len(shear_matrix)
    elif scaling is not None:
        ndim = len(scaling)
    elif translation is not None:
        ndim = len(translation)
    elif angles is not None:
        ndim = 3 if len(angles) == 3 else 2
    else:
        raise ValueError('No parameters supplied.')

    if angles is not None:
        if len(angles) == 1 and ndim == 2:
            rotation_matrix = rotation_matrix_2d(*angles)
        elif len(angles) == 3 and ndim == 3:
            rotation_matrix = rotation_matrix_3d(*angles)
        else:
            raise ValueError(
                'Number of angles ({}) not '
                'supported.'.format(len(angles)))
    else:
        rotation_matrix = np.eye(ndim)

    if shear_matrix is not None:
        shear_matrix = np.array(shear_matrix).astype(DTYPE)
        if shear_matrix.shape != (ndim, ndim):
            raise ValueError(
                'Number of dimensions in the shear matrix {} does not match '
                'ndim {}'.format(shear_matrix.shape, ndim))

        shear_det = np.linalg.det(shear_matrix)
        if shear_det != 1:
            print('WARNING: Shear matrix has a scale component. '
                  'Determinant not equal to 1, but {}.'.format(shear_det))
    else:
        shear_matrix = np.eye(ndim)

    if scaling is not None:
        scaling = np.array(scaling).astype(DTYPE)
        if len(scaling) != ndim:
            raise ValueError(
                'Number of dimensions in the scaling array {} does not match '
                'ndim {}'.format(len(scaling), ndim))
    else:
        scaling = np.ones(ndim)
    scaling_matrix = np.diag(scaling)

    if translation is not None:
        translation = np.array(translation).astype(DTYPE)
        if len(translation) != ndim:
            raise ValueError(
                'Number of dimensions in the translation array {} does not '
                'match ndim {}'.format(len(translation), ndim))
    else:
        translation = np.zeros(ndim)

    pre_translation = np.eye(ndim + 1)
    if center is not None:
        center = np.array(center)
        translation += center
        pre_translation[:ndim, -1] = -center

    transform_matrix = np.zeros((ndim, ndim + 1))
    transform_matrix[:ndim, :ndim] = np.eye(ndim)
    mat = np.dot(rotation_matrix, np.dot(shear_matrix, scaling_matrix))

    transform_matrix[:, :-1] = mat
    transform_matrix[:, -1] = translation

    matrix = np.dot(transform_matrix, pre_translation)
    return matrix.astype('float32')


def rotation_matrix_2d(theta):
    """2D rotation matrix for a single rotation angle theta."""
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])


def rotation_matrix_3d(alpha, beta, gamma):
    """3D rotation matrix for three rotation angles alpha, beta, gamma."""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])
    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])
    return np.dot(np.dot(Rx, Ry), Rz)
