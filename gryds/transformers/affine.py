#! /usr/bin/env python
#
# Affine transformation
# For the sake of simplicity, the ND generalization does not exist for this
# transformation, as it severely complicates the implementation.
#
# However, 4D affine transformations can be implemented by subclassing
# LinearTransformation, or simply defining a LinearTransformation with the
# appropriate transformation matrix.


from __future__ import division, print_function, absolute_import

import numpy as np
from ..config import DTYPE
from .linear import LinearTransformation


class AffineTransformation(LinearTransformation):
    """Affine transformation for 2D or 3D augmented coordinates. Subclasses
    LinearTransformation, as this is merely a filling in of the linear
    transformation's matrix instance variable.

    Attributes:
        ndim (int): The number of dimensions.
        parameters (np.ndarray): An (ndim ) x (ndim + 1) array
            representing the augmented affine matrix, where ndim is either
            2 or 3.
    """

    def __init__(self, ndim, center=None, center_of=None, scaling=None,
                 angles=None, translation=None, shear_matrix=None):
        """
        Given a shear matrix G, a center c, a scaling s, angles a, and
        translation t computes on a point x:

            R * G * S * (x - c) + c

        where S = s * np.eye(ndim) and R = rotation_matrix_nd(a)

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
        if center_of is not None:
            center = _center_of(center_of)
        matrix = _affine_matrix(
            ndim=ndim, center=center, scaling=scaling, angles=angles,
            translation=translation, shear_matrix=shear_matrix
        )
        super(AffineTransformation, self).__init__(matrix)

    def _transform_points(self, points):
        augmented_points = np.ones((self.ndim + 1, points.shape[1]))
        augmented_points[:self.ndim] = points
        transformed_points = np.dot(self.parameters, augmented_points)
        return transformed_points[:self.ndim, :]


def _center_of(image):
    """Returns the center coordinate of an image, i.e. (shape - 1) / 2."""
    return [(x - 1) / (2. * x) for x in image.shape]


def _affine_matrix(ndim, center=None, shear_matrix=None, scaling=None,
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
    Warnings:
        When shear_matrix contains a scaling components (i.e. determinant != 0).
    """
    if angles is not None:
        angles = np.array(angles, dtype=DTYPE)
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
        shear_matrix = np.array(shear_matrix, dtype=DTYPE)
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
        scaling = np.array(scaling, dtype=DTYPE)
        if len(scaling) != ndim:
            raise ValueError(
                'Number of dimensions in the scaling array {} does not match '
                'ndim {}'.format(len(scaling), ndim))
    else:
        scaling = np.ones(ndim)
    scaling_matrix = np.diag(scaling)

    if translation is not None:
        translation = np.array(translation, dtype=DTYPE)
        if len(translation) != ndim:
            raise ValueError(
                'Number of dimensions in the translation array {} does not '
                'match ndim {}'.format(len(translation), ndim))
    else:
        translation = np.zeros(ndim)

    pre_translation = np.eye(ndim + 1, dtype=DTYPE)
    if center is not None:
        center = np.array(center, dtype=DTYPE)
        translation += center
        pre_translation[:ndim, -1] = -center

    transform_matrix = np.zeros((ndim, ndim + 1), dtype=DTYPE)
    transform_matrix[:ndim, :ndim] = np.eye(ndim, dtype=DTYPE)
    mat = np.dot(rotation_matrix, np.dot(shear_matrix, scaling_matrix))

    transform_matrix[:, :-1] = mat
    transform_matrix[:, -1] = translation

    matrix = np.dot(transform_matrix, pre_translation)
    return matrix.astype(DTYPE)


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
