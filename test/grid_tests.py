from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
import os, sys
currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import spatial_transformations as tr


def test_bspline_on_grid():
    bsp_grid = np.zeros((3, 127, 127, 127))
    bsp_grid[0, 63:64, 63:64, 63:64] = 1/126.
    trf = tr.BSplineTransformation(bsp_grid, order=1)
    im = np.zeros((127, 127, 127))
    im[63, 63, 63] = 1.
    trf_im = tr.Interpolator(im, order=3).transform(trf)
    print(trf_im[60:65, 60:65, 60:65])
    # assert np.all(trf_im[62, 62, 62] == 1.)


N = 10
M = 3
def test_rotation():
    im = np.zeros((N, N))
    im[M:-M] = 1
    im[:, M:-M] = 1
    im[0] = 0
    im[-1] = 0
    im[:, 0] = 0
    im[:, -1] = 0
    # im[::4] = 0
    # im[:, ::4] = 0
    trf = tr.AffineTransformation(
        ndim=2,
        angles=[np.pi / 4.],
        center_of=im)
    trf_im = tr.Interpolator(im, order=3).transform(trf, trf, trf, trf)
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(im, vmin=0, vmax=1)
    ax[1].imshow(trf_im, vmin=0, vmax=1)
    ax[2].imshow(trf_im - im)#, vmin=-1, vmax=1)
    ax[3].hist(trf_im - im)
    plt.tight_layout()
    plt.show()


def test_translation():
    im = np.zeros((N, N))
    im[M:-M] = 1
    im[:, M:-M] = 1
    im[0] = 0
    im[-1] = 0
    im[:, 0] = 0
    im[:, -1] = 0
    # im[::4] = 0
    # im[:, ::4] = 0w
    trf = tr.TranslationTransformation([0.5, 0])
    trf_im = tr.Interpolator(im, order=1).transform(trf)
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(im, vmin=0, vmax=1)
    ax[1].imshow(trf_im, vmin=0, vmax=1)
    ax[2].imshow(trf_im - im)#, vmin=-1, vmax=1)
    ax[3].hist(trf_im - im)
    plt.tight_layout()
    plt.show()


def test_bspline():
    im = np.zeros((N, N))
    im[M:-M] = 1
    im[:, M:-M] = 1
    im[0] = 0
    im[-1] = 0
    im[:, 0] = 0
    im[:, -1] = 0
    # im[::4] = 0
    # im[:, ::4] = 0
    grid = np.zeros((2, 3, 3))
    grid[0, :2] = 0.5
    # grid[0, 2] = 0.1
    # grid[0, 3] = 0
    # grid[0, 4] = 0.1
    # grid[0, 5] = 0
    # grid[0, 4] = 0.1
    trf = tr.BSplineTransformation(grid, order=0, mode='constant' )
    sampling_grid = tr.Grid(im.shape).transform(trf)
    print(sampling_grid.grid - tr.Grid(im.shape).grid)

    trf_im = tr.Interpolator(im, order=1).transform(trf)
    trf_trs = tr.TranslationTransformation([0.5, 0])
    trf_trs_im = tr.Interpolator(im, order=3).transform(trf_trs)
    fig, ax = plt.subplots(1, 5)
    ax[0].imshow(im, vmin=0, vmax=1)
    ax[1].imshow(trf_im, vmin=0, vmax=1)
    ax[2].imshow(trf_trs_im, vmin=0, vmax=1)
    ax[3].imshow(trf_im - trf_trs_im)#, vmin=-1, vmax=1)
    ax[4].hist(trf_im - im)
    plt.tight_layout()
    plt.show()


def test_bspline2():
    im = np.zeros((N, N))
    im[M:-M] = 1
    im[:, M:-M] = 1
    im[0] = 0
    im[-1] = 0
    im[:, 0] = 0
    im[:, -1] = 0
    # im[::4] = 0
    # im[:, ::4] = 0
    grid = np.zeros((2, N + 1, N + 1))
    grid[0, 8, 8] = 0.5
    trf = tr.BSplineTransformation(grid, order=0)
    sampling_grid = tr.Grid(im.shape).transform(trf)
    print(sampling_grid.grid - tr.Grid(im.shape).grid)
    # trf_im = tr.Interpolator(im, order=1).transform(trf)
    # trf_trs = tr.TranslationTransformation([0.5, 0])
    # trf_trs_im = tr.Interpolator(im, order=3).transform(trf_trs)
    # fig, ax = plt.subplots(1, 5)
    # ax[0].imshow(im, vmin=0, vmax=1)
    # ax[1].imshow(trf_im, vmin=0, vmax=1)
    # ax[2].imshow(trf_trs_im, vmin=0, vmax=1)
    # ax[3].imshow(trf_im - trf_trs_im)#, vmin=-1, vmax=1)
    # ax[4].hist(trf_im - im)
    # plt.tight_layout()
    # plt.show()


def test_bspline_scaled_similarity():
    im = np.zeros((N, N))
    grid = np.zeros((2, N, N))
    grid[0, 0] = 0.1
    grid[0, 1] = 0.0
    grid[0, 2] = 0.2
    grid[0, 3] = 0.0
    grid[0, 4] = 0.1
    grid[0, 5] = 0.2
    trf = tr.BSplineTransformation(grid, order=1, mode='constant')
    dfm = tr.Grid(im.shape).transform(trf).grid
    # dsp = np.zeros([2] + [x + 1 for x in im.shape])
    # dsp[:, :-1, :-1] = dfm - tr.Grid(im.shape).grid
    dsp = dfm - tr.Grid(im.shape).grid
    print(dsp) # - tr.Grid(im.shape).grid

    trf2 = tr.BSplineTransformation.from_deformation_field(dsp, order=1, mode='constant')
    dfm2 = tr.Grid(im.shape).transform(trf2).grid
    dsp2 = dfm2 - tr.Grid(im.shape).grid
    print(dsp2)


if __name__ == '__main__':
    
    # test_rotation()
    test_bspline()
    test_bspline2()

    test_translation()
    test_rotation()

    test_bspline_scaled_similarity()
