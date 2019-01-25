from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
import os, sys
currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import spatial_transformations as tr

im = np.zeros((200, 200))
for i in range(3):
    im[i::40] = 1
    im[:, i::40] = 1

bsp = tr.BSplineTransformation(0.4 * (np.random.rand(2, 2, 2) - 0.5),
    order=1, cval=0)
bil_im = tr.LinearInterpolator(im).transform(bsp, bsp)
bsp_im = tr.BSplineInterpolator(im).transform(bsp, bsp)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(bil_im, vmin=0, vmax=1)
ax[1].imshow(bsp_im, vmin=0, vmax=1)
ax[2].imshow(bsp_im - bil_im, vmin=-1, vmax=1)
plt.show()


im = np.zeros((200, 200, 200))
for i in range(3):
    im[i::40] = 1
    im[:, i::40] = 1
    im[:, :, i::40] = 1

bsp = tr.BSplineTransformation(0.4 * (np.random.rand(3, 2, 2, 2) - 0.5),
    order=1, cval=0)
tril_im = tr.LinearInterpolator(im).transform(bsp)
bsp_im = tr.BSplineInterpolator(im).transform(bsp)

print(np.abs(tril_im - bsp_im).max())

fig, ax = plt.subplots(1, 3)
ax[0].imshow(tril_im[100], vmin=0, vmax=1)
ax[1].imshow(bsp_im[100], vmin=0, vmax=1)
ax[2].imshow((bsp_im - tril_im)[100], vmin=-1, vmax=1)
plt.show()
