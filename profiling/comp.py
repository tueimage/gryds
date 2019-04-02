import os
import sys
sys.path.append(os.path.abspath('..'))
import gryds
import numpy as np
from cProfile import Profile
from pstats import Stats
import time
from gryds.interpolators import cuda
import matplotlib.pyplot as plt
import seaborn as sns


bsp = gryds.BSplineTransformation(0.01 * (np.random.rand(2, 32, 32) - 0.5), order=1)
# bsp = gryds.TranslationTransformation([0.1, 0.3])

image = np.zeros((128, 128))
image[32:-32] = 0.5
image[:, 32:-32] += 0.5
intp_cpu = gryds.BSplineInterpolator(image, order=1).transform(bsp)
intp_gpu = cuda.BSplineInterpolatorCuda(image).transform(bsp)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(intp_cpu, vmin=0, vmax=1)
ax[1].imshow(intp_gpu, vmin=0, vmax=1)
ax[2].imshow(intp_cpu - intp_gpu, vmin=-1, vmax=1)
plt.show()


