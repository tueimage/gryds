import os
import sys
sys.path.append(os.path.abspath('..'))
import gryds
import numpy as np
from cProfile import Profile
from pstats import Stats
import time
import matplotlib.pyplot as plt
import seaborn as sns


# bsp = gryds.BSplineTransformation(np.random.rand(3, 32, 32, 32), order=1)
bsp = gryds.TranslationTransformation([0.1, 0.2, 0.3])
N = 1

Ns = range(0, 151, 1)
M = 10

image = np.random.rand(N, 128, 128)
intp = gryds.BSplineInterpolatorCuda(image)
intp.transform(bsp)

times = []
for i in range(M):
    ts = []
    for N in Ns:
        print(i, N)
        image = np.random.rand(N, 128, 128)
        intp = gryds.Interpolator(image, order=1)
        t0 = time.time()
        intp.transform(bsp)
        ts.append(time.time() - t0)
    times.append(ts)
times = np.median(times, axis=0)

times_cuda = []
for i in range(M):
    ts = []
    for N in Ns:
        print(i, N)
        image = np.random.rand(N, 128, 128)
        intp = gryds.BSplineInterpolatorCuda(image, order=1)
        t0 = time.time()
        intp.transform(bsp)
        ts.append(time.time() - t0)
    times_cuda.append(ts)
times_cuda = np.median(times_cuda, axis=0)


plt.plot(Ns, times, '-')
plt.plot(Ns, times_cuda, '-')
plt.xlabel('Number of slices (volume = N x 128 x 128)')
plt.ylabel('Seconds')
plt.title('Time of interpolation as function of image size\nTiming is median of 10 trials')
plt.legend(['CPU', 'GPU'])
plt.grid()
plt.show()
