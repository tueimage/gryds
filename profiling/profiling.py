import os
import sys
sys.path.append(os.path.abspath('..'))
import gryds
import numpy as np
from cProfile import Profile
from pstats import Stats
import gryds.interpolators.cuda


# bsp = gryds.BSplineTransformation(np.random.rand(3, 32, 32, 32), order=1)
bsp = gryds.TranslationTransformation([0.1, 0.2, 0.3])
image = np.random.rand(128, 128, 128)

prf = Profile()
intp = gryds.BSplineInterpolator(image, order=1)
prf.runcall(intp.transform, bsp)

prf_cuda = Profile()
intp_cuda = gryds.interpolators.cuda.BSplineInterpolatorCuda(image, order=1)
prf_cuda.runcall(intp_cuda.transform, bsp)

stats = Stats(prf)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats()

stats_cuda = Stats(prf_cuda)
stats_cuda.strip_dirs()
stats_cuda.sort_stats('cumulative')
stats_cuda.print_stats()
