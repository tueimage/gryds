import os
import sys
sys.path.append(os.path.abspath('..'))
import gryds
import numpy as np
from cProfile import Profile
from pstats import Stats

prf = Profile()

bsp = gryds.BSplineTransformation(np.random.rand(3, 128, 128, 128), order=3)
intp = gryds.Interpolator(np.random.rand(256, 256, 256), order=3)
prf.runcall(intp.transform, bsp)

stats = Stats(prf)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats()
