import os
import sys
sys.path.append(os.path.abspath('..'))
import gryds
import numpy as np
from cProfile import Profile
from pstats import Stats

prf = Profile()

bsp = gryds.BSplineTransformation(np.random.rand(3, 32, 32, 32), order=1)
intp = gryds.Interpolator(np.random.rand(64, 64, 64), order=1)
prf.runcall(intp.transform, bsp)

stats = Stats(prf)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats()
