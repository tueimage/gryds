# Profiling of B-spline code

The code in this folder profiles third-order B-spline interpolation and transformation.

Currently, the profiling tool has the following output:

```python  
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    1    0.076    0.076   29.532   29.532 bspline.py:101(transform)
    4    0.000    0.000   27.591    6.898 interpolation.py:266(map_coordinates)
    4   26.314    6.578   26.314    6.578 {built-in method scipy.ndimage._nd_image.geometric_transform}
    1    0.000    0.000   24.299   24.299 grid.py:69(transform)
    1    0.064    0.064   24.066   24.066 base.py:88(__call__)
    1    0.429    0.429   24.002   24.002 base.py:64(transform)
    1    0.204    0.204   23.256   23.256 bspline.py:60(_transform_points)
    1    0.011    0.011    5.144    5.144 bspline.py:76(resample)
    1    0.010    0.010    4.794    4.794 bspline.py:48(sample)
    4    0.000    0.000    1.277    0.319 interpolation.py:108(spline_filter)
   12    0.000    0.000    1.272    0.106 interpolation.py:54(spline_filter1d)
   12    1.272    0.106    1.272    0.106 {built-in method scipy.ndimage._nd_image.spline_filter1d}
    7    0.482    0.069    0.482    0.069 {method 'astype' of 'numpy.ndarray' objects}
   30    0.457    0.015    0.457    0.015 {built-in method numpy.core.multiarray.array}
    1    0.039    0.039    0.327    0.327 grid.py:43(scaled_to)
    2    0.000    0.000    0.231    0.116 grid.py:19(__init__)
    1    0.120    0.120    0.120    0.120 {method 'copy' of 'numpy.ndarray' objects}
    1    0.049    0.049    0.049    0.049 grid.py:65(<listcomp>)
   20    0.000    0.000    0.005    0.000 _ni_support.py:71(_get_output)
   24    0.005    0.000    0.005    0.000 {built-in method numpy.core.multiarray.zeros}
   24    0.000    0.000    0.000    0.000 type_check.py:250(iscomplexobj)
   24    0.000    0.000    0.000    0.000 numeric.py:433(asarray)
    2    0.000    0.000    0.000    0.000 {method 'reshape' of 'numpy.ndarray' objects}
   24    0.000    0.000    0.000    0.000 {built-in method builtins.issubclass}
   16    0.000    0.000    0.000    0.000 _ni_support.py:38(_extend_mode_to_code)
   12    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
    1    0.000    0.000    0.000    0.000 base.py:42(_dimension_check)
   12    0.000    0.000    0.000    0.000 _ni_support.py:86(_check_axis)
    3    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
    6    0.000    0.000    0.000    0.000 {built-in method builtins.len}
    1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```

Because B-splines are used for the transformation and the interpolation, there are four calls (three dimensions + interpolation) to `scipy.ndimage.map_coordinates`, which takes up about 93% of the time required for the full script. Accelerating this function would therefore be the best way to improve speed, however, it is already very much optimized: `map_coordinates` calls compiled C-code.
