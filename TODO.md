- [ ] Because of similarity, the cuda versions of the B-Spline interpolator and transformer should be refactored to be subclasses of their respective non-cuda equivalent
- [ ] The B-spline transformers (both gpu/cpu) should be implemented to have only one call to `map_coordinates` instead of one per dimension.