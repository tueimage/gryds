# Spatial transformations for augmentations in deep learning

This package enables you to make fast spatial transformations for the purpose of data augmentation in deep learning. The supported spatial transformations are

* `TranslationTransform`
* `LinearTransform`, supporting
    * Rigid transformations (translation + rotation)
    * Similarity transformations (translation + rotation + isotropic scaling)
    * Affine transformations (translation + rotation + arbitrary scaling + shearing)
* `BSplineTransform`: deformable transformations for image warping

which can be applied to `Interpolator` objects that wrap an image, and automatically perform B-spline interpolation. The package has been designed such that images of arbitrary dimensions (up from 2D) can be used, but it has only been extensively tested on 2D and 3D images.

### A minimal working example for randomly warping an image

Assuming you have a 2D image in the `image` variable:

```python
import numpy as np
import spatial_transformations as tr

# Define a random 3x3 B-spline grid for a 2D image:
random_grid = np.random.rand(2, 3, 3)
random_grid -= 0.5
random_grid /= 5

# Define a B-spline transformation object
bspline = tr.BSplineTransformation(random_grid)

# Define an interpolator object for the image:
interpolator = tr.Interpolator(image)

# Transform the image using the B-spline transformation
transformed_image = interpolator.transform(bspline)
```

### Making many random transformations

```
import matplotlib.pyplot as plt
import numpy as np
import spatial_transformations as tr

fig, ax = plt.subplots(5, 5, figsize=(15, 15));
ax = ax.flatten()
[x.set_axis_off() for x in ax];

for i in range(25):
    random_grid = np.random.rand(2, 3, 3) # Make a random 2D 3 x 3 grid
    random_grid -= 0.5 # Move the displacements to the -0.5 to 0.5 grid
    random_grid /= 5 # Scale the grid to -0.1 to 0.1 displacements

    my_augmentation = tr.BSplineTransformation(random_grid)
    my_augmented_image = my_image_interpolator.transform(my_augmentation)

    ax[i].imshow(my_augmented_image)
    plt.show()
```

![](examples.png)
