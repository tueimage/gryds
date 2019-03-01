import setuptools


setuptools.setup(
    name="gryds",
    version="0.0.1",
    author="Koen A. J. Eppenhof",
    author_email="k.a.j.eppenhof@tue.nl",
    description="Gryds: a Python package for geometric transformations of images for data augmentation in deep learning",
    long_description="""
Gryds enables you to make fast geometric transformations of images for the purpose of data augmentation in deep learning.
The supported geometric transformations include translations, rigid transformations (translation + rotation),
similarity transformations (translation + rotation + isotropic scaling),
affine transformations (translation + rotation + arbitrary scaling + shearing),
and deformable transformations (modeled as B-splines). It is also possible to apply the transformations to
coordinates in the image domain, and to inspect the deformation vector field.""",
    long_description_content_type="text/markdown",
    url="https://github.com/tueimage/gryds",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
