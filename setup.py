import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spatial_transformations",
    version="0.0.1",
    author="Koen A. J. Eppenhof",
    author_email="k.a.j.eppenhof@tue.nl",
    description="Fast spatial transformations for data augmentation in deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tueimage/Spatial-transformations",
    packages=[
        numpy, scipy
    ]
    classifiers=[
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
