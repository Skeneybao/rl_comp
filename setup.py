import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

# Define an Extension object with desired compile options
ext_modules = [
    Extension(
        "training.util.sumtree",
        ["training/util/sumtree.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='SumTree Module',
    ext_modules=cythonize(ext_modules),
    zip_safe=False,
)
