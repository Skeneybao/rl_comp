import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

# Define an Extension object with desired compile options
ext_modules = [
    Extension(
        "training.util.sumtree",
        ["training/util/sumtree.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=["-O3"]
    )
]

setup(
    name='SumTree Module',
    ext_modules=cythonize(ext_modules),
    zip_safe=False,
)
