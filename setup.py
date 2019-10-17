from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

# Is this still the recommended method? Or should it use setuptools?
path = os.path.dirname(os.path.realpath(__file__))
extension = Extension("dtw",
                      [os.path.join(path, "input_reps/dtw.pyx")],
                      extra_compile_args=["-O3"],
                      include_dirs=[numpy.get_include()])
setup(name="DTW implementation in cython", ext_modules=cythonize(extension))
