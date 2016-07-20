from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import os

#os.environ["CC"] = "clang-omp"
#os.environ["CXX"] = "clang-omp"

sourcefiles = [ 'fast_likelihood_caller.pyx', 'fast_likelihood_threaded.cpp']
ext_modules = [Extension("fast_likelihood",
                          sourcefiles,
                          include_dirs = [np.get_include()],
                          extra_compile_args=['-O3', '-fopenmp', '-lc++'],
                          extra_link_args=['-fopenmp'],
                          language='c++')]

setup(
  name = 'fastgmm',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)