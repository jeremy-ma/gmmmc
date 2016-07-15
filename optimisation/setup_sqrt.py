from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy as np

os.environ["CC"] = "clang-omp"
os.environ["CXX"] = "clang-omp"
setup(
  name = "sqrt",
  ext_modules=[ Extension("sqrt", ["wrapper.pyx", "sqrt.c"],
                          include_dirs = [np.get_include()],
                          extra_compile_args=["-O3", "-ffast-math", "-fopenmp"],
                          extra_link_args=["-fopenmp", "-lm"])],
  cmdclass = {'build_ext': build_ext}
)