from distutils.core import setup
from distutils.extension import Extension
import numpy as np
import os
from Cython.Build import cythonize

sourcefiles = ['./gmmmc/fastgmm/fast_likelihood.pyx']
ext_modules = [Extension("fast_likelihood",
                          sourcefiles,
                          include_dirs = [np.get_include()],
                          extra_compile_args=['-O3', '-fopenmp', '-lc++'],
                          extra_link_args=['-fopenmp'],
                          language='c++')]
# ,'./gmmmc/fastgmm/fast_likelihood_threaded.cpp'
setup(
    name='gmmmc',
    version='0.2',
    packages=['gmmmc', 'gmmmc.tests', 'gmmmc.priors', 'gmmmc.fastgmm', 'gmmmc.proposals'],
    url='',
    license='',
    author='Jeremy Ma',
    author_email='jeremy.ma@student.unsw.edu.au',
    description='Functions for drawing Monte Carlo samples from GMM parameter space',
    ext_modules= cythonize(ext_modules)
)
